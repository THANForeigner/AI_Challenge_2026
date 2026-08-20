"""
OCR retriever (C4): tìm keyframe theo chữ xuất hiện trong ảnh.

Dữ liệu mong đợi (do thành viên khác chạy OCR cung cấp — xem
PLACEHOLDERS.md, hiện là MÃ GIẢ nếu chưa có):
    data/ocr/<video_id>/<keyframe>.json
    {"texts": ["dòng chữ 1", "dòng chữ 2", ...]}
    (hoặc {"text": "toàn bộ chữ"})

Khi chưa có thư mục data/ocr/: available() = False và fusion bỏ qua
modality này, không làm search thất bại.
"""

import json
import re
import sqlite3
import unicodedata
from pathlib import Path

from search_types import (
    ARTIFACTS_DIR,
    PROJECT_DIR,
    load_mapping,
    mapping_sha256,
    result_from_row,
)
from object_retriever import STOPWORDS
from stdio_setup import configure_stdio


configure_stdio()


OCR_DIR = PROJECT_DIR / "data" / "ocr"
OCR_INDEX_PATH = ARTIFACTS_DIR / "ocr_index.sqlite3"
OCR_MIN_CONFIDENCE = 0.3


def tokenize(text):
    """Tách token Unicode, không làm vỡ tiếng Việt thành ký tự rời."""

    normalized = unicodedata.normalize("NFC", str(text).casefold())
    return set(re.findall(r"[^\W_]+", normalized, re.UNICODE))


class OcrRetriever:
    name = "ocr"

    def __init__(self):
        self._index = None
        self._mapping = None
        self._connection = None
        self._vector_by_keyframe = None

    def available(self):
        if OCR_INDEX_PATH.exists():
            return True
        return OCR_DIR.exists() and any(OCR_DIR.rglob("*.json"))

    def _open_sqlite_index(self):
        uri = f"file:{OCR_INDEX_PATH.resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("schema") != "aic_ocr_sqlite_v1":
            connection.close()
            raise RuntimeError(
                f"OCR index sai schema: {metadata.get('schema')!r}. "
                "Chạy lại scripts/build_ocr_index.py."
            )
        if metadata.get("mapping_sha256") != mapping_sha256():
            connection.close()
            raise RuntimeError(
                "OCR index không khớp clip_row_mapping.jsonl. "
                "Chạy lại scripts/build_ocr_index.py."
            )
        return connection

    def _ensure_loaded(self):
        if self._index is not None:
            return

        self._mapping = load_mapping()

        if OCR_INDEX_PATH.exists():
            self._connection = self._open_sqlite_index()
            self._index = {}
            self._vector_by_keyframe = {
                (row["video_id"], Path(row["keyframe_name"]).stem): int(
                    row["vector_index"]
                )
                for row in self._mapping
            }
            return

        self._index = {}

        for row in self._mapping:
            stem = Path(row["keyframe_name"]).stem
            json_path = OCR_DIR / row["video_id"] / f"{stem}.json"

            if not json_path.exists():
                continue

            raw = json.loads(json_path.read_text(encoding="utf-8"))

            if isinstance(raw, dict):
                lines = raw.get("texts") or raw.get("text") or []
            else:
                lines = raw

            if isinstance(lines, str):
                lines = [lines]

            text_parts = []

            for line in lines:
                if isinstance(line, dict):
                    confidence = line.get("confidence")

                    if (
                        confidence is not None
                        and float(confidence) < OCR_MIN_CONFIDENCE
                    ):
                        continue

                    value = line.get("text", "")
                else:
                    value = line

                if value:
                    text_parts.append(str(value))

            text = " ".join(text_parts)
            tokens = tokenize(text)

            if tokens:
                self._index[row["vector_index"]] = tokens

    def search(self, query, top_k=100):
        if not self.available():
            return None

        self._ensure_loaded()

        if self._connection is None and not self._index:
            return None

        query_tokens = tokenize(query) - STOPWORDS

        if not query_tokens:
            return None

        if self._connection is not None:
            placeholders = ",".join("?" for _ in query_tokens)
            sql = f"""
                SELECT vector_index, COUNT(*) AS overlap
                FROM postings
                WHERE token IN ({placeholders})
                GROUP BY vector_index
                ORDER BY overlap DESC, vector_index ASC
                LIMIT ?
            """
            rows = self._connection.execute(
                sql, (*sorted(query_tokens), int(top_k))
            ).fetchall()
            scored = [
                (int(vector_index), float(overlap) / len(query_tokens))
                for vector_index, overlap in rows
            ]
        else:
            scored = []

            for vector_index, tokens in self._index.items():
                overlap = len(query_tokens & tokens)

                if overlap == 0:
                    continue

                score = overlap / len(query_tokens)
                scored.append((vector_index, score))

        if not scored:
            return None

        scored.sort(key=lambda item: item[1], reverse=True)

        results = []

        for rank, (vector_index, score) in enumerate(
            scored[:top_k], start=1
        ):
            row = self._mapping[int(vector_index)]
            result = result_from_row(row, ocr_score=float(score))
            result.rank = rank
            results.append(result)

        return results

    def context_for_keyframes(self, video_id, keyframe_names):
        """Lấy OCR text cho Q&A, ưu tiên SQLite thay vì mở lại JSON."""

        if not self.available():
            return ""
        self._ensure_loaded()

        if self._connection is not None:
            vector_indices = [
                self._vector_by_keyframe.get((video_id, Path(name).stem))
                for name in keyframe_names
            ]
            vector_indices = [value for value in vector_indices if value is not None]
            if not vector_indices:
                return ""
            placeholders = ",".join("?" for _ in vector_indices)
            rows = self._connection.execute(
                f"SELECT vector_index, text FROM documents "
                f"WHERE vector_index IN ({placeholders})",
                vector_indices,
            ).fetchall()
            by_vector = {int(vector): text for vector, text in rows}
            texts = [by_vector[value] for value in vector_indices if value in by_vector]
            return " ".join(dict.fromkeys(filter(None, texts)))

        texts = []
        for keyframe_name in keyframe_names:
            json_path = OCR_DIR / video_id / f"{Path(keyframe_name).stem}.json"
            if not json_path.exists():
                continue
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            lines = raw.get("texts") or raw.get("text") or [] if isinstance(raw, dict) else raw
            if isinstance(lines, str):
                lines = [lines]
            for line in lines:
                if isinstance(line, dict):
                    confidence = line.get("confidence")
                    if confidence is not None and float(confidence) < OCR_MIN_CONFIDENCE:
                        continue
                    value = line.get("text", "")
                else:
                    value = line
                if value:
                    texts.append(str(value).strip())
        return " ".join(dict.fromkeys(filter(None, texts)))
