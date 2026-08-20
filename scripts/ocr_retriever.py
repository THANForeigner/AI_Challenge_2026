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
import unicodedata
from pathlib import Path

from search_types import PROJECT_DIR, load_mapping, result_from_row
from object_retriever import STOPWORDS
from stdio_setup import configure_stdio


configure_stdio()


OCR_DIR = PROJECT_DIR / "data" / "ocr"
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

    def available(self):
        return OCR_DIR.exists() and any(OCR_DIR.rglob("*.json"))

    def _ensure_loaded(self):
        if self._index is not None:
            return

        self._mapping = load_mapping()
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

        if not self._index:
            return None

        query_tokens = tokenize(query) - STOPWORDS

        if not query_tokens:
            return None

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
