"""Build SQLite OCR index cho tập dữ liệu lớn.

Mặc định đọc ``data/ocr/<video_id>/<keyframe>.json``. Cũng hỗ trợ một
JSON/JSONL tổng hợp nếu mỗi record có ``video_id`` và một trong các trường
``keyframe_id``, ``keyframe_name``, ``frame_id`` hoặc ``frame_index``.
"""

import argparse
import json
import os
import sqlite3
import tempfile
from pathlib import Path

from ocr_retriever import OCR_MIN_CONFIDENCE, tokenize
from search_types import (
    ARTIFACTS_DIR,
    PROJECT_DIR,
    load_mapping,
    make_keyframe_id,
    mapping_sha256,
)
from stdio_setup import configure_stdio


configure_stdio()

DEFAULT_INPUT = PROJECT_DIR / "data" / "ocr"
DEFAULT_OUTPUT = ARTIFACTS_DIR / "ocr_index.sqlite3"
RECORD_LIST_KEYS = ("records", "items", "data", "keyframes", "results")


def extract_text(raw):
    """Chuẩn hóa các schema OCR phổ biến thành một chuỗi."""

    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        lines = raw
    elif isinstance(raw, dict):
        nested = raw.get("ocr")
        if nested is not None and not any(k in raw for k in ("text", "texts")):
            return extract_text(nested)
        lines = raw.get("texts")
        if lines is None:
            lines = raw.get("text")
        if lines is None:
            lines = raw.get("ocr_text")
        if lines is None:
            lines = raw.get("results")
        if lines is None and isinstance(raw.get("rec_texts"), list):
            scores = raw.get("rec_scores") or [None] * len(raw["rec_texts"])
            lines = [
                {"text": text, "confidence": score}
                for text, score in zip(raw["rec_texts"], scores)
            ]
        if lines is None:
            lines = []
    else:
        return ""

    if isinstance(lines, str):
        lines = [lines]

    parts = []
    for line in lines or []:
        if isinstance(line, dict):
            confidence = line.get("confidence", line.get("score"))
            if confidence is not None and float(confidence) < OCR_MIN_CONFIDENCE:
                continue
            value = line.get("text", line.get("value", ""))
        else:
            value = line
        if value:
            parts.append(str(value).strip())
    return " ".join(dict.fromkeys(filter(None, parts)))


def iter_records(path):
    """Yield (record, source_path); JSONL được đọc streaming."""

    paths = sorted(path.rglob("*.json")) + sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    for source in paths:
        if source.suffix.lower() == ".jsonl":
            with source.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, 1):
                    if line.strip():
                        yield json.loads(line), source, line_number
            continue

        raw = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for position, record in enumerate(raw, 1):
                yield record, source, position
            continue

        if isinstance(raw, dict):
            if any(
                key in raw
                for key in (
                    "video_id", "keyframe_id", "keyframe_name",
                    "frame_id", "frame_index", "text", "texts",
                    "ocr_text", "rec_texts",
                )
            ):
                yield raw, source, 1
                continue

            records = next(
                (
                    raw[key]
                    for key in RECORD_LIST_KEYS
                    if isinstance(raw.get(key), list)
                    and raw[key]
                    and isinstance(raw[key][0], dict)
                    and any(
                        locator in raw[key][0]
                        for locator in (
                            "video_id", "keyframe_id", "keyframe_name",
                            "frame_id", "frame_index",
                        )
                    )
                ),
                None,
            )
            if records is not None:
                for position, record in enumerate(records, 1):
                    yield record, source, position
            else:
                position = 0
                for video_or_key, value in raw.items():
                    if not isinstance(value, dict):
                        continue
                    # {"video/keyframe": {OCR...}}
                    if "/" in str(video_or_key):
                        video_id, keyframe_id = str(video_or_key).rsplit("/", 1)
                        record = dict(value)
                        record.setdefault("video_id", video_id)
                        record.setdefault("keyframe_id", keyframe_id)
                        position += 1
                        yield record, source, position
                        continue
                    # {"video_id": {"keyframe": {OCR...}, ...}}
                    for keyframe_id, ocr_value in value.items():
                        if not isinstance(ocr_value, (dict, list, str)):
                            continue
                        record = (
                            dict(ocr_value)
                            if isinstance(ocr_value, dict)
                            else {"texts": ocr_value}
                        )
                        record.setdefault("video_id", str(video_or_key))
                        record.setdefault("keyframe_id", str(keyframe_id))
                        position += 1
                        yield record, source, position


def mapping_lookups(mapping):
    by_video_key = {}
    by_keyframe_id = {}
    for row in mapping:
        video_id = str(row["video_id"])
        stem = Path(row["keyframe_name"]).stem
        by_video_key[(video_id, stem)] = int(row["vector_index"])
        by_video_key[(video_id, str(row["frame_index"]))] = int(row["vector_index"])
        identifiers = {
            row.get("keyframe_id"),
            make_keyframe_id(video_id, row["keyframe_name"]),
        }
        for identifier in filter(None, identifiers):
            by_keyframe_id[str(identifier)] = int(row["vector_index"])
    return by_video_key, by_keyframe_id


def resolve_vector(record, source, input_root, by_video_key, by_keyframe_id):
    if not isinstance(record, dict):
        return None

    identifier = record.get("keyframe_id")
    if identifier is not None and str(identifier) in by_keyframe_id:
        return by_keyframe_id[str(identifier)]

    candidates = []
    raw_video = record.get("video_id")
    raw_key = (
        record.get("keyframe_name")
        or record.get("keyframe_id")
        or record.get("frame_id")
        or record.get("frame_index")
    )
    if raw_video is not None and raw_key is not None:
        candidates.append((str(raw_video), Path(str(raw_key)).stem))

    if input_root.is_dir() and source.parent != input_root:
        candidates.append((source.parent.name, source.stem))

    for candidate in candidates:
        if candidate in by_video_key:
            return by_video_key[candidate]
    return None


def build(input_path, output_path):
    mapping = load_mapping()
    by_video_key, by_keyframe_id = mapping_lookups(mapping)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix="ocr_index_", suffix=".sqlite3", dir=output_path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    indexed = missing = empty = duplicate = 0

    try:
        connection = sqlite3.connect(temporary)
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE documents (
                vector_index INTEGER PRIMARY KEY,
                text TEXT NOT NULL
            );
            CREATE TABLE postings (
                token TEXT NOT NULL,
                vector_index INTEGER NOT NULL,
                PRIMARY KEY (token, vector_index)
            ) WITHOUT ROWID;
            """
        )

        for record, source, _ in iter_records(input_path):
            vector_index = resolve_vector(
                record, source, input_path, by_video_key, by_keyframe_id
            )
            if vector_index is None:
                missing += 1
                continue
            text = extract_text(record)
            tokens = tokenize(text)
            if not tokens:
                empty += 1
                continue

            cursor = connection.execute(
                "INSERT OR IGNORE INTO documents(vector_index, text) VALUES (?, ?)",
                (vector_index, text),
            )
            if cursor.rowcount == 0:
                duplicate += 1
                continue
            connection.executemany(
                "INSERT OR IGNORE INTO postings(token, vector_index) VALUES (?, ?)",
                ((token, vector_index) for token in tokens),
            )
            indexed += 1
            if indexed % 10000 == 0:
                connection.commit()
                print(f"Đã index {indexed:,} OCR keyframe...")

        connection.execute("CREATE INDEX postings_token_idx ON postings(token)")
        metadata = {
            "schema": "aic_ocr_sqlite_v1",
            "mapping_sha256": mapping_sha256(),
            "mapping_rows": str(len(mapping)),
            "indexed_documents": str(indexed),
            "min_confidence": str(OCR_MIN_CONFIDENCE),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
        connection.close()

        if indexed == 0:
            raise RuntimeError(
                "Không index được OCR nào. Kiểm tra schema và video/keyframe ID."
            )
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print("Tạo OCR index thành công:", output_path)
    print(
        f"  Indexed: {indexed:,}; rỗng: {empty:,}; "
        f"trùng: {duplicate:,}; không map được: {missing:,}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build persistent OCR SQLite index")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Không tìm thấy OCR input: {args.input}")
    build(args.input, args.output)


if __name__ == "__main__":
    main()
