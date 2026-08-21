"""Dựng SQLite Object index có fingerprint mapping và thống kê IDF.

Input mặc định: ``data/objects/<video_id>/<keyframe>.json``. Ngoài thư mục,
script nhận trực tiếp một hay nhiều ZIP để phù hợp pipeline chia batch.

Catalog nhãn tạm được build riêng bằng ``build_object_label_catalog.py``;
retrieval index này chỉ dành cho Object JSON khớp đúng keyframe mapping.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

from object_io import iter_object_documents, parse_detections
from object_label_catalog import normalize_text
from search_types import ARTIFACTS_DIR, MAPPING_PATH, PROJECT_DIR
from stdio_setup import configure_stdio


configure_stdio()


OBJECT_DIR = PROJECT_DIR / "data" / "objects"
INDEX_PATH = ARTIFACTS_DIR / "object_index.sqlite3"
LEGACY_INDEX_PATH = ARTIFACTS_DIR / "object_index.json"
BASELINE_MIN_SCORE = 0.4
SCHEMA = "aic_object_sqlite_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(path=MAPPING_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mapping: {path}. Chạy build_mapping.py trước."
        )

    rows = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("video_id", "keyframe_name", "vector_index"):
                if field not in row:
                    raise ValueError(
                        f"Mapping dòng {line_number} thiếu trường {field!r}"
                    )
            rows.append(row)

    rows.sort(key=lambda row: int(row["vector_index"]))
    actual = [int(row["vector_index"]) for row in rows]
    if actual != list(range(len(rows))):
        raise ValueError("vector_index trong mapping phải liên tục từ 0")
    return rows


def mapping_lookup(mapping):
    lookup = {}
    for row in mapping:
        key = (
            str(row["video_id"]),
            Path(str(row["keyframe_name"])).stem,
        )
        if key in lookup:
            raise ValueError(f"Mapping trùng video/keyframe: {key}")
        lookup[key] = int(row["vector_index"])
    return lookup


def verify_manifest(manifest_path, mapping_hash):
    if manifest_path is None:
        return None
    manifest_path = Path(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    expected = raw.get("mapping_sha256")
    if expected is None and isinstance(raw.get("metadata"), dict):
        expected = raw["metadata"].get("mapping_sha256")
    if not expected:
        raise ValueError(
            f"Object manifest không có mapping_sha256: {manifest_path}"
        )
    if str(expected).casefold() != mapping_hash.casefold():
        raise RuntimeError(
            "Object manifest không khớp clip_row_mapping.jsonl. "
            "Không được ghép Object của bộ keyframe khác."
        )
    return raw


def _create_schema(connection):
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE documents (
            vector_index INTEGER PRIMARY KEY,
            logical_id TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL
        );

        CREATE TABLE classes (
            label_key TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            document_frequency INTEGER NOT NULL,
            detection_count INTEGER NOT NULL,
            idf REAL NOT NULL
        );

        CREATE TABLE postings (
            label_key TEXT NOT NULL,
            vector_index INTEGER NOT NULL,
            max_confidence REAL NOT NULL,
            max_area REAL NOT NULL,
            box_count INTEGER NOT NULL,
            PRIMARY KEY (label_key, vector_index)
        ) WITHOUT ROWID;
        """
    )


def build(
    inputs,
    output_path=INDEX_PATH,
    mapping_path=MAPPING_PATH,
    min_score=BASELINE_MIN_SCORE,
    manifest_path=None,
    allow_unmapped=False,
    require_complete=False,
):
    """Build index atomically; trả metadata dùng cho test/notebook."""

    inputs = [Path(value) for value in inputs]
    mapping_path = Path(mapping_path)
    output_path = Path(output_path)
    mapping = load_mapping(mapping_path)
    lookup = mapping_lookup(mapping)
    mapping_hash = file_sha256(mapping_path)
    manifest = verify_manifest(manifest_path, mapping_hash)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix="object_index_", suffix=".sqlite3", dir=output_path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)

    seen_documents = {}
    vector_sources = {}
    display_names = defaultdict(Counter)
    invalid_detections = filtered_detections = 0
    raw_detections = indexed_detections = 0
    mapped_documents = duplicate_documents = conflict_documents = 0
    unmapped_documents = invalid_documents = 0
    unmapped_examples = []

    try:
        connection = sqlite3.connect(temporary)
        _create_schema(connection)

        for document in iter_object_documents(inputs):
            previous_hash = seen_documents.get(document.logical_id)
            if previous_hash is not None:
                if previous_hash == document.content_sha256:
                    duplicate_documents += 1
                    continue
                conflict_documents += 1
                raise RuntimeError(
                    "Object JSON trùng ID nhưng khác nội dung: "
                    f"{document.logical_id}"
                )
            seen_documents[document.logical_id] = document.content_sha256

            vector_index = lookup.get(
                (document.video_id, document.keyframe_stem)
            )
            if vector_index is None:
                unmapped_documents += 1
                if len(unmapped_examples) < 10:
                    unmapped_examples.append(document.logical_id)
                continue

            if vector_index in vector_sources:
                conflict_documents += 1
                raise RuntimeError(
                    f"Hai Object JSON cùng map vector {vector_index}: "
                    f"{vector_sources[vector_index]} và {document.logical_id}"
                )

            try:
                detections = parse_detections(document.raw, document.source)
            except (TypeError, ValueError, KeyError):
                invalid_documents += 1
                raise

            connection.execute(
                "INSERT INTO documents(vector_index, logical_id, source) "
                "VALUES (?, ?, ?)",
                (vector_index, document.logical_id, document.source),
            )
            vector_sources[vector_index] = document.logical_id
            mapped_documents += 1

            aggregates = {}
            for display_name, score, area in detections:
                raw_detections += 1
                if (
                    not display_name
                    or not math.isfinite(score)
                    or score < 0.0
                    or score > 1.0
                    or not math.isfinite(area)
                ):
                    invalid_detections += 1
                    continue
                if score <= min_score:
                    filtered_detections += 1
                    continue

                label_key = normalize_text(display_name)
                if not label_key:
                    invalid_detections += 1
                    continue
                display_names[label_key][display_name] += 1
                current = aggregates.setdefault(
                    label_key,
                    {
                        "max_confidence": 0.0,
                        "max_area": 0.0,
                        "box_count": 0,
                    },
                )
                current["max_confidence"] = max(
                    current["max_confidence"], float(score)
                )
                current["max_area"] = max(
                    current["max_area"], max(0.0, float(area))
                )
                current["box_count"] += 1
                indexed_detections += 1

            connection.executemany(
                "INSERT INTO postings("
                "label_key, vector_index, max_confidence, max_area, box_count"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    (
                        label_key,
                        vector_index,
                        values["max_confidence"],
                        values["max_area"],
                        values["box_count"],
                    )
                    for label_key, values in aggregates.items()
                ),
            )

            if mapped_documents % 10000 == 0:
                connection.commit()
                print(f"Đã index {mapped_documents:,} Object keyframe...")

        if unmapped_documents and not allow_unmapped:
            examples = ", ".join(unmapped_examples)
            raise RuntimeError(
                f"Có {unmapped_documents:,} Object JSON không map được. "
                f"Ví dụ: {examples}. Nếu đây thật sự là file dư, chạy lại "
                "với --allow_unmapped; không dùng tùy chọn này để ghép hai "
                "bộ keyframe khác nhau."
            )

        missing_documents = len(mapping) - mapped_documents
        if require_complete and missing_documents:
            raise RuntimeError(
                f"Object index chưa đủ: thiếu {missing_documents:,}/"
                f"{len(mapping):,} keyframe"
            )
        if mapped_documents == 0:
            raise RuntimeError("Không có Object JSON nào map được với mapping")

        grouped = connection.execute(
            "SELECT label_key, COUNT(*), SUM(box_count) "
            "FROM postings GROUP BY label_key"
        ).fetchall()
        for label_key, document_frequency, detection_count in grouped:
            display_name = (
                display_names[label_key].most_common(1)[0][0]
                if display_names[label_key]
                else label_key
            )
            idf = math.log(
                (mapped_documents + 1) / (int(document_frequency) + 1)
            ) + 1.0
            connection.execute(
                "INSERT INTO classes("
                "label_key, display_name, document_frequency, "
                "detection_count, idf"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    label_key,
                    display_name,
                    int(document_frequency),
                    int(detection_count),
                    float(idf),
                ),
            )

        complete = missing_documents == 0 and unmapped_documents == 0
        metadata = {
            "schema": SCHEMA,
            "mapping_sha256": mapping_hash,
            "mapping_rows": str(len(mapping)),
            "min_score_exclusive": str(float(min_score)),
            "mapped_object_documents": str(mapped_documents),
            "missing_object_documents": str(missing_documents),
            "unmapped_object_documents": str(unmapped_documents),
            "duplicate_object_documents": str(duplicate_documents),
            "conflicting_object_documents": str(conflict_documents),
            "invalid_object_documents": str(invalid_documents),
            "raw_detections": str(raw_detections),
            "indexed_detections": str(indexed_detections),
            "filtered_detections": str(filtered_detections),
            "invalid_detections": str(invalid_detections),
            "class_count": str(len(grouped)),
            "complete": "1" if complete else "0",
            "retrieval_ready": "1" if complete else "0",
            "sources_json": json.dumps(
                [str(path) for path in inputs], ensure_ascii=False
            ),
            "manifest_json": (
                json.dumps(manifest, ensure_ascii=False)
                if manifest is not None
                else ""
            ),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        connection.execute(
            "CREATE INDEX postings_class_score_idx ON postings("
            "label_key, max_confidence DESC)"
        )
        connection.commit()
        connection.close()
        os.replace(temporary, output_path)

    finally:
        try:
            connection.close()
        except (NameError, sqlite3.Error):
            pass
        if temporary.exists():
            temporary.unlink()

    print("Đã tạo Object SQLite index")
    print(f"  Mapping: {len(mapping):,} keyframe")
    print(f"  Object đã map: {mapped_documents:,}")
    print(f"  Object còn thiếu: {missing_documents:,}")
    print(f"  Object không map được: {unmapped_documents:,}")
    print(f"  Nhãn: {len(grouped):,}")
    print(f"  Detection được index: {indexed_detections:,}")
    print(f"  Hoàn chỉnh: {'có' if complete else 'chưa'}")
    print(f"  Output: {output_path}")
    return metadata


def build_index(min_score=BASELINE_MIN_SCORE):
    """API cũ: build từ ``data/objects`` với cấu hình mặc định."""

    return build(
        inputs=[OBJECT_DIR],
        output_path=INDEX_PATH,
        mapping_path=MAPPING_PATH,
        min_score=min_score,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dựng SQLite Object index khớp clip_row_mapping"
    )
    parser.add_argument(
        "--input",
        action="append",
        help="Thư mục/ZIP/JSON Object; có thể truyền nhiều lần",
    )
    parser.add_argument("--output", type=Path, default=INDEX_PATH)
    parser.add_argument("--mapping", type=Path, default=MAPPING_PATH)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--min_score", type=float, default=BASELINE_MIN_SCORE)
    parser.add_argument(
        "--allow_unmapped",
        action="store_true",
        help="Cho phép file Object dư không map được (mặc định dừng)",
    )
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="Dừng nếu còn keyframe chưa có Object JSON",
    )
    args = parser.parse_args()

    if not 0.0 <= args.min_score < 1.0:
        raise ValueError("--min_score phải nằm trong [0, 1)")

    build(
        inputs=args.input or [OBJECT_DIR],
        output_path=args.output,
        mapping_path=args.mapping,
        min_score=args.min_score,
        manifest_path=args.manifest,
        allow_unmapped=args.allow_unmapped,
        require_complete=args.require_complete,
    )


if __name__ == "__main__":
    main()
