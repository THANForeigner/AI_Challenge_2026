"""Dựng catalog nhãn Object song ngữ từ một hoặc nhiều batch JSON/ZIP.

Catalog không cần FAISS/mapping và không phải retrieval index. Vì vậy có thể
dùng batch Object tạm để khóa vocabulary trong khi detector vẫn đang chạy.
"""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import tempfile

from object_io import (
    CLASS_KEYS,
    SCORE_KEYS,
    clean_entity,
    iter_object_documents,
    parse_detections,
)
from object_label_catalog import (
    ALIASES_PATH,
    CATALOG_PATH,
    canonical_concepts,
    load_alias_config,
    normalize_text,
)
from stdio_setup import configure_stdio


configure_stdio()


DEFAULT_MIN_SCORE = 0.4


def _iter_label_scores(raw, source):
    """Fast path cho schema TF Hub: không tạo tuple bbox trung gian."""

    if isinstance(raw, dict) and not isinstance(raw.get("objects"), list):
        classes = next((raw[key] for key in CLASS_KEYS if key in raw), None)
        scores = next((raw[key] for key in SCORE_KEYS if key in raw), None)
        if classes is not None and scores is not None:
            if len(classes) != len(scores):
                raise ValueError(
                    f"{source} có {len(classes)} class nhưng "
                    f"{len(scores)} score"
                )
            for entity, score in zip(classes, scores):
                yield clean_entity(entity), float(score)
            return

    for display_name, score, _area in parse_detections(
        raw, source, include_area=False
    ):
        yield display_name, score


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix="object_catalog_", suffix=".json", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_catalog(
    inputs,
    output_path=CATALOG_PATH,
    aliases_path=ALIASES_PATH,
    min_score=DEFAULT_MIN_SCORE,
    translate_vi=False,
    translator=None,
    strict=True,
):
    """Scan streaming batch Object và ghi catalog deterministic."""

    raw_counts = Counter()
    active_counts = Counter()
    raw_document_counts = Counter()
    active_document_counts = Counter()
    display_names = defaultdict(Counter)
    max_scores = defaultdict(float)
    seen_documents = {}
    documents = duplicate_documents = conflicting_documents = 0
    invalid_documents = invalid_detections = 0
    raw_detections = detections_above_threshold = 0

    def handle_read_error(source, error):
        nonlocal invalid_documents
        invalid_documents += 1
        print(f"Cảnh báo: bỏ JSON Object không đọc được {source}: {error}")

    for document in iter_object_documents(
        inputs,
        on_error=None if strict else handle_read_error,
    ):
        try:
            # ``_iter_label_scores`` là generator nên lỗi schema chỉ phát sinh
            # khi duyệt. Materialize tối đa vài trăm detection của một JSON để
            # ``--skip_invalid`` thực sự bắt và bỏ đúng tài liệu lỗi.
            detections = list(
                _iter_label_scores(document.raw, document.source)
            )
        except (TypeError, ValueError, KeyError) as error:
            invalid_documents += 1
            if strict:
                raise
            print(f"Cảnh báo: bỏ JSON Object lỗi {document.source}: {error}")
            continue

        # Chỉ khóa logical_id sau khi parse thành công. Nếu batch trước có bản
        # lỗi và batch sau có bản tốt cùng ID, bản tốt vẫn phải được sử dụng.
        previous_hash = seen_documents.get(document.logical_id)
        if previous_hash is not None:
            if previous_hash == document.content_sha256:
                duplicate_documents += 1
                continue
            conflicting_documents += 1
            message = (
                f"Object JSON trùng ID nhưng khác nội dung: "
                f"{document.logical_id} ({document.source})"
            )
            if strict:
                raise ValueError(message)
            print("Cảnh báo:", message)
            continue
        seen_documents[document.logical_id] = document.content_sha256

        documents += 1
        document_raw_labels = set()
        document_active_labels = set()

        for display_name, score in detections:
            if (
                not display_name
                or not math.isfinite(score)
                or score < 0.0
                or score > 1.0
            ):
                invalid_detections += 1
                continue
            label_key = normalize_text(display_name)
            if not label_key:
                invalid_detections += 1
                continue

            raw_counts[label_key] += 1
            display_names[label_key][display_name] += 1
            max_scores[label_key] = max(max_scores[label_key], float(score))
            document_raw_labels.add(label_key)
            raw_detections += 1

            if score > min_score:
                active_counts[label_key] += 1
                document_active_labels.add(label_key)
                detections_above_threshold += 1

        raw_document_counts.update(document_raw_labels)
        active_document_counts.update(document_active_labels)

        if documents == 1000 or documents % 10000 == 0:
            print(
                f"Đã đọc {documents:,} Object JSON — "
                f"{len(raw_counts):,} nhãn..."
            )

    if not raw_counts:
        raise RuntimeError("Không tìm được nhãn Object hợp lệ trong input")

    labels = {}
    for label_key in sorted(raw_counts):
        display_name = display_names[label_key].most_common(1)[0][0]
        labels[label_key] = {
            "key": label_key,
            "display_name": display_name,
            "aliases": [],
            "sample_stats": {
                "documents_raw": raw_document_counts[label_key],
                "detections_raw": raw_counts[label_key],
                "documents_above_threshold": active_document_counts[
                    label_key
                ],
                "detections_above_threshold": active_counts[label_key],
                "max_score": round(max_scores[label_key], 8),
            },
            "active": active_counts[label_key] > 0,
        }

    concepts = canonical_concepts(labels)

    if translate_vi:
        if translator is None:
            from query_translator import translate_en_to_vi_batch

            translator = translate_en_to_vi_batch
        keys = sorted(labels)
        display_names = [labels[key]["display_name"] for key in keys]
        print(f"Đang dịch {len(display_names):,} nhãn Object Anh→Vi...")
        translations = translator(display_names)
        if len(translations) != len(display_names):
            raise RuntimeError(
                "Bộ dịch trả số kết quả khác số nhãn Object đầu vào"
            )
        concept_by_id = {concept["id"]: concept for concept in concepts}
        for label_key, translated in zip(keys, translations):
            translated = str(translated).strip()
            if not normalize_text(translated):
                continue
            labels[label_key]["aliases"].append(
                {
                    "lang": "vi",
                    "text": translated,
                    "weight": 0.88,
                    "source": "opus-mt-en-vi",
                }
            )
            concept_by_id[f"label:{label_key}"]["aliases"]["vi"].append(
                {
                    "text": translated,
                    "weight": 0.88,
                    "source": "opus-mt-en-vi",
                }
            )

    manual_concepts = load_alias_config(aliases_path)
    pending_targets = sorted(
        {
            str(target.get("label", "")).casefold()
            for concept in manual_concepts
            for target in concept.get("targets", [])
            if str(target.get("label", "")).strip()
            and str(target.get("label", "")).casefold() not in labels
        }
    )
    concepts.extend(manual_concepts)

    payload = {
        "schema": "aic_object_label_catalog_v1",
        "config": {
            "min_score_exclusive": float(min_score),
            "normalization": "nfkc-casefold-accent-fold-v1",
            "translated_vi": bool(translate_vi),
        },
        "stats": {
            "documents": documents,
            "duplicate_documents": duplicate_documents,
            "conflicting_documents": conflicting_documents,
            "invalid_documents": invalid_documents,
            "raw_detections": raw_detections,
            "detections_above_threshold": detections_above_threshold,
            "invalid_detections": invalid_detections,
            "unique_labels": len(labels),
            "active_labels": sum(record["active"] for record in labels.values()),
            "pending_curated_targets": pending_targets,
        },
        "sources": [str(Path(value)) for value in inputs],
        "labels": [labels[key] for key in sorted(labels)],
        "concepts": sorted(concepts, key=lambda concept: concept["id"]),
        "embeddings": None,
    }
    _atomic_write_json(output_path, payload)

    print("Đã tạo Object label catalog")
    print(f"  JSON hợp lệ: {documents:,}")
    print(f"  Nhãn thô: {len(labels):,}")
    print(f"  Nhãn score > {min_score}: {payload['stats']['active_labels']:,}")
    print(f"  Detection score > {min_score}: {detections_above_threshold:,}")
    print(f"  Output: {output_path}")
    if pending_targets:
        print(
            "  Alias chờ label ở batch sau: " + ", ".join(pending_targets)
        )
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Dựng vocabulary/alias Việt–Anh từ batch Object JSON"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Thư mục, JSON hoặc ZIP Object; có thể truyền nhiều lần",
    )
    parser.add_argument("--output", type=Path, default=CATALOG_PATH)
    parser.add_argument("--aliases", type=Path, default=ALIASES_PATH)
    parser.add_argument("--min_score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument(
        "--translate_vi",
        action="store_true",
        help="Dịch tự động từng nhãn Anh→Vi bằng Helsinki-NLP/opus-mt-en-vi",
    )
    parser.add_argument(
        "--skip_invalid",
        action="store_true",
        help="Bỏ JSON lỗi thay vì dừng build",
    )
    args = parser.parse_args()

    if not 0.0 <= args.min_score < 1.0:
        raise ValueError("--min_score phải nằm trong [0, 1)")

    build_catalog(
        inputs=args.input,
        output_path=args.output,
        aliases_path=args.aliases,
        min_score=args.min_score,
        translate_vi=args.translate_vi,
        strict=not args.skip_invalid,
    )


if __name__ == "__main__":
    main()
