"""
Bước 6 của pipeline: re-ranking kết quả CLIP bằng vật thể.

Luồng chạy:
1. python scripts/search_text_query.py "<câu truy vấn>"
   → lưu top ứng viên CLIP vào artifacts/last_search_results.json
2. python scripts/rerank_objects.py
   → tính điểm vật thể cho từng ứng viên, kết hợp với điểm CLIP
     và xếp hạng lại. Kết quả cuối lưu vào
     artifacts/last_rerank_results.json (mục submission là danh sách
     <video_id>, <frame_id> theo định dạng nộp bài BTC).

Tùy chọn:
- --objects person,car   chỉ định vật thể thủ công (không tự đoán)
- --weight 0.5           trọng số điểm object (0..1). combined =
                         (1-w)*clip_norm + w*object_norm

Đối tượng dùng chung ObjectLabelCatalog/ObjectRetriever với search_engine:
alias Việt–Anh, cụm dài, accent-fold, translation fallback và IDF.
"""

import argparse
import json
from pathlib import Path

from object_retriever import ObjectRetriever
from stdio_setup import configure_stdio


configure_stdio()


DEFAULT_WEIGHT = 0.5

PROJECT_DIR = Path(__file__).resolve().parents[1]

RESULTS_PATH = PROJECT_DIR / "artifacts" / "last_search_results.json"
RERANK_PATH = PROJECT_DIR / "artifacts" / "last_rerank_results.json"

def normalize_term(term):
    return str(term).strip().lower()


def load_json(path, hint):
    if not path.exists():
        raise FileNotFoundError(f"{hint}: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def normalize_min_max(values):
    low = min(values)
    high = max(values)

    if high - low <= 1e-9:
        return [1.0 for _ in values]

    return [(value - low) / (high - low) for value in values]


def main():
    parser = argparse.ArgumentParser(
        description="Re-ranking kết quả CLIP bằng vật thể (Faster R-CNN)"
    )
    parser.add_argument(
        "--objects",
        default=None,
        help="Danh sách vật thể, phân tách bằng dấu phẩy. "
        "Nếu bỏ trống sẽ tự đoán từ câu query.",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=DEFAULT_WEIGHT,
        help="Trọng số điểm object trong điểm tổng (0..1, mặc định 0.5)",
    )

    args = parser.parse_args()

    if not 0.0 <= args.weight <= 1.0:
        raise ValueError("--weight phải nằm trong khoảng 0..1")

    search_data = load_json(
        RESULTS_PATH,
        "Chạy scripts/search_text_query.py trước. Thiếu kết quả CLIP",
    )

    query = search_data["query"]
    candidates = search_data["results"]

    if not candidates:
        raise ValueError("Không có ứng viên nào trong kết quả CLIP")

    if args.objects:
        objects = [
            normalize_term(term)
            for term in args.objects.split(",")
            if term.strip()
        ]
        print("Vật thể (chỉ định):", ", ".join(objects))
    else:
        objects = None

    retriever = ObjectRetriever()
    try:
        matches = retriever.resolve_query(query, objects=objects)
        print(
            "Class OpenImages khớp:",
            ", ".join(
                f"{match.display_name} ({match.method})" for match in matches
            ) or "(không có)",
        )
        scored = retriever.score_map(matches)
    finally:
        retriever.close()

    object_scores = {
        vector_index: values[0] for vector_index, values in scored.items()
    }
    classes_by_vector = {
        vector_index: set(values[1]) for vector_index, values in scored.items()
    }

    clip_scores = [candidate["score"] for candidate in candidates]
    clip_norms = normalize_min_max(clip_scores)

    raw_object_scores = [
        object_scores.get(candidate["vector_index"], 0.0)
        for candidate in candidates
    ]

    max_object = max(raw_object_scores) if raw_object_scores else 0.0

    rows = []

    for candidate, clip_norm, raw_object in zip(
        candidates, clip_norms, raw_object_scores
    ):
        object_norm = raw_object / max_object if max_object > 0 else 0.0

        combined = (
            (1.0 - args.weight) * clip_norm
            + args.weight * object_norm
        )

        rows.append(
            {
                "video_id": candidate["video_id"],
                "frame_id": candidate["frame_id"],
                "vector_index": candidate["vector_index"],
                "combined_score": round(combined, 6),
                "clip_score": candidate["score"],
                "object_score": round(raw_object, 4),
                "matched_classes": sorted(
                    classes_by_vector.get(candidate["vector_index"], set())
                ),
                "keyframe_path": candidate["keyframe_path"],
            }
        )

    rows.sort(key=lambda row: row["combined_score"], reverse=True)

    print()
    print("=" * 60)
    print("QUERY:", query)
    print(
        f"Re-rank {len(rows)} ứng viên "
        f"(weight object = {args.weight})"
    )
    print("=" * 60)
    print("Kết quả theo định dạng nộp bài BTC: <video_id>, <frame_id>")
    print()

    for rank, row in enumerate(rows, start=1):
        classes = ",".join(row["matched_classes"]) or "-"

        print(
            f"{rank:>3}. "
            f"{row['video_id']}, {row['frame_id']} "
            f"(combined={row['combined_score']:.4f}, "
            f"clip={row['clip_score']:.4f}, "
            f"obj={row['object_score']:.4f}, "
            f"classes={classes})"
        )

    output = {
        "query": query,
        "objects": objects,
        "object_label_matches": [match.to_dict() for match in matches],
        "weight": args.weight,
        "top_k": len(rows),
        "submission": [
            {
                "video_id": row["video_id"],
                "frame_id": row["frame_id"],
            }
            for row in rows
        ],
        "results": rows,
    }

    RERANK_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Đã lưu kết quả re-rank tại:", RERANK_PATH)


if __name__ == "__main__":
    main()
