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

Đối tượng khớp theo tên class OpenImages: khớp đúng từ
("car" khớp "Car", "bicycle wheel" khớp "Wheel" của bicycle...)
và tự bỏ số ít/số nhiều đơn giản.
"""

import argparse
import json
import re
from pathlib import Path

from stdio_setup import configure_stdio


configure_stdio()


DEFAULT_WEIGHT = 0.5

PROJECT_DIR = Path(__file__).resolve().parents[1]

OBJECT_INDEX_PATH = PROJECT_DIR / "artifacts" / "object_index.json"
RESULTS_PATH = PROJECT_DIR / "artifacts" / "last_search_results.json"
RERANK_PATH = PROJECT_DIR / "artifacts" / "last_rerank_results.json"

STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "with", "and", "or",
    "is", "are", "was", "were", "to", "from", "by", "for", "about",
    "that", "this", "there", "here", "it", "its", "as", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "into",
    "over", "under", "near", "while", "when", "who", "whom", "which",
    "what", "where", "how", "some", "any", "no", "not", "very",
}


def normalize_term(term):
    return str(term).strip().lower()


def variants(term):
    """Sinh các biến thể số ít/số nhiều đơn giản của một từ."""

    term = normalize_term(term)
    result = {term}

    if term.endswith("s") and len(term) > 3:
        result.add(term[:-1])
    else:
        result.add(term + "s")

    return result


def term_matches_class(term, class_name):
    """Khớp từ khóa với tên class theo từng từ (tránh 'car' ăn 'cardboard')."""

    term_variants = variants(term)
    class_lower = class_name.lower()

    if class_lower in term_variants:
        return True

    words = re.split(r"[\s\-_/]+", class_lower)

    for word in words:
        if word in term_variants:
            return True

    return False


def extract_query_objects(query, vocabulary):
    """Tự tìm các từ trong query khớp với class có trong object index."""

    tokens = re.findall(r"[a-zA-Z]+", query.lower())

    matched = []

    for token in tokens:
        if token in STOPWORDS:
            continue

        for class_key in vocabulary:
            if term_matches_class(token, vocabulary[class_key]):
                matched.append(token)
                break

    # Giữ nguyên thứ tự, bỏ trùng lặp
    seen = set()
    unique = []

    for term in matched:
        if term not in seen:
            seen.add(term)
            unique.append(term)

    return unique


def load_json(path, hint):
    if not path.exists():
        raise FileNotFoundError(f"{hint}: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def compute_object_scores(candidates, inverted_index, vocabulary, objects):
    """
    Với mỗi class cần tìm, cộng dồn detection score vào các keyframe
    chứa class đó. Trả về {vector_index: (score, [class khớp])}.
    """

    matched_classes = set()

    for term in objects:
        for class_key, display_name in vocabulary.items():
            if term_matches_class(term, display_name):
                matched_classes.add(class_key)

    scores = {}
    classes_by_vector = {}

    for class_key in matched_classes:
        for vector_index, score, _area in inverted_index[class_key]:
            scores[vector_index] = scores.get(vector_index, 0.0) + score

            classes_by_vector.setdefault(vector_index, set()).add(
                vocabulary[class_key]
            )

    return scores, classes_by_vector, matched_classes


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

    object_data = load_json(
        OBJECT_INDEX_PATH,
        "Chạy scripts/build_object_index.py trước. Thiếu object index",
    )
    search_data = load_json(
        RESULTS_PATH,
        "Chạy scripts/search_text_query.py trước. Thiếu kết quả CLIP",
    )

    inverted_index = object_data["inverted_index"]
    vocabulary = object_data["vocabulary"]

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
        objects = extract_query_objects(query, vocabulary)
        print("Vật thể tự đoán từ query:", ", ".join(objects) or "(không có)")

        if not objects:
            print(
                "Không tìm thấy vật thể nào khớp query. "
                "Dùng --objects để chỉ định thủ công. "
                "Giữ nguyên thứ hạng CLIP."
            )

    object_scores, classes_by_vector, matched_classes = (
        compute_object_scores(
            candidates, inverted_index, vocabulary, objects
        )
    )

    if matched_classes:
        print(
            "Class OpenImages khớp:",
            ", ".join(
                vocabulary[class_key]
                for class_key in sorted(matched_classes)
            ),
        )

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
