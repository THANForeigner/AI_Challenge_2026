"""Kiểm tra dữ liệu CLIP/FAISS/Object theo các bất biến của baseline BTC."""

import json
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

from stdio_setup import configure_stdio


configure_stdio()


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAPPING_PATH = PROJECT_DIR / "artifacts" / "clip_row_mapping.jsonl"
INDEX_PATH = PROJECT_DIR / "artifacts" / "clip.index"
OBJECT_INDEX_PATH = PROJECT_DIR / "artifacts" / "object_index.json"
FEATURES_DIR = PROJECT_DIR / "features" / "clip"
KEYFRAME_DIR = PROJECT_DIR / "data" / "keyframes"
BASELINE_OBJECT_THRESHOLD = 0.4


def fail(message):
    raise RuntimeError("BASELINE SETUP LỖI: " + message)


def load_mapping():
    if not MAPPING_PATH.exists():
        fail(f"thiếu mapping {MAPPING_PATH}")

    rows = [
        json.loads(line)
        for line in MAPPING_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: row["vector_index"])

    if [row["vector_index"] for row in rows] != list(range(len(rows))):
        fail("vector_index không liên tục từ 0")

    return rows


def validate_files_and_features(rows):
    by_video = defaultdict(list)

    for row in rows:
        by_video[row["video_id"]].append(row)
        image_path = (
            KEYFRAME_DIR / row["video_id"] / row["keyframe_name"]
        )

        if not image_path.exists():
            fail(f"thiếu keyframe {image_path}")

        try:
            int(Path(row["keyframe_name"]).stem)
        except ValueError:
            fail(f"tên keyframe không phải frame số: {image_path.name}")

    dimensions = set()
    total_vectors = 0

    for video_id, video_rows in sorted(by_video.items()):
        feature_path = FEATURES_DIR / f"{video_id}.npy"

        if not feature_path.exists():
            fail(f"thiếu CLIP feature {feature_path}")

        features = np.load(feature_path, mmap_mode="r")

        if features.ndim != 2:
            fail(f"{feature_path.name} phải là ma trận 2 chiều")

        if features.shape[0] != len(video_rows):
            fail(
                f"{video_id}: {len(video_rows)} keyframe nhưng "
                f"{features.shape[0]} vector"
            )

        if not np.isfinite(features).all():
            fail(f"{feature_path.name} chứa NaN hoặc Inf")

        norms = np.linalg.norm(features, axis=1)
        if np.any(norms == 0):
            fail(f"{feature_path.name} chứa zero vector")

        dimensions.add(int(features.shape[1]))
        total_vectors += int(features.shape[0])

        normalized = bool(np.allclose(norms, 1.0, atol=1e-3))
        status = "đã L2 normalize" if normalized else "FAISS sẽ normalize"
        print(
            f"  ✓ {video_id}: {len(video_rows)} ảnh = "
            f"{features.shape[0]} vector, dim={features.shape[1]} ({status})"
        )

    if len(dimensions) != 1:
        fail(f"các file feature không cùng dimension: {sorted(dimensions)}")

    return total_vectors, dimensions.pop()


def validate_faiss(expected_count, expected_dimension):
    if not INDEX_PATH.exists():
        fail(f"thiếu FAISS index {INDEX_PATH}")

    index = faiss.read_index(str(INDEX_PATH))

    if index.ntotal != expected_count:
        fail(f"FAISS có {index.ntotal} vector, cần {expected_count}")

    if index.d != expected_dimension:
        fail(f"FAISS dim={index.d}, feature dim={expected_dimension}")

    print(
        f"  ✓ FAISS: {type(index).__name__}, "
        f"ntotal={index.ntotal}, dim={index.d}"
    )


def validate_object_index():
    if not OBJECT_INDEX_PATH.exists():
        print("  ! Chưa có object_index.json (không cản Visual baseline)")
        return

    data = json.loads(OBJECT_INDEX_PATH.read_text(encoding="utf-8"))
    threshold = data.get("config", {}).get("min_score_exclusive")

    if threshold != BASELINE_OBJECT_THRESHOLD:
        fail(
            "object index chưa xác nhận ngưỡng score > 0.4; "
            "chạy lại scripts/build_object_index.py"
        )

    posting_scores = [
        float(posting[1])
        for postings in data.get("inverted_index", {}).values()
        for posting in postings
    ]

    if not posting_scores:
        fail("object index không có detection nào")

    if min(posting_scores) <= BASELINE_OBJECT_THRESHOLD:
        fail("object index vẫn chứa detection có score <= 0.4")

    indexed = data.get("stats", {}).get("indexed_detections", "?")
    print(f"  ✓ Object: score > {threshold}, {indexed} detection được index")


def main():
    rows = load_mapping()

    if not rows:
        fail("mapping rỗng")

    print("Kiểm tra setup baseline BTC")
    print(f"  ✓ Mapping: {len(rows)} keyframe")
    vector_count, dimension = validate_files_and_features(rows)
    validate_faiss(vector_count, dimension)
    validate_object_index()
    print()
    print("SETUP BASELINE HỢP LỆ")
    print(
        "Lưu ý: video_id/frame_id chỉ hợp lệ để nộp khi tên thư mục và "
        "metadata là định danh thật do BTC cung cấp."
    )


if __name__ == "__main__":
    main()
