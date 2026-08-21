"""
Bước 3 của pipeline (C1): gộp features từng video thành MỘT FAISS
index chung.

Bất biến bắt buộc:
    index.ntotal = tổng số keyframe = số dòng clip_row_mapping.jsonl
"""

import json
from pathlib import Path

import faiss
import numpy as np

from stdio_setup import configure_stdio


configure_stdio()


PROJECT_DIR = Path(__file__).resolve().parents[1]

FEATURES_DIR = PROJECT_DIR / "features" / "clip"
MAPPING_PATH = PROJECT_DIR / "artifacts" / "clip_row_mapping.jsonl"
INDEX_PATH = PROJECT_DIR / "artifacts" / "clip.index"
MANIFEST_PATH = PROJECT_DIR / "artifacts" / "clip_index_manifest.json"


def load_mapping():
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mapping: {MAPPING_PATH}. "
            "Chạy scripts/build_mapping.py trước."
        )

    rows = []

    with MAPPING_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    rows.sort(key=lambda row: row["vector_index"])
    return rows


def load_features_in_mapping_order(mapping_rows):
    """Ghép features/clip/<video_id>.npy theo đúng thứ tự mapping."""

    if not mapping_rows:
        raise ValueError("File mapping không có dữ liệu")

    chunks = []
    current_video = None
    current_expected = 0
    loaded_counts = {}

    for row in mapping_rows:
        video_id = row["video_id"]

        if video_id != current_video:
            if current_video is not None:
                chunks.append((current_video, current_expected))

            current_video = video_id
            current_expected = 0

        current_expected += 1

    if current_video is not None:
        chunks.append((current_video, current_expected))

    arrays = []

    for video_id, expected_count in chunks:
        npy_path = FEATURES_DIR / f"{video_id}.npy"

        if not npy_path.exists():
            raise FileNotFoundError(
                f"Thiếu features của video {video_id}: {npy_path}. "
                "Chạy scripts/encode_clip_features.py trước."
            )

        features = np.load(npy_path)

        if features.ndim != 2:
            raise ValueError(
                f"Video {video_id}: features phải là ma trận 2 chiều, "
                f"nhận shape {features.shape}"
            )

        if arrays and features.shape[1] != arrays[0].shape[1]:
            raise ValueError(
                f"Video {video_id}: dimension {features.shape[1]} không "
                f"khớp dimension {arrays[0].shape[1]} của video trước"
            )

        if features.shape[0] != expected_count:
            raise ValueError(
                f"Video {video_id}: file .npy có {features.shape[0]} "
                f"vector nhưng mapping có {expected_count} keyframe. "
                "Chạy lại encode_clip_features.py."
            )

        loaded_counts[video_id] = features.shape[0]
        arrays.append(features)

    return np.concatenate(arrays, axis=0), loaded_counts


def main():
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    mapping_rows = load_mapping()

    features, loaded_counts = load_features_in_mapping_order(
        mapping_rows
    )

    vector_count, dimension = features.shape

    # === Bất biến C1 ===
    if vector_count != len(mapping_rows):
        raise RuntimeError(
            f"BẤT BIẾN C1 LỖI: features có {vector_count} vector "
            f"nhưng mapping có {len(mapping_rows)} dòng"
        )

    vectors = np.array(
        features,
        dtype=np.float32,
        order="C",
        copy=True,
    )

    if not np.isfinite(vectors).all():
        raise ValueError("Features chứa NaN hoặc Inf")

    norms_before = np.linalg.norm(vectors, axis=1)

    if np.any(norms_before == 0):
        raise ValueError("Features chứa zero vector")

    # Chuẩn hóa lại để chắc chắn mỗi vector có norm bằng 1
    faiss.normalize_L2(vectors)

    # Tạo index dùng inner product
    index = faiss.IndexFlatIP(dimension)

    # Dòng 0 của features được gán ID 0, dòng 1 có ID 1...
    index.add(vectors)

    if index.ntotal != vector_count:
        raise RuntimeError(
            f"BẤT BIẾN C1 LỖI: index có {index.ntotal} vectors, "
            f"features có {vector_count}"
        )

    faiss.write_index(index, str(INDEX_PATH))

    manifest = {
        "index_type": "IndexFlatIP",
        "similarity": "cosine_via_normalized_inner_product",
        "vector_count": int(vector_count),
        "dimension": int(dimension),
        "dtype": "float32",
        "normalized": True,
        "invariant": "index.ntotal == tổng keyframe == số dòng mapping",
        "videos": loaded_counts,
        "features_dir": FEATURES_DIR.relative_to(PROJECT_DIR).as_posix(),
        "index_path": INDEX_PATH.relative_to(PROJECT_DIR).as_posix(),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Tạo FAISS index thành công")
    print("Index type:", type(index).__name__)
    print("Index dimension:", index.d)
    print("Index ntotal:", index.ntotal)
    print(
        f"Bất biến C1: index.ntotal ({index.ntotal}) = "
        f"tổng keyframe = số dòng mapping ({len(mapping_rows)}) ✓"
    )
    print("Index path:", INDEX_PATH)
    print("Manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
