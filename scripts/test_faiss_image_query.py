"""
Kiểm tra nhanh FAISS index bằng image query: lấy feature của một
keyframe có sẵn làm query, kết quả hạng 1 phải là chính nó (score ~1).
"""

import json
from pathlib import Path

import faiss
import numpy as np

from stdio_setup import configure_stdio


configure_stdio()


PROJECT_DIR = Path(__file__).resolve().parents[1]

FEATURES_DIR = PROJECT_DIR / "features" / "clip"
INDEX_PATH = PROJECT_DIR / "artifacts" / "clip.index"
MAPPING_PATH = PROJECT_DIR / "artifacts" / "clip_row_mapping.jsonl"

TEST_ROW = 5
TOP_K = 5


def load_mapping():
    rows = []

    with MAPPING_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    rows.sort(key=lambda row: row["vector_index"])
    return rows


def load_feature_row(mapping, vector_index):
    """Đọc 1 dòng feature từ file .npy của video tương ứng."""

    row = mapping[vector_index]
    video_id = row["video_id"]

    npy_path = FEATURES_DIR / f"{video_id}.npy"

    if not npy_path.exists():
        raise FileNotFoundError(f"Thiếu features: {npy_path}")

    features = np.load(npy_path, mmap_mode="r")

    # Vị trí của keyframe trong file .npy của video nó
    in_video_index = sum(
        1
        for other in mapping[:vector_index]
        if other["video_id"] == video_id
    )

    return np.array(
        features[in_video_index:in_video_index + 1],
        dtype=np.float32,
        order="C",
        copy=True,
    )


def main():
    index = faiss.read_index(str(INDEX_PATH))
    mapping = load_mapping()

    if len(mapping) != index.ntotal:
        raise ValueError(
            f"Số dòng mapping ({len(mapping)}) không bằng "
            f"số vector trong index ({index.ntotal})"
        )

    if index.ntotal == 0:
        raise ValueError("FAISS index và mapping không có vector nào")

    test_row = min(TEST_ROW, index.ntotal - 1)

    query = load_feature_row(mapping, test_row)
    faiss.normalize_L2(query)

    k = min(TOP_K, index.ntotal)
    scores, indices = index.search(query, k)

    expected = mapping[test_row]

    print("Ảnh dùng làm query:")
    print("Vector index:", test_row)
    print("Video:", expected["video_id"])
    print("Keyframe:", expected["keyframe_name"])
    print("Frame index:", expected["frame_index"])
    print("Path:", expected["keyframe_path"])

    print()
    print(f"Top-{k} kết quả:")

    for rank, (vector_index, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1,
    ):
        row = mapping[int(vector_index)]

        print(
            f"{rank}. "
            f"index={vector_index} | "
            f"score={score:.6f} | "
            f"{row['video_id']}/{row['keyframe_name']} | "
            f"frame_id={row['frame_index']}"
        )


if __name__ == "__main__":
    main()
