"""
Bước 2 của pipeline (C1): encode CLIP features.

Mỗi video được lưu thành MỘT file .npy riêng trong features/clip/
(vd: features/clip/L21_V008.npy), thứ tự vector trong file theo thứ
tự keyframe tăng dần. Bước build_faiss_index.py sẽ gộp tất cả thành
một FAISS index chung theo đúng thứ tự của clip_row_mapping.jsonl.
"""

import json
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from stdio_setup import configure_stdio


configure_stdio()


MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"
BATCH_SIZE = int(os.getenv("AIC_CLIP_BATCH_SIZE", "128"))
RESUME = os.getenv("AIC_CLIP_RESUME", "1") == "1"
VALIDATE_KEYFRAMES = os.getenv("AIC_VALIDATE_KEYFRAMES", "sample")

PROJECT_DIR = Path(__file__).resolve().parents[1]

MAPPING_PATH = PROJECT_DIR / "artifacts" / "clip_row_mapping.jsonl"
FEATURES_DIR = PROJECT_DIR / "features" / "clip"
MANIFEST_PATH = PROJECT_DIR / "features" / "clip_features_manifest.json"
KEYFRAME_DIR = PROJECT_DIR / "data" / "keyframes"


def resolve_keyframe_path(row):
    """
    Ưu tiên đường dẫn theo layout chuẩn BTC
    (data/keyframes/<video_id>/<keyframe_name>) để không phụ thuộc
    đường dẫn tuyệt đối lưu trong mapping — mapping cũ từ máy khác
    hoặc sau khi di chuyển thư mục vẫn chạy được.
    """

    canonical = KEYFRAME_DIR / row["video_id"] / row["keyframe_name"]

    if canonical.exists():
        return canonical

    stored = Path(row["keyframe_path"])

    if stored.is_absolute():
        return stored

    return PROJECT_DIR / stored


def validate_keyframe_paths(mapping_rows):
    """Kiểm tra đủ ảnh TRƯỚC khi tải model (fail fast, liệt kê đủ)."""

    if VALIDATE_KEYFRAMES == "none":
        print("Bỏ qua kiểm tra đường dẫn keyframe theo cấu hình")
        return

    rows_to_check = mapping_rows

    if VALIDATE_KEYFRAMES == "sample":
        boundaries = {}

        for row in mapping_rows:
            video_id = row["video_id"]

            if video_id not in boundaries:
                boundaries[video_id] = [row, row]
            else:
                boundaries[video_id][1] = row

        rows_to_check = []

        for first, last in boundaries.values():
            rows_to_check.append(first)

            if last["vector_index"] != first["vector_index"]:
                rows_to_check.append(last)

        print(
            f"Kiểm tra mẫu {len(rows_to_check)} đường dẫn "
            f"của {len(boundaries)} video"
        )

    missing = [
        f"{row['video_id']}/{row['keyframe_name']}"
        for row in rows_to_check
        if not resolve_keyframe_path(row).exists()
    ]

    if missing:
        preview = "\n".join(missing[:10])

        if len(missing) > 10:
            preview += f"\n... và {len(missing) - 10} file khác"

        raise FileNotFoundError(
            f"Không tìm thấy {len(missing)} ảnh keyframe:\n{preview}\n"
            f"Kiểm tra thư mục {KEYFRAME_DIR} hoặc chạy lại "
            "scripts/build_mapping.py nếu dữ liệu vừa thay đổi."
        )


def load_mapping():
    rows = []

    with MAPPING_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    rows.sort(key=lambda row: row["vector_index"])

    actual_indices = [
        row["vector_index"]
        for row in rows
    ]

    expected_indices = list(range(len(rows)))

    if actual_indices != expected_indices:
        raise ValueError(
            "vector_index phải liên tục từ 0 đến số keyframe - 1"
        )

    return rows


def group_by_video(mapping_rows):
    """Mapping đã sắp theo video_id; tách thành {video_id: [rows]}."""

    groups = {}

    for row in mapping_rows:
        groups.setdefault(row["video_id"], []).append(row)

    return groups


def encode_rows(rows, model, preprocess, device):
    feature_batches = []

    for start in range(0, len(rows), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(rows))
        batch_rows = rows[start:end]

        image_tensors = []

        for row in batch_rows:
            image_path = resolve_keyframe_path(row)

            if not image_path.exists():
                raise FileNotFoundError(
                    "Không tìm thấy ảnh: "
                    f"{row['video_id']}/{row['keyframe_name']} "
                    f"({image_path})"
                )

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensors.append(preprocess(image))

        image_batch = torch.stack(image_tensors).to(device)

        with torch.inference_mode(), torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=device == "cuda",
        ):
            features = model.encode_image(image_batch).float()

            # Chuẩn hóa mỗi vector về độ dài 1
            features = features / features.norm(
                dim=-1,
                keepdim=True,
            )

        feature_batches.append(
            features.cpu().numpy().astype(np.float32)
        )

    return np.concatenate(feature_batches, axis=0)


def main():
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mapping: {MAPPING_PATH}. "
            "Chạy scripts/build_mapping.py trước."
        )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    mapping_rows = load_mapping()

    if not mapping_rows:
        raise ValueError("File mapping không có dữ liệu")

    # Kiểm tra đủ ảnh trước khi mất thời gian tải model
    validate_keyframe_paths(mapping_rows)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Thiết bị:", device)
    print("Model:", MODEL_NAME)
    print("Checkpoint:", PRETRAINED)
    print("Số keyframe:", len(mapping_rows))
    print("Đang tải CLIP model...")

    model, _, preprocess = (
        open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
        )
    )

    model = model.to(device)
    model.eval()

    groups = group_by_video(mapping_rows)
    video_files = {}
    total_vectors = 0
    dimension = None

    for video_id in sorted(groups.keys()):
        rows = groups[video_id]
        out_path = FEATURES_DIR / f"{video_id}.npy"

        if RESUME and out_path.exists():
            try:
                existing = np.load(out_path, mmap_mode="r")
            except (OSError, ValueError):
                print(f"  - {video_id}: NPY lỗi, đang encode lại")
            else:
                if existing.ndim == 2 and existing.shape[0] == len(rows):
                    existing_dimension = int(existing.shape[1])

                    if dimension is None:
                        dimension = existing_dimension
                    elif dimension != existing_dimension:
                        raise ValueError(
                            f"Video {video_id}: dimension "
                            f"{existing_dimension} không khớp {dimension}"
                        )

                    video_files[video_id] = {
                        "path": out_path.relative_to(PROJECT_DIR).as_posix(),
                        "vector_count": int(existing.shape[0]),
                    }
                    total_vectors += int(existing.shape[0])
                    print(
                        f"  - {video_id}: đã có {existing.shape[0]} vector, "
                        "bỏ qua (resume)"
                    )
                    continue

        features = encode_rows(rows, model, preprocess, device)

        if features.shape[0] != len(rows):
            raise RuntimeError(
                f"Video {video_id}: encode được {features.shape[0]} "
                f"vector nhưng có {len(rows)} keyframe"
            )

        if dimension is None:
            dimension = int(features.shape[1])

        np.save(out_path, features)

        video_files[video_id] = {
            "path": out_path.relative_to(PROJECT_DIR).as_posix(),
            "vector_count": int(features.shape[0]),
        }
        total_vectors += features.shape[0]

        print(
            f"  - {video_id}: {features.shape[0]} vector "
            f"→ {out_path.name}"
        )

    if dimension is None:
        raise RuntimeError(
            "Không có video nào được encode — kiểm tra lại mapping"
        )

    manifest = {
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "dimension": dimension,
        "dtype": "float32",
        "normalized": True,
        "vector_count": int(total_vectors),
        "videos": video_files,
        "mapping_path": MAPPING_PATH.relative_to(PROJECT_DIR).as_posix(),
        "features_dir": FEATURES_DIR.relative_to(PROJECT_DIR).as_posix(),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Đã tạo features theo từng video")
    print("Thư mục:", FEATURES_DIR)
    print("Tổng số vector:", total_vectors)
    print("Manifest:", MANIFEST_PATH)


if __name__ == "__main__":
    main()
