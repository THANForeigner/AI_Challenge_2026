"""
Bước 5 của pipeline: dựng inverted index vật thể từ kết quả
Faster R-CNN pretrained trên OpenImages V4 (định dạng JSON theo
tutorial object detection của TensorFlow Hub).

Dữ liệu đầu vào (theo quy tắc BTC):
- data/objects/<video_id>/<keyframe>.json — tên file JSON trùng tên
  file keyframe (vd: keyframe L21_V008/001.jpg có L21_V008/001.json).
- Mỗi JSON chứa: detection_boxes ([ymin, xmin, ymax, xmax] chuẩn hóa),
  detection_scores, detection_class_entities.

Kết quả:
- artifacts/object_index.json: inverted index
  {tên class: [[vector_index, score, area], ...]} + vocabulary.
"""

import argparse
import json
import math
from pathlib import Path

from stdio_setup import configure_stdio


configure_stdio()


PROJECT_DIR = Path(__file__).resolve().parents[1]

OBJECT_DIR = PROJECT_DIR / "data" / "objects"
MAPPING_PATH = PROJECT_DIR / "artifacts" / "clip_row_mapping.jsonl"
INDEX_PATH = PROJECT_DIR / "artifacts" / "object_index.json"

# Notebook baseline của BTC chỉ đưa detection có confidence > 0.4 vào
# FiftyOne. Giữ cùng ngưỡng mặc định để object index không bị nhiễu bởi
# hàng nghìn detection độ tin cậy thấp.
BASELINE_MIN_SCORE = 0.4

# Các biến thể tên trường có thể gặp trong JSON
CLASS_KEYS = (
    "detection_class_entities",
    "detection_class_names",
    "classes",
    "names",
)
SCORE_KEYS = ("detection_scores", "scores")
BOX_KEYS = ("detection_boxes", "boxes")


def clean_entity(entity):
    """Chuẩn hóa tên class: bỏ dạng byte repr như b'Person'."""

    text = str(entity).strip()

    if text.startswith(("b'", 'b"')) and text.endswith(("'", '"')):
        text = text[2:-1]

    return text.strip()


def box_area(box):
    """Diện tích box chuẩn hóa [ymin, xmin, ymax, xmax]."""

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return 0.0

    ymin, xmin, ymax, xmax = box

    height = max(0.0, float(ymax) - float(ymin))
    width = max(0.0, float(xmax) - float(xmin))

    return height * width


def parse_detections(raw, source):
    """
    Đọc danh sách detection từ JSON. Hỗ trợ 2 dạng:
    - dict các mảng (định dạng TF Hub): {"detection_boxes": [...], ...}
    - list các dict: [{"class": ..., "score": ..., "box": [...]}, ...]
    """

    detections = []

    if isinstance(raw, dict):
        # Format do pipeline object detection của đội xuất:
        # {"video_id": ..., "keyframe_id": ..., "objects": [
        #   {"label": "Person", "score": 0.9, "bbox": [...]}
        # ]}
        if isinstance(raw.get("objects"), list):
            return parse_detections(raw["objects"], source)

        classes = None
        scores = None
        boxes = None

        for key in CLASS_KEYS:
            if key in raw:
                classes = raw[key]
                break

        for key in SCORE_KEYS:
            if key in raw:
                scores = raw[key]
                break

        for key in BOX_KEYS:
            if key in raw:
                boxes = raw[key]
                break

        if classes is None or scores is None:
            raise ValueError(
                f"{source} thiếu trường class/score. "
                f"Cần một trong {CLASS_KEYS} và {SCORE_KEYS}. "
                f"Nhận được các khóa: {list(raw.keys())}"
            )

        if len(classes) != len(scores):
            raise ValueError(
                f"{source} có {len(classes)} class "
                f"nhưng {len(scores)} score"
            )

        if boxes is not None and len(boxes) != len(classes):
            raise ValueError(
                f"{source} có {len(classes)} class "
                f"nhưng {len(boxes)} bounding box"
            )

        for position in range(len(classes)):
            box = boxes[position] if boxes is not None else [0, 0, 1, 1]

            detections.append(
                (
                    clean_entity(classes[position]),
                    float(scores[position]),
                    box_area(box),
                )
            )

        return detections

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError(
                    f"{source}: phần tử trong list phải là dict, "
                    f"nhận được {type(item).__name__}"
                )

            entity = (
                item.get("class")
                or item.get("name")
                or item.get("label")
                or item.get("detection_class_entities")
            )
            score = (
                item.get("score")
                if item.get("score") is not None
                else item.get("detection_scores")
            )
            box = (
                item.get("box")
                or item.get("bbox")
                or item.get("detection_boxes")
                or [0, 0, 1, 1]
            )

            if entity is None or score is None:
                raise ValueError(
                    f"{source}: phần tử thiếu class hoặc score: {item}"
                )

            detections.append(
                (clean_entity(entity), float(score), box_area(box))
            )

        return detections

    raise ValueError(
        f"{source}: JSON phải là dict hoặc list, "
        f"nhận được {type(raw).__name__}"
    )


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


def build_index(min_score=BASELINE_MIN_SCORE):
    if not OBJECT_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục object: {OBJECT_DIR}. "
            "Cần đặt JSON theo layout: "
            "data/objects/<video_id>/<keyframe>.json"
        )

    mapping_rows = load_mapping()

    inverted_index = {}
    vocabulary = {}
    raw_detections = 0
    total_detections = 0
    filtered_detections = 0
    missing_count = 0
    video_stats = {}

    for row in mapping_rows:
        vector_index = row["vector_index"]
        video_id = row["video_id"]
        stem = Path(row["keyframe_name"]).stem

        json_path = OBJECT_DIR / video_id / f"{stem}.json"

        if not json_path.exists():
            missing_count += 1
            continue

        raw = json.loads(json_path.read_text(encoding="utf-8"))
        detections = parse_detections(raw, str(json_path))

        video_stats[video_id] = video_stats.get(video_id, 0) + 1

        for entity, score, area in detections:
            raw_detections += 1

            # Baseline dùng điều kiện nghiêm ngặt score > 0.4.
            if not math.isfinite(score) or score <= min_score:
                filtered_detections += 1
                continue

            class_key = entity.lower()

            if class_key not in inverted_index:
                inverted_index[class_key] = []
                vocabulary[class_key] = entity

            inverted_index[class_key].append(
                [vector_index, float(score), round(area, 6)]
            )

            total_detections += 1

    if not inverted_index:
        raise ValueError(
            "Không đọc được detection nào. Kiểm tra lại định dạng "
            "JSON trong data/objects/."
        )

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "min_score_exclusive": float(min_score),
            "score_rule": f"score > {min_score}",
        },
        "stats": {
            "raw_detections": raw_detections,
            "indexed_detections": total_detections,
            "filtered_detections": filtered_detections,
            "mapped_keyframes": len(mapping_rows),
            "missing_object_files": missing_count,
        },
        "vocabulary": vocabulary,
        "inverted_index": inverted_index,
    }

    INDEX_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Đã tạo object index")
    print(f"Ngưỡng baseline: score > {min_score}")
    print(f"Số class: {len(inverted_index)}")
    print(f"Số detection gốc: {raw_detections}")
    print(f"Số detection được index: {total_detections}")
    print(f"Số detection bị lọc: {filtered_detections}")
    print(
        f"Keyframe có object: "
        f"{len(mapping_rows) - missing_count}/{len(mapping_rows)}"
    )

    if missing_count:
        print(
            f"Cảnh báo: {missing_count} keyframe thiếu file JSON "
            f"trong {OBJECT_DIR}"
        )

    for video_id in sorted(video_stats.keys()):
        print(f"  - {video_id}: {video_stats[video_id]} keyframe")

    print(f"Index: {INDEX_PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="Tạo object index theo ngưỡng confidence của baseline"
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=BASELINE_MIN_SCORE,
        help="Chỉ index detection có score lớn hơn giá trị này (mặc định 0.4)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.min_score < 1.0:
        raise ValueError("--min_score phải nằm trong khoảng [0, 1)")

    build_index(min_score=args.min_score)


if __name__ == "__main__":
    main()
