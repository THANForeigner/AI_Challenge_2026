"""
Bước 1 của pipeline CLIP + FAISS: dựng file mapping keyframe.

Dữ liệu đầu vào theo layout của BTC:
- data/keyframes/<video_id>/0000.jpg, 0001.jpg, ...
  (thư mục đặt theo tên file video, keyframe đánh số tăng dần)
- data/metadata/<video_id>.json hoặc MỘT file JSON gộp tất cả video,
  ghi frame index thực của mỗi keyframe trong video.
- Hoặc data/metadata/keyframes.jsonl, mỗi dòng là một keyframe có các trường
  video_id, frame_index, timestamp_ms, keyframe_path và shot_id.

Kết quả:
- artifacts/clip_row_mapping.jsonl, mỗi dòng một keyframe:
  vector_index, video_id, keyframe_name, frame_index, keyframe_path

frame_index là frame thật trong video — đây chính là giá trị dùng
làm <frame_id> khi nộp bài cho BTC.
"""

import json
from pathlib import Path

from stdio_setup import configure_stdio


configure_stdio()


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

PROJECT_DIR = Path(__file__).resolve().parents[1]

KEYFRAME_DIR = PROJECT_DIR / "data" / "keyframes"
METADATA_DIR = PROJECT_DIR / "data" / "metadata"
MAPPING_PATH = PROJECT_DIR / "artifacts" / "clip_row_mapping.jsonl"


def parse_keyframe_number(path):
    """Lấy số thứ tự keyframe từ tên file (0000.jpg -> 0)."""

    try:
        return int(path.stem)
    except ValueError:
        return None


def scan_keyframes():
    """Quét data/keyframes/<video_id>/xxxx.jpg theo layout BTC."""

    if not KEYFRAME_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục keyframe: {KEYFRAME_DIR}"
        )

    videos = {}

    for video_dir in sorted(KEYFRAME_DIR.iterdir()):
        if not video_dir.is_dir():
            continue

        keyframes = []

        for image_path in video_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            number = parse_keyframe_number(image_path)

            if number is None:
                print(
                    f"Cảnh báo: bỏ qua file không đánh số: {image_path}"
                )
                continue

            keyframes.append((number, image_path))

        if not keyframes:
            continue

        keyframes.sort(key=lambda item: item[0])

        numbers = [number for number, _ in keyframes]

        if len(set(numbers)) != len(numbers):
            raise ValueError(
                f"Thư mục {video_dir.name} có keyframe trùng số thứ tự"
            )

        videos[video_dir.name] = keyframes

    if not videos:
        raise ValueError(
            f"Không có keyframe nào trong {KEYFRAME_DIR}. "
            "Dữ liệu phải đặt theo layout BTC: "
            "data/keyframes/<video_id>/0000.jpg ..."
        )

    return videos


def parse_frame_index_map(raw, video_id, keyframe_numbers=None):
    """
    Chuyển metadata JSON về dạng {số thứ tự keyframe: frame index}.

    Hỗ trợ 2 định dạng:
    - dict: {"0": 60, "1": 120} hoặc {"0000.jpg": 60}
    - list: [60, 120, ...] (frame index theo thứ tự keyframe)
    """

    if isinstance(raw, list):
        numbers = (
            list(range(len(raw)))
            if keyframe_numbers is None
            else list(keyframe_numbers)
        )

        if len(raw) != len(numbers):
            raise ValueError(
                f"Metadata của {video_id} có {len(raw)} frame index "
                f"nhưng có {len(numbers)} keyframe"
            )

        return {
            number: int(frame_index)
            for number, frame_index in zip(numbers, raw)
        }

    if isinstance(raw, dict):
        frame_index_map = {}

        for key, value in raw.items():
            number = parse_keyframe_number(Path(str(key)))

            if number is None:
                raise ValueError(
                    f"Metadata của {video_id} có khóa không phải "
                    f"số thứ tự keyframe: {key!r}"
                )

            frame_index_map[number] = int(value)

        return frame_index_map

    raise ValueError(
        f"Metadata của {video_id} phải là dict hoặc list, "
        f"nhận được: {type(raw).__name__}"
    )


def load_jsonl_metadata(video_ids):
    """Đọc metadata JSONL gộp theo kiểu streaming và chỉ giữ video cần dùng."""

    wanted = set(video_ids)
    frame_index_maps = {video_id: {} for video_id in video_ids}
    details = {}
    jsonl_candidates = sorted(METADATA_DIR.glob("*.jsonl"))

    for jsonl_path in jsonl_candidates:
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSONL lỗi tại {jsonl_path}:{line_number}: {error}"
                    ) from error

                if not isinstance(row, dict):
                    raise ValueError(
                        f"JSONL {jsonl_path}:{line_number} phải là object"
                    )

                video_id = str(row.get("video_id", "")).strip()

                if video_id not in wanted:
                    continue

                keyframe_path = Path(str(row.get("keyframe_path", "")))
                number = parse_keyframe_number(keyframe_path)

                if number is None:
                    keyframe_id = str(row.get("keyframe_id", ""))
                    marker = keyframe_id.rsplit("_K", 1)

                    if len(marker) == 2 and marker[1].isdigit():
                        number = int(marker[1])

                if number is None or "frame_index" not in row:
                    raise ValueError(
                        f"Metadata thiếu keyframe/frame_index tại "
                        f"{jsonl_path}:{line_number}"
                    )

                frame_index = int(row["frame_index"])
                existing = frame_index_maps[video_id].get(number)

                if existing is not None and existing != frame_index:
                    raise ValueError(
                        f"Metadata trùng nhưng khác frame_index cho "
                        f"{video_id}/{number}: {existing} và {frame_index}"
                    )

                frame_index_maps[video_id][number] = frame_index
                detail = {}

                for field in ("keyframe_id", "timestamp_ms", "shot_id"):
                    if row.get(field) is not None:
                        detail[field] = row[field]

                details[(video_id, number)] = detail

    return frame_index_maps, details, jsonl_candidates


def load_metadata(video_ids, keyframe_numbers_by_video=None):
    """
    Tìm metadata cho từng video. Ưu tiên file riêng
    data/metadata/<video_id>.json; nếu không có thì tìm file JSON
    gộp chứa dict {video_id: {...}}.
    """

    frame_index_maps = {}
    metadata_details = {}

    if not METADATA_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục metadata: {METADATA_DIR}. "
            "Cần file JSON ghi frame index của mỗi keyframe."
        )

    aggregate_candidates = [
        path
        for path in sorted(METADATA_DIR.glob("*.json"))
        if path.stem not in video_ids
    ]

    aggregate_data = None
    aggregate_path = None
    jsonl_maps, jsonl_details, jsonl_candidates = load_jsonl_metadata(video_ids)
    jsonl_videos_used = set()

    for video_id in video_ids:
        keyframe_numbers = (
            keyframe_numbers_by_video.get(video_id)
            if keyframe_numbers_by_video is not None
            else None
        )
        per_video_path = METADATA_DIR / f"{video_id}.json"

        if per_video_path.exists():
            raw = json.loads(
                per_video_path.read_text(encoding="utf-8")
            )
            frame_index_maps[video_id] = parse_frame_index_map(
                raw, video_id, keyframe_numbers
            )
            continue

        if aggregate_data is None and aggregate_candidates:
            aggregate_path = aggregate_candidates[0]
            aggregate_data = json.loads(
                aggregate_path.read_text(encoding="utf-8")
            )

        if (
            isinstance(aggregate_data, dict)
            and video_id in aggregate_data
        ):
            frame_index_maps[video_id] = parse_frame_index_map(
                aggregate_data[video_id], video_id, keyframe_numbers
            )
            continue

        if jsonl_maps.get(video_id):
            frame_index_maps[video_id] = jsonl_maps[video_id]
            jsonl_videos_used.add(video_id)
            continue

        raise FileNotFoundError(
            f"Không tìm thấy metadata cho video {video_id}. "
            f"Cần file {per_video_path} hoặc file gộp "
            f"{aggregate_candidates[0] if aggregate_candidates else '(chưa có)'} "
            f"chứa khóa {video_id!r}, hoặc JSONL trong "
            f"{jsonl_candidates if jsonl_candidates else '(chưa có)'}."
        )

    metadata_details.update(
        {
            key: value
            for key, value in jsonl_details.items()
            if key[0] in jsonl_videos_used
        }
    )
    return frame_index_maps, metadata_details


def main():
    videos = scan_keyframes()
    keyframe_numbers_by_video = {
        video_id: [number for number, _path in keyframes]
        for video_id, keyframes in videos.items()
    }
    frame_index_maps, metadata_details = load_metadata(
        list(videos.keys()), keyframe_numbers_by_video
    )

    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)

    mapping_rows = []
    vector_index = 0

    for video_id in sorted(videos.keys()):
        keyframes = videos[video_id]
        frame_index_map = frame_index_maps[video_id]

        for number, image_path in keyframes:
            if number not in frame_index_map:
                raise ValueError(
                    f"Keyframe {video_id}/{image_path.name} không có "
                    f"frame index trong metadata"
                )

            mapping_row = {
                "vector_index": vector_index,
                "video_id": video_id,
                "keyframe_name": image_path.name,
                "frame_index": int(frame_index_map[number]),
                # Lưu tương đối theo PROJECT_DIR để artifact có thể mang
                # giữa máy local và Kaggle mà không phụ thuộc mount path.
                "keyframe_path": (
                    Path("data")
                    / "keyframes"
                    / video_id
                    / image_path.name
                ).as_posix(),
            }
            mapping_row.update(metadata_details.get((video_id, number), {}))
            mapping_rows.append(mapping_row)

            vector_index += 1

    with MAPPING_PATH.open("w", encoding="utf-8") as file:
        for row in mapping_rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    print("Đã tạo mapping theo layout BTC")
    print(f"Số video: {len(videos)}")
    print(f"Số keyframe: {len(mapping_rows)}")
    print(f"Mapping: {MAPPING_PATH}")

    for video_id in sorted(videos.keys()):
        print(f"  - {video_id}: {len(videos[video_id])} keyframe")


if __name__ == "__main__":
    main()
