"""Đọc và chuẩn hóa Object Detection JSON từ thư mục hoặc ZIP.

Module này không phụ thuộc mapping. Vì vậy nó được dùng chung cho hai luồng:

- dựng catalog nhãn từ một batch Object tạm;
- dựng retrieval index khi Object JSON đã khớp với keyframe mapping.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


CLASS_KEYS = (
    "detection_class_entities",
    "detection_class_names",
    "classes",
    "names",
)
SCORE_KEYS = ("detection_scores", "scores")
BOX_KEYS = ("detection_boxes", "boxes")
SIDECAR_NAMES = {
    "object_manifest.json",
    "dataset_manifest.json",
    "validation_report.json",
}


@dataclass(frozen=True)
class ObjectDocument:
    """Một JSON Object cùng định danh keyframe suy ra từ nội dung/đường dẫn."""

    source: str
    logical_id: str
    video_id: str
    keyframe_stem: str
    raw: object
    content_sha256: str


@lru_cache(maxsize=4096)
def clean_entity(entity):
    """Chuẩn hóa tên class, kể cả chuỗi dạng ``b'Person'`` của TF Hub."""

    if entity is None or not isinstance(entity, (str, bytes)):
        return ""
    if isinstance(entity, bytes):
        text = entity.decode("utf-8", errors="replace")
    else:
        text = str(entity)
    text = " ".join(text.strip().split())

    if text.startswith(("b'", 'b"')) and text.endswith(("'", '"')):
        text = text[2:-1]

    return " ".join(text.strip().split())


def box_area(box):
    """Diện tích box chuẩn hóa ``[ymin, xmin, ymax, xmax]``.

    Khi nguồn không có bounding box, trả 0 thay vì giả định box phủ toàn ảnh.
    """

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return 0.0

    ymin, xmin, ymax, xmax = box
    height = max(0.0, float(ymax) - float(ymin))
    width = max(0.0, float(xmax) - float(xmin))
    return height * width


def parse_detections(raw, source="<memory>", include_area=True):
    """Trả ``[(class_name, score, area), ...]`` từ các schema phổ biến."""

    detections = []

    if isinstance(raw, dict):
        if isinstance(raw.get("objects"), list):
            return parse_detections(
                raw["objects"], source, include_area=include_area
            )

        classes = next((raw[key] for key in CLASS_KEYS if key in raw), None)
        scores = next((raw[key] for key in SCORE_KEYS if key in raw), None)
        boxes = next((raw[key] for key in BOX_KEYS if key in raw), None)

        if classes is None or scores is None:
            raise ValueError(
                f"{source} thiếu trường class/score. Cần một trong "
                f"{CLASS_KEYS} và {SCORE_KEYS}. Nhận được: {list(raw.keys())}"
            )
        if not isinstance(classes, (list, tuple)) or not isinstance(
            scores, (list, tuple)
        ):
            raise ValueError(f"{source}: class và score phải là danh sách")
        if len(classes) != len(scores):
            raise ValueError(
                f"{source} có {len(classes)} class nhưng {len(scores)} score"
            )
        if boxes is not None and len(boxes) != len(classes):
            raise ValueError(
                f"{source} có {len(classes)} class nhưng {len(boxes)} box"
            )

        for position, entity in enumerate(classes):
            box = boxes[position] if boxes is not None else None
            detections.append(
                (
                    clean_entity(entity),
                    float(scores[position]),
                    box_area(box) if include_area else 0.0,
                )
            )
        return detections

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError(
                    f"{source}: phần tử detection phải là dict, nhận "
                    f"{type(item).__name__}"
                )
            entity = (
                item.get("class")
                or item.get("name")
                or item.get("label")
                or item.get("detection_class_entities")
            )
            score = item.get("score")
            if score is None:
                score = item.get("detection_scores")
            box = item.get("box") or item.get("bbox") or item.get(
                "detection_boxes"
            )

            if entity is None or score is None:
                raise ValueError(
                    f"{source}: phần tử thiếu class hoặc score: {item}"
                )
            detections.append(
                (
                    clean_entity(entity),
                    float(score),
                    box_area(box) if include_area else 0.0,
                )
            )
        return detections

    raise ValueError(
        f"{source}: JSON phải là dict hoặc list, nhận "
        f"{type(raw).__name__}"
    )


_VIDEO_ID_PATTERN = re.compile(r"^(?:[a-z]+\d+_)?v\d+$", re.IGNORECASE)
_COMPOSITE_KEYFRAME_PATTERN = re.compile(
    r"^(?P<video>.+)_k(?P<stem>\d+)$",
    re.IGNORECASE,
)


def _raw_keyframe_locator(value):
    """Chuẩn hóa keyframe_name/keyframe_id; không dùng frame_index mơ hồ."""

    if value is None:
        return None, None
    key_path = PurePosixPath(str(value).replace("\\", "/"))
    embedded_video = (
        key_path.parent.name
        if str(key_path.parent) not in ("", ".")
        else None
    )
    stem = Path(key_path.name).stem
    composite = _COMPOSITE_KEYFRAME_PATTERN.fullmatch(stem)
    if composite:
        embedded_video = embedded_video or composite.group("video")
        stem = composite.group("stem")
    return embedded_video, stem or None


def _locator_from_raw(raw, fallback_path):
    """Suy locator và hard-error nếu ID JSON mâu thuẫn đường dẫn AIC."""

    fallback = PurePosixPath(str(fallback_path).replace("\\", "/"))
    path_video = fallback.parent.name
    path_stem = fallback.stem

    raw_video = None
    raw_stem = None
    if isinstance(raw, dict):
        raw_video = raw.get("video_id")
        raw_keyframe = raw.get("keyframe_name") or raw.get("keyframe_id")
        embedded_video, raw_stem = _raw_keyframe_locator(raw_keyframe)
        if raw_video and embedded_video and str(raw_video) != embedded_video:
            raise ValueError(
                "video_id trong JSON mâu thuẫn keyframe ID: "
                f"{raw_video!r} != {embedded_video!r}"
            )
        raw_video = str(raw_video or embedded_video or "").strip() or None

    # Với cấu trúc chuẩn <video_id>/<keyframe>.json, đường dẫn là nguồn định
    # danh chính. Raw metadata chỉ được dùng để đối chiếu, không được âm thầm
    # chuyển detection sang frame khác.
    structured_path = bool(_VIDEO_ID_PATTERN.fullmatch(path_video or ""))
    if raw_video and raw_video == path_video:
        structured_path = True
    if structured_path:
        if raw_video and raw_video != path_video:
            raise ValueError(
                "video_id trong JSON mâu thuẫn đường dẫn: "
                f"{raw_video!r} != {path_video!r}"
            )
        if raw_stem and raw_stem != path_stem:
            raise ValueError(
                "keyframe trong JSON mâu thuẫn tên file: "
                f"{raw_stem!r} != {path_stem!r}"
            )
        video_id, keyframe_stem = path_video, path_stem
    else:
        # Hỗ trợ một JSON standalone nếu nó tự mang video_id/keyframe_name.
        video_id = raw_video or path_video
        keyframe_stem = raw_stem or path_stem

    if not video_id or not keyframe_stem:
        raise ValueError(f"Không suy ra được video/keyframe từ {fallback_path}")

    return str(video_id), str(keyframe_stem)


def _decode_document(payload, source, fallback_path):
    raw = json.loads(payload.decode("utf-8-sig"))
    video_id, keyframe_stem = _locator_from_raw(raw, fallback_path)
    canonical = json.dumps(
        raw,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ObjectDocument(
        source=source,
        logical_id=f"{video_id}/{keyframe_stem}",
        video_id=video_id,
        keyframe_stem=keyframe_stem,
        raw=raw,
        content_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def iter_object_documents(inputs, on_error=None):
    """Duyệt streaming Object JSON trong một hay nhiều file/thư mục/ZIP.

    Mặc định mọi lỗi đọc/parse đều được ném ra để retrieval index không thể
    âm thầm thiếu dữ liệu. Catalog thăm dò có thể truyền ``on_error`` để ghi
    nhận một tài liệu hỏng rồi tiếp tục quét các tài liệu còn lại.
    """

    for raw_input in inputs:
        input_path = Path(raw_input)

        if not input_path.exists():
            raise FileNotFoundError(f"Không tìm thấy nguồn Object: {input_path}")

        if input_path.is_dir():
            sources = sorted(
                path
                for path in input_path.rglob("*")
                if path.is_file() and path.suffix.casefold() == ".json"
                and path.name.casefold() not in SIDECAR_NAMES
            )
            for source in sources:
                try:
                    payload = source.read_bytes()
                    yield _decode_document(payload, str(source), source)
                except Exception as error:
                    if on_error is None:
                        raise
                    on_error(str(source), error)
            continue

        if zipfile.is_zipfile(input_path):
            with zipfile.ZipFile(input_path) as archive:
                # Giữ thứ tự vật lý trong ZIP. Sort 177k entry rồi đọc ngẫu
                # nhiên làm Windows phải seek liên tục và chậm hơn rất nhiều;
                # output cuối vẫn deterministic vì labels/concepts được sort.
                entries = (
                    entry
                    for entry in archive.infolist()
                    if not entry.is_dir()
                    and PurePosixPath(entry.filename).suffix.casefold()
                    == ".json"
                    and PurePosixPath(entry.filename).name.casefold()
                    not in SIDECAR_NAMES
                )
                for entry in entries:
                    source = f"{input_path}!/{entry.filename}"
                    try:
                        payload = archive.read(entry)
                        yield _decode_document(payload, source, entry.filename)
                    except Exception as error:
                        if on_error is None:
                            raise
                        on_error(source, error)
            continue

        if input_path.suffix.casefold() != ".json":
            raise ValueError(
                f"Nguồn Object phải là thư mục, ZIP hoặc JSON: {input_path}"
            )

        try:
            payload = input_path.read_bytes()
            yield _decode_document(payload, str(input_path), input_path)
        except Exception as error:
            if on_error is None:
                raise
            on_error(str(input_path), error)
