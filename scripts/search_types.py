"""
Kiểu dữ liệu và tiện ích dùng chung (C2).

Mọi retriever (visual / object / ocr / asr) trả về list[SearchResult];
fusion và search_engine điền các cột điểm tương ứng.
"""

import csv
import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from stdio_setup import configure_stdio


configure_stdio()


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MAPPING_PATH = ARTIFACTS_DIR / "clip_row_mapping.jsonl"
INDEX_PATH = ARTIFACTS_DIR / "clip.index"


@lru_cache(maxsize=1)
def mapping_sha256():
    """Dấu vân tay dùng để ngăn các index ghép nhầm với mapping khác."""

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy mapping: {MAPPING_PATH}")

    digest = hashlib.sha256()
    with MAPPING_PATH.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class SearchResult:
    rank: int = 0
    video_id: str = ""
    keyframe_id: str = ""
    frame_index: int = 0
    timestamp_ms: Optional[int] = None
    clip_score: float = 0.0
    object_score: float = 0.0
    ocr_score: float = 0.0
    asr_score: float = 0.0
    final_score: float = 0.0
    keyframe_path: str = ""
    # Nội bộ: vị trí của keyframe trong FAISS index
    vector_index: int = -1

    @property
    def answer(self):
        """Cặp trả lời theo định dạng nộp bài BTC."""

        return {
            "video_id": self.video_id,
            "frame_id": self.frame_index,
        }


def make_keyframe_id(video_id, keyframe_name):
    """keyframe_id thống nhất: '<video_id>/<số keyframe>'."""

    return f"{video_id}/{Path(keyframe_name).stem}"


def relative_keyframe_path(video_id, keyframe_name):
    """Đường dẫn keyframe ổn định để trao đổi với backend."""

    return (
        Path("data") / "keyframes" / video_id / Path(keyframe_name).name
    ).as_posix()


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

    expected = list(range(len(rows)))
    actual = [row["vector_index"] for row in rows]

    if actual != expected:
        raise ValueError(
            "vector_index trong mapping không liên tục từ 0. "
            "Chạy lại scripts/build_mapping.py."
        )

    return rows


def result_from_row(row, **scores):
    """Tạo SearchResult từ một dòng mapping."""

    return SearchResult(
        video_id=row["video_id"],
        keyframe_id=make_keyframe_id(
            row["video_id"], row["keyframe_name"]
        ),
        frame_index=int(row["frame_index"]),
        timestamp_ms=row.get("timestamp_ms"),
        # Không truyền absolute path từ máy build/Kaggle sang backend.
        keyframe_path=relative_keyframe_path(
            row["video_id"], row["keyframe_name"]
        ),
        vector_index=int(row["vector_index"]),
        **scores,
    )


def video_keyframes(mapping):
    """
    {video_id: [dòng mapping sắp theo frame_index tăng dần]}.
    Dùng cho Q&A (lấy frame lân cận) và TRAKE (dò chuỗi).
    """

    videos = {}

    for row in mapping:
        videos.setdefault(row["video_id"], []).append(row)

    for video_id in videos:
        videos[video_id].sort(key=lambda row: row["frame_index"])

    return videos


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def submission_from_results(results):
    return [result.answer for result in results]


def normalize_qa_answer(answer, max_length=100):
    """Chuẩn hóa Answer thành một dòng và giới hạn theo luật submission."""

    normalized = unicodedata.normalize("NFC", str(answer))
    normalized = " ".join(normalized.split())
    return normalized[:max_length], len(normalized) > max_length


def save_submission_csv(path, query_type, answers, event_count=None):
    """Ghi submission UTF-8, dấu phẩy, LF và không có header."""

    path = Path(path)
    query_type = str(query_type).strip().casefold()

    if query_type not in {"kis", "qa", "trake"}:
        raise ValueError(f"Loại submission không hỗ trợ: {query_type!r}")

    answers = list(answers or [])
    rows = []
    truncated_answers = 0
    skipped_rows = 0

    for position, answer in enumerate(answers, start=1):
        if not isinstance(answer, dict):
            raise ValueError(f"Kết quả thứ {position} phải là object")

        video_id = str(answer.get("video_id", "")).strip()
        if not video_id:
            skipped_rows += 1
            continue

        if query_type == "trake":
            frame_ids = answer.get("frame_ids")
            if not isinstance(frame_ids, (list, tuple)):
                skipped_rows += 1
                continue

            try:
                frame_ids = [int(frame_id) for frame_id in frame_ids]
            except (TypeError, ValueError):
                skipped_rows += 1
                continue

            if event_count is not None and len(frame_ids) != int(event_count):
                skipped_rows += 1
                continue

            if not frame_ids or any(
                current <= previous
                for previous, current in zip(frame_ids, frame_ids[1:])
            ):
                skipped_rows += 1
                continue

            rows.append([video_id, *frame_ids])
            continue

        try:
            frame_id = int(answer["frame_id"])
        except (KeyError, TypeError, ValueError):
            skipped_rows += 1
            continue

        if query_type == "kis":
            rows.append([video_id, frame_id])
            continue

        normalized_answer, truncated = normalize_qa_answer(
            answer.get("answer", "")
        )
        if not normalized_answer:
            skipped_rows += 1
            continue

        truncated_answers += int(truncated)
        rows.append([video_id, frame_id, normalized_answer])

    if answers and not rows:
        raise ValueError("Không có dòng submission hợp lệ để ghi CSV")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(
            file,
            delimiter=",",
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writerows(rows)

    return {
        "path": str(path),
        "row_count": len(rows),
        "truncated_answers": truncated_answers,
        "skipped_rows": skipped_rows,
    }


def print_submission_results(results, header_lines=None):
    print()
    print("=" * 60)

    for line in header_lines or []:
        print(line)

    print("=" * 60)
    print("Kết quả theo định dạng nộp bài BTC: <video_id>, <frame_id>")
    print()

    for result in results:
        print(
            f"{result.rank:>3}. "
            f"{result.video_id}, {result.frame_index} "
            f"(final={result.final_score:.4f}, "
            f"clip={result.clip_score:.4f}, "
            f"obj={result.object_score:.4f}, "
            f"ocr={result.ocr_score:.4f}, "
            f"asr={result.asr_score:.4f})"
        )


def serialize_results(results):
    return [asdict(result) for result in results]
