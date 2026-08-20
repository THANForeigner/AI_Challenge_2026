"""ASR retriever: tìm keyframe theo lời thoại trong video.

Mỗi video dùng một file ``data/asr/<video_id>.json``. Hỗ trợ cả
``start_ms``/``end_ms`` (mili giây) và ``start``/``end`` (giây).

Nếu mapping có ``timestamp_ms`` thật, retriever dùng trực tiếp. Nếu chưa có,
retriever phân bố các keyframe theo thứ tự trên toàn thời lượng transcript.
Đây là fallback để chạy thử; kết quả nộp bài vẫn nên dùng timestamp thật.
"""

import json
import re
import unicodedata
from pathlib import Path

from search_types import PROJECT_DIR, load_mapping, result_from_row
from object_retriever import STOPWORDS
from stdio_setup import configure_stdio


configure_stdio()


ASR_DIR = PROJECT_DIR / "data" / "asr"


def tokenize(text):
    """Tách token Unicode để giữ được tiếng Việt có dấu."""

    normalized = unicodedata.normalize("NFC", str(text).casefold())
    return set(re.findall(r"[^\W_]+", normalized, re.UNICODE))


def segment_bounds_ms(segment):
    """Đọc thời gian segment và chuẩn hóa về mili giây."""

    if segment.get("start_ms") is not None:
        start_ms = int(float(segment["start_ms"]))
    elif segment.get("start") is not None:
        start_ms = int(float(segment["start"]) * 1000)
    else:
        raise ValueError("segment thiếu start_ms/start")

    if segment.get("end_ms") is not None:
        end_ms = int(float(segment["end_ms"]))
    elif segment.get("end") is not None:
        end_ms = int(float(segment["end"]) * 1000)
    else:
        raise ValueError("segment thiếu end_ms/end")

    if start_ms < 0 or end_ms < start_ms:
        raise ValueError(
            f"segment có khoảng thời gian không hợp lệ: {start_ms}-{end_ms}"
        )

    return start_ms, end_ms


class AsrRetriever:
    name = "asr"

    def __init__(self):
        self._segments = None
        self._mapping = None
        self._usable = None

    def available(self):
        if self._usable is not None:
            return self._usable

        json_paths = list(ASR_DIR.glob("*.json")) if ASR_DIR.exists() else []

        if not json_paths:
            self._usable = False
            return False

        mapping = load_mapping()
        self._usable = bool(mapping)
        return self._usable

    @staticmethod
    def _resolve_video_id(json_path, raw, mapping_video_ids, file_count):
        """Ghép transcript với video trong mapping."""

        candidates = [json_path.stem]

        if isinstance(raw, dict) and raw.get("video_id"):
            candidates.append(str(raw["video_id"]))

        for candidate in candidates:
            if candidate in mapping_video_ids:
                return candidate

        # Hỗ trợ bộ test có thư mục keyframe tên ``testing`` nhưng ASR vẫn
        # mang video_id gốc. Chỉ tự ghép khi hoàn toàn không mơ hồ.
        if len(mapping_video_ids) == 1 and file_count == 1:
            resolved = next(iter(mapping_video_ids))
            print(
                f"ASR: ánh xạ transcript {json_path.name} "
                f"→ video {resolved} (bộ dữ liệu chỉ có một video)."
            )
            return resolved

        return None

    @staticmethod
    def _keyframes_with_timestamps(keyframes, duration_ms, video_id):
        """Bổ sung timestamp xấp xỉ cho các keyframe còn thiếu."""

        if all(row.get("timestamp_ms") is not None for row in keyframes):
            return keyframes

        if duration_ms <= 0:
            return []

        ordered = sorted(
            keyframes,
            key=lambda row: (int(row["frame_index"]), row["vector_index"]),
        )
        frame_values = [int(row["frame_index"]) for row in ordered]
        first_frame = min(frame_values)
        last_frame = max(frame_values)
        denominator = last_frame - first_frame
        timestamped = []

        for position, row in enumerate(ordered):
            copied = dict(row)

            if copied.get("timestamp_ms") is None:
                if denominator > 0:
                    ratio = (
                        (int(copied["frame_index"]) - first_frame)
                        / denominator
                    )
                elif len(ordered) > 1:
                    ratio = position / (len(ordered) - 1)
                else:
                    ratio = 0.0

                copied["timestamp_ms"] = round(ratio * duration_ms)

            timestamped.append(copied)

        print(
            f"ASR: {video_id} thiếu timestamp thật; đang nội suy "
            f"{len(timestamped)} keyframe trên {duration_ms / 1000:.1f} giây."
        )
        return timestamped

    def _ensure_loaded(self):
        if self._segments is not None:
            return

        self._mapping = load_mapping()
        self._segments = []
        self._keyframes_by_video = {}
        mapping_keyframes = {}

        for row in self._mapping:
            mapping_keyframes.setdefault(row["video_id"], []).append(row)

        mapping_video_ids = set(mapping_keyframes)
        json_paths = sorted(ASR_DIR.glob("*.json"))
        durations = {}

        for json_path in json_paths:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            video_id = self._resolve_video_id(
                json_path, raw, mapping_video_ids, len(json_paths)
            )

            if video_id is None:
                print(
                    f"ASR: bỏ qua {json_path.name} vì không tìm thấy "
                    "video_id tương ứng trong mapping."
                )
                continue

            segments = (
                raw.get("segments", [])
                if isinstance(raw, dict)
                else raw
            )

            for position, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    raise ValueError(
                        f"{json_path.name}: segment {position} phải là object"
                    )

                start_ms, end_ms = segment_bounds_ms(segment)
                durations[video_id] = max(
                    durations.get(video_id, 0), end_ms
                )
                self._segments.append(
                    {
                        "video_id": video_id,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": str(segment.get("text", "")).strip(),
                        "tokens": tokenize(segment.get("text", "")),
                    }
                )

        for video_id, keyframes in mapping_keyframes.items():
            self._keyframes_by_video[video_id] = (
                self._keyframes_with_timestamps(
                    keyframes, durations.get(video_id, 0), video_id
                )
            )

    def context_for_frame(self, video_id, frame_index, window_ms=15000):
        """Lấy lời thoại gần một frame để bổ sung bằng chứng cho Q&A."""

        self._ensure_loaded()
        keyframes = self._keyframes_by_video.get(video_id, [])

        if not keyframes:
            return ""

        nearest = min(
            keyframes,
            key=lambda row: abs(int(row["frame_index"]) - int(frame_index)),
        )
        center = int(nearest["timestamp_ms"])
        texts = [
            segment.get("text", "")
            for segment in self._segments
            if segment["video_id"] == video_id
            and segment["end_ms"] >= center - window_ms
            and segment["start_ms"] <= center + window_ms
            and segment.get("text")
        ]
        return " ".join(dict.fromkeys(texts))

    def _keyframes_in_segment(self, segment):
        keyframes = self._keyframes_by_video.get(
            segment["video_id"], []
        )

        matched = [
            row
            for row in keyframes
            if segment["start_ms"]
            <= row["timestamp_ms"]
            <= segment["end_ms"]
        ]

        # Không có keyframe nào rơi đúng segment: lấy keyframe gần nhất
        if not matched and keyframes:
            center = (segment["start_ms"] + segment["end_ms"]) // 2
            matched = [
                min(
                    keyframes,
                    key=lambda row: abs(row["timestamp_ms"] - center),
                )
            ]

        return matched

    def search(self, query, top_k=100):
        if not self.available():
            return None

        self._ensure_loaded()

        mapping = self._mapping
        if mapping is None:
            raise RuntimeError("ASR mapping was not loaded")

        query_tokens = tokenize(query) - STOPWORDS

        if not query_tokens or not self._segments:
            return None

        scores = {}

        for segment in self._segments:
            overlap = len(query_tokens & segment["tokens"])

            if overlap == 0:
                continue

            segment_score = overlap / len(query_tokens)

            for row in self._keyframes_in_segment(segment):
                vector_index = row["vector_index"]
                scores[vector_index] = max(
                    scores.get(vector_index, 0.0), segment_score
                )

        if not scores:
            return None

        ranked = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )[:top_k]

        results = []

        for rank, (vector_index, score) in enumerate(ranked, start=1):
            row = mapping[int(vector_index)]
            result = result_from_row(row, asr_score=float(score))
            result.rank = rank
            results.append(result)

        return results
