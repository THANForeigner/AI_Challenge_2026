"""
TRAKE sequence search (C8).

Với mỗi event trong chuỗi:
    Event j → Top-K_j ứng viên CLIP
Sau đó tìm chuỗi thỏa:
    cùng video_id và frame_1 < frame_2 < ... < frame_n
bằng Dynamic Programming (tổng điểm CLIP lớn nhất).

Nếu không video nào có đủ ứng viên cho mọi event: chọn video có
chuỗi dài nhất, các event thiếu được NỘI SUY theo frame_index rồi
bám vào keyframe gần nhất.

Tách truy vấn thành events (C7) nằm trong query_router.decompose_events.
"""

import os

from search_types import (
    ARTIFACTS_DIR,
    make_keyframe_id,
    relative_keyframe_path,
    save_json,
    video_keyframes,
)
from stdio_setup import configure_stdio


configure_stdio()


def collect_candidates(events, engine, candidate_k):
    """Mỗi event → top ứng viên multimodal, nhóm theo video."""

    from query_translator import translate_for_visual

    per_event = []
    configured_modalities = os.getenv("AIC_TRAKE_MODALITIES", "auto")
    text_cues = (
        "biển báo", "biển cảnh báo", "dòng chữ", "chữ gì", "ghi gì",
        "logo", "màn hình", "phụ đề", "văn bản",
    )

    for event in events:
        if configured_modalities == "auto":
            modalities = ["visual"]
            if any(cue in event.casefold() for cue in text_cues):
                modalities.append("ocr")
        else:
            modalities = [
                name.strip()
                for name in configured_modalities.split(",")
                if name.strip()
            ]

        visual_query = translate_for_visual(event)
        if visual_query != event:
            print(f"  CLIP event: {event} → {visual_query}")
        print(f"  Modalities: {', '.join(modalities)}")
        candidates = engine.search_kis(
            event,
            visual_query=visual_query,
            modalities=modalities,
            top_k=candidate_k,
            pool_k=candidate_k,
            min_gap_frames=0,
            save=False,
        )
        per_event.append(candidates)

    videos = {}

    for event_index, candidates in enumerate(per_event):
        for result in candidates:
            videos.setdefault(result.video_id, {}).setdefault(
                event_index, []
            ).append(result)

    # Trong mỗi video/event: giữ 1 ứng viên cho mỗi frame_index
    # (lấy điểm cao nhất), sắp theo frame tăng dần
    for video_id, by_event in videos.items():
        for event_index, results in by_event.items():
            best_by_frame = {}

            for result in results:
                current = best_by_frame.get(result.frame_index)

                if current is None or candidate_score(result) > candidate_score(current):
                    best_by_frame[result.frame_index] = result

            by_event[event_index] = sorted(
                best_by_frame.values(),
                key=lambda result: result.frame_index,
            )

    return per_event, videos


def candidate_score(result):
    """Điểm dùng cho TRAKE: ưu tiên fusion, fallback về CLIP."""

    return result.final_score if result.final_score > 0 else result.clip_score


def k_best_complete_chains(by_event, event_count, beam_size=30):
    """Beam search các chuỗi đủ event, cùng video và frame tăng nghiêm ngặt."""

    if any(not by_event.get(index) for index in range(event_count)):
        return []

    beams = [
        (candidate_score(candidate), [candidate])
        for candidate in by_event[0]
    ]
    beams.sort(key=lambda item: item[0], reverse=True)
    beams = beams[:beam_size]

    for event_index in range(1, event_count):
        expanded = []

        for score, chain in beams:
            previous_frame = chain[-1].frame_index

            for candidate in by_event[event_index]:
                if candidate.frame_index <= previous_frame:
                    continue
                expanded.append((score + candidate_score(candidate), chain + [candidate]))

        if not expanded:
            return []

        expanded.sort(key=lambda item: item[0], reverse=True)
        unique = []
        seen = set()

        for score, chain in expanded:
            frame_tuple = tuple(result.frame_index for result in chain)
            if frame_tuple in seen:
                continue
            seen.add(frame_tuple)
            unique.append((score, chain))
            if len(unique) >= beam_size:
                break

        beams = unique

    return beams


def best_chain_in_video(by_event, event_count):
    """
    DP trên các event có ứng viên trong video (giữ đúng thứ tự).
    Trả về (event_indices, results) của chuỗi tốt nhất.
    """

    available_events = [
        event_index
        for event_index in range(event_count)
        if event_index in by_event and by_event[event_index]
    ]

    if not available_events:
        return [], []

    # Mỗi state lưu (số event đã phủ, tổng điểm, state trước).
    # Tối ưu độ dài trước rồi mới tối ưu điểm; CLIP có thể
    # trả điểm âm nên chỉ so tổng điểm sẽ làm mất event.
    states = {}
    best_key = None

    for event_index in available_events:
        states[event_index] = []

        for candidate_index, result in enumerate(by_event[event_index]):
            best_state = (1, candidate_score(result), None)

            for previous_event in available_events:
                if previous_event >= event_index:
                    break

                for previous_index, previous_result in enumerate(
                    by_event[previous_event]
                ):
                    if previous_result.frame_index >= result.frame_index:
                        continue

                    previous_state = states[previous_event][previous_index]
                    proposed = (
                        previous_state[0] + 1,
                        previous_state[1] + candidate_score(result),
                        (previous_event, previous_index),
                    )

                    if proposed[:2] > best_state[:2]:
                        best_state = proposed

            states[event_index].append(best_state)
            state_key = (event_index, candidate_index)

            if best_key is None:
                best_key = state_key
            else:
                current_best = states[best_key[0]][best_key[1]]

                if best_state[:2] > current_best[:2]:
                    best_key = state_key

    if best_key is None:
        return [], []

    covered_events = []
    chosen = []
    cursor = best_key

    while cursor is not None:
        event_index, candidate_index = cursor
        covered_events.append(event_index)
        chosen.append(by_event[event_index][candidate_index])
        cursor = states[event_index][candidate_index][2]

    covered_events.reverse()
    chosen.reverse()
    return covered_events, chosen


def nearest_keyframe_frame(video_rows, target_frame):
    """Bám frame nội suy vào keyframe gần nhất của video."""

    best_row = min(
        video_rows,
        key=lambda row: abs(row["frame_index"] - target_frame),
    )

    return int(best_row["frame_index"]), make_keyframe_id(
        best_row["video_id"], best_row["keyframe_name"]
    )


def interpolate_missing(events, chosen_map, video_rows, median_gap):
    """Nội suy frame cho các event thiếu rồi bám keyframe gần nhất."""

    if not video_rows:
        raise ValueError("Không có keyframe trong mapping để nội suy TRAKE")

    known = sorted(chosen_map.keys())
    frames = {}
    keyframe_ids = {}

    for event_index in chosen_map:
        frames[event_index] = chosen_map[event_index].frame_index
        keyframe_ids[event_index] = chosen_map[event_index].keyframe_id

    for event_index in range(len(events)):
        if event_index in frames:
            continue

        previous = next(
            (j for j in reversed(known) if j < event_index), None
        )
        following = next(
            (j for j in known if j > event_index), None
        )

        if previous is not None and following is not None:
            ratio = (event_index - previous) / (following - previous)
            target = frames[previous] + (
                frames[following] - frames[previous]
            ) * ratio
        elif previous is not None:
            target = frames[previous] + median_gap * (
                event_index - previous
            )
        elif following is not None:
            target = frames[following] - median_gap * (
                following - event_index
            )
        else:
            target = 0

        video_frames = [row["frame_index"] for row in video_rows]
        target = max(min(video_frames), min(max(video_frames), target))

        frame, keyframe_id = nearest_keyframe_frame(
            video_rows, int(round(target))
        )

        frames[event_index] = frame
        keyframe_ids[event_index] = keyframe_id

    return frames, keyframe_ids


def video_median_gap(video_rows):
    frames = sorted(row["frame_index"] for row in video_rows)

    diffs = [
        current - previous
        for previous, current in zip(frames, frames[1:])
        if current > previous
    ]

    if not diffs:
        return 1

    diffs.sort()
    return max(1, diffs[len(diffs) // 2])


def search_trake(events, engine, candidate_k=300, answer_k=100):
    """
    Interface C8:
        result = search_trake(
            events=["athlete starts running", "athlete takes off", ...],
            candidate_k=300,
        )
    Trả {"video_id", "frame_ids", ...}.
    """

    if not events:
        raise ValueError("Danh sách events trống")

    if candidate_k <= 0:
        raise ValueError("candidate_k phải lớn hơn 0")

    if answer_k <= 0:
        raise ValueError("answer_k phải lớn hơn 0")

    print("TRAKE — chuỗi sự kiện:")

    for index, event in enumerate(events, start=1):
        print(f"  Event {index}: {event}")

    print()
    print(f"Đang lấy top-{candidate_k} ứng viên cho mỗi event...")

    _per_event, videos = collect_candidates(
        events, engine, candidate_k
    )

    videos_info = video_keyframes(engine.mapping)

    # Ưu tiên chuỗi đầy đủ. Mỗi video đóng góp nhiều chuỗi để có thể nộp
    # Top-100, thay vì đánh cược toàn bộ vào đúng một chuỗi.
    complete_candidates = []

    for video_id, by_event in videos.items():
        for chain_score, chain in k_best_complete_chains(
            by_event, len(events), beam_size=min(answer_k, 100)
        ):
            frame_ids = [result.frame_index for result in chain]
            complete_candidates.append(
                {
                    "video_id": video_id,
                    "frame_ids": frame_ids,
                    "keyframe_ids": [result.keyframe_id for result in chain],
                    "keyframe_paths": [result.keyframe_path for result in chain],
                    "chain_score": chain_score / len(events),
                }
            )

    if complete_candidates:
        complete_candidates.sort(
            key=lambda item: item["chain_score"], reverse=True
        )
        complete_candidates = complete_candidates[:answer_k]
        best = complete_candidates[0]
        answers = [
            {
                "video_id": item["video_id"],
                "frame_ids": item["frame_ids"],
            }
            for item in complete_candidates
        ]

        print(f"Video chọn: {best['video_id']}")
        print("Chuỗi đầy đủ: có")
        for index, (event, frame_id) in enumerate(
            zip(events, best["frame_ids"]), start=1
        ):
            print(f"  Event {index}: frame {frame_id} — {event}")

        outcome = {
            "type": "trake",
            "video_id": best["video_id"],
            "frame_ids": best["frame_ids"],
            "keyframe_ids": best["keyframe_ids"],
            "keyframe_paths": best["keyframe_paths"],
            "events": list(events),
            "complete_chain": True,
            "chain_score": round(best["chain_score"], 4),
            "submission": answers[0],
            "answers": answers,
            "candidates": complete_candidates,
        }
        save_json(ARTIFACTS_DIR / "trake_results.json", outcome)
        return outcome

    best_video = None
    best_chain_length = -1
    best_chain_score = float("-inf")
    best_chosen_map = None

    for video_id, by_event in videos.items():
        chain_events, chain_results = best_chain_in_video(
            by_event, len(events)
        )

        if not chain_events:
            continue

        chain_score = sum(
            candidate_score(result) for result in chain_results
        )

        if (
            len(chain_events) > best_chain_length
            or (
                len(chain_events) == best_chain_length
                and chain_score > best_chain_score
            )
        ):
            best_video = video_id
            best_chain_length = len(chain_events)
            best_chain_score = chain_score
            best_chosen_map = dict(zip(chain_events, chain_results))

    if best_video is None:
        return {
            "type": "trake",
            "video_id": None,
            "frame_ids": [],
            "error": "Không video nào có ứng viên cho chuỗi event",
        }

    complete_chain = best_chain_length == len(events)

    video_rows = videos_info.get(best_video, [])

    frames, keyframe_ids = interpolate_missing(
        events,
        best_chosen_map,
        video_rows,
        video_median_gap(video_rows),
    )

    frame_ids = [frames[index] for index in range(len(events))]
    keyframe_list = [
        keyframe_ids[index] for index in range(len(events))
    ]
    path_by_keyframe_id = {
        make_keyframe_id(row["video_id"], row["keyframe_name"]):
        relative_keyframe_path(row["video_id"], row["keyframe_name"])
        for row in video_rows
    }
    keyframe_paths = [
        path_by_keyframe_id[keyframe_id]
        for keyframe_id in keyframe_list
    ]

    print(f"Video chọn: {best_video}")
    print(
        f"Chuỗi đầy đủ: {'có' if complete_chain else 'không'} "
        f"({best_chain_length}/{len(events)} event có ứng viên)"
    )

    for index, event in enumerate(events):
        marker = "" if index in best_chosen_map else " (nội suy)"
        print(
            f"  Event {index + 1}: frame {frame_ids[index]} "
            f"[{keyframe_list[index]}]{marker}"
        )

    outcome = {
        "type": "trake",
        "video_id": best_video,
        "frame_ids": frame_ids,
        "keyframe_ids": keyframe_list,
        "keyframe_paths": keyframe_paths,
        "events": list(events),
        "complete_chain": complete_chain,
        "chain_score": round(best_chain_score, 4),
        "submission": {
            "video_id": best_video,
            "frame_ids": frame_ids,
        },
        "answers": [
            {
                "video_id": best_video,
                "frame_ids": frame_ids,
            }
        ],
    }
    save_json(ARTIFACTS_DIR / "trake_results.json", outcome)
    return outcome
