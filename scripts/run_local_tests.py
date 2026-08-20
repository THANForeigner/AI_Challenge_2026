"""Chạy bộ test nhỏ trên video local ``testing`` và báo accuracy chẩn đoán.

Đây là bộ kiểm thử phát triển, không phải điểm chính thức của BTC. Ground truth
được ghi theo các khoảng cảnh trong 101 keyframe local hiện tại.
"""

import argparse
from pathlib import Path

from query_translator import translate_for_visual
from search_engine import SearchEngine
from search_types import ARTIFACTS_DIR, save_json


KIS_CASES = [
    ("kis_01", "con đường ven sông bị sạt lở", 1, 15),
    ("kis_02", "biển cảnh báo sạt lở nguy hiểm", 16, 22),
    ("kis_03", "bờ biển và rừng nhìn từ trên cao", 27, 34),
    ("kis_04", "vòi nước phun cao lên trời", 47, 60),
    ("kis_05", "hai người đàn ông khiêng một chiếc thùng", 64, 73),
    ("kis_06", "đàn cá trong ao nuôi", 76, 85),
    ("kis_07", "đàn trâu đang ăn cỏ trên cánh đồng", 90, 101),
]

QA_CASES = [
    {
        "id": "qa_01",
        "event": "cảnh có biển cảnh báo sạt lở nguy hiểm",
        "question": "biển cảnh báo có màu gì?",
        "window": (16, 22),
        "answers": {"vàng", "yellow"},
    },
    {
        "id": "qa_02",
        "event": "hai người đàn ông đang khiêng một chiếc thùng",
        "question": "có bao nhiêu người đang khiêng thùng?",
        "window": (64, 73),
        "answers": {"2", "hai", "two"},
    },
    {
        "id": "qa_03",
        "event": "đàn trâu đang ăn cỏ trên cánh đồng",
        "question": "có bao nhiêu con trâu?",
        "window": (90, 101),
        "answers": {"3", "ba", "three"},
    },
]

TRAKE_CASES = [
    {
        "id": "trake_01",
        "events": [
            "con đường ven sông bị sạt lở",
            "biển cảnh báo sạt lở nguy hiểm",
            "bờ biển và rừng nhìn từ trên cao",
        ],
        "windows": [(1, 15), (16, 22), (27, 34)],
    },
    {
        "id": "trake_02",
        "events": [
            "vòi nước phun cao lên trời",
            "hai người đàn ông khiêng một chiếc thùng",
            "đàn cá trong ao nuôi",
        ],
        "windows": [(47, 60), (64, 73), (76, 85)],
    },
    {
        "id": "trake_03",
        "events": [
            "biển cảnh báo sạt lở nguy hiểm",
            "hai người đàn ông khiêng một chiếc thùng",
            "đàn trâu đang ăn cỏ trên cánh đồng",
        ],
        "windows": [(16, 22), (64, 73), (90, 101)],
    },
]


def in_window(frame_id, window):
    return window[0] <= int(frame_id) <= window[1]


def run_kis(engine, top_k, modalities):
    reports = []

    print("\n=== KIS ===")
    for query_id, query, start, end in KIS_CASES:
        results = engine.search_kis(
            query,
            visual_query=translate_for_visual(query),
            modalities=modalities,
            top_k=top_k,
            pool_k=max(100, top_k),
            save=False,
        )
        ranks = [
            index
            for index, result in enumerate(results, start=1)
            if result.video_id == "testing"
            and start <= result.frame_index <= end
        ]
        first_rank = ranks[0] if ranks else None
        report = {
            "query_id": query_id,
            "query": query,
            "expected_window": [start, end],
            "first_correct_rank": first_rank,
            "R@1": int(first_rank == 1),
            "R@5": int(first_rank is not None and first_rank <= 5),
            "R@20": int(first_rank is not None and first_rank <= 20),
            "top_results": [result.answer for result in results[:5]],
        }
        reports.append(report)
        print(
            f"{query_id}: rank đúng đầu tiên={first_rank or '-'} "
            f"| cửa sổ={start}-{end} | {query}"
        )

    return reports


def run_qa(engine):
    import qa_engine

    reports = []
    print("\n=== Q&A ===")

    for case in QA_CASES:
        outcome = qa_engine.answer_question(
            event_description=case["event"],
            question=case["question"],
            engine=engine,
            top_k=5,
        )
        location_correct = (
            outcome.get("video_id") == "testing"
            and in_window(outcome.get("frame_id", -1), case["window"])
        )
        answer = str(outcome.get("answer", "")).strip().casefold()
        answer_correct = answer in {
            expected.casefold() for expected in case["answers"]
        }
        report = {
            "query_id": case["id"],
            "event": case["event"],
            "question": case["question"],
            "expected_window": list(case["window"]),
            "expected_answers": sorted(case["answers"]),
            "video_id": outcome.get("video_id"),
            "frame_id": outcome.get("frame_id"),
            "answer": outcome.get("answer"),
            "location_correct": location_correct,
            "answer_correct": answer_correct,
            "fully_correct": location_correct and answer_correct,
        }
        reports.append(report)
        print(
            f"{case['id']}: frame={outcome.get('frame_id')} "
            f"answer={outcome.get('answer')!r} "
            f"| vị trí={'đúng' if location_correct else 'SAI'} "
            f"| đáp án={'đúng' if answer_correct else 'SAI'}"
        )

    return reports


def run_trake(engine):
    import trake_engine

    reports = []
    print("\n=== TRAKE ===")

    for case in TRAKE_CASES:
        outcome = trake_engine.search_trake(
            events=case["events"],
            engine=engine,
            candidate_k=100,
            answer_k=5,
        )
        frame_ids = outcome.get("frame_ids", [])
        event_hits = [
            index < len(frame_ids) and in_window(frame_ids[index], window)
            for index, window in enumerate(case["windows"])
        ]
        report = {
            "query_id": case["id"],
            "events": case["events"],
            "expected_windows": [list(window) for window in case["windows"]],
            "video_id": outcome.get("video_id"),
            "frame_ids": frame_ids,
            "event_hits": event_hits,
            "event_accuracy": sum(event_hits) / len(event_hits),
            "fully_correct": (
                outcome.get("video_id") == "testing" and all(event_hits)
            ),
        }
        reports.append(report)
        print(
            f"{case['id']}: frames={frame_ids} "
            f"| event đúng={sum(event_hits)}/{len(event_hits)}"
        )

    return reports


def average(rows, field):
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def main():
    parser = argparse.ArgumentParser(description="Bộ test local KIS/Q&A/TRAKE")
    parser.add_argument(
        "--group", choices=["kis", "qa", "trake", "all"], default="all"
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--modalities",
        default="visual,object,ocr,asr",
        help="Modalities cho nhóm KIS",
    )
    args = parser.parse_args()

    modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    engine = SearchEngine()
    report = {"dataset": "testing", "groups": {}}

    if args.group in {"kis", "all"}:
        rows = run_kis(engine, args.top_k, modalities)
        report["groups"]["kis"] = {
            "cases": rows,
            "R@1": average(rows, "R@1"),
            "R@5": average(rows, "R@5"),
            "R@20": average(rows, "R@20"),
        }

    if args.group in {"qa", "all"}:
        rows = run_qa(engine)
        report["groups"]["qa"] = {
            "cases": rows,
            "location_accuracy": average(rows, "location_correct"),
            "answer_accuracy": average(rows, "answer_correct"),
            "full_accuracy": average(rows, "fully_correct"),
        }

    if args.group in {"trake", "all"}:
        rows = run_trake(engine)
        report["groups"]["trake"] = {
            "cases": rows,
            "event_accuracy": average(rows, "event_accuracy"),
            "full_accuracy": average(rows, "fully_correct"),
        }

    output_path = ARTIFACTS_DIR / "local_test_suite_results.json"
    save_json(output_path, report)
    print(f"\nĐã lưu báo cáo: {output_path}")


if __name__ == "__main__":
    main()
