"""
Evaluation (C10): chấm điểm theo đúng công thức vòng Sơ tuyển BTC.

KIS   : R-Score(r) = I(video khớp ∧ frame ∈ [s, e])
Q&A   : R-Score(r) = I(video khớp ∧ frame ∈ [s, e] ∧ answer khớp)
TRAKE : R-Score(r) = (tỉ lệ event có frame ∈ [s_j, e_j]) nếu đúng
        video, ngược lại 0.
Final : R@k = max R-Score trong k câu trả lời đầu; báo cáo
        R@1/5/20/50/100 và trung bình.

Định dạng file (xem evals/sample_*.json):
GT:
[
  {"query_id", "type": "kis",  "video_id", "frame_start", "frame_end"},
  {"query_id", "type": "qa",   "video_id", "frame_start", "frame_end", "answer"},
  {"query_id", "type": "trake","video_id", "events": [{"frame_start","frame_end"}, ...]}
]
Dự đoán:
[
  {"query_id", "answers": [{"video_id","frame_id"}]                       }  # kis
  {"query_id", "answers": [{"video_id","frame_id","answer"}]              }  # qa
  {"query_id", "answers": [{"video_id","frame_ids":[...]}]                }  # trake
]

Chạy:
    python evaluation.py --gt evals/sample_gt.json \
                         --pred evals/sample_predictions.json
"""

import argparse
import json
from pathlib import Path

from stdio_setup import configure_stdio


configure_stdio()


PROJECT_DIR = Path(__file__).resolve().parents[1]
K_VALUES = [1, 5, 20, 50, 100]


def normalize_answer(text):
    return str(text).strip().lower()


def r_score_kis(answer, ground_truth):
    if answer.get("video_id") != ground_truth["video_id"]:
        return 0.0

    frame_id = int(answer.get("frame_id", -1))

    if ground_truth["frame_start"] <= frame_id <= ground_truth["frame_end"]:
        return 1.0

    return 0.0


def r_score_qa(answer, ground_truth):
    if r_score_kis(answer, ground_truth) != 1.0:
        return 0.0

    predicted = normalize_answer(answer.get("answer", ""))
    expected = normalize_answer(ground_truth["answer"])

    return 1.0 if predicted == expected else 0.0


def r_score_trake(answer, ground_truth):
    if answer.get("video_id") != ground_truth["video_id"]:
        return 0.0

    frame_ids = answer.get("frame_ids", [])
    events = ground_truth["events"]
    total = len(events)

    if total == 0:
        return 0.0

    hits = 0

    for index, event in enumerate(events):
        if index >= len(frame_ids):
            continue

        frame_id = int(frame_ids[index])

        if event["frame_start"] <= frame_id <= event["frame_end"]:
            hits += 1

    return hits / total


R_SCORE_FUNCTIONS = {
    "kis": r_score_kis,
    "qa": r_score_qa,
    "trake": r_score_trake,
}


def r_at_k(scores, k):
    window = scores[:k]

    if not window:
        return 0.0

    return max(window)


def evaluate_query(answers, ground_truth):
    query_type = ground_truth["type"]
    score_function = R_SCORE_FUNCTIONS[query_type]

    scores = [
        score_function(answer, ground_truth)
        for answer in answers
    ]

    metrics = {
        "query_id": ground_truth["query_id"],
        "type": query_type,
        "num_answers": len(scores),
    }

    for k in K_VALUES:
        metrics[f"R@{k}"] = r_at_k(scores, k)

    metrics["final"] = sum(
        metrics[f"R@{k}"] for k in K_VALUES
    ) / len(K_VALUES)

    # Chẩn đoán riêng TRAKE (top-1). Luôn tạo các trường
    # này để truy vấn không có dự đoán vẫn in báo cáo được.
    if query_type == "trake":
        top_answer = answers[0] if answers else {}
        video_correct = bool(answers) and (
            top_answer.get("video_id") == ground_truth["video_id"]
        )
        frame_ids = top_answer.get("frame_ids", [])
        order_correct = bool(answers) and all(
            previous < current
            for previous, current in zip(frame_ids, frame_ids[1:])
        )

        metrics["top1_video_correct"] = video_correct
        metrics["top1_order_correct"] = order_correct
        metrics["top1_event_accuracy"] = (
            r_score_trake(top_answer, ground_truth) if answers else 0.0
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Chấm điểm KIS / Q&A / TRAKE theo công thức BTC"
    )
    parser.add_argument("--gt", required=True, help="File ground truth JSON")
    parser.add_argument("--pred", required=True, help="File dự đoán JSON")

    args = parser.parse_args()

    ground_truths = json.loads(
        Path(args.gt).read_text(encoding="utf-8")
    )
    predictions = json.loads(
        Path(args.pred).read_text(encoding="utf-8")
    )

    predictions_by_id = {
        item["query_id"]: item.get("answers", [])
        for item in predictions
    }

    all_metrics = []

    for ground_truth in ground_truths:
        query_id = ground_truth["query_id"]
        answers = predictions_by_id.get(query_id, [])

        metrics = evaluate_query(answers, ground_truth)
        all_metrics.append(metrics)

        print(
            f"{metrics['query_id']} ({metrics['type']}): "
            f"R@1={metrics['R@1']:.2f} "
            f"R@5={metrics['R@5']:.2f} "
            f"R@20={metrics['R@20']:.2f} "
            f"R@50={metrics['R@50']:.2f} "
            f"R@100={metrics['R@100']:.2f} "
            f"→ final={metrics['final']:.3f}"
        )

        if metrics["type"] == "trake":
            print(
                "    chẩn đoán top-1: "
                f"video={'đúng' if metrics['top1_video_correct'] else 'SAI'} | "
                f"thứ tự={'đúng' if metrics['top1_order_correct'] else 'SAI'} | "
                f"tỉ lệ event đúng={metrics['top1_event_accuracy']:.2f}"
            )

    print()

    for query_type in ["kis", "qa", "trake"]:
        group = [
            metrics
            for metrics in all_metrics
            if metrics["type"] == query_type
        ]

        if not group:
            continue

        average = {}

        for k in K_VALUES:
            average[f"R@{k}"] = sum(
                metrics[f"R@{k}"] for metrics in group
            ) / len(group)

        average["final"] = sum(
            metrics["final"] for metrics in group
        ) / len(group)

        summary = " | ".join(
            f"R@{k}={average[f'R@{k}']:.3f}" for k in K_VALUES
        )

        print(
            f"Trung bình {query_type.upper()} "
            f"({len(group)} truy vấn): {summary} "
            f"→ final={average['final']:.3f}"
        )


if __name__ == "__main__":
    main()
