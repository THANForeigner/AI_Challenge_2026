"""
Tìm kiếm bằng câu mô tả tiếng Anh (truy vấn Textual KIS của BTC).

Kết quả được xếp hạng theo điểm CLIP giảm dần và in theo đúng
định dạng nộp bài của BTC: mỗi câu trả lời là một cặp
<video_id>, <frame_id> — trong đó frame_id là frame index thật
trong video (lấy từ metadata), KHÔNG phải số thứ tự keyframe.

BTC cho phép nộp tối đa 100 câu trả lời mỗi truy vấn; điểm cuối
cùng là trung bình R@1/5/20/50/100 nên thứ tự xếp hạng rất quan
trọng. TOP_K mặc định đặt bằng 100.

Cách chạy:
- python search_text_query.py                (chế độ tương tác)
- python search_text_query.py "câu truy vấn" (chạy 1 truy vấn)
"""

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch

from stdio_setup import configure_stdio


configure_stdio()


MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"
TOP_K = 100

PROJECT_DIR = Path(__file__).resolve().parents[1]

INDEX_PATH = PROJECT_DIR / "artifacts" / "clip.index"
MAPPING_PATH = PROJECT_DIR / "artifacts" / "clip_row_mapping.jsonl"
RESULTS_PATH = PROJECT_DIR / "artifacts" / "last_search_results.json"


@dataclass
class SearchResult:
    rank: int
    score: float
    vector_index: int
    video_id: str
    frame_id: int
    keyframe_name: str
    keyframe_path: str


def load_mapping():
    rows = []

    with MAPPING_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    rows.sort(key=lambda row: row["vector_index"])

    expected_indices = list(range(len(rows)))
    actual_indices = [
        row["vector_index"]
        for row in rows
    ]

    if actual_indices != expected_indices:
        raise ValueError(
            "vector_index trong mapping không liên tục từ 0. "
            "Chạy lại scripts/build_mapping.py."
        )

    return rows


def encode_query(query, model, tokenizer, device):
    # Chuyển câu thành token
    text_tokens = tokenizer([query]).to(device)

    # Chuyển token thành embedding
    with torch.inference_mode():
        text_features = model.encode_text(
            text_tokens,
            normalize=True,
        )

    # FAISS cần float32 và vùng nhớ liên tục
    query_vector = (
        text_features
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    query_vector = np.ascontiguousarray(query_vector)

    return query_vector


def search(
    query,
    model,
    tokenizer,
    device,
    index,
    mapping,
):
    query_vector = encode_query(
        query,
        model,
        tokenizer,
        device,
    )

    if query_vector.shape[1] != index.d:
        raise ValueError(
            f"Vector câu có {query_vector.shape[1]} chiều, "
            f"nhưng FAISS index có {index.d} chiều"
        )

    k = min(TOP_K, index.ntotal)

    scores, indices = index.search(
        query_vector,
        k,
    )

    results = []

    for rank, (vector_index, score) in enumerate(
        zip(indices[0], scores[0]),
        start=1,
    ):
        if vector_index < 0:
            continue

        row = mapping[int(vector_index)]

        results.append(
            SearchResult(
                rank=rank,
                score=float(score),
                vector_index=int(vector_index),
                video_id=row["video_id"],
                frame_id=int(row["frame_index"]),
                keyframe_name=row["keyframe_name"],
                keyframe_path=row["keyframe_path"],
            )
        )

    return results


def display_results(query, results):
    print()
    print("=" * 60)
    print("QUERY:", query)
    print("=" * 60)
    print("Kết quả theo định dạng nộp bài BTC: <video_id>, <frame_id>")
    print()

    for result in results:
        print(
            f"{result.rank:>3}. "
            f"{result.video_id}, {result.frame_id} "
            f"(score={result.score:.4f}, "
            f"keyframe={result.keyframe_name})"
        )


def save_results(query, results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "query": query,
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "top_k": len(results),
        # Danh sách nộp bài: mỗi phần tử là một câu trả lời
        # <video_id>, <frame_id> theo thứ tự xếp hạng.
        "submission": [
            {
                "video_id": result.video_id,
                "frame_id": result.frame_id,
            }
            for result in results
        ],
        "results": [
            asdict(result)
            for result in results
        ],
    }

    RESULTS_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def open_result_images(results):
    if os.name != "nt":
        print("Chức năng mở ảnh tự động hiện chỉ dùng trên Windows.")
        return

    for result in results:
        image_path = Path(result.keyframe_path)

        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path

        image_path = image_path.resolve()

        if image_path.exists():
            os.startfile(image_path)
        else:
            print(f"Không tìm thấy ảnh: {image_path}")


def setup():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy FAISS index: {INDEX_PATH}"
        )

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mapping: {MAPPING_PATH}"
        )

    mapping = load_mapping()
    index = faiss.read_index(str(INDEX_PATH))

    if len(mapping) != index.ntotal:
        raise ValueError(
            f"Mapping có {len(mapping)} dòng, "
            f"nhưng index có {index.ntotal} vector. "
            "Chạy lại build_mapping.py, encode_clip_features.py "
            "và build_faiss_index.py cho đồng bộ."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Thiết bị:", device)
    print("Model:", MODEL_NAME)
    print("Checkpoint:", PRETRAINED)
    print("FAISS vectors:", index.ntotal)
    print("Dimension:", index.d)
    print("Đang tải CLIP text encoder...")

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )

    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    return mapping, index, model, tokenizer, device


def run_query(query, model, tokenizer, device, index, mapping):
    results = search(
        query,
        model,
        tokenizer,
        device,
        index,
        mapping,
    )

    display_results(query, results)
    save_results(query, results)

    print()
    print("Đã lưu kết quả tại:", RESULTS_PATH)

    return results


def main():
    mapping, index, model, tokenizer, device = setup()

    # Chế độ 1 truy vấn từ command line
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
        run_query(query, model, tokenizer, device, index, mapping)
        return

    print()
    print("Đã sẵn sàng tìm kiếm.")
    print("Nhập câu tiếng Anh mô tả cảnh cần tìm.")
    print("Nhập exit để kết thúc.")

    while True:
        print()
        query = input("Query: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Đã kết thúc.")
            break

        if not query:
            print("Query không được để trống.")
            continue

        results = run_query(
            query, model, tokenizer, device, index, mapping
        )

        print()
        answer = input(
            "Bạn có muốn mở các ảnh kết quả không? (y/n): "
        ).strip().lower()

        if answer == "y":
            open_result_images(results)


if __name__ == "__main__":
    main()
