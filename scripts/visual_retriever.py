"""
Visual retriever: CLIP text encoder + FAISS.

Trả về list[SearchResult] với clip_score = cosine similarity.
Model được tải trễ (chỉ khi search lần đầu) để import nhanh.
"""

from pathlib import Path

import numpy as np

from search_types import (
    INDEX_PATH,
    SearchResult,
    load_mapping,
    result_from_row,
)
from stdio_setup import configure_stdio


configure_stdio()


MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"


class VisualRetriever:
    name = "visual"

    def __init__(self):
        import faiss

        self._mapping = load_mapping()
        self._index = faiss.read_index(str(INDEX_PATH))

        if len(self._mapping) != self._index.ntotal:
            raise ValueError(
                f"Mapping có {len(self._mapping)} dòng nhưng index có "
                f"{self._index.ntotal} vector. Chạy lại "
                "build_mapping.py, encode_clip_features.py và "
                "build_faiss_index.py cho đồng bộ."
            )

        self._model = None
        self._tokenizer = None
        self._device = None

    @property
    def mapping(self):
        return self._mapping

    @property
    def ntotal(self):
        return self._index.ntotal

    def available(self):
        return True

    def _ensure_model(self):
        if self._model is not None:
            return

        import open_clip
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Đang tải CLIP text encoder ({MODEL_NAME})...")

        model, _, _ = open_clip.create_model_and_transforms(
            MODEL_NAME,
            pretrained=PRETRAINED,
        )

        self._model = model.to(self._device).eval()
        self._tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    def encode_text(self, query):
        import torch

        self._ensure_model()

        text_tokens = self._tokenizer([query]).to(self._device)

        with torch.inference_mode():
            text_features = self._model.encode_text(
                text_tokens,
                normalize=True,
            )

        query_vector = (
            text_features
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return np.ascontiguousarray(query_vector)

    def search(self, query, top_k=100):
        if top_k <= 0:
            return []

        query_vector = self.encode_text(query)

        if query_vector.shape[1] != self._index.d:
            raise ValueError(
                f"Vector câu có {query_vector.shape[1]} chiều, "
                f"index có {self._index.d} chiều"
            )

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, k)

        results = []

        for vector_index, score in zip(indices[0], scores[0]):
            if vector_index < 0:
                continue

            row = self._mapping[int(vector_index)]

            results.append(
                result_from_row(row, clip_score=float(score))
            )

        for rank, result in enumerate(results, start=1):
            result.rank = rank

        return results
