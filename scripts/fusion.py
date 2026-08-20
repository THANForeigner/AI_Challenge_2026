"""
Multimodal fusion (C4): kết hợp các retriever bằng Reciprocal Rank
Fusion.

    RRF(d) = Σ_r  w_r / (k + rank_r(d))

- Modality nào trả None (không có dữ liệu / không khớp) thì bị bỏ
  qua, KHÔNG làm search thất bại.
- Điểm từng modality (clip/object/ocr/asr score) được giữ lại trên
  SearchResult để giải thích kết quả; final_score = điểm RRF.
"""

from stdio_setup import configure_stdio


configure_stdio()


# Hằng số chuẩn của RRF (Cormack et al.)
RRF_K = 60

DEFAULT_WEIGHTS = {
    "visual": 1.0,
    "object": 0.7,
    "ocr": 0.6,
    "asr": 0.6,
}

SCORE_FIELDS = {
    "visual": "clip_score",
    "object": "object_score",
    "ocr": "ocr_score",
    "asr": "asr_score",
}


def rrf_fuse(ranked_by_modality, weights=None, k=RRF_K):
    """
    ranked_by_modality: {"visual": [SearchResult], "object": [...], ...}
    Modality nào None hoặc list rỗng thì bỏ qua.

    Trả về list[SearchResult] theo final_score (RRF) giảm dần,
    chưa gán rank.
    """

    if k < 0:
        raise ValueError("Hằng số RRF k phải không âm")

    weights = DEFAULT_WEIGHTS if weights is None else weights

    fused = {}

    for name, rows in ranked_by_modality.items():
        if not rows:
            continue

        weight = weights.get(name, 1.0)
        field = SCORE_FIELDS.get(name)

        for rank, row in enumerate(rows, start=1):
            contribution = weight / (k + rank)

            if row.vector_index not in fused:
                base = row
                fused[row.vector_index] = {
                    "result": base,
                    "rrf": 0.0,
                }

            fused[row.vector_index]["rrf"] += contribution

            if field is not None:
                current = getattr(fused[row.vector_index]["result"], field)
                new_value = getattr(row, field)

                if new_value > current:
                    setattr(
                        fused[row.vector_index]["result"],
                        field,
                        new_value,
                    )

    results = []

    for entry in fused.values():
        result = entry["result"]
        result.final_score = entry["rrf"]
        results.append(result)

    results.sort(key=lambda result: result.final_score, reverse=True)

    return results
