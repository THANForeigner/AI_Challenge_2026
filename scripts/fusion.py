"""
Multimodal fusion (C4): kết hợp các retriever bằng Reciprocal Rank
Fusion.

    RRF(d) = Σ_r  w_r * gate_r(d) / (k + rank_r(d))

- Modality nào trả None (không có dữ liệu / không khớp) thì bị bỏ
  qua, KHÔNG làm search thất bại.
- Điểm từng modality (clip/object/ocr/asr score) được giữ lại trên
  SearchResult để giải thích kết quả; final_score = điểm RRF.
- Score gating (bật mặc định, tắt bằng AIC_RRF_SCORE_GATE=0): đóng
  góp của mỗi nhánh nhân với độ tin của chính nhánh đó. OCR/ASR có
  overlap dưới ngưỡng AIC_TEXT_MIN_OVERLAP (mặc định 0.3) bị coi là
  nhiễu và không đóng góp — tránh việc phụ đề/transcript khớp 1-2
  token đẩy frame sai lên trên kết quả visual mạnh.
"""

import os

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


def text_min_overlap():
    try:
        return float(os.getenv("AIC_TEXT_MIN_OVERLAP", "0.3"))
    except ValueError:
        return 0.3


def modality_gate(name, result, score_gating):
    """Hệ số độ tin của một kết quả trong nhánh của nó.

    Visual luôn 1.0 (score CLIP không so trực tiếp với overlap được).
    Object/OCR/ASR lấy chính score của nhánh (đã kẹp [0, 1]); OCR/ASR
    dưới ngưỡng overlap tối thiểu bị chặn hẳn để không gây nhiễu.
    """

    if not score_gating or name == "visual":
        return 1.0

    field = SCORE_FIELDS.get(name)
    if field is None:
        return 1.0

    value = float(getattr(result, field, 0.0) or 0.0)

    if name in ("ocr", "asr") and value < text_min_overlap():
        return 0.0

    return max(0.0, min(1.0, value))

SCORE_FIELDS = {
    "visual": "clip_score",
    "object": "object_score",
    "ocr": "ocr_score",
    "asr": "asr_score",
}


def rrf_fuse(ranked_by_modality, weights=None, k=RRF_K, score_gating=None):
    """
    ranked_by_modality: {"visual": [SearchResult], "object": [...], ...}
    Modality nào None hoặc list rỗng thì bỏ qua.

    Trả về list[SearchResult] theo final_score (RRF) giảm dần,
    chưa gán rank.

    score_gating=None đọc env AIC_RRF_SCORE_GATE (mặc định bật).
    """

    if k < 0:
        raise ValueError("Hằng số RRF k phải không âm")

    weights = DEFAULT_WEIGHTS if weights is None else weights

    if score_gating is None:
        score_gating = os.getenv("AIC_RRF_SCORE_GATE", "1") == "1"

    fused = {}

    for name, rows in ranked_by_modality.items():
        if not rows:
            continue

        weight = weights.get(name, 1.0)
        field = SCORE_FIELDS.get(name)

        for rank, row in enumerate(rows, start=1):
            gate = modality_gate(name, row, score_gating)
            if gate <= 0.0:
                continue
            contribution = weight * gate / (k + rank)

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
