"""Q&A: truy xuất cảnh, gom bằng chứng và trả lời bằng mô hình VQA cục bộ."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from search_types import (
    ARTIFACTS_DIR,
    PROJECT_DIR,
    make_keyframe_id,
    relative_keyframe_path,
    save_json,
    video_keyframes,
)
from stdio_setup import configure_stdio


configure_stdio()


DEFAULT_VQA_MODEL = "dandelin/vilt-b32-finetuned-vqa"
OCR_DIR = PROJECT_DIR / "data" / "ocr"

_vqa_model = None
_vqa_processor = None
_vqa_device = None


@dataclass
class EvidenceBundle:
    video_id: str
    frame_index: int
    keyframe_ids: List[str] = field(default_factory=list)
    keyframe_paths: List[str] = field(default_factory=list)
    focus_keyframe_id: str = ""
    ocr_text: str = ""
    asr_text: str = ""


def _read_ocr_text(video_id, keyframe_names):
    texts = []

    for keyframe_name in keyframe_names:
        json_path = OCR_DIR / video_id / f"{Path(keyframe_name).stem}.json"

        if not json_path.exists():
            continue

        raw = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            lines = raw.get("texts") or raw.get("text") or []
        else:
            lines = raw

        if isinstance(lines, str):
            lines = [lines]

        for line in lines:
            if isinstance(line, dict):
                confidence = line.get("confidence")
                if confidence is not None and float(confidence) < 0.3:
                    continue
                value = line.get("text", "")
            else:
                value = line

            if value:
                texts.append(str(value).strip())

    return " ".join(dict.fromkeys(filter(None, texts)))


def gather_evidence(top_result, engine, window=1):
    """Lấy keyframe kết quả và các keyframe lân cận cùng OCR/ASR."""

    video_rows = video_keyframes(engine.mapping).get(top_result.video_id, [])

    if not video_rows:
        raise ValueError(f"Không có keyframe cho video {top_result.video_id!r}")

    position = next(
        (
            index
            for index, row in enumerate(video_rows)
            if row["vector_index"] == top_result.vector_index
        ),
        None,
    )

    if position is None:
        position = min(
            range(len(video_rows)),
            key=lambda index: abs(
                int(video_rows[index]["frame_index"]) - top_result.frame_index
            ),
        )

    neighbors = video_rows[
        max(0, position - window):min(len(video_rows), position + window + 1)
    ]
    keyframe_names = [row["keyframe_name"] for row in neighbors]

    asr_text = ""
    if engine.asr.available():
        asr_text = engine.asr.context_for_frame(
            top_result.video_id,
            top_result.frame_index,
        )

    return EvidenceBundle(
        video_id=top_result.video_id,
        frame_index=top_result.frame_index,
        keyframe_ids=[
            make_keyframe_id(row["video_id"], row["keyframe_name"])
            for row in neighbors
        ],
        keyframe_paths=[
            relative_keyframe_path(row["video_id"], row["keyframe_name"])
            for row in neighbors
        ],
        focus_keyframe_id=top_result.keyframe_id,
        ocr_text=_read_ocr_text(top_result.video_id, keyframe_names),
        asr_text=asr_text,
    )


def _ensure_vqa_model():
    global _vqa_model, _vqa_processor, _vqa_device

    if _vqa_model is not None:
        return

    import torch
    from transformers import ViltForQuestionAnswering, ViltProcessor

    model_name = os.getenv("AIC_VQA_MODEL", DEFAULT_VQA_MODEL)
    offline = os.getenv("AIC_OFFLINE", "0") == "1"
    _vqa_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Đang tải VQA ({model_name}) trên {_vqa_device}...")

    try:
        _vqa_processor = ViltProcessor.from_pretrained(
            model_name, local_files_only=True
        )
        _vqa_model = ViltForQuestionAnswering.from_pretrained(
            model_name, local_files_only=True
        )
    except OSError as error:
        if offline:
            raise RuntimeError(
                "Không tìm thấy model VQA offline. Hãy gắn model vào Kaggle "
                "Dataset và đặt AIC_VQA_MODEL tới thư mục model."
            ) from error

        _vqa_processor = ViltProcessor.from_pretrained(model_name)
        _vqa_model = ViltForQuestionAnswering.from_pretrained(model_name)

    _vqa_model = _vqa_model.to(_vqa_device).eval()


def _resolve_image_path(keyframe_id, stored_path):
    path = Path(stored_path)
    if path.exists():
        return path

    if not path.is_absolute():
        project_relative = PROJECT_DIR / path
        if project_relative.exists():
            return project_relative

    video_id, keyframe_stem = keyframe_id.split("/", 1)
    for extension in (".jpg", ".jpeg", ".png"):
        canonical = PROJECT_DIR / "data" / "keyframes" / video_id / (
            keyframe_stem + extension
        )
        if canonical.exists():
            return canonical

    raise FileNotFoundError(f"Không tìm thấy ảnh bằng chứng {keyframe_id}")


def normalize_vqa_answer(answer):
    """Ưu tiên định dạng ngắn, ổn định và dễ chấm ngữ nghĩa."""

    normalized = str(answer).strip().casefold()
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10",
    }
    vietnamese = {
        "yes": "có", "no": "không", "red": "đỏ", "blue": "xanh dương",
        "green": "xanh lá", "yellow": "vàng", "black": "đen",
        "white": "trắng", "orange": "cam", "brown": "nâu",
        "gray": "xám", "grey": "xám", "purple": "tím", "pink": "hồng",
    }
    return number_words.get(normalized, vietnamese.get(normalized, normalized))


def vqa_answer(question, evidence):
    """Chạy ViLT trên từng ảnh bằng chứng và lấy dự đoán tự tin nhất."""

    import torch
    from PIL import Image
    from query_translator import translate_vi_to_en

    _ensure_vqa_model()
    question_en = translate_vi_to_en(question)
    images = []

    try:
        for keyframe_id, stored_path in zip(
            evidence.keyframe_ids, evidence.keyframe_paths
        ):
            path = _resolve_image_path(keyframe_id, stored_path)
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        encoded = _vqa_processor(
            images=images,
            text=[question_en] * len(images),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(_vqa_device)

        with torch.inference_mode():
            probabilities = torch.softmax(_vqa_model(**encoded).logits, dim=-1)
            confidence_by_image, label_by_image = probabilities.max(dim=-1)

        if evidence.focus_keyframe_id in evidence.keyframe_ids:
            best_image = evidence.keyframe_ids.index(evidence.focus_keyframe_id)
        else:
            best_image = int(confidence_by_image.argmax().item())
        label_id = int(label_by_image[best_image].item())
        answer_en = _vqa_model.config.id2label[label_id]

        return {
            "answer": normalize_vqa_answer(answer_en),
            "answer_en": answer_en,
            "confidence": float(confidence_by_image[best_image].item()),
            "question_en": question_en,
            "evidence_keyframe_id": evidence.keyframe_ids[best_image],
        }
    finally:
        for image in images:
            image.close()


def answer_question(event_description, question, engine, top_k=5):
    """Trả đáp án Q&A đã xếp hạng cùng bằng chứng để kiểm tra."""

    if top_k <= 0:
        raise ValueError("top_k phải lớn hơn 0")

    question = question.strip() or event_description.strip()
    retrieval_query = event_description.strip() or question
    print("Q&A — mô tả sự kiện:", retrieval_query)
    print("Q&A — câu hỏi:", question)
    print()

    from query_translator import translate_for_visual

    visual_query = translate_for_visual(retrieval_query)
    if visual_query != retrieval_query:
        print("Q&A — CLIP query:", visual_query)

    qa_modalities = [
        name.strip()
        for name in os.getenv("AIC_QA_MODALITIES", "visual").split(",")
        if name.strip()
    ]
    print("Q&A — retrieval modalities:", ", ".join(qa_modalities))

    results = engine.search_kis(
        retrieval_query,
        visual_query=visual_query,
        modalities=qa_modalities,
        top_k=top_k,
        save=False,
    )

    if not results:
        return {
            "type": "qa", "video_id": None, "frame_id": None,
            "answer": "", "answers": [],
            "error": "retrieval không có kết quả",
        }

    candidates = []
    vqa_question = question.rsplit(",", 1)[-1].strip()

    # Giới hạn 5 cảnh để test tương tác không quá chậm; model được giữ trong
    # RAM sau lần gọi đầu tiên.
    for retrieval_rank, result in enumerate(results[:5], start=1):
        evidence = gather_evidence(result, engine, window=1)
        prediction = vqa_answer(vqa_question, evidence)
        combined_score = 0.55 / retrieval_rank + 0.45 * prediction["confidence"]
        candidates.append(
            {
                "video_id": result.video_id,
                "frame_id": result.frame_index,
                "answer": prediction["answer"],
                "answer_en": prediction["answer_en"],
                "vqa_confidence": prediction["confidence"],
                "combined_score": combined_score,
                "retrieval_rank": retrieval_rank,
                "question_en": prediction["question_en"],
                "evidence_keyframe_id": prediction["evidence_keyframe_id"],
                "evidence": {
                    "keyframe_ids": evidence.keyframe_ids,
                    "keyframe_paths": evidence.keyframe_paths,
                    "ocr_text": evidence.ocr_text,
                    "asr_text": evidence.asr_text,
                },
            }
        )

    candidates.sort(key=lambda item: item["combined_score"], reverse=True)
    best = candidates[0]
    answers = [
        {
            "video_id": item["video_id"],
            "frame_id": item["frame_id"],
            "answer": item["answer"],
        }
        for item in candidates
    ]

    print(
        f"Q&A chọn: {best['video_id']}, frame {best['frame_id']}, "
        f"answer={best['answer']!r}, confidence={best['vqa_confidence']:.3f}"
    )

    outcome = {
        "type": "qa",
        "video_id": best["video_id"],
        "frame_id": best["frame_id"],
        "answer": best["answer"],
        "vqa_status": "ok",
        "submission": answers[0],
        "answers": answers,
        "candidates": candidates,
    }
    save_json(ARTIFACTS_DIR / "qa_results.json", outcome)
    return outcome
