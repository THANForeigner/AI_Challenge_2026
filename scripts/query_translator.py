"""Dịch tiếng Việt sang tiếng Anh cho các model dùng vocabulary tiếng Anh."""

from functools import lru_cache
import re

from stdio_setup import configure_stdio


configure_stdio()


MODEL_NAME = "Helsinki-NLP/opus-mt-vi-en"

# Từ chức năng/lượng từ không nên dịch riêng lẻ: ví dụ ``chiếc`` có thể bị
# Marian dịch nhầm thành ``umbrella`` và tạo false match Object.
VIETNAMESE_FUNCTION_WORDS = {
    "à", "bị", "các", "cái", "cho", "chiếc", "có", "của", "đã",
    "đang", "đến", "được", "hai", "hay", "khi", "là", "lại", "mà",
    "một", "những", "này", "ở", "ra", "sau", "sẽ", "theo", "thì",
    "trên", "trong", "từ", "và", "vào", "với",
}

_model = None
_tokenizer = None
_device = None


def _ensure_model():
    global _model, _tokenizer, _device

    if _model is not None:
        return

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang tải bộ dịch Việt→Anh ({MODEL_NAME})...")

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    _model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    ).to(_device).eval()


def looks_vietnamese(text):
    """Nhận biết thô để không đưa câu tiếng Anh qua model Việt→Anh."""

    normalized = str(text).casefold()
    vietnamese_marks = set(
        "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệ"
        "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
    )
    return any(character in vietnamese_marks for character in normalized)


@lru_cache(maxsize=256)
def translate_vi_to_en(text):
    """Dịch đúng một câu; dùng cho CLIP/VQA, không sinh thêm n-gram."""

    import torch

    source = str(text).strip()

    if not source or not looks_vietnamese(source):
        return source

    _ensure_model()
    encoded = _tokenizer(
        [source],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(_device)

    with torch.inference_mode():
        generated = _model.generate(
            **encoded,
            max_length=128,
            num_beams=4,
            renormalize_logits=True,
        )

    return _tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()


@lru_cache(maxsize=256)
def translate_for_visual(text):
    """Dịch và sửa vài lỗi Marian thường gây hại cho CLIP retrieval."""

    source = str(text).casefold()
    translated = translate_vi_to_en(text)
    replacements = (
        (r"\bin view of\b", "a scene of"),
        (r"\byou two\b", "two people"),
        (r"\byou guys\b", "people"),
        (r"\ba barrel\b", "a box"),
        (r"\bbarrel\b", "box"),
        (r"\bcarry boxes\b", "carrying a box"),
        (r"\bcarrying boxes\b", "carrying a box"),
    )

    for pattern, replacement in replacements:
        translated = re.sub(
            pattern, replacement, translated, flags=re.IGNORECASE
        )

    # Marian đôi khi lược số lượng trong cụm "hai người", trong khi đây là
    # tín hiệu rất quan trọng cho retrieval/VQA.
    if "hai người" in source and "two people" not in translated.casefold():
        translated = re.sub(
            r"\bpeople\b", "two people", translated, count=1,
            flags=re.IGNORECASE,
        )

    return translated.strip()


@lru_cache(maxsize=256)
def translate_for_object(query):
    """Dịch toàn câu và các cụm ngắn để không bỏ sót tên vật thể."""

    import torch

    text = str(query).strip()

    if not text:
        return ""

    _ensure_model()

    words = re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)
    phrases = [text]

    # Marian đôi khi lược chủ thể trong bản dịch toàn câu. Dịch thêm các
    # unigram/bigram/trigram giúp giữ các cụm vật thể như "người", "ô tô",
    # "đèn giao thông"; ObjectRetriever sẽ tự bỏ từ không thuộc vocabulary.
    for size in (1, 2, 3):
        for start in range(len(words) - size + 1):
            phrase_words = words[start:start + size]

            if any(
                word in VIETNAMESE_FUNCTION_WORDS
                for word in phrase_words
            ):
                continue

            phrases.append(" ".join(phrase_words))

    phrases = list(dict.fromkeys(phrases))[:64]

    encoded = _tokenizer(
        phrases,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(_device)

    with torch.inference_mode():
        generated = _model.generate(
            **encoded,
            max_length=128,
            num_beams=4,
            renormalize_logits=True,
        )

    translations = _tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )
    translations = [translation.strip() for translation in translations]
    return ". ".join(dict.fromkeys(filter(None, translations)))
