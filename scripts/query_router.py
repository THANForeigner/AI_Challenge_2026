"""
Query router (C6/C7): phân loại truy vấn và tách thành phần.

Ba dạng truy vấn vòng Sơ tuyển AIC 2026:
- kis  : Textual KIS — tìm <video_id>, <frame_id>
- qa   : Q&A — tìm <video_id>, <frame_id> và trả lời câu hỏi
- trake: TRAKE — tìm <video_id> và chuỗi frame_id theo sự kiện

Hiện dùng RULE-BASED. Bản nâng cấp dùng LLM được để làm mã giả —
xem PLACEHOLDERS.md (hàm llm_parse_query).
"""

import re

from stdio_setup import configure_stdio


configure_stdio()


QUESTION_MARKERS_EN = [
    "how many", "how much", "what color", "what colour", "what is",
    "what are", "what does", "what did", "what time", "who is",
    "who are", "who won", "where is", "where does", "when does",
    "which", "how long", "how often", "is there", "are there",
    "does the", "do the", "can you see",
]

QUESTION_MARKERS_VI = [
    "bao nhiêu", "màu gì", "là gì", "là ai", "ở đâu", "khi nào",
    "lúc nào", "ai là", "có bao nhiêu", "bao lâu", "phải không",
    "gì vậy", "thế nào", "ra sao",
]

TRAKE_MARKERS_EN = [
    "sequence of events", "series of events", "moments:",
    "key moments", "key events", "event 1", "event 2",
    "then", "after that", "finally",
]

TRAKE_MARKERS_VI = [
    "chuỗi sự kiện", "chuỗi khoảnh khắc", "các khoảnh khắc",
    "trình tự", "lần lượt", "sau đó", "cuối cùng",
    "khoảnh khắc 1", "khoảnh khắc 2",
]

# Không dùng capture group để re.split không nhét số vào kết quả
NUMBERED_ITEM = re.compile(r"(?:^|\s)(?:\(\d+\)|\d+[\.\):])\s+")


def split_qa(query):
    """
    Tách truy vấn Q&A thành (event_description, question).

    Quy tắc: câu chứa '?' là question; phần còn lại mô tả sự kiện.
    """

    stripped_query = query.strip()
    lowered_query = stripped_query.casefold()

    # Dạng thường gặp của đề: "Trong cảnh ..., có bao nhiêu ...?". Phần
    # trước dấu phẩy là mô tả để retrieval; giữ nguyên cả câu làm question
    # để VQA không mất chủ thể được hỏi.
    marker_positions = [
        lowered_query.find(marker)
        for marker in QUESTION_MARKERS_EN + QUESTION_MARKERS_VI
        if lowered_query.find(marker) >= 0
    ]
    if marker_positions:
        first_marker = min(marker_positions)
        comma = stripped_query.rfind(",", 0, first_marker + 1)
        if comma > 10:
            event_description = stripped_query[:comma].strip(" ,.;:")
            if event_description:
                return event_description, stripped_query

    sentences = re.split(r"(?<=[\.\?!])\s+", stripped_query)

    question_parts = []
    event_parts = []

    for sentence in sentences:
        if "?" in sentence or any(
            marker in sentence.lower()
            for marker in QUESTION_MARKERS_EN + QUESTION_MARKERS_VI
        ):
            question_parts.append(sentence)
        else:
            event_parts.append(sentence)

    question = " ".join(question_parts).strip()
    event_description = " ".join(event_parts).strip()

    # Query chỉ có câu hỏi, không có mô tả sự kiện
    if not event_description:
        event_description = stripped_query

    return event_description, question


def decompose_events(query):
    """
    Tách truy vấn TRAKE thành danh sách event (C7, rule-based).

    Ưu tiên:
    1. Các mục đánh số "1." "2." hoặc "(1)" "(2)".
    2. Câu phân tách bằng dấu chấm chấm phẩy.
    """

    text = query.strip()

    # Cách 1: mục đánh số. Phần đứng trước mục số đầu tiên là câu dẫn, không
    # phải event (vd. "Tìm các khoảnh khắc sau: 1. chạy 2. nhảy").
    matches = list(NUMBERED_ITEM.finditer(text))

    if len(matches) >= 2:
        pieces = []

        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            piece = text[start:end].strip(" .;:")
            if piece:
                pieces.append(piece)

        if len(pieces) >= 2:
            return pieces

    # Cách 2: tách câu
    sentences = re.split(r"(?<=[\.!?;])\s+", text)
    sentences = [
        sentence.strip(" .;:")
        for sentence in sentences
        if sentence.strip()
    ]

    if len(sentences) >= 2:
        return sentences

    return [text]


def route(query):
    """
    Trả về:
    {
        "type": "kis" | "qa" | "trake",
        "event_description": ...,
        "question": ... (qa),
        "events": [...] (trake),
    }
    """

    query = query.strip()
    lowered = query.lower()

    has_numbered = len(NUMBERED_ITEM.split(query)) >= 3
    has_trake_marker = any(
        marker in lowered for marker in TRAKE_MARKERS_EN + TRAKE_MARKERS_VI
    )

    if has_numbered or has_trake_marker:
        return {
            "type": "trake",
            "events": decompose_events(query),
        }

    has_question = "?" in query or any(
        marker in lowered
        for marker in QUESTION_MARKERS_EN + QUESTION_MARKERS_VI
    )

    if has_question:
        event_description, question = split_qa(query)

        return {
            "type": "qa",
            "event_description": event_description,
            "question": question,
        }

    return {
        "type": "kis",
        "event_description": query,
    }


def llm_parse_query(query):
    """
    MÃ GIẢ: điểm tích hợp LLM khi rule-based không đủ (C6/C7).
    Xem PLACEHOLDERS.md — cần API key của một LLM (vd: Qwen/GPT).
    """

    raise NotImplementedError(
        "Chưa cấu hình LLM. Hiện tại dùng rule-based; "
        "xem PLACEHOLDERS.md mục 'LLM phân tích query'."
    )
