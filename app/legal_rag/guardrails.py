from __future__ import annotations


OUT_OF_SCOPE_MESSAGE = (
    "Câu hỏi này có vẻ không thuộc phạm vi Bộ luật Hình sự mà hệ thống hiện đang tra cứu. "
    "Mình chưa đưa ra câu trả lời pháp lý để tránh sai lệch. "
    "Bạn có thể hỏi về tội phạm, hình phạt, trách nhiệm hình sự, tình tiết giảm nhẹ, "
    "người dưới 18 tuổi phạm tội hoặc các tội danh trong Bộ luật Hình sự."
)

GENERAL_OFF_TOPIC_MESSAGE = (
    "Mình hiện chỉ hỗ trợ tra cứu thông tin từ Bộ luật Hình sự Việt Nam. "
    "Bạn có thể hỏi về khái niệm tội phạm, hình phạt, trách nhiệm hình sự, "
    "tình tiết giảm nhẹ, người dưới 18 tuổi phạm tội hoặc các tội danh cụ thể."
)

UNSAFE_MESSAGE = (
    "Mình không thể hỗ trợ cách thực hiện, che giấu hoặc né tránh trách nhiệm pháp luật. "
    "Mình có thể giúp tra cứu quy định liên quan đến trách nhiệm hình sự, tự thú, "
    "khắc phục hậu quả, tình tiết giảm nhẹ hoặc quyền và nghĩa vụ theo quy định pháp luật."
)


def _text_of(understanding) -> str:
    original = getattr(understanding, "original_question", "") or ""
    normalized = getattr(understanding, "normalized_question", "") or ""
    terms = " ".join(getattr(understanding, "legal_terms", []) or [])
    queries = " ".join(getattr(understanding, "search_queries", []) or [])
    return f"{original} {normalized} {terms} {queries}".lower()


def detect_unsafe_question(text: str) -> bool:
    unsafe_patterns = [
        "tron toi",
        "trốn tội",
        "khong bi phat hien",
        "không bị phát hiện",
        "che giau",
        "che giấu",
        "xoa dau vet",
        "xóa dấu vết",
        "qua mat cong an",
        "qua mặt công an",
        "lam sao de giet",
        "làm sao để giết",
        "cach lua dao",
        "cách lừa đảo",
        "cach trom",
        "cách trộm",
        "cach cuop",
        "cách cướp",
        "khai gian",
        "lam chung gia",
        "làm chứng giả",
    ]
    return any(pattern in text for pattern in unsafe_patterns)


def classify_scope(understanding) -> tuple[bool, str | None]:
    """
    Returns:
        allow_answer: bool
        refusal_message: str | None
    """
    domain = (getattr(understanding, "domain", "unknown") or "unknown").strip()
    intent = (getattr(understanding, "intent", "unknown") or "unknown").strip()
    confidence = float(getattr(understanding, "confidence", 0.0) or 0.0)

    combined_text = _text_of(understanding)

    if detect_unsafe_question(combined_text):
        return False, UNSAFE_MESSAGE

    if intent in {"unsafe", "evade_law", "criminal_instruction", "harmful_instruction"}:
        return False, UNSAFE_MESSAGE

    if domain in {"general_chat", "off_topic", "weather", "joke", "math"}:
        return False, GENERAL_OFF_TOPIC_MESSAGE

    if domain not in {"criminal_law", "unknown", "unclear"} and confidence >= 0.65:
        return False, OUT_OF_SCOPE_MESSAGE

    return True, None