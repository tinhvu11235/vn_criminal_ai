from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

VI_STOPWORDS = {
    "la", "gi", "co", "khong", "ko", "k", "va", "hoac", "thi", "the", "nao", "nhu",
    "ve", "cua", "cho", "trong", "theo", "duoc", "bi", "neu", "toi", "moi", "mot",
    "cac", "nhung", "nay", "do", "ra", "vao", "voi", "tu", "den", "hay", "hoi",
}

LEGAL_SYNONYMS = {
    # Theft / property crimes
    "an cap": "trộm cắp tài sản chiếm đoạt tài sản",
    "trom": "trộm cắp tài sản chiếm đoạt tài sản",
    "trom cap": "trộm cắp tài sản chiếm đoạt tài sản",
    "lay do": "trộm cắp tài sản chiếm đoạt tài sản",
    "cuop": "cướp tài sản chiếm đoạt tài sản",
    "cuop giat": "cướp giật tài sản chiếm đoạt tài sản",
    "lua dao": "lừa đảo chiếm đoạt tài sản",
    "lua tien": "lừa đảo chiếm đoạt tài sản",

    # Violence / health / life
    "danh nguoi": "cố ý gây thương tích xâm phạm sức khỏe người khác",
    "danh nhau": "cố ý gây thương tích gây rối trật tự công cộng",
    "gay thuong tich": "cố ý gây thương tích tổn hại sức khỏe",
    "giet nguoi": "giết người xâm phạm tính mạng",
    "dam nguoi": "cố ý gây thương tích giết người",

    # Drugs and other common legal topics
    "ma tuy": "ma túy chất ma túy tội phạm về ma túy",
    "chua 18": "người dưới 18 tuổi phạm tội nguyên tắc xử lý",
    "duoi 18": "người dưới 18 tuổi phạm tội nguyên tắc xử lý",
    "tre em": "người dưới 18 tuổi",
    "tu thu": "tự thú tình tiết giảm nhẹ trách nhiệm hình sự",
    "dau thu": "đầu thú tình tiết giảm nhẹ trách nhiệm hình sự",
    "giam an": "giảm nhẹ trách nhiệm hình sự hình phạt",
    "mien trach nhiem": "miễn trách nhiệm hình sự",
    "cong ty": "pháp nhân thương mại trách nhiệm hình sự",
    "doanh nghiep": "pháp nhân thương mại trách nhiệm hình sự",
}


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_text(text: str) -> str:
    text = strip_accents(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return [tok for tok in normalized.split() if len(tok) > 1 and tok not in VI_STOPWORDS]


def expand_query_with_synonyms(question: str) -> str:
    normalized = normalize_text(question)
    additions = []
    for key, value in LEGAL_SYNONYMS.items():
        if key in normalized:
            additions.append(value)
    if not additions:
        return question
    return question + "\nThuật ngữ pháp lý liên quan: " + "; ".join(additions)


def keyword_score(query: str, text: str) -> float:
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    t_tokens = set(tokenize(text))
    overlap = len(q_tokens & t_tokens)
    return overlap / max(len(q_tokens), 1)


def compact_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
