from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

from legal_rag.gemini_client import generate_text
from legal_rag.text_utils import expand_query_with_synonyms


@dataclass
class QueryUnderstanding:
    original_question: str
    normalized_question: str
    domain: str = "unknown"
    intent: str = "tra_cuu_quy_dinh"
    legal_terms: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    confidence: float = 0.0
    note: str = ""
    used_gemini: bool = False


def _json_from_text(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _unique_clean(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen = set()
    for value in values:
        value = str(value or "").strip()
        if not value:
            continue
        key = value.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(value)
    return cleaned


def understand_question(user_question: str) -> QueryUnderstanding:
    """Use Gemini as the AI query understanding layer.

    This function does NOT answer the legal question. It only rewrites noisy,
    informal, non-diacritic Vietnamese into a clearer criminal-law search query.
    The retriever then uses the rewritten question + legal terms to find sources.
    """
    fallback_query = expand_query_with_synonyms(user_question)
    fallback = QueryUnderstanding(
        original_question=user_question,
        normalized_question=fallback_query,
        search_queries=[fallback_query],
        note="Không dùng được Gemini query understanding, hệ thống dùng câu hỏi gốc và từ điển thuật ngữ nội bộ.",
    )

    prompt = f"""
Bạn là bộ phân tích câu hỏi cho hệ thống RAG tra cứu Bộ luật Hình sự Việt Nam.

Vai trò của bạn: HIỂU CÂU HỎI, không được trả lời câu hỏi pháp lý.

Nhiệm vụ:
1. Viết lại câu hỏi người dùng thành tiếng Việt có dấu, rõ nghĩa, trung lập.
2. Nếu người dùng dùng ngôn ngữ đời thường, hãy ánh xạ sang thuật ngữ hình sự phù hợp.
3. Không kết luận ai có tội/không có tội.
4. Không tạo căn cứ pháp lý nếu không chắc.
5. Trích xuất legal_terms dùng để tra cứu trong Bộ luật Hình sự.
6. Tạo 3-5 search_queries ngắn, mỗi query ưu tiên thuật ngữ pháp lý hoặc tên tội danh.
7. Phân loại domain, chỉ chọn một giá trị:
   - criminal_law: thuộc Bộ luật Hình sự
   - labor_law: lao động
   - civil_law: dân sự
   - administrative_law: hành chính
   - land_law: đất đai
   - marriage_family_law: hôn nhân gia đình
   - tax_law: thuế
   - general_chat: trò chuyện thông thường
   - off_topic: ngoài phạm vi pháp luật hoặc không liên quan
   - unknown: không rõ

8. Phân loại intent, chỉ chọn một giá trị:
   - legal_lookup: tra cứu quy định
   - legal_explanation: giải thích quy định
   - penalty_question: hỏi về hình phạt/mức xử lý
   - case_assessment: hỏi đánh giá tình huống
   - unsafe: yêu cầu nguy hiểm hoặc trái pháp luật
   - evade_law: hỏi cách trốn tránh/che giấu trách nhiệm pháp luật
   - criminal_instruction: hỏi cách thực hiện hành vi phạm tội
   - off_topic: ngoài phạm vi
   - unclear: chưa rõ
9. Trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.

Ví dụ 1:
Input: an cap nho co bi di tu khong
Output JSON:
{{
  "normalized_question": "Hành vi trộm cắp tài sản có thể bị xử lý như thế nào theo Bộ luật Hình sự?",
  "domain": "criminal_law",
  "intent": "tra_cuu_toi_danh_hinh_phat",
  "legal_terms": ["trộm cắp tài sản", "trách nhiệm hình sự", "hình phạt tù", "chiếm đoạt tài sản"],
  "search_queries": ["trộm cắp tài sản", "tội trộm cắp tài sản", "hình phạt tù đối với trộm cắp tài sản", "chiếm đoạt tài sản"],
  "confidence": 0.9,
  "note": "Câu hỏi đời thường được ánh xạ sang thuật ngữ trộm cắp tài sản."
}}

Ví dụ 2:
Input: danh nguoi bi xu sao
Output JSON:
{{
  "normalized_question": "Hành vi đánh người có thể bị xử lý như thế nào theo Bộ luật Hình sự?",
  "domain": "criminal_law",
  "intent": "tra_cuu_toi_danh_hinh_phat",
  "legal_terms": ["cố ý gây thương tích", "xâm phạm sức khỏe", "trách nhiệm hình sự"],
  "search_queries": ["cố ý gây thương tích", "xâm phạm sức khỏe người khác", "tội cố ý gây thương tích", "gây tổn hại cho sức khỏe"],
  "confidence": 0.86,
  "note": "Câu hỏi có thể liên quan nhóm tội xâm phạm sức khỏe."
}}

Ví dụ 3:
Input: cty co bi di tu khong
Output JSON:
{{
  "normalized_question": "Pháp nhân thương mại có thể phải chịu trách nhiệm hình sự theo Bộ luật Hình sự trong trường hợp nào?",
  "domain": "criminal_law",
  "intent": "tra_cuu_chu_the_chiu_trach_nhiem_hinh_su",
  "legal_terms": ["pháp nhân thương mại", "trách nhiệm hình sự của pháp nhân thương mại"],
  "search_queries": ["pháp nhân thương mại chịu trách nhiệm hình sự", "trách nhiệm hình sự của pháp nhân thương mại", "điều kiện chịu trách nhiệm hình sự của pháp nhân thương mại"],
  "confidence": 0.82,
  "note": "Câu hỏi đời thường về công ty được ánh xạ sang pháp nhân thương mại."
}}

Ví dụ 4:
Input: 17 tuoi pham toi thi xu ly sao
Output JSON:
{{
  "normalized_question": "Người dưới 18 tuổi phạm tội thì được xử lý như thế nào theo Bộ luật Hình sự?",
  "domain": "criminal_law",
  "intent": "tra_cuu_nguoi_duoi_18_tuoi_pham_toi",
  "legal_terms": ["người dưới 18 tuổi phạm tội", "nguyên tắc xử lý", "trách nhiệm hình sự"],
  "search_queries": ["người dưới 18 tuổi phạm tội", "nguyên tắc xử lý người dưới 18 tuổi phạm tội", "trách nhiệm hình sự người dưới 18 tuổi"],
  "confidence": 0.9,
  "note": "Câu hỏi thuộc nhóm quy định về người dưới 18 tuổi phạm tội."
}}

Câu hỏi người dùng:
{user_question}

JSON:
""".strip()

    result = generate_text(prompt, temperature=0.0)
    if not result.ok:
        return fallback

    try:
        data = _json_from_text(result.text)
        normalized = str(data.get("normalized_question") or user_question).strip()

        legal_terms = data.get("legal_terms") or []
        if isinstance(legal_terms, str):
            legal_terms = [legal_terms]

        search_queries = data.get("search_queries") or []
        if isinstance(search_queries, str):
            search_queries = [search_queries]

        # Retrieval should use the rewritten question, extracted legal terms,
        # generated search queries, and a rule-based synonym expansion fallback.
        base_queries = _unique_clean([normalized, *legal_terms, *search_queries])
        expanded = expand_query_with_synonyms("\n".join(base_queries))
        final_queries = _unique_clean([*base_queries, expanded])

        return QueryUnderstanding(
            original_question=user_question,
            normalized_question=normalized,
            domain=str(data.get("domain", "unknown")),
            intent=str(data.get("intent", "tra_cuu_quy_dinh")),
            legal_terms=_unique_clean([str(x) for x in legal_terms]),
            search_queries=final_queries,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            note=str(data.get("note", "")),
            used_gemini=True,
        )
    except Exception as exc:
        fallback.note = f"Gemini trả về dữ liệu không parse được, dùng fallback. Lỗi: {exc}"
        return fallback
