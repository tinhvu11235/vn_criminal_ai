from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from legal_rag.gemini_client import generate_text
from legal_rag.hybrid_retriever import HybridLegalRetriever, RetrievedChunk, source_label
from legal_rag.query_understanding import QueryUnderstanding, understand_question
from legal_rag.reranker import gemini_rerank
from legal_rag.guardrails import classify_scope

@dataclass
class QAResult:
    answer: str
    understanding: QueryUnderstanding
    sources: List[RetrievedChunk] = field(default_factory=list)
    error: str = ""


SYSTEM_RULES = """
Bạn là Trợ lý tra cứu Bộ luật Hình sự Việt Nam.

Quy tắc bắt buộc:
1. Chỉ trả lời dựa trên CONTEXT được cung cấp.
2. Không tự bịa điều luật, số điều, khung hình phạt hoặc mức phạt.
3. Không kết luận một người cụ thể có tội/không có tội.
4. Không hướng dẫn trốn tránh pháp luật, che giấu tội phạm, khai gian hoặc lách luật.
5. Nếu context không đủ căn cứ, nói rõ: "Tôi chưa tìm thấy căn cứ rõ ràng trong bộ dữ liệu hiện có".
6. Trả lời bằng tiếng Việt, rõ ràng, mạch lạc, phù hợp người dùng phổ thông.
7. Câu trả lời chính chỉ cần MỘT câu trả lời thống nhất, không đưa nhiều phương án dài dòng nếu câu hỏi đủ rõ.
8. Nếu câu hỏi mơ hồ, hãy trả lời phần chắc chắn dựa trên context và nêu điều kiện cần biết thêm, nhưng không hỏi lại quá nhiều.
""".strip()


def _build_context(sources: List[RetrievedChunk]) -> str:
    parts = []
    for idx, item in enumerate(sources, start=1):
        c = item.chunk
        label = source_label(c)
        text = c.get("text", "")
        parts.append(f"[{idx}] {label}\n{text}")
    return "\n\n---\n\n".join(parts)


def _main_source_line(sources: List[RetrievedChunk]) -> str:
    if not sources:
        return ""
    labels = []
    for item in sources[:2]:
        labels.append(source_label(item.chunk))
    return "; ".join(labels)


def _fallback_answer(sources: List[RetrievedChunk]) -> str:
    if not sources:
        return "Tôi chưa tìm thấy căn cứ rõ ràng trong bộ dữ liệu hiện có."
    source_line = _main_source_line(sources)
    return (
        "Tôi đã tìm được một số điều luật có thể liên quan, nhưng Gemini chưa được cấu hình hoặc đang lỗi nên chưa thể tạo câu trả lời đầy đủ. "
        f"Bạn có thể kiểm tra nguồn chính ở mục bên dưới: {source_line}."
    )


def answer_question(
    user_question: str,
    retriever: HybridLegalRetriever,
    use_rerank: bool = True,
    top_k: int = 5,
) -> QAResult:
    # 1) AI query understanding: rewrite noisy/informal question into legal search queries.
    understanding = understand_question(user_question)
    allow_answer, refusal_message = classify_scope(understanding)
    if not allow_answer:
        return QAResult(
            answer=refusal_message or "Mình chưa thể trả lời câu hỏi này trong phạm vi Bộ luật Hình sự.",
            understanding=understanding,
            sources=[],
        )
    search_queries = understanding.search_queries or [understanding.normalized_question, user_question]

    # 2) Hybrid retrieval: semantic + keyword + metadata.
    candidates = retriever.search(search_queries, top_k=max(top_k * 2, 10), candidate_k=30)

    # 3) Optional Gemini rerank: choose the most relevant legal provisions.
    sources = gemini_rerank(understanding.normalized_question, candidates, top_k=top_k) if use_rerank else candidates[:top_k]

    if not sources:
        return QAResult(
            answer="Tôi chưa tìm thấy căn cứ rõ ràng trong bộ dữ liệu Bộ luật Hình sự hiện có.",
            understanding=understanding,
            sources=[],
        )

    context = _build_context(sources)
    source_line = _main_source_line(sources)

    domain_warning = ""
    if understanding.domain != "criminal_law" and understanding.confidence >= 0.65:
        domain_warning = (
            f"\nLưu ý nội bộ: bộ phân tích câu hỏi nhận diện domain là {understanding.domain}. "
            "Nếu context không đủ căn cứ hình sự, hãy nói rõ giới hạn và không cố trả lời ngoài phạm vi.\n"
        )

    prompt = f"""
{SYSTEM_RULES}
{domain_warning}

CÂU HỎI GỐC:
{understanding.original_question}

CÂU HỎI HỆ THỐNG HIỂU:
{understanding.normalized_question}

CONTEXT LUẬT ĐƯỢC TRUY XUẤT:
{context}

Yêu cầu format câu trả lời:
- Chỉ tạo MỘT câu trả lời chính, không tạo nhiều phiên bản trả lời.
- Trả lời trực tiếp vào câu hỏi trước.
- Sau đó giải thích ngắn gọn 2-5 câu hoặc 2-4 gạch đầu dòng nếu cần.
- Không lặp lại toàn bộ điều luật dài dòng; diễn giải dễ hiểu nhưng không làm sai nghĩa.
- Không tự thêm thông tin ngoài context.
- Không cần tạo mục "Nguồn tham khảo" dài trong câu trả lời chính, vì UI đã có phần nguồn riêng.
- Cuối câu trả lời chỉ thêm một dòng ngắn: "Căn cứ chính: {source_line}."
- Thêm một câu lưu ý ngắn: "Thông tin chỉ nhằm mục đích tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp."
""".strip()

    result = generate_text(prompt, temperature=0.1)
    if not result.ok:
        return QAResult(
            answer=_fallback_answer(sources),
            understanding=understanding,
            sources=sources,
            error=result.error or "Gemini error",
        )
    return QAResult(answer=result.text, understanding=understanding, sources=sources)
