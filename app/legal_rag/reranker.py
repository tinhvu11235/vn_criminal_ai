from __future__ import annotations

import json
import re
from typing import List

from legal_rag.gemini_client import generate_text
from legal_rag.hybrid_retriever import RetrievedChunk, source_label


def _parse_ids(text: str) -> List[str]:
    text = re.sub(r"^```json\s*", "", text.strip())
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        ids = data.get("chunk_ids") or data.get("ids") or []
        return [str(x) for x in ids]
    except Exception:
        return re.findall(r"[a-zA-Z0-9_\-]+", text)


def gemini_rerank(question: str, candidates: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
    if len(candidates) <= top_k:
        return candidates
    listing = []
    for item in candidates:
        c = item.chunk
        excerpt = c.get("text", "")[:1100]
        listing.append(
            f"ID: {c.get('chunk_id')}\nSOURCE: {source_label(c)}\nTEXT: {excerpt}"
        )
    prompt = f"""
Bạn là bộ rerank nguồn cho RAG tra cứu Bộ luật Hình sự Việt Nam.
Chọn {top_k} chunk liên quan nhất để trả lời câu hỏi. Ưu tiên chunk chứa điều luật/tội danh trực tiếp.
Không trả lời câu hỏi. Chỉ trả về JSON hợp lệ.

Câu hỏi:
{question}

Danh sách chunk ứng viên:
{chr(10).join(listing)}

JSON format:
{{"chunk_ids": ["id1", "id2", "id3"]}}
""".strip()
    result = generate_text(prompt, temperature=0.0)
    if not result.ok:
        return candidates[:top_k]
    ids = _parse_ids(result.text)
    by_id = {str(item.chunk.get("chunk_id")): item for item in candidates}
    reranked = [by_id[i] for i in ids if i in by_id]
    for item in candidates:
        if item not in reranked:
            reranked.append(item)
        if len(reranked) >= top_k:
            break
    for idx, item in enumerate(reranked[:top_k], start=1):
        item.rank = idx
    return reranked[:top_k]
