from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from legal_rag.config import PROCESSED_DIR, get_embedding_model
from legal_rag.text_utils import keyword_score, normalize_text


@dataclass
class RetrievedChunk:
    chunk: dict
    semantic_score: float
    keyword_score: float
    metadata_score: float
    final_score: float
    rank: int = 0


class HybridLegalRetriever:
    """
    Hybrid retrieval for demo:
    - Semantic search using sentence-transformers embeddings.
    - Lexical keyword matching on normalized Vietnamese text.
    - Metadata/title boost for article titles and legal terms.

    This is intentionally lightweight for Streamlit demo deployment.
    """

    def __init__(self, chunks_path: Path | None = None, embedding_model_name: str | None = None):
        self.chunks_path = chunks_path or (PROCESSED_DIR / "legal_chunks.jsonl")
        self.embedding_model_name = embedding_model_name or get_embedding_model()
        self.chunks: List[dict] = self._load_chunks()
        self._model = None
        self._embeddings = None

    def _load_chunks(self) -> List[dict]:
        if not self.chunks_path.exists():
            return []
        rows = []
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _texts_for_embedding(self) -> List[str]:
        texts = []
        for c in self.chunks:
            meta = " ".join(
                str(c.get(k, ""))
                for k in ["document_title", "document_number", "article", "article_title", "clause"]
            )
            texts.append(f"passage: {meta}\n{c.get('text', '')}")
        return texts

    def build_embeddings(self) -> np.ndarray:
        if not self.chunks:
            self._embeddings = np.empty((0, 1), dtype=np.float32)
            return self._embeddings
        texts = self._texts_for_embedding()
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self._embeddings = np.asarray(embeddings, dtype=np.float32)
        return self._embeddings

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            return self.build_embeddings()
        return self._embeddings

    def _metadata_score(self, query: str, chunk: dict) -> float:
        metadata_text = " ".join(
            str(chunk.get(k, ""))
            for k in ["article", "article_title", "clause", "document_title", "document_number"]
        )
        score = keyword_score(query, metadata_text)
        qn = normalize_text(query)
        if "dieu" in qn:
            score += 0.08
        if chunk.get("article_title") and keyword_score(query, chunk.get("article_title", "")) > 0:
            score += 0.12
        return min(score, 1.0)

    def search(self, queries: List[str], top_k: int = 6, candidate_k: int = 30) -> List[RetrievedChunk]:
        if not self.chunks:
            return []
        queries = [q for q in queries if q and q.strip()]
        if not queries:
            return []

        # Semantic score uses max similarity across all rewritten/search queries.
        q_texts = [f"query: {q}" for q in queries]
        q_emb = self.model.encode(q_texts, show_progress_bar=False, normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sim_matrix = cosine_similarity(q_emb, self.embeddings)
        semantic_scores = sim_matrix.max(axis=0)

        combined_query = "\n".join(queries)
        results: List[RetrievedChunk] = []
        for idx, chunk in enumerate(self.chunks):
            content_for_keyword = " ".join(
                str(chunk.get(k, ""))
                for k in ["article", "article_title", "clause", "text"]
            )
            lex = keyword_score(combined_query, content_for_keyword)
            meta = self._metadata_score(combined_query, chunk)
            sem = float(semantic_scores[idx])
            # Semantic matters most, but law needs exact keywords and article titles too.
            final = 0.62 * sem + 0.28 * lex + 0.10 * meta
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    semantic_score=sem,
                    keyword_score=lex,
                    metadata_score=meta,
                    final_score=final,
                )
            )
        results.sort(key=lambda x: x.final_score, reverse=True)
        deduped: List[RetrievedChunk] = []
        seen = set()
        for item in results[:candidate_k]:
            cid = item.chunk.get("chunk_id")
            if cid in seen:
                continue
            seen.add(cid)
            item.rank = len(deduped) + 1
            deduped.append(item)
            if len(deduped) >= top_k:
                break
        return deduped


def source_label(chunk: dict) -> str:
    parts = [chunk.get("document_title", "Văn bản")]
    if chunk.get("article"):
        parts.append(chunk["article"])
    if chunk.get("clause"):
        parts.append(chunk["clause"])
    if chunk.get("article_title"):
        parts.append(chunk["article_title"])
    return " - ".join(str(x) for x in parts if x)
