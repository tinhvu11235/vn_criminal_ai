from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import fitz  # PyMuPDF

from legal_rag.config import PROCESSED_DIR, RAW_DIR
from legal_rag.text_utils import compact_whitespace

ARTICLE_RE = re.compile(r"(?m)^\s*Điều\s+(\d+[a-zA-Z]?)\.\s*(.*)$")
CLAUSE_RE = re.compile(r"(?m)^\s*(\d+)\.\s+(.+)")


@dataclass
class LegalChunk:
    chunk_id: str
    document_id: str
    document_title: str
    document_number: str
    document_type: str
    article: str
    article_title: str
    clause: str
    text: str
    page_start: int
    page_end: int
    source_url: str


def load_source_info() -> Dict[str, dict]:
    info_path = RAW_DIR / "source_info.json"
    if not info_path.exists():
        return {}
    return json.loads(info_path.read_text(encoding="utf-8"))


def extract_pages(pdf_path: Path) -> List[dict]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = clean_extracted_text(text)
            if text:
                pages.append({"file_name": pdf_path.name, "page": idx, "text": text})
    return pages


def clean_extracted_text(text: str) -> str:
    text = compact_whitespace(text)
    lines = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        # Remove obvious page numbers / repeated decorative lines.
        if re.fullmatch(r"\d{1,3}", raw):
            continue
        if len(raw) <= 2 and not raw.isalnum():
            continue
        lines.append(raw)
    text = "\n".join(lines)
    # Merge hyphenated line breaks conservatively.
    text = re.sub(r"-\n(?=\w)", "", text)
    return compact_whitespace(text)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _doc_id_from_name(name: str) -> str:
    base = re.sub(r"\.[^.]+$", "", name.lower())
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    return base or "document"


def parse_articles_from_pages(pages: List[dict], source_info: dict) -> List[LegalChunk]:
    all_text_parts = []
    page_for_offset = []
    cursor = 0
    for page in pages:
        marker = f"\n\n[[PAGE:{page['page']}]]\n"
        txt = marker + page["text"]
        all_text_parts.append(txt)
        for _ in txt:
            page_for_offset.append(page["page"])
        cursor += len(txt)
    full_text = "".join(all_text_parts)

    matches = list(ARTICLE_RE.finditer(full_text))
    chunks: List[LegalChunk] = []
    file_name = pages[0]["file_name"] if pages else "unknown.pdf"
    doc_meta = source_info.get(file_name, {})
    document_id = _doc_id_from_name(file_name)

    if not matches:
        # Fallback: page chunks.
        for page in pages:
            chunks.append(
                LegalChunk(
                    chunk_id=f"{document_id}_page_{page['page']}",
                    document_id=document_id,
                    document_title=doc_meta.get("document_title", file_name),
                    document_number=doc_meta.get("document_number", ""),
                    document_type=doc_meta.get("document_type", ""),
                    article="",
                    article_title="",
                    clause="",
                    text=page["text"],
                    page_start=page["page"],
                    page_end=page["page"],
                    source_url=doc_meta.get("source_url", ""),
                )
            )
        return chunks

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        article_number = match.group(1)
        article_title = (match.group(2) or "").strip()
        article_body = full_text[match.end():end]
        article_body = re.sub(r"\[\[PAGE:\d+\]\]", "\n", article_body)
        article_body = compact_whitespace(article_body)
        page_start = page_for_offset[start] if start < len(page_for_offset) else 0
        page_end = page_for_offset[end - 1] if end - 1 < len(page_for_offset) else page_start

        article_heading = f"Điều {article_number}. {article_title}".strip()
        clause_matches = list(CLAUSE_RE.finditer(article_body))

        # If article is short or clause parsing looks weak, keep article as one chunk.
        if len(article_body) < 1200 or len(clause_matches) <= 1:
            text = compact_whitespace(article_heading + "\n" + article_body)
            chunks.append(
                LegalChunk(
                    chunk_id=f"{document_id}_dieu_{article_number}",
                    document_id=document_id,
                    document_title=doc_meta.get("document_title", file_name),
                    document_number=doc_meta.get("document_number", ""),
                    document_type=doc_meta.get("document_type", ""),
                    article=f"Điều {article_number}",
                    article_title=article_title,
                    clause="",
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    source_url=doc_meta.get("source_url", ""),
                )
            )
            continue

        for j, cm in enumerate(clause_matches):
            c_start = cm.start()
            c_end = clause_matches[j + 1].start() if j + 1 < len(clause_matches) else len(article_body)
            clause_num = cm.group(1)
            clause_text = compact_whitespace(article_body[c_start:c_end])
            text = compact_whitespace(article_heading + "\n" + clause_text)
            chunks.append(
                LegalChunk(
                    chunk_id=f"{document_id}_dieu_{article_number}_khoan_{clause_num}",
                    document_id=document_id,
                    document_title=doc_meta.get("document_title", file_name),
                    document_number=doc_meta.get("document_number", ""),
                    document_type=doc_meta.get("document_type", ""),
                    article=f"Điều {article_number}",
                    article_title=article_title,
                    clause=f"Khoản {clause_num}",
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    source_url=doc_meta.get("source_url", ""),
                )
            )
    return chunks


def ingest_all() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    source_info = load_source_info()
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    all_pages: List[dict] = []
    all_chunks: List[LegalChunk] = []

    for pdf in pdfs:
        pages = extract_pages(pdf)
        all_pages.extend(pages)
        all_chunks.extend(parse_articles_from_pages(pages, source_info))

    write_jsonl(PROCESSED_DIR / "extracted_pages.jsonl", all_pages)
    write_jsonl(PROCESSED_DIR / "legal_chunks.jsonl", [c.__dict__ for c in all_chunks])

    article_set = sorted({c.article for c in all_chunks if c.article})
    report = {
        "total_pdf_files": len(pdfs),
        "total_pages": len(all_pages),
        "total_chunks": len(all_chunks),
        "total_articles_detected": len(article_set),
        "sample_articles": article_set[:20],
        "pdf_files": [p.name for p in pdfs],
        "note": "Nếu số điều quá thấp hoặc chunk bị nhiễu, hãy kiểm tra chất lượng PDF/text extraction.",
    }
    (PROCESSED_DIR / "parser_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = ingest_all()
    print(json.dumps(result, ensure_ascii=False, indent=2))
