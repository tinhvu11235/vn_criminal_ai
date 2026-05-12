from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF

from legal_rag.config import PROCESSED_DIR, RAW_DIR


# ---------------------------------------------------------------------------
# Regex rules
# ---------------------------------------------------------------------------
# Some Vietnamese legal PDFs extract "Điều" as "Ðiều". Accept both.
ARTICLE_RE = re.compile(r"(?m)^\s*[ĐÐ]iều\s+(\d+[a-zA-Z]?)\.\s*(.*)$")

# Match real top-level clauses only after footnote superscripts have been removed.
# Keep this intentionally conservative: legal clauses are normally 1., 2., ...
CLAUSE_RE = re.compile(r"(?m)^\s*(\d{1,2})\.\s+(.+)$")

# Chapter/section headings are metadata, not article body. They are useful if you
# later want to enrich metadata, but should not be treated as article text.
STRUCTURAL_HEADING_RE = re.compile(
    r"^\s*(Phần thứ|Chương\s+[IVXLCDM]+|Mục\s+\d+|Tiểu mục\s+\d+)\b",
    re.IGNORECASE,
)

# Inline amendment notes sometimes get interleaved into the main text by PDF
# extraction order. They are provenance notes, not normative article content.
AMENDMENT_NOTE_RE = re.compile(
    r"(?:"
    r"Tên\s+Điều\s+này|Điểm\s+này|Khoản\s+này|Điều\s+này|"
    r"Cụm\s+từ\s+[“\"].{0,80}?[”\"]|Từ\s+[“\"].{0,80}?[”\"]|"
    r"Dấu\s+[“\"].{0,80}?[”\"]|Các\s+từ\s+[“\"].{0,80}?[”\"]"
    r")"
    r".{0,900}?"
    r"(?:có\s+hiệu\s+lực\s+kể\s+từ\s+ngày\s+\d{2}\s+tháng\s+\d{2}\s+năm\s+\d{4}\.|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


# A footnote block often starts with a superscript number followed by one of
# these phrases. After the first footnote line, the rest of the page is normally
# footnote text, so the whole tail is skipped.
FOOTNOTE_START_RE = re.compile(
    r"^\s*\d{1,4}\s*("
    r"Điểm này|Khoản này|Điều này|Chương này|Cụm từ|Từ\s+|Dấu\s+|Các từ|"
    r"Việc thi hành|Hiệu lực thi hành|Luật số|Nghị quyết|Điều\s+\d+\s+của\s+Luật"
    r")\b",
    re.IGNORECASE,
)

# If we have already dropped the superscript digit span, these are still clear
# signs that a line belongs to a footnote block.
FOOTNOTE_TEXT_START_RE = re.compile(
    r"^\s*("
    r"Điểm này|Khoản này|Điều này|Chương này|Cụm từ|Từ\s+|Dấu\s+|Các từ|"
    r"Việc thi hành|Hiệu lực thi hành|Luật số|Nghị quyết"
    r")\b"
)


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


@dataclass
class ArticleSpan:
    number: str
    title: str
    body: str
    start: int
    end: int
    body_start: int
    page_start: int
    page_end: int


# ---------------------------------------------------------------------------
# General text normalization
# ---------------------------------------------------------------------------
def compact_whitespace(text: str) -> str:
    """Collapse excessive spaces while preserving meaningful newlines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines: List[str] = []
    blank = False
    for line in text.split("\n"):
        line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
        if not line:
            if not blank and lines:
                lines.append("")
                blank = True
            continue
        lines.append(line)
        blank = False
    return "\n".join(lines).strip()


def compact_for_chunk(text: str) -> str:
    """RAG-friendly paragraph text: preserve paragraph breaks, remove line-wrap noise."""
    text = normalize_ocr(text)
    text = strip_amendment_notes(text)
    text = re.sub(r"-\n(?=\w)", "", text)
    # Join normal wrapped lines, but keep paragraph-ish breaks around articles/clauses/points.
    text = re.sub(r"\n(?!(?:\d{1,2}\.\s|[a-zđ]\)\s|Điều\s+\d+\.))", " ", text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_ocr(text: str) -> str:
    """Fix common OCR/extraction variants seen in Vietnamese legal PDFs."""
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "Ðiều": "Điều",
        "ÐIỀU": "ĐIỀU",
        "Ð": "Đ",
        "điều": "điều",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text



VIETNAMESE_DIACRITIC_RE = re.compile(
    r"[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩị"
    r"òóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊ"
    r"ÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ]"
)
SUSPICIOUS_TEXT_RE = re.compile(r"[~<>◊@{}\[\]\|^`�]")


def _looks_garbled_vietnamese_line(line: str) -> bool:
    """Detect badly decoded/OCR-corrupted Vietnamese lines.

    The last page of some official PDFs can contain a scanned/signature layer
    whose text extraction is unusable. Keeping it hurts RAG more than omitting it.
    """
    if len(line) < 25:
        return False
    has_diacritic = bool(VIETNAMESE_DIACRITIC_RE.search(line))
    suspicious = len(SUSPICIOUS_TEXT_RE.findall(line))
    if suspicious >= 2:
        return True
    if suspicious >= 1 and not has_diacritic:
        return True
    if not has_diacritic and len(line) >= 80:
        legal_ascii_noise = re.search(
            r"\b(lu[~a]t|hinh|phap|quy|dinh|thang|nam|toa|nhan|dan|chinh|phu)\b",
            line,
            flags=re.IGNORECASE,
        )
        if legal_ascii_noise:
            return True
    return False


def _looks_low_quality_page(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 180:
        return False
    if re.search(r"VAN PH<|HQP NHAT|CHUNH|LU\'UHC|VBI IN", text):
        return True
    diacritics = len(VIETNAMESE_DIACRITIC_RE.findall(text))
    suspicious = len(SUSPICIOUS_TEXT_RE.findall(text))
    # Vietnamese legal text should contain many diacritics. If a long page has
    # almost none and contains extraction artifacts, it is safer to drop it.
    return diacritics / max(len(compact), 1) < 0.005 and suspicious >= 2


def strip_amendment_notes(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = AMENDMENT_NOTE_RE.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text)


def _join_spans_as_line(spans: Sequence[dict], *, drop_superscript_digits: bool) -> str:
    """Join spans into a readable line and optionally drop small numeric footnote marks."""
    parts: List[str] = []
    last_x1: Optional[float] = None
    for span in spans:
        piece = span.get("text", "")
        if not piece:
            continue
        stripped = piece.strip()
        if not stripped:
            continue

        size = float(span.get("size", 0.0) or 0.0)
        # Footnote markers inside the main text are usually tiny digit-only spans,
        # e.g. "1." + "395" + " Người nào" or title + "400".
        if drop_superscript_digits and size <= 9.5 and re.fullmatch(r"\d{1,4}", stripped):
            continue

        x0 = float(span.get("bbox", [0, 0, 0, 0])[0])
        if last_x1 is not None:
            gap = x0 - last_x1
            # PDF spans often omit spaces between adjacent spans. Add one when
            # there is a visible horizontal gap and the previous/current text are
            # not punctuation-bound.
            if gap > 1.8 and parts and not parts[-1].endswith((" ", "(", "[", "{", "“", "‘")):
                if not stripped.startswith((".", ",", ";", ":", ")", "]", "}", "%")):
                    parts.append(" ")
        parts.append(piece)
        last_x1 = float(span.get("bbox", [0, 0, 0, 0])[2])

    return normalize_ocr(re.sub(r"[ \t\u00a0]+", " ", "".join(parts)).strip())


def _first_nonempty_span_info(spans: Sequence[dict]) -> Tuple[str, float]:
    for span in spans:
        text = span.get("text", "").strip()
        if text:
            return text, float(span.get("size", 0.0) or 0.0)
    return "", 0.0


def _is_footnote_start_line(raw_line: str, clean_line: str, spans: Sequence[dict], line_bbox: Sequence[float]) -> bool:
    first_text, first_size = _first_nonempty_span_info(spans)
    y0 = float(line_bbox[1]) if len(line_bbox) > 1 else 0.0
    starts_with_tiny_note_number = bool(re.fullmatch(r"\d{1,4}", first_text)) and first_size <= 9.5
    is_bottom_note_area = y0 >= 720.0

    if FOOTNOTE_START_RE.match(raw_line) and (starts_with_tiny_note_number or is_bottom_note_area):
        return True
    if FOOTNOTE_TEXT_START_RE.match(clean_line) and is_bottom_note_area:
        return True
    return False


def extract_page_text(page: fitz.Page) -> str:
    """Extract one page while removing page numbers and footnote blocks.

    This is more reliable than page.get_text('text') for Vietnamese consolidated
    legal documents because superscript footnote numbers can otherwise become
    fake clause/article numbers.
    """
    data = page.get_text("dict", sort=True)
    kept_lines: List[str] = []
    in_footnote = False

    for block in data.get("blocks", []):
        if "lines" not in block:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            raw_line = _join_spans_as_line(spans, drop_superscript_digits=False)
            clean_line = _join_spans_as_line(spans, drop_superscript_digits=True)
            raw_line = raw_line.strip()
            clean_line = clean_line.strip()

            if not raw_line or not clean_line:
                continue
            if re.fullmatch(r"\d{1,4}", clean_line):
                continue
            if len(clean_line) <= 2 and not clean_line.isalnum():
                continue

            # Once a real footnote block starts, skip the rest of this page.
            # Guard with font-size/bottom-page checks so wrapped body text like
            # "1 Điều này..." is not mistaken for a footnote.
            if _is_footnote_start_line(raw_line, clean_line, spans, line.get("bbox", [0, 0, 0, 0])):
                in_footnote = True
                continue
            if in_footnote:
                continue
            if _looks_garbled_vietnamese_line(clean_line):
                continue

            kept_lines.append(clean_line)

    page_text = compact_whitespace("\n".join(kept_lines))
    if _looks_low_quality_page(page_text):
        return ""
    return page_text


def clean_extracted_text(text: str) -> str:
    """Fallback cleaner for callers that already extracted plain text.

    Prefer extract_page_text(page) because it can remove superscript spans.
    """
    text = normalize_ocr(text)
    lines: List[str] = []
    in_footnote = False
    for line in text.splitlines():
        raw = re.sub(r"[ \t\u00a0]+", " ", line).strip()
        if not raw:
            continue
        if re.fullmatch(r"\d{1,4}", raw):
            continue
        if FOOTNOTE_START_RE.match(raw) or FOOTNOTE_TEXT_START_RE.match(raw):
            in_footnote = True
            continue
        if in_footnote:
            continue
        lines.append(raw)
    return compact_whitespace("\n".join(lines))


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_source_info() -> Dict[str, dict]:
    info_path = RAW_DIR / "source_info.json"
    if not info_path.exists():
        return {}
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_page_overrides() -> Dict[str, dict]:
    """Optional manual OCR/text overrides for pages with broken embedded text.

    Format in RAW_DIR/page_overrides.json:
    {
      "135-vbhn-vpqh.pdf": {
        "277": "Bộ luật này có hiệu lực thi hành ..."
      }
    }
    """
    override_path = RAW_DIR / "page_overrides.json"
    if not override_path.exists():
        return {}
    return json.loads(override_path.read_text(encoding="utf-8"))


def extract_pages(pdf_path: Path) -> List[dict]:
    pages: List[dict] = []
    page_overrides = load_page_overrides().get(pdf_path.name, {})
    with fitz.open(pdf_path) as doc:
        for idx, page in enumerate(doc, start=1):
            override_text = page_overrides.get(str(idx)) or page_overrides.get(idx)
            if override_text:
                text = clean_extracted_text(str(override_text))
            else:
                text = extract_page_text(page)
            if text:
                pages.append({"file_name": pdf_path.name, "page": idx, "text": text})
    return pages


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


def _slug_piece(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_")
    return value or "x"


# ---------------------------------------------------------------------------
# Article and clause parsing
# ---------------------------------------------------------------------------
def _build_full_text(pages: List[dict]) -> Tuple[str, List[Tuple[int, int, int]]]:
    parts: List[str] = []
    spans: List[Tuple[int, int, int]] = []
    offset = 0
    for page in pages:
        txt = f"\n\n[[PAGE:{page['page']}]]\n{page['text']}\n"
        start = offset
        end = start + len(txt)
        parts.append(txt)
        spans.append((start, end, int(page["page"])))
        offset = end
    return "".join(parts), spans


def _page_at(offset: int, page_spans: List[Tuple[int, int, int]]) -> int:
    for start, end, page_no in page_spans:
        if start <= offset < end:
            return page_no
    return page_spans[-1][2] if page_spans else 0


def _article_num_key(article_no: str) -> Tuple[int, str]:
    m = re.match(r"(\d+)([a-zA-Z]?)", article_no)
    if not m:
        return 0, ""
    return int(m.group(1)), m.group(2).lower()


def _filter_article_matches(matches: List[re.Match]) -> Tuple[List[re.Match], List[str]]:
    """Remove fake article headings from footnotes or quoted amendment texts.

    The main law's article numbers should be non-decreasing. In this document,
    quoted amendment provisions after Article 426 include "Điều 3" and "Điều 4";
    those must not split the Code into fake articles.
    """
    kept: List[re.Match] = []
    skipped: List[str] = []
    last_num = -1

    for match in matches:
        article_no = match.group(1)
        num, suffix = _article_num_key(article_no)
        heading = compact_for_chunk(match.group(0))

        if kept and num < last_num:
            skipped.append(heading)
            continue
        # Avoid duplicate article markers unless they are lettered articles like 123a.
        if kept and num == last_num and not suffix:
            skipped.append(heading)
            continue

        kept.append(match)
        last_num = max(last_num, num)

    return kept, skipped


def _looks_like_title_continuation(line: str) -> bool:
    """Heuristic for multi-line legal article headings."""
    stripped = line.strip()
    if not stripped:
        return False
    if ARTICLE_RE.match(stripped):
        return False
    if STRUCTURAL_HEADING_RE.match(stripped):
        return False
    if CLAUSE_RE.match(stripped):
        return False
    if re.match(r"^[a-zA-ZđĐ]\)\s", stripped):
        return False
    # Most heading continuations in these PDFs begin with lowercase text, e.g.
    # "trên lãnh thổ...", "hoặc trong huấn luyện", "bị kỹ thuật quân sự".
    first = stripped[0]
    if first.islower():
        return True
    # Also handle rare continuation lines beginning with a connector.
    return bool(re.match(r"^(hoặc|và|đối với|trong|của)\b", stripped, re.IGNORECASE))


def _split_article_title_and_body(raw_title: str, raw_body: str, match_end: int) -> Tuple[str, str, int]:
    title_parts = [raw_title.strip()] if raw_title.strip() else []
    body = raw_body.lstrip("\n")
    consumed = len(raw_body) - len(body)

    lines = body.splitlines(keepends=True)
    idx = 0
    # Consume at most 3 continuation lines to avoid swallowing real article text.
    while idx < len(lines) and len(title_parts) < 4:
        line_with_end = lines[idx]
        line = line_with_end.strip()
        if not _looks_like_title_continuation(line):
            break
        title_parts.append(line)
        consumed += len(line_with_end)
        idx += 1

    new_body = "".join(lines[idx:])
    title = compact_for_chunk(" ".join(title_parts))
    return title, new_body, match_end + consumed


def _remove_structural_headings(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if STRUCTURAL_HEADING_RE.match(line.strip()):
            continue
        lines.append(line)
    return compact_whitespace("\n".join(lines))


def _make_article_spans(full_text: str, page_spans: List[Tuple[int, int, int]]) -> Tuple[List[ArticleSpan], List[str]]:
    matches_raw = list(ARTICLE_RE.finditer(full_text))
    matches, skipped = _filter_article_matches(matches_raw)
    articles: List[ArticleSpan] = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        article_number = match.group(1)
        raw_title = match.group(2) or ""
        raw_body = full_text[match.end():end]
        raw_body = re.sub(r"\[\[PAGE:\d+\]\]", "\n", raw_body)
        title, body, body_start = _split_article_title_and_body(raw_title, raw_body, match.end())
        body = _remove_structural_headings(body)
        body = strip_amendment_notes(body)
        body = compact_whitespace(body)

        articles.append(
            ArticleSpan(
                number=article_number,
                title=title,
                body=body,
                start=start,
                end=end,
                body_start=body_start,
                page_start=_page_at(start, page_spans),
                page_end=_page_at(max(start, end - 1), page_spans),
            )
        )

    return articles, skipped


def _valid_clause_matches(article_body: str) -> List[re.Match]:
    matches = list(CLAUSE_RE.finditer(article_body))
    if len(matches) < 2:
        return []

    valid: List[re.Match] = []
    expected_min = 1
    for match in matches:
        num = int(match.group(1))
        if not (1 <= num <= 30):
            continue
        # Legal clauses usually appear in ascending order. Allow gaps, but not
        # reversals caused by noisy lines.
        if valid and num <= int(valid[-1].group(1)):
            continue
        if num < expected_min:
            continue
        valid.append(match)
        expected_min = num + 1

    return valid if len(valid) >= 2 else []


def _chunk_id(document_id: str, article_number: str, clause_number: str = "") -> str:
    base = f"{document_id}_dieu_{_slug_piece(article_number)}"
    if clause_number:
        return f"{base}_khoan_{_slug_piece(clause_number)}"
    return base


def parse_articles_from_pages(pages: List[dict], source_info: dict) -> List[LegalChunk]:
    chunks: List[LegalChunk] = []
    file_name = pages[0]["file_name"] if pages else "unknown.pdf"
    doc_meta = source_info.get(file_name, {})
    document_id = _doc_id_from_name(file_name)

    if not pages:
        return chunks

    full_text, page_spans = _build_full_text(pages)
    articles, _skipped = _make_article_spans(full_text, page_spans)

    if not articles:
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
                    text=compact_for_chunk(page["text"]),
                    page_start=page["page"],
                    page_end=page["page"],
                    source_url=doc_meta.get("source_url", ""),
                )
            )
        return chunks

    for article in articles:
        article_heading = f"Điều {article.number}. {article.title}".strip()
        clause_matches = _valid_clause_matches(article.body)

        if not clause_matches:
            text = compact_for_chunk(article_heading + "\n" + article.body)
            chunks.append(
                LegalChunk(
                    chunk_id=_chunk_id(document_id, article.number),
                    document_id=document_id,
                    document_title=doc_meta.get("document_title", file_name),
                    document_number=doc_meta.get("document_number", ""),
                    document_type=doc_meta.get("document_type", ""),
                    article=f"Điều {article.number}",
                    article_title=article.title,
                    clause="",
                    text=text,
                    page_start=article.page_start,
                    page_end=article.page_end,
                    source_url=doc_meta.get("source_url", ""),
                )
            )
            continue

        for j, cm in enumerate(clause_matches):
            c_start = cm.start()
            c_end = clause_matches[j + 1].start() if j + 1 < len(clause_matches) else len(article.body)
            clause_num = cm.group(1)
            clause_text = compact_for_chunk(article.body[c_start:c_end])
            text = compact_for_chunk(article_heading + "\n" + clause_text)
            chunks.append(
                LegalChunk(
                    chunk_id=_chunk_id(document_id, article.number, clause_num),
                    document_id=document_id,
                    document_title=doc_meta.get("document_title", file_name),
                    document_number=doc_meta.get("document_number", ""),
                    document_type=doc_meta.get("document_type", ""),
                    article=f"Điều {article.number}",
                    article_title=article.title,
                    clause=f"Khoản {clause_num}",
                    text=text,
                    page_start=article.page_start,
                    page_end=article.page_end,
                    source_url=doc_meta.get("source_url", ""),
                )
            )

    return chunks


# ---------------------------------------------------------------------------
# Ingest entrypoint
# ---------------------------------------------------------------------------
def ingest_all() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    source_info = load_source_info()
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    all_pages: List[dict] = []
    all_chunks: List[LegalChunk] = []
    skipped_by_file: Dict[str, List[str]] = {}

    for pdf in pdfs:
        pages = extract_pages(pdf)
        all_pages.extend(pages)
        all_chunks.extend(parse_articles_from_pages(pages, source_info))

        if pages:
            full_text, page_spans = _build_full_text(pages)
            _, skipped = _make_article_spans(full_text, page_spans)
            if skipped:
                skipped_by_file[pdf.name] = skipped[:30]

    write_jsonl(PROCESSED_DIR / "extracted_pages.jsonl", all_pages)
    write_jsonl(PROCESSED_DIR / "legal_chunks.jsonl", [c.__dict__ for c in all_chunks])

    article_set = sorted(
        {c.article for c in all_chunks if c.article},
        key=lambda x: _article_num_key(x.replace("Điều ", "")),
    )
    clause_count = sum(1 for c in all_chunks if c.clause)
    article_only_count = sum(1 for c in all_chunks if c.article and not c.clause)

    report = {
        "total_pdf_files": len(pdfs),
        "total_pages": len(all_pages),
        "total_chunks": len(all_chunks),
        "total_articles_detected": len(article_set),
        "total_clause_chunks": clause_count,
        "total_article_only_chunks": article_only_count,
        "first_articles": article_set[:20],
        "last_articles": article_set[-20:],
        "pdf_files": [p.name for p in pdfs],
        "skipped_article_like_markers": skipped_by_file,
        "note": (
            "Parser đã lọc footnote/superscript, chuẩn hóa Đ/Ð, giữ tiêu đề Điều nhiều dòng, "
            "và bỏ các tiêu đề Điều giả trong phần trích dẫn luật sửa đổi. Nếu PDF scan lỗi font nặng, "
            "hãy kiểm tra extracted_pages.jsonl ở các trang được báo nghi ngờ."
        ),
    }
    (PROCESSED_DIR / "parser_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = ingest_all()
    print(json.dumps(result, ensure_ascii=False, indent=2))
