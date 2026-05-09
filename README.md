# Trợ lý tra cứu Bộ luật Hình sự Việt Nam - AI RAG Core

Demo này là một chatbot RAG về Bộ luật Hình sự Việt Nam. Mục tiêu là để người dùng phổ thông chỉ cần nhập câu hỏi, hệ thống sẽ:

1. Dùng Gemini để hiểu lại câu hỏi đời thường/không dấu.
2. Tạo các truy vấn pháp lý như `trộm cắp tài sản`, `cố ý gây thương tích`, `người dưới 18 tuổi phạm tội`.
3. Tìm điều luật bằng hybrid retrieval: semantic embedding + keyword + metadata.
4. Có thể dùng Gemini để chọn lại nguồn phù hợp nhất.
5. Trả lời một câu trả lời chính, rõ ràng, kèm căn cứ chính.
6. Cho phép xem nguồn luật trong phần mở rộng.

## Cấu trúc

```text
app/
  streamlit_app.py
  ingest.py
  legal_rag/
    query_understanding.py   # Gemini hiểu câu hỏi
    hybrid_retriever.py      # semantic + keyword retrieval
    answer_chain.py          # Gemini trả lời dựa trên nguồn
    reranker.py              # Gemini rerank nguồn
    ingest.py                # extract PDF và tạo legal_chunks.jsonl

data/
  raw/
    135-vbhn-vpqh.pdf        # đặt PDF tại đây
    source_info.json
  processed/
    legal_chunks.jsonl       # sinh ra sau ingest
    parser_report.json       # sinh ra sau ingest
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

## Cấu hình Gemini

Tạo `.env` ở thư mục gốc:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Hoặc tạo `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_key_here"
GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

## Chạy ingest

Đặt PDF vào:

```text
data/raw/135-vbhn-vpqh.pdf
```

Rồi chạy:

```bash
python app/ingest.py
```

## Chạy app

```bash
streamlit run app/streamlit_app.py
```

## Câu hỏi test

```text
an cap nho co bi di tu khong
danh nguoi bi xu ly nhu the nao
17 tuoi pham toi thi sao
cong ty co phai chiu trach nhiem hinh su khong
tu thu co duoc giam nhe khong
```

## Lưu ý

Đây là demo tra cứu thông tin pháp luật, không thay thế tư vấn pháp lý chuyên nghiệp. Hệ thống chỉ trả lời dựa trên văn bản đã nạp vào `data/processed/legal_chunks.jsonl`.
