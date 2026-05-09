from __future__ import annotations

import json

import streamlit as st

from legal_rag.answer_chain import answer_question
from legal_rag.config import PROCESSED_DIR
from legal_rag.hybrid_retriever import HybridLegalRetriever, source_label

st.set_page_config(
    page_title="Trợ lý tra cứu Bộ luật Hình sự",
    page_icon="⚖️",
    layout="centered",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Đang khởi tạo hệ thống tra cứu...")
def load_retriever() -> HybridLegalRetriever:
    retriever = HybridLegalRetriever()
    retriever.build_embeddings()
    return retriever


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Xin chào! Mình là trợ lý tra cứu Bộ luật Hình sự Việt Nam. "
                    "Bạn có thể đặt câu hỏi bằng tiếng Việt có dấu hoặc không dấu. "
                    "Mình sẽ tra cứu trong dữ liệu luật đã nạp và trả lời ngắn gọn kèm căn cứ khi có."
                ),
            }
        ]


init_state()

st.markdown(
    """
    <h1 style="
        font-size: clamp(26px, 4vw, 40px);
        line-height: 1.2;
        margin-bottom: 0.25rem;
    ">
        ⚖️ Trợ lý Luật Hình sự
    </h1>
    """,
    unsafe_allow_html=True,
)
st.caption("Tra cứu Bộ luật Hình sự Việt Nam")
st.caption(
    "Thông tin chỉ mang tính tham khảo từ văn bản luật đã nạp, không thay thế tư vấn pháp lý chuyên nghiệp."
)

if not (PROCESSED_DIR / "legal_chunks.jsonl").exists():
    st.error("Chưa có dữ liệu đã xử lý. Hãy chạy: python app/ingest.py")
    st.stop()

try:
    retriever = load_retriever()
except Exception as exc:
    st.error(f"Không thể khởi tạo hệ thống tra cứu: {exc}")
    st.stop()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu và tạo câu trả lời..."):
            result = answer_question(
                user_input,
                retriever,
                use_rerank=True,
                top_k=5,
            )

        st.markdown(result.answer)

        if result.sources:
            with st.expander("Xem căn cứ pháp lý được sử dụng", expanded=False):
                for item in result.sources:
                    c = item.chunk
                    st.markdown(f"**{item.rank}. {source_label(c)}**")

                    excerpt = c.get("text", "")
                    if excerpt:
                        st.write(excerpt[:1200] + ("..." if len(excerpt) > 1200 else ""))

                    if c.get("source_url"):
                        st.markdown(f"[Nguồn văn bản]({c['source_url']})")

                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": result.answer})