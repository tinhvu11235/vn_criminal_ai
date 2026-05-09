from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read config from Streamlit secrets first, then environment variables."""
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


def get_gemini_api_key() -> Optional[str]:
    return get_config("GEMINI_API_KEY")


def get_gemini_model() -> str:
    return get_config("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash"


def get_embedding_model() -> str:
    return (
        get_config(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
