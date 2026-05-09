from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from legal_rag.config import get_gemini_api_key, get_gemini_model

try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None


@dataclass
class GeminiResult:
    ok: bool
    text: str
    error: Optional[str] = None


def has_gemini_key() -> bool:
    return bool(get_gemini_api_key())


def generate_text(prompt: str, temperature: float = 0.1) -> GeminiResult:
    api_key = get_gemini_api_key()
    if not api_key:
        return GeminiResult(False, "", "GEMINI_API_KEY chưa được cấu hình.")
    if genai is None:
        return GeminiResult(False, "", "Chưa cài google-genai. Hãy chạy pip install -r requirements.txt")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=get_gemini_model(),
            contents=prompt,
            config={"temperature": temperature},
        )
        return GeminiResult(True, (response.text or "").strip())
    except Exception as exc:  # pragma: no cover
        return GeminiResult(False, "", str(exc))
