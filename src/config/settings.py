"""Application settings loaded from environment variables.

The defaults are intentionally local-development friendly. Docker Compose overrides
OLLAMA_BASE_URL so the application container can reach the Ollama container.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _get_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


APP_TITLE = os.getenv("APP_TITLE", "Document Intelligence XAi")
APP_SUBTITLE = os.getenv(
    "APP_SUBTITLE",
    "Extração com Tesseract + RAG local com Ollama",
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

OCR_LANG = os.getenv("OCR_LANG", "por+eng")
def _find_tesseract() -> str:
    """Return Tesseract path: env var first, then auto-detect common locations."""
    cmd = os.getenv("TESSERACT_CMD", "").strip()
    if cmd:
        return cmd
    # Auto-detect common Windows installation paths
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\Public\Tesseract-OCR\tesseract.exe",
    ]
    for path in windows_paths:
        if os.path.isfile(path):
            return path
    return ""  # Linux/Mac: pytesseract finds it automatically via PATH

TESSERACT_CMD = _find_tesseract()
OCR_MIN_CONFIDENCE = _get_int("OCR_MIN_CONFIDENCE", 40)
PDF_RENDER_ZOOM = float(os.getenv("PDF_RENDER_ZOOM", "2.0"))

VECTOR_INDEX_DIR = Path(os.getenv("VECTOR_INDEX_DIR", str(BASE_DIR / "storage" / "faiss_index")))
RETRIEVER_K = _get_int("RETRIEVER_K", 30)
ALLOW_DANGEROUS_DESERIALIZATION = _get_bool("ALLOW_DANGEROUS_DESERIALIZATION", False)

ALLOWED_FILE_TYPES = ["pdf", "png", "jpg", "jpeg"]
