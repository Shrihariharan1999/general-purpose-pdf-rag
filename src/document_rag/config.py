
"""Central configuration and project paths."""

import os
from pathlib import Path

from dotenv import load_dotenv


# Resolve the project root from src/document_rag/config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load local environment variables without overriding existing variables.
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with a safe default."""
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    """Read a floating-point environment variable."""
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be a number."
        ) from exc


# Input, processed data, and index directories.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCLING_OUTPUT_DIR = PROCESSED_DIR / "docling"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"

# Images used for source-page previews and extracted figures.
FIGURES_DIR = PROJECT_ROOT / "artifacts" / "figures"

# Existing locally downloaded Docling and Hugging Face model directories.
DOCLING_MODEL_DIR = PROJECT_ROOT / "models" / "docling"
HUGGINGFACE_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface"

# OCR is enabled by default for general-purpose PDF ingestion.
# Set DOCLING_ENABLE_OCR=false in .env for known text-based PDFs.
DOCLING_ENABLE_OCR = _get_bool("DOCLING_ENABLE_OCR", True)

# Lower resolution reduces processing time and storage for page previews.
PAGE_IMAGE_SCALE = _get_float("PAGE_IMAGE_SCALE", 0.5)

# The initial implementation prioritizes speed when reconstructing tables.
DOCLING_TABLE_MODE = os.getenv("DOCLING_TABLE_MODE", "FAST").upper()

# Local LLM API configuration, read only when needed by later modules.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash")

# Cross-encoder reranking runs on hybrid-retrieval candidates.
RERANKER_ENABLED = _get_bool("RERANKER_ENABLED", True)
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANKER_CANDIDATES = int(os.getenv("RERANKER_CANDIDATES", "30"))

# Create project output directories when the module is imported.
for directory in (
    RAW_DATA_DIR,
    DOCLING_OUTPUT_DIR,
    CHUNKS_DIR,
    INDEX_DIR,
    FIGURES_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
