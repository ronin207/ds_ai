"""
Shared configuration for all RAG implementations.
This module provides paths and settings that are common across different implementations.
"""

from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Common settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector store settings
VECTOR_STORE_DIR = PROCESSED_DATA_DIR / "vector_store"
VECTOR_STORE_DIR.mkdir(exist_ok=True)

def get_pdf_path(filename: str) -> Path:
    """Get the full path for a PDF file in the raw data directory."""
    return RAW_DATA_DIR / filename

def get_vector_store_path(name: str) -> Path:
    """Get the full path for a vector store in the processed data directory."""
    return VECTOR_STORE_DIR / name
