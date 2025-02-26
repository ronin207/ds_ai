from rag.models.rag_advanced import AdvancedRAG
import os
from pathlib import Path
from config import RAW_DATA_DIR

# Get your Gemini API key from environment variable
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Get all PDF files from the raw data directory
pdf_paths = list(RAW_DATA_DIR.glob("*.pdf"))
if not pdf_paths:
    raise ValueError(f"No PDF files found in {RAW_DATA_DIR}")

print(f"Found {len(pdf_paths)} PDF files: {[p.name for p in pdf_paths]}")

# Initialize RAG with your PDF paths
rag = AdvancedRAG(pdf_paths=pdf_paths, gemini_api_key=gemini_api_key)

# Rebuild the vector store
print("Rebuilding vector store...")
rag.rebuild_vector_store()
print("Done!")
