#!/usr/bin/env python3
"""
Script to update the vector index for the RAG system.
This can be run independently to refresh the vector store when documents are added or modified.
"""

import os
import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from rag.models.rag_advanced import AdvancedRAG
from config import RAW_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('index_update.log')
    ]
)

def get_document_stats(directory: Path) -> dict:
    """Get statistics about documents in the directory."""
    stats = {
        'total_files': 0,
        'by_type': {},
        'total_size': 0
    }
    
    for file in directory.glob('*.*'):
        if file.is_file() and not file.name.startswith('.'):
            ext = file.suffix.lower()
            stats['total_files'] += 1
            stats['by_type'][ext] = stats['by_type'].get(ext, 0) + 1
            stats['total_size'] += file.stat().st_size
    
    return stats

def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def update_index(args):
    """Update the vector index with current documents."""
    try:
        # Get document statistics before update
        raw_dir = Path(RAW_DATA_DIR)
        before_stats = get_document_stats(raw_dir)
        
        logging.info("Starting vector index update...")
        logging.info(f"Documents found: {before_stats['total_files']}")
        for ext, count in before_stats['by_type'].items():
            logging.info(f"  {ext}: {count} files")
        logging.info(f"Total size: {format_size(before_stats['total_size'])}")
        
        # Initialize RAG with API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Get list of PDF files
        pdf_paths = [str(p) for p in raw_dir.glob("*.pdf")]
        
        # Initialize RAG system
        rag = AdvancedRAG(pdf_paths=pdf_paths, gemini_api_key=api_key)
        
        # Force reindex if specified
        if args.force:
            logging.info("Forcing complete reindex...")
            if Path(rag.vector_store_path).exists():
                import shutil
                shutil.rmtree(rag.vector_store_path)
        
        # Update knowledge base
        rag.setup_knowledge_base()
        
        # Get statistics after update
        after_stats = get_document_stats(raw_dir)
        
        logging.info("\nIndex update completed successfully!")
        logging.info(f"Documents indexed: {after_stats['total_files']}")
        for ext, count in after_stats['by_type'].items():
            logging.info(f"  {ext}: {count} files")
        
        if args.verify:
            logging.info("\nVerifying index...")
            # Try a simple query to verify the index
            test_query = "What topics are covered in these documents?"
            response = rag.get_answer_rag_token(test_query)
            logging.info("Index verification successful!")
        
    except Exception as e:
        logging.error(f"Error updating index: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Update vector index for RAG system')
    parser.add_argument('--force', action='store_true',
                       help='Force complete reindex by removing existing vector store')
    parser.add_argument('--verify', action='store_true',
                       help='Verify index after update by running a test query')
    
    args = parser.parse_args()
    
    try:
        update_index(args)
    except Exception as e:
        logging.error(f"Failed to update index: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
