"""
Command-line interface for the RAG system.
Provides interactive question-answering functionality.
"""

import os
from pathlib import Path
import logging
from typing import List, Optional

from ..models.rag_advanced import AdvancedRAG
from config import RAW_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_pdf_files() -> List[str]:
    """Get all PDF files from the raw directory."""
    raw_dir = Path(RAW_DATA_DIR)
    pdf_files = [f.name for f in raw_dir.glob("*.pdf")]
    
    if not pdf_files:
        raise ValueError("No PDF files found in the raw directory")
    
    logging.info(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
    return pdf_files

def initialize_rag(api_key: Optional[str] = None) -> AdvancedRAG:
    """Initialize the RAG system with available PDF files."""
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    pdf_files = get_pdf_files()
    return AdvancedRAG(pdf_files, api_key)

def main():
    """Main function to run the interactive RAG system."""
    try:
        rag = initialize_rag()
        
        print("\nWelcome to the RAG System!")
        print("You can ask questions about your documents.")
        print("Type 'quit' to exit, 'stats' to see cache statistics.")
        
        while True:
            question = input("\nQuestion: ").strip()
            
            if question.lower() == 'quit':
                break
            elif question.lower() == 'stats':
                stats = rag.cache.get_cache_stats()
                print("\nCache Statistics:")
                print(f"Total Requests: {stats['total_requests']}")
                print(f"Cache Hits: {stats['cache_hits']}")
                print(f"Cache Misses: {stats['cache_misses']}")
                print(f"Hit Rate: {stats['hit_rate']:.2%}")
                print(f"L1 Cache Size: {stats['l1_size']}")
                print(f"L2 Cache Size: {stats['l2_size']}")
                continue
            
            try:
                answer = rag.get_answer_rag_token(question)
                print("\nAnswer:", answer)
            except Exception as e:
                logging.error(f"Error processing question: {str(e)}")
                print(f"\nError: {str(e)}")
    
    except Exception as e:
        logging.error(f"Initialization error: {str(e)}")
        print(f"Failed to initialize the RAG system: {str(e)}")

if __name__ == "__main__":
    main()
