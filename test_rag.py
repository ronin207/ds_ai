import os
from pathlib import Path
from rag.models.rag_advanced import AdvancedRAG
from config import RAW_DATA_DIR

def test_rag_system():
    # Initialize the RAG system
    pdf_paths = list(RAW_DATA_DIR.glob('*.pdf'))
    gemini_api_key = os.getenv('GOOGLE_API_KEY')
    
    print(f"Found {len(pdf_paths)} PDF files")
    
    rag = AdvancedRAG(pdf_paths=pdf_paths, gemini_api_key=gemini_api_key)
    
    # Test queries
    test_queries = [
        "What are the fundamental concepts of machine learning?",
        "Explain the difference between supervised and unsupervised learning",
        "What is deep learning and how does it work?"
    ]
    
    for query in test_queries:
        print("\n" + "="*50)
        print(f"Query: {query}")
        print("="*50)
        
        # Get answer with fact-checking
        print("\nGenerating verified answer...")
        verified_answer = rag.get_answer_with_verification(query)
        print(verified_answer)
        
        # Show relevant documents for reference
        print("\nRelevant Documents:")
        print("-" * 30)
        docs, scores = rag.get_relevant_docs(query, k=2)
        print(f"Found {len(docs)} relevant documents")
        
        for i, (doc, score) in enumerate(zip(docs, scores)):
            print(f"\nDocument {i+1}:")
            print(f"Similarity: {score:.3f}")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Preview: {doc.page_content[:200]}...")

if __name__ == "__main__":
    test_rag_system()
