"""
RAG (Retrieval Augmented Generation) Chatbot implementation.
This chatbot uses a local PDF document as a knowledge base and Gemini Pro for generation.
"""

import os
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache

# Import shared configuration
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    get_pdf_path,
    get_vector_store_path,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL
)

# Load environment variables from root .env file
root_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(root_dir / '.env')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class RAGChatbot:
    def __init__(self, pdf_path: str, gemini_api_key: str):
        """
        Initialize the RAG chatbot with a PDF document and Gemini API key.
        
        Args:
            pdf_path: Path to the PDF document to use as knowledge base
            gemini_api_key: Google Gemini API key
        """
        self.pdf_path = pdf_path
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        # Initialize embeddings and vector store
        self.setup_knowledge_base()

    def setup_knowledge_base(self):
        """Set up the knowledge base from the PDF document."""
        # Load and split document
        loader = PyPDFLoader(str(get_pdf_path(self.pdf_path)))
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(docs)
        
        # Create vector store with persistence
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        store_path = get_vector_store_path("basic_rag")
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(store_path)
        )
        self.vectorstore.persist()
        self.retriever = self.vectorstore.as_retriever()

    def generate_subquestions(self, question: str) -> List[str]:
        """
        Break down the main question into sub-questions.
        
        Args:
            question: The main question to decompose
            
        Returns:
            List of sub-questions
        """
        template = """You are a helpful assistant that generates multiple sub-questions related to an input question.
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation.
        Generate multiple search queries related to: {question}
        Output (3 queries):"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser() | (lambda x: x.split("\n"))
        return chain.invoke({"question": question})

    @staticmethod
    def reciprocal_rank_fusion(results: List[List], k: int = 60) -> List:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.
        
        Args:
            results: List of ranked document lists
            k: Constant in RRF formula
            
        Returns:
            Combined and re-ranked list of documents
        """
        doc_scores: Dict = {}
        
        for doc_list in results:
            for rank, doc in enumerate(doc_list):
                doc_id = doc.page_content
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += 1.0 / (rank + k)

        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_docs]

    def get_answer(self, question: str) -> str:
        """
        Get an answer for the given question using RAG.
        
        Args:
            question: The question to answer
            
        Returns:
            Generated answer
        """
        # Generate sub-questions
        sub_questions = self.generate_subquestions(question)
        
        # Get relevant documents for each sub-question
        doc_lists = [self.retriever.get_relevant_documents(q) for q in sub_questions]
        
        # Combine results using RRF
        combined_docs = self.reciprocal_rank_fusion(doc_lists)
        
        # Create context from combined documents
        context = "\n\n".join([doc.page_content for doc in combined_docs[:3]])
        
        # Final answer template
        template = """Answer the following question based on the provided context.
        If the context doesn't contain relevant information, say so.
        
        Question: {question}
        
        Context:
        {context}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "question": question,
            "context": context
        })

def main():
    """Main function to demonstrate usage."""
    # Replace these with actual values
    pdf_path = "path/to/your/document.pdf"
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    chatbot = RAGChatbot(pdf_path, api_key)
    
    while True:
        question = input("\nAsk me anything (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        print("\nProcessing your question...\n")
        answer = chatbot.get_answer(question)
        print(f"\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
