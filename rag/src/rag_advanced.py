"""
Advanced RAG (Retrieval Augmented Generation) implementation with query translation and fusion techniques.
Features:
- Query decomposition and translation
- Multiple retrieval strategies
- Reciprocal Rank Fusion
- Sub-question answering
"""

import os
from typing import List, Dict, Tuple, Any
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
import subprocess
import tempfile

# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import shared configuration
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    get_pdf_path,
    get_vector_store_path,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    RAW_DATA_DIR
)

# Load environment variables from root .env file
root_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(root_dir / '.env')

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import (
    PyPDFLoader,
    PDFMinerLoader,
    UnstructuredPDFLoader,
    PDFPlumberLoader
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tiktoken

@dataclass
class RetrievalResult:
    """Container for retrieval results with metadata."""
    documents: List[Any]
    scores: List[float] = None
    metadata: Dict = None

class RateLimitTracker:
    def __init__(self, cooldown_minutes=1):
        self.last_error_time = None
        self.cooldown_minutes = cooldown_minutes
        
    def record_error(self):
        self.last_error_time = datetime.now()
        
    def get_remaining_cooldown(self):
        if not self.last_error_time:
            return 0
        
        elapsed = datetime.now() - self.last_error_time
        remaining_seconds = max(0, self.cooldown_minutes * 60 - elapsed.total_seconds())
        return remaining_seconds
    
    def can_retry(self):
        return self.get_remaining_cooldown() == 0

class AdvancedRAG:
    def __init__(self, pdf_paths: List[str], gemini_api_key: str):
        """Initialize the RAG system with PDF paths and API key."""
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]  # Convert single path to list
        self.pdf_paths = [get_pdf_path(path) for path in pdf_paths]
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        # Initialize rate limit tracker
        self.rate_tracker = RateLimitTracker(cooldown_minutes=1)
        
        # Initialize LLM
        self.setup_llm()
        
        # Initialize embeddings and knowledge base
        self.setup_knowledge_base()
        
        # Initialize prompts
        self.setup_prompts()

    def setup_llm(self):
        """Set up the language model with appropriate configuration."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_output_tokens=2048,
            top_p=0.8,
            top_k=40,
            convert_system_message_to_human=True
        )
        
        # Initialize token LLM for RAG-Token approach
        self.token_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.7,
            max_output_tokens=2048,
            top_p=0.8,
            top_k=40,
            convert_system_message_to_human=True
        )

    def load_pdf_with_fallback(self, pdf_path: Path) -> List[Any]:
        """
        Try multiple methods to load a PDF file.
        Returns a list of documents or raises an exception if all methods fail.
        """
        methods_tried = []
        
        # Method 1: Try PyPDFLoader (fastest)
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from {pdf_path.name} using PyPDFLoader")
            return documents
        except Exception as e:
            methods_tried.append(f"PyPDFLoader failed: {str(e)}")
        
        # Method 2: Try PDFMiner
        try:
            loader = PDFMinerLoader(str(pdf_path))
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from {pdf_path.name} using PDFMinerLoader")
            return documents
        except Exception as e:
            methods_tried.append(f"PDFMinerLoader failed: {str(e)}")
            
        # Method 3: Try PDFPlumber
        try:
            loader = PDFPlumberLoader(str(pdf_path))
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from {pdf_path.name} using PDFPlumberLoader")
            return documents
        except Exception as e:
            methods_tried.append(f"PDFPlumberLoader failed: {str(e)}")
        
        # Method 4: Try Unstructured
        try:
            loader = UnstructuredPDFLoader(str(pdf_path))
            documents = loader.load()
            print(f"Loaded {len(documents)} pages from {pdf_path.name} using UnstructuredPDFLoader")
            return documents
        except Exception as e:
            methods_tried.append(f"UnstructuredPDFLoader failed: {str(e)}")
        
        # Method 5: Try pdftotext command line tool
        try:
            # Check if pdftotext is available
            subprocess.run(['which', 'pdftotext'], check=True, capture_output=True)
            
            # Create temporary file for text output
            with tempfile.NamedTemporaryFile(suffix='.txt') as temp_txt:
                # Convert PDF to text
                subprocess.run(['pdftotext', str(pdf_path), temp_txt.name], check=True)
                
                # Read the text file
                with open(temp_txt.name, 'r') as f:
                    text = f.read()
                
                # Split into pages (assuming form feeds separate pages)
                pages = text.split('\f')
                documents = []
                
                for i, page_text in enumerate(pages):
                    if page_text.strip():  # Skip empty pages
                        documents.append(Document(
                            page_content=page_text,
                            metadata={"source": pdf_path.name, "page": i}
                        ))
                
                print(f"Loaded {len(documents)} pages from {pdf_path.name} using pdftotext")
                return documents
        except Exception as e:
            methods_tried.append(f"pdftotext failed: {str(e)}")
        
        # If all methods failed, raise an exception with details
        raise ValueError(f"Failed to load PDF {pdf_path.name}. Methods tried:\n" + "\n".join(methods_tried))

    def setup_knowledge_base(self):
        """Initialize the vector store and document retriever."""
        all_documents = []
        
        # Load all documents
        for pdf_path in self.pdf_paths:
            try:
                documents = self.load_pdf_with_fallback(pdf_path)
                # Add source information to metadata
                for doc in documents:
                    doc.metadata["source"] = pdf_path.name
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading {pdf_path.name}: {str(e)}")
        
        if not all_documents:
            raise ValueError("No documents were successfully loaded")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(all_documents)
        print(f"Split into {len(texts)} chunks")
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Create or load vector store
        vector_store_dir = get_vector_store_path("advanced_rag")
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=str(vector_store_dir)
        )
        
        # Initialize retriever with search kwargs
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5, "fetch_k": 10}  # Fetch more candidates for better diversity
        )

    def setup_prompts(self):
        """Initialize the prompts for different stages of RAG."""
        self.qa_prompt = ChatPromptTemplate.from_template(
            """Answer the following question based on the provided context.
            If you cannot find the answer in the context, say so.
            
            Context: {context}
            Question: {question}
            
            Answer:"""
        )
        
        self.enhance_prompt = ChatPromptTemplate.from_template(
            """Based on the following context, provide a detailed answer to the question.
            Include specific details and references from the context.
            
            Context: {context}
            Question: {question}
            
            Detailed Answer:"""
        )
        
        self.synthesis_prompt = ChatPromptTemplate.from_template(
            """Based on the following detailed information, provide a clear and concise final answer.
            
            Information: {context}
            Question: {question}
            
            Final Answer:"""
        )

    @staticmethod
    def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
        """Count the number of tokens in a string."""
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def generate_sub_questions(self, question: str) -> List[str]:
        """Generate sub-questions from the main question."""
        chain = self.qa_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": question})
        return [q.strip() for q in result.split('\n') if q.strip()]

    @staticmethod
    def reciprocal_rank_fusion(results: List[List[Any]], k: int = 60) -> List[Any]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion.
        
        Args:
            results: List of ranked document lists
            k: Constant in RRF formula
            
        Returns:
            Combined and re-ranked list of documents
        """
        doc_scores: Dict[str, float] = {}
        
        for doc_list in results:
            for rank, doc in enumerate(doc_list):
                doc_id = doc.page_content
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += 1.0 / (rank + k)

        # Sort documents by score
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs]

    def retrieve_with_fusion(self, questions: List[str]) -> List[Any]:
        """
        Retrieve documents using multiple questions and combine results using RRF.
        
        Args:
            questions: List of questions to use for retrieval
            
        Returns:
            Combined list of retrieved documents
        """
        # Get documents for each question
        all_docs = [self.retriever.get_relevant_documents(q) for q in questions]
        
        # Combine using RRF
        return self.reciprocal_rank_fusion(all_docs)

    def answer_sub_question(self, question: str, context: List[Any]) -> str:
        """
        Answer a single sub-question using the provided context.
        
        Args:
            question: Question to answer
            context: List of relevant documents
            
        Returns:
            Generated answer
        """
        context_str = "\n\n".join(doc.page_content for doc in context[:3])
        chain = self.qa_prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "context": context_str,
            "question": question
        })

    def format_qa_pairs(self, questions: List[str], answers: List[str]) -> str:
        """Format question-answer pairs for the synthesis prompt."""
        pairs = []
        for q, a in zip(questions, answers):
            pairs.append(f"Q: {q}\nA: {a}")
        return "\n\n".join(pairs)

    def get_answer(self, question: str) -> str:
        """
        Get an answer for the given question using the advanced RAG pipeline.
        
        Args:
            question: The question to answer
            
        Returns:
            Generated answer
        """
        # Generate sub-questions
        sub_questions = self.generate_sub_questions(question)
        
        # Retrieve documents using fusion
        docs = self.retrieve_with_fusion(sub_questions)
        
        # Answer each sub-question
        sub_answers = []
        for sub_q in sub_questions:
            answer = self.answer_sub_question(sub_q, docs)
            sub_answers.append(answer)
        
        # Format QA pairs
        qa_pairs = self.format_qa_pairs(sub_questions, sub_answers)
        
        # Generate final answer
        chain = self.synthesis_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "original_question": question,
            "qa_pairs": qa_pairs
        })

    def get_relevant_docs(self, query: str, k: int = 5, max_distance: float = 2.0) -> Tuple[List[Any], List[float]]:
        """
        Get relevant documents with distance-based filtering and deduplication.
        Lower distance means higher similarity.
        """
        # Get more documents than needed to allow for filtering and ensure diversity
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k*4)
        
        # Filter by distance
        filtered = [(doc, score) for doc, score in docs_and_scores if score <= max_distance]
        
        # Group by source to ensure diversity
        docs_by_source = {}
        for doc, score in filtered:
            source = doc.metadata['source']
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append((doc, score))
        
        # Sort each source's documents by score
        for source in docs_by_source:
            docs_by_source[source].sort(key=lambda x: x[1])
        
        # Take top documents from each source in round-robin fashion
        final_docs = []
        while len(final_docs) < k and any(docs for docs in docs_by_source.values()):
            for source in list(docs_by_source.keys()):
                if docs_by_source[source]:
                    final_docs.append(docs_by_source[source].pop(0))
                else:
                    del docs_by_source[source]
                if len(final_docs) >= k:
                    break
        
        # Sort final selection by score
        final_docs.sort(key=lambda x: x[1])
        
        if not final_docs:
            return [], []
        
        # Deduplicate based on content similarity
        deduped_docs = []
        seen_contents = set()
        for doc, score in final_docs:
            # Create a simplified version of the content for comparison
            content = ' '.join(doc.page_content.split())[:100]  # First 100 chars after whitespace normalization
            if content not in seen_contents:
                seen_contents.add(content)
                deduped_docs.append((doc, score))
        
        if not deduped_docs:
            return [], []
            
        return zip(*deduped_docs)

    def format_references(self, docs: List[Any], scores: List[float] = None) -> str:
        """
        Format reference documents with their scores.
        Scores from Chroma are actually L2 distances, smaller means more similar.
        We need to convert them to cosine similarity (0-1 range).
        """
        if not scores:
            return "\n\n".join(f"[{doc.metadata['source']} - Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}" 
                             for doc in docs)
        
        references = []
        seen_contents = set()  # For additional deduplication
        
        for doc, score in zip(docs, scores):
            # Skip if we've seen very similar content
            content = ' '.join(doc.page_content.split())[:100]
            if content in seen_contents:
                continue
            seen_contents.add(content)
            
            source = doc.metadata['source']
            page = doc.metadata.get('page', 'N/A')
            # Convert L2 distance to similarity score (0-1 range)
            # Using 1 / (1 + distance) to convert distance to similarity
            similarity = 1 / (1 + score)
            similarity_pct = f"{similarity:.2%}"
            ref = f"[{source} - Page {page} (Similarity: {similarity_pct})]\n{doc.page_content}"
            references.append((ref, similarity))
        
        # Sort references by similarity score in descending order
        references.sort(key=lambda x: x[1], reverse=True)
        return "\n\n".join(ref for ref, _ in references)

    def get_answer_rag_token(self, question: str) -> str:
        """
        Get an answer using the RAG-Token approach.
        This method retrieves different documents for each generated token.
        """
        # Check if we're still in cooldown
        cooldown = self.rate_tracker.get_remaining_cooldown()
        if cooldown > 0:
            minutes = int(cooldown // 60)
            seconds = int(cooldown % 60)
            print(f"\nRate limit cooldown in progress. Please wait {minutes} minutes and {seconds} seconds before trying again.\n")
            return
            
        try:
            # Retrieve initial context with scores
            docs, scores = self.get_relevant_docs(question, k=5, max_distance=2.0)
            if not docs:
                return "No relevant documents found within the similarity threshold."
                
            context = self.format_references(docs, scores)
            
            print("\nRetrieved Documents (with similarity scores):")
            print("=" * 50)
            print(context)
            print("=" * 50 + "\n")
            
            # Create the chain for initial answer
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.enhance_prompt
                | self.token_llm
                | StrOutputParser()
            )
            
            try:
                # Get initial detailed answer
                detailed_answer = chain.invoke({
                    "context": context,
                    "question": question
                })
                
                # Get additional context for key points in the detailed answer
                add_docs, add_scores = self.get_relevant_docs(detailed_answer, k=3, max_distance=2.0)
                
                if add_docs:
                    additional_context = self.format_references(add_docs, add_scores)
                    print("\nAdditional Supporting Documents:")
                    print("=" * 50)
                    print(additional_context)
                    print("=" * 50 + "\n")
                    detailed_answer += f"\n\nAdditional Supporting References:\n{additional_context}"
                
                # Create synthesis chain for final answer
                synthesis_chain = (
                    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                    | self.synthesis_prompt
                    | self.token_llm
                    | StrOutputParser()
                )
                
                # Get final synthesized answer
                final_answer = synthesis_chain.invoke({
                    "context": detailed_answer,
                    "question": question
                })
                
                # Format final output
                output = f"""
                Final Answer:
                {final_answer}

                Note: Documents are shown with their similarity scores, which range from 0% (completely different) to 100% (exact match).
                The scores are calculated based on the semantic similarity between your question and the document content.
                """
                return output
                
            except Exception as e:
                if "429" in str(e) or "Resource has been exhausted" in str(e):
                    self.rate_tracker.record_error()
                    cooldown = self.rate_tracker.get_remaining_cooldown()
                    minutes = int(cooldown // 60)
                    seconds = int(cooldown % 60)
                    print(f"\nRate limit exceeded. Please wait {minutes} minutes and {seconds} seconds before trying again.\n")
                    return
                print(f"\nError enhancing response: {str(e)}\n")
                return
                
        except Exception as e:
            if "429" in str(e) or "Resource has been exhausted" in str(e):
                self.rate_tracker.record_error()
                cooldown = self.rate_tracker.get_remaining_cooldown()
                minutes = int(cooldown // 60)
                seconds = int(cooldown % 60)
                print(f"\nRate limit exceeded. Please wait {minutes} minutes and {seconds} seconds before trying again.\n")
                return
            print(f"\nError processing question: {str(e)}\n")
            return

def main():
    """Main function to demonstrate usage."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    # Get all PDF files from the raw directory
    raw_dir = Path(RAW_DATA_DIR)
    pdf_files = [f.name for f in raw_dir.glob("*.pdf")]
    
    if not pdf_files:
        raise ValueError("No PDF files found in the raw directory")
        
    print(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
    
    # Initialize RAG with all PDF files
    rag = AdvancedRAG(pdf_files, api_key)
    
    while True:
        question = input("\nAsk me anything about the documents (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        print("\nProcessing your question using RAG-Token method...\n")
        answer = rag.get_answer_rag_token(question)
        if answer:
            print(answer)

if __name__ == "__main__":
    main()
