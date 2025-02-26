"""
Advanced RAG (Retrieval Augmented Generation) implementation with query translation and fusion techniques.
Features:
- Query decomposition and translation
- Multiple retrieval strategies
- Reciprocal Rank Fusion
- Sub-question answering
"""

import os
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
import subprocess
import tempfile
import logging
import re
from .cache import EnhancedCacheStore

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
    RAW_DATA_DIR,
    VECTOR_STORE_DIR
)

# Load environment variables from root .env file
root_dir = Path(__file__).resolve().parent.parent.parent
load_dotenv(root_dir / '.env')

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import (
    PyPDFLoader,
    PDFMinerLoader,
    UnstructuredPDFLoader,
    PDFPlumberLoader
)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tiktoken

@dataclass
class RetrievalResult:
    """Container for retrieval results with metadata."""
    documents: List[Any]
    scores: List[float] = None
    metadata: Dict = None

class RateTracker:
    """Track API call rates and implement cooldown."""
    
    def __init__(self, cooldown_seconds: int = 60):
        self.last_error_time = 0
        self.cooldown_seconds = cooldown_seconds
        self.error_count = 0
        self.max_retries = 3
        self.retry_delay = 5  # seconds
    
    def can_make_request(self) -> bool:
        """Check if enough time has passed since the last error."""
        if self.error_count == 0:
            return True
        
        time_since_error = time.time() - self.last_error_time
        return time_since_error >= self.cooldown_seconds
    
    def record_error(self):
        """Record an error occurrence."""
        self.last_error_time = time.time()
        self.error_count += 1
    
    def record_success(self):
        """Record a successful request."""
        self.error_count = 0
    
    def get_remaining_cooldown(self) -> float:
        """Get remaining cooldown time in seconds."""
        if self.error_count == 0:
            return 0
        
        elapsed = time.time() - self.last_error_time
        remaining = max(0, self.cooldown_seconds - elapsed)
        return remaining

class AdvancedRAG:
    def __init__(self, pdf_paths: List[str], gemini_api_key: str):
        """Initialize the RAG system with PDF paths and API key."""
        if isinstance(pdf_paths, str):
            pdf_paths = [pdf_paths]  # Convert single path to list
        self.pdf_paths = [get_pdf_path(path) for path in pdf_paths]
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        
        # Initialize rate limit tracker
        self.rate_tracker = RateTracker(cooldown_seconds=60)
        
        # Initialize components
        self.setup_llm()
        self.setup_knowledge_base()
        self.setup_prompts()
        self.setup_reranker()
        
        # Initialize cache
        self.embedding_dimension = 384  # dimension for all-MiniLM-L6-v2
        self.cache = EnhancedCacheStore(dimension=self.embedding_dimension)

    def setup_llm(self):
        """Set up the LLM with backup models and retry mechanism."""
        try:
            # Try Gemini first as primary model
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.7,
                max_output_tokens=2048,
                top_p=0.8,
                top_k=40,
                convert_system_message_to_human=True,
                max_retries=5,
                retry_delay=10,
            )
            self.token_llm = self.llm
            logging.info("Successfully initialized Gemini model")
            
        except Exception as e:
            logging.error(f"Failed to initialize Gemini: {str(e)}")
            # Fall back to local model
            try:
                from langchain.llms import LlamaCpp
                self.llm = LlamaCpp(
                    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    temperature=0.7,
                    max_tokens=2000,
                    top_p=0.95,
                    n_ctx=4096,
                    n_gpu_layers=0,  # Use CPU only
                    n_batch=512,
                    verbose=False,
                )
                self.token_llm = self.llm
            except Exception as e2:
                raise Exception(f"Failed to initialize any LLM: {str(e2)}")

    def load_pdf_with_fallback(self, pdf_path: str) -> List[Document]:
        """Load a PDF file using multiple strategies with fallback and validation."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Check if file is empty
        if os.path.getsize(pdf_path) == 0:
            raise ValueError(f"PDF file is empty: {pdf_path}")
            
        # Check if file is actually a PDF (check magic numbers)
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                raise ValueError(f"File is not a valid PDF: {pdf_path}")
        
        errors = []
        loaders = [
            (PyPDFLoader, "PyPDFLoader"),
            (PDFMinerLoader, "PDFMinerLoader"),
            (PDFPlumberLoader, "PDFPlumberLoader"),
            (UnstructuredPDFLoader, "UnstructuredPDFLoader"),
        ]
        
        for loader_class, loader_name in loaders:
            try:
                loader = loader_class(pdf_path)
                documents = loader.load()
                if documents:  # Only return if we actually got some content
                    logging.info(f"Successfully loaded {pdf_path} with {loader_name}")
                    return documents
            except Exception as e:
                errors.append(f"{loader_name} failed: {str(e)}")
                continue
        
        # If we get here, all loaders failed
        error_msg = f"Failed to load PDF {os.path.basename(pdf_path)}. Methods tried:\n"
        error_msg += "\n".join(errors)
        raise ValueError(error_msg)

    def setup_knowledge_base(self):
        """Initialize or update the knowledge base with documents."""
        # Load and process documents
        documents = []
        for pdf_path in self.pdf_paths:
            try:
                docs = self.load_pdf_with_fallback(pdf_path)
                if docs:
                    documents.extend(docs)
            except Exception as e:
                logging.error(f"Error loading {pdf_path}: {str(e)}")
        
        if not documents:
            logging.warning("No documents were successfully loaded")
            return
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
        
        splits = text_splitter.split_documents(documents)
        
        # Initialize embeddings
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Initialize Chroma with basic configuration
        self.vector_store = Chroma(
            collection_name="pdf_collection",
            embedding_function=embedding_function,
            persist_directory=str(VECTOR_STORE_DIR),
        )
        
        # Add documents to the vector store
        self.vector_store.add_documents(splits)
        
        # Create a basic retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        logging.info(f"Knowledge base initialized with {len(splits)} document chunks")

    def rebuild_vector_store(self):
        """Rebuild the vector store from scratch."""
        # Clear existing vector store
        if hasattr(self, 'vector_store'):
            self.vector_store = None
        
        # Reinitialize knowledge base
        self.setup_knowledge_base()
        
        logging.info("Vector store rebuilt successfully")

    def update_knowledge_base_incremental(self, new_pdf_paths: List[str]):
        """
        Update the knowledge base with new documents without reprocessing existing ones.
        
        Args:
            new_pdf_paths: List of paths to new PDF files to add
        """
        new_documents = []
        successful_loads = 0
        failed_loads = 0
        
        for pdf_path in new_pdf_paths:
            try:
                documents = self.load_pdf_with_fallback(pdf_path)
                new_documents.extend(documents)
                successful_loads += 1
                logging.info(f"Successfully loaded new document: {pdf_path}")
            except Exception as e:
                failed_loads += 1
                logging.error(f"Error loading new document {os.path.basename(pdf_path)}: {str(e)}")
                continue
        
        if not new_documents:
            logging.warning("No new documents were successfully loaded.")
            return
        
        logging.info(f"Successfully loaded {successful_loads} new documents. {failed_loads} files failed.")
        
        # Split new documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        new_splits = text_splitter.split_documents(new_documents)
        
        # Add new documents to existing vector store
        vector_store_dir = get_vector_store_path("advanced_rag")
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        if not hasattr(self, 'vector_store'):
            # Initialize vector store if not already done
            self.vector_store = Chroma(
                persist_directory=str(vector_store_dir),
                embedding_function=embedding_function
            )
        
        # Add new documents to vector store
        self.vector_store.add_documents(new_splits)
        logging.info(f"Added {len(new_splits)} new chunks to vector store")
        
        # Update retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5, "fetch_k": 10}
        )
        
        # Update pdf_paths list
        self.pdf_paths.extend([get_pdf_path(path) for path in new_pdf_paths])

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

    def setup_reranker(self):
        """Initialize the cross-encoder reranker."""
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank_documents(self, query: str, docs: List[Document], 
                        scores: List[float], top_k: int = None) -> Tuple[List[Document], List[float]]:
        """
        Rerank documents using a cross-encoder model for more accurate relevance scoring.
        """
        if not docs:
            return [], []
            
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in docs]
        
        # Get cross-encoder scores
        cross_scores = self.reranker.predict(pairs)
        
        # Combine with original scores (weighted average)
        combined_scores = []
        for orig_score, cross_score in zip(scores, cross_scores):
            # Convert distance to similarity (0-1 range)
            orig_sim = 1 / (1 + orig_score)
            # Normalize cross-encoder score (already 0-1)
            cross_sim = (cross_score + 1) / 2
            # Weighted average (favoring cross-encoder)
            combined_score = 0.3 * orig_sim + 0.7 * cross_sim
            combined_scores.append(combined_score)
        
        # Sort by combined scores
        reranked = list(zip(docs, combined_scores))
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k if specified
        if top_k:
            reranked = reranked[:top_k]
        
        # Unzip results
        docs, scores = zip(*reranked)
        return list(docs), list(scores)

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
        sub_question_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that breaks down complex questions into simpler sub-questions."),
            ("human", """Break down the following question into 2-3 specific sub-questions that will help provide a comprehensive answer.
            Focus on different aspects of the main question.
            
            Question: {question}
            
            Output the sub-questions as a numbered list:
            1.
            2.
            3. (optional)""")
        ])
        
        chain = sub_question_prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({"question": question})
            # Parse numbered list into separate questions
            sub_questions = []
            for line in result.split('\n'):
                if line.strip() and line[0].isdigit():
                    sub_questions.append(line.split('.', 1)[1].strip())
            return sub_questions if sub_questions else [question]
        except Exception as e:
            logging.error(f"Failed to generate sub-questions: {str(e)}")
            return [question]

    def reciprocal_rank_fusion(self, results: List[List[Any]], k: int = 60) -> List[Any]:
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
        try:
            # Get documents for each question
            all_docs = []
            for q in questions:
                try:
                    docs = self.retriever.get_relevant_documents(q)
                    all_docs.append(docs)
                except Exception as e:
                    logging.error(f"Error retrieving documents for question '{q}': {str(e)}")
                    continue
            
            if not all_docs:
                logging.warning("No documents retrieved for any questions")
                return []
            
            # Combine and deduplicate results
            combined_docs = []
            seen_content = set()
            
            for docs in all_docs:
                for doc in docs:
                    if doc.page_content not in seen_content:
                        combined_docs.append(doc)
                        seen_content.add(doc.page_content)
            
            return combined_docs[:4]  # Return top 4 unique documents
            
        except Exception as e:
            logging.error(f"Error in retrieve_with_fusion: {str(e)}")
            return []

    def initialize_retriever(self):
        """Initialize the retriever with proper configuration."""
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # Initialize Chroma with basic configuration
        self.vector_store = Chroma(
            collection_name="pdf_collection",
            embedding_function=embedding_function,
            persist_directory=str(VECTOR_STORE_DIR),
        )
        
        # Create a basic retriever without MMR
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Removed fetch_k and other MMR parameters
        )
        
        # Persist the vector store
        self.vector_store.persist()

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
        try:
            # Generate sub-questions
            sub_questions = self.generate_sub_questions(question)
            
            # Get documents using fusion retrieval
            docs = self.retrieve_with_fusion(sub_questions)
            
            if not docs:
                return "I couldn't find any relevant information to answer your question."
            
            # Format context from documents
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # Create the answer chain
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant that provides accurate answers based on the given context."),
                ("human", """Please answer the following question using only the provided context. If you cannot answer the question from the context, say so.

Context:
{context}

Question: {question}""")
            ])
            
            chain = answer_prompt | self.llm | StrOutputParser()
            
            # Generate answer
            return chain.invoke({
                "context": context,
                "question": question
            })
            
        except Exception as e:
            logging.error(f"Error in get_answer: {str(e)}")
            if "429" in str(e) or "quota" in str(e).lower():
                return "The service is currently experiencing high demand. Please try again in a few minutes."
            return f"Error: {str(e)}"

    def get_relevant_docs(self, query: str, k: int = 5, max_distance: float = 0.8, 
                         metadata_filters: Optional[Dict[str, Any]] = None) -> Tuple[List[Document], List[float]]:
        """
        Get relevant documents for a query using similarity search.
        
        Args:
            query: The search query
            k: Number of documents to return
            max_distance: Maximum cosine distance for filtering (lower means more similar)
            metadata_filters: Optional filters for document metadata
            
        Returns:
            Tuple of (documents, similarity_scores)
        """
        try:
            # Get documents with scores
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k*3  # Get more docs initially for filtering and deduplication
            )
            
            if not results:
                return [], []
            
            # Filter by similarity threshold and metadata
            filtered_results = []
            seen_content = set()  # Track unique content
            
            for doc, score in results:
                # Convert score to cosine similarity (0-1 range)
                similarity = 1.0 - score  # Chroma returns distance, convert to similarity
                
                # Check similarity threshold
                if similarity >= (1.0 - max_distance):
                    # Check metadata filters if provided
                    if metadata_filters and not all(doc.metadata.get(k) == v for k, v in metadata_filters.items()):
                        continue
                    
                    # Create a hash of the content to detect duplicates
                    content_hash = hash(doc.page_content.strip())
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        filtered_results.append((doc, similarity))
            
            # Sort by similarity and take top k
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            filtered_results = filtered_results[:k]
            
            # Separate documents and scores
            docs, scores = zip(*filtered_results) if filtered_results else ([], [])
            return list(docs), list(scores)
            
        except Exception as e:
            logging.error(f"Error in get_relevant_docs: {str(e)}")
            return [], []

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

    def format_sources(self, sources: List[Tuple[Document, float]]) -> str:
        """Format sources in a clean, readable way with similarity scores."""
        if not sources:
            return ""
        
        formatted_sources = []
        for doc, score in sources:
            # Extract page number if available
            page_info = f" - Page {doc.metadata.get('page', 'N/A')}" if 'page' in doc.metadata else ""
            
            # Format the source entry
            source_entry = f" {doc.metadata.get('source', 'Unknown')}{page_info}"
            if score is not None:
                source_entry += f" (Relevance: {score:.1%})"
            
            formatted_sources.append(source_entry)
        
        return "\n".join([
            "\n Sources:",
            "─" * 50,
            *formatted_sources,
            "─" * 50
        ])

    def format_inline_citation(self, doc: Document, score: float) -> str:
        """Create an inline citation for a source."""
        page = doc.metadata.get('page', 'N/A')
        source = doc.metadata.get('source', 'Unknown')
        return f"[{source} - P{page} ({score:.1%})]"

    def get_answer_rag_token(self, question: str) -> str:
        """Get answer using RAG with inline citations and formatted sources."""
        try:
            # Get relevant documents
            docs_with_scores = self.get_relevant_docs(question)
            if not docs_with_scores:
                return "I couldn't find any relevant information to answer your question."

            # Create context with inline citations
            context_parts = []
            for doc, score in docs_with_scores:
                citation = self.format_inline_citation(doc, score)
                context_parts.append(f"{doc.page_content} {citation}")
            
            context = "\n\n".join(context_parts)
            
            # Generate answer
            prompt = self.qa_prompt_template.format(
                context=context,
                question=question
            )
            
            response = self.safe_llm_call(self.qa_prompt_template, context=context, question=question)
            
            # Add formatted sources at the end
            sources_section = self.format_sources(docs_with_scores)
            
            return f"{response}\n\n{sources_section}"
            
        except Exception as e:
            logging.error(f"Error in get_answer_rag_token: {str(e)}")
            return f"Error: {str(e)}"

    def expand_query(self, query: str) -> List[str]:
        """
        Expand the query using HyDE (Hypothetical Document Embeddings) technique.
        This generates multiple variations of the query to improve retrieval.
        """
        hyde_prompt = PromptTemplate.from_template("""
        Generate 3 different versions of the following query that capture the same meaning but use different words or phrasings.
        Make the variations more specific and detailed than the original query.
        
        Original query: {query}
        
        Variations (numbered list):
        """)
        
        # Generate query variations
        response = self.llm.invoke(hyde_prompt.format(query=query))
        variations = [query]  # Always include original query
        
        # Extract variations from response content
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
            
        for line in content.split('\n'):
            if line.strip() and any(line.startswith(str(i)) for i in range(1, 4)):
                variation = line.split('.', 1)[1].strip() if '.' in line else line.strip()
                variations.append(variation)
        
        return variations

    def get_expanded_relevant_docs(self, query: str, k: int = 5, max_distance: float = 1.5) -> Tuple[List[Document], List[float]]:
        """
        Get relevant documents using query expansion and tighter similarity threshold.
        A lower max_distance means higher similarity requirement.
        """
        # Generate query variations
        expanded_queries = self.expand_query(query)
        
        all_docs = []
        all_scores = []
        
        # Get docs for each query variation
        for expanded_query in expanded_queries:
            docs, scores = self.get_relevant_docs(
                expanded_query, 
                k=k,
                max_distance=max_distance  # More strict similarity requirement
            )
            all_docs.extend(docs)
            all_scores.extend(scores)
        
        # Deduplicate while preserving order and keeping best scores
        seen_contents = {}
        final_docs = []
        final_scores = []
        
        for doc, score in zip(all_docs, all_scores):
            content_hash = hash(doc.page_content)
            if content_hash not in seen_contents or score < seen_contents[content_hash][1]:
                seen_contents[content_hash] = (doc, score)
        
        # Sort by score and take top k
        results = sorted(seen_contents.values(), key=lambda x: x[1])[:k]
        
        if not results:
            return [], []
        
        final_docs, final_scores = zip(*results)
        return list(final_docs), list(final_scores)

    def condition_context(self, question: str, docs: List[Document]) -> str:
        """
        Condition and structure the context for better answer generation.
        Now includes special handling for code and math expressions.
        """
        # Detect if question is about code or math
        is_code_question = bool(re.search(r'\b(code|function|class|implement|program)\b', question.lower()))
        is_math_question = bool(re.search(r'\b(equation|formula|calculate|solve|math)\b', question.lower()))
        
        structured_context = []
        
        for doc in docs:
            content = doc.page_content
            
            # Handle code blocks
            if is_code_question and doc.metadata.get('has_code'):
                code_blocks = re.finditer(r'```(.*?)```', content, re.DOTALL)
                for block in code_blocks:
                    structured_context.append(f"Code example:\n```\n{block.group(1).strip()}\n```\n")
            
            # Handle math expressions
            if is_math_question and doc.metadata.get('has_math'):
                math_exprs = re.finditer(r'\$(.*?)\$|\\\[(.*?)\\\]|\\begin{equation}(.*?)\\end{equation}', 
                                       content, re.DOTALL)
                for expr in math_exprs:
                    math_content = expr.group(1) or expr.group(2) or expr.group(3)
                    structured_context.append(f"Mathematical expression: {math_content}\n")
            
            # Include regular text
            text_content = re.sub(r'```.*?```|\$.*?\$|\\\[.*?\\\]|\\begin{equation}.*?\\end{equation}', 
                                '', content, flags=re.DOTALL)
            if text_content.strip():
                structured_context.append(text_content.strip())
        
        return "\n\n".join(structured_context)

    def detect_hallucinations(self, answer: str, context: str) -> Tuple[str, List[str]]:
        """
        Detect and mitigate potential hallucinations in the generated answer.
        """
        verification_prompt = PromptTemplate.from_template("""
        Analyze the following answer for potential hallucinations by comparing it with the provided context.
        A hallucination is any statement that cannot be directly supported by the context.
        
        Answer to verify:
        {answer}
        
        Context:
        {context}
        
        Instructions:
        1. List any statements in the answer that are not supported by the context
        2. Identify any numerical claims that don't match the context
        3. Flag any entities (names, places, dates) that aren't mentioned in the context
        4. Note any causal relationships that aren't explicitly stated in the context
        
        Format your response as:
        Hallucinations:
        - [List each unsupported statement]
        
        Cleaned Answer:
        [Provide a revised version of the answer with hallucinations removed]
        """)
        
        # Run hallucination detection
        verification = self.llm.invoke(verification_prompt.format(
            answer=answer,
            context=context
        ))
        verification_text = self.get_content(verification)
        
        # Extract hallucinations and cleaned answer
        parts = verification_text.split("Cleaned Answer:")
        if len(parts) != 2:
            return answer, []
        
        hallucinations_section = parts[0].split("Hallucinations:")[1].strip()
        hallucinations = [h.strip("- ").strip() for h in hallucinations_section.split("\n") if h.strip()]
        cleaned_answer = parts[1].strip()
        
        return cleaned_answer, hallucinations

    def generate_enhanced_answer(self, question: str, context: str) -> str:
        """
        Generate an enhanced answer using structured context and multiple prompts.
        """
        initial_prompt = PromptTemplate.from_template("""
        Based on the following structured context, provide a comprehensive answer to the question.
        If the context doesn't contain enough information, clearly state what is missing.
        
        Question: {question}
        
        Structured Context:
        {context}
        
        Instructions:
        1. Start with a direct answer to the question
        2. Provide supporting evidence from the context
        3. Mention any uncertainties or missing information
        4. Add relevant qualifiers or conditions
        
        Answer:
        """)
        
        initial_response = self.llm.invoke(initial_prompt.format(
            question=question,
            context=context
        ))
        initial_answer = self.get_content(initial_response)
        
        # Detect and mitigate hallucinations
        cleaned_answer, hallucinations = self.detect_hallucinations(initial_answer, context)
        
        # If hallucinations were detected, add a note
        if hallucinations:
            note = "\n\nNote: Some statements were removed or modified due to lack of supporting evidence in the source documents."
            cleaned_answer += note
        
        return cleaned_answer

    def get_answer_rag_token(self, question: str) -> str:
        """
        Get an answer using the enhanced RAG pipeline with:
        - Cache lookup
        - Query expansion
        - Cross-encoder reranking
        - Context conditioning
        - Enhanced answer generation
        - Hallucination detection and mitigation
        """
        try:
            # Check rate limit
            if not self.rate_tracker.can_make_request():
                cooldown = self.rate_tracker.get_remaining_cooldown()
                logging.warning(f"Rate limit cooldown in effect. {cooldown}s remaining")
                # Force use local model if rate limited
                try:
                    from langchain.llms import LlamaCpp
                    local_llm = LlamaCpp(
                        model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                        temperature=0.7,
                        max_tokens=2000,
                        top_p=0.95,
                        n_ctx=4096,
                        n_gpu_layers=0,  # Use CPU only
                        n_batch=512,
                        verbose=False,
                    )
                    return local_llm.invoke(self.qa_prompt_template.format(
                        context="",
                        question=question
                    ))
                except Exception as e:
                    logging.error(f"Failed to use local model fallback: {str(e)}")
                    raise Exception("Rate limited and local model fallback failed")
        
            # Get question embedding for cache lookup
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            question_embedding = embeddings.embed_query(question)
            
            # Try cache lookup first
            cache_result = self.cache.search_cache(
                query_embedding=np.array(question_embedding),
                threshold=0.85  # Higher threshold for stricter cache matching
            )
            
            if cache_result:
                cached_answer, quality_score = cache_result
                if quality_score > 0.8:  # Only use high-quality cache hits
                    return f"{cached_answer}\n\n[Source: Cache hit with {quality_score:.2f} confidence]"
            
            # Use expanded retrieval for cache miss
            docs, scores = self.get_expanded_relevant_docs(question, k=5, max_distance=1.5)
            if not docs:
                return "No relevant documents found within the similarity threshold."
            
            # Rerank documents
            reranked_docs, reranked_scores = self.rerank_documents(question, docs, scores)
            
            # Condition and structure context
            structured_context = self.condition_context(question, reranked_docs)
            
            # Generate answer prompt
            answer_prompt = PromptTemplate.from_template("""
            Based on the following context, provide a clear and concise answer to the question.
            If the context doesn't contain enough information to fully answer the question, say so.
            
            Question: {question}
            
            Context:
            {context}
            
            Instructions:
            1. Answer the question directly and specifically
            2. Use only information from the provided context
            3. If multiple approaches are mentioned, organize them clearly
            4. Include any relevant code examples from the context
            5. If the context doesn't fully answer the question, acknowledge what's missing
            
            Answer:
            """)
            
            # Generate initial answer with rate limiting
            initial_answer = self.safe_llm_call(
                answer_prompt,
                question=question,
                context=structured_context
            )
            
            # Add source references
            references = self.format_references(reranked_docs[:3], reranked_scores[:3])
            response = f"{initial_answer}\n\nSources:\n{references}"
            
            # Cache the successful response
            # Calculate quality score based on reranking scores
            quality_score = 1.0 - min(reranked_scores[:3]) if reranked_scores else 0.7
            self.cache.add_to_cache(
                query=question,
                answer=response,
                embedding=np.array(question_embedding),
                quality_score=quality_score
            )
            
            return response
            
        except Exception as e:
            logging.error(f"Error in get_answer_rag_token: {str(e)}")
            if "429" in str(e) or "quota" in str(e).lower():
                return "The service is currently experiencing high demand. Please try again in a few minutes."
            return f"Error: {str(e)}"

    def safe_llm_call(self, prompt_template: PromptTemplate, **kwargs) -> str:
        """
        Make an LLM call with rate limiting and retries.
        Now includes exponential backoff and quota management.
        """
        if not self.rate_tracker.can_make_request():
            cooldown = self.rate_tracker.get_remaining_cooldown()
            logging.warning(f"Rate limit cooldown in effect. {cooldown}s remaining")
            # Force use local model if rate limited
            try:
                from langchain.llms import LlamaCpp
                local_llm = LlamaCpp(
                    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                    temperature=0.7,
                    max_tokens=2000,
                    top_p=0.95,
                    n_ctx=4096,
                    n_gpu_layers=0,  # Use CPU only
                    n_batch=512,
                    verbose=False,
                )
                return local_llm.invoke(prompt_template.format(**kwargs))
            except Exception as e:
                logging.error(f"Failed to use local model fallback: {str(e)}")
                raise Exception("Rate limited and local model fallback failed")
        
        try:
            response = self.llm.invoke(prompt_template.format(**kwargs))
            self.rate_tracker.record_success()
            return response
        except Exception as e:
            self.rate_tracker.record_error()
            logging.error(f"LLM call failed: {str(e)}")
            raise
    
    def get_content(self, response) -> str:
        """Extract content from LLM response, handling different response types."""
        if hasattr(response, 'content'):
            return response.content
        return str(response)

    def visualize_system(self, output_path: str = "rag_system_diagram.png"):
        """
        Generate and save a visualization of the hybrid RAG-CAG system architecture using Graphviz.
        
        Args:
            output_path: Path to save the visualization PNG file
        """
        try:
            from graphviz import Digraph
            
            # Create graph
            dot = Digraph(comment='Hybrid RAG-CAG System Architecture')
            dot.attr(rankdir='TB')
            
            # Add nodes
            # Input and Processing
            dot.node('query', 'User Query', shape='ellipse', style='filled', fillcolor='lightblue')
            dot.node('subq', 'Sub-Questions\nGenerator', shape='box', style='filled', fillcolor='lightgreen')
            
            # Cache (CAG) components
            dot.node('cache_lookup', 'Cache\nLookup', shape='diamond', style='filled', fillcolor='yellow')
            dot.node('cache_store', 'Cache Store\n(Vector + Quality)', shape='cylinder', style='filled', fillcolor='yellow')
            
            # RAG components
            dot.node('retriever', 'Document\nRetriever', shape='box', style='filled', fillcolor='orange')
            dot.node('vectorstore', 'Vector Store\n(Chroma)', shape='cylinder', style='filled', fillcolor='orange')
            dot.node('reranker', 'Cross-Encoder\nReranker', shape='box', style='filled', fillcolor='pink')
            
            # Answer Generation
            dot.node('context', 'Context\nProcessor', shape='box', style='filled', fillcolor='lightgreen')
            dot.node('answer_gen', 'Answer\nGenerator', shape='box', style='filled', fillcolor='lightgreen')
            dot.node('gemini', 'Primary LLM\n(Gemini-1.5-Pro)', shape='hexagon', style='filled', fillcolor='purple')
            dot.node('mistral', 'Fallback LLM\n(Mistral-7B)', shape='hexagon', style='filled', fillcolor='purple')
            dot.node('final', 'Final Answer', shape='ellipse', style='filled', fillcolor='lightblue')
            
            # Add edges with flow
            # Initial flow
            dot.edge('query', 'subq')
            dot.edge('subq', 'cache_lookup')
            
            # Cache (CAG) flow
            dot.edge('cache_lookup', 'cache_store')
            dot.edge('cache_store', 'final', 'Cache Hit')
            
            # RAG flow (on cache miss)
            dot.edge('cache_lookup', 'retriever', 'Cache Miss')
            dot.edge('retriever', 'vectorstore')
            dot.edge('retriever', 'reranker')
            dot.edge('reranker', 'context')
            dot.edge('context', 'answer_gen')
            
            # LLM integration
            dot.edge('answer_gen', 'gemini')
            dot.edge('answer_gen', 'mistral', 'Fallback')
            dot.edge('answer_gen', 'final')
            
            # Cache update
            dot.edge('final', 'cache_store', 'Update Cache')
            
            # Save visualization
            dot.render(output_path, format='png', cleanup=True)
            print(f"System diagram saved to: {output_path}.png")
            
            return f"{output_path}.png"
            
        except ImportError:
            print("Please install required package: pip install graphviz")
            return None
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
            return None

    def verify_answer_consistency(self, answer: str, question: str, context: str) -> Dict[str, Any]:
        """
        Verify if the generated answer is consistent with the source documents.
        
        Args:
            answer: Generated answer to verify
            question: Original question
            context: Source context used to generate the answer
            
        Returns:
            Dictionary containing verification results
        """
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fact-checking assistant. Your task is to:
1. Verify if the provided answer is consistent with the source context
2. Check if the answer correctly acknowledges what information is or isn't in the context
3. Identify any statements in the answer that aren't supported by the context
4. Note any relevant information from the context that was omitted in the answer

Be specific and cite the relevant parts of the context when possible."""),
            ("human", """Question: {question}

Source Context:
{context}

Generated Answer:
{answer}

Please analyze the answer's consistency with the source context and provide:
1. Consistency Assessment (Is the answer faithful to the context?)
2. Unsupported Claims (Are there statements not backed by the context?)
3. Missing Information (What relevant information from the context was omitted?)
4. Suggested Corrections (If needed)""")
        ])
        
        try:
            chain = verification_prompt | self.llm | StrOutputParser()
            verification_result = chain.invoke({
                "question": question,
                "context": context,
                "answer": answer
            })
            
            return {
                "verified": True,
                "verification_result": verification_result
            }
        except Exception as e:
            logging.error(f"Error in answer verification: {str(e)}")
            return {
                "verified": False,
                "error": str(e)
            }

    def get_answer_with_verification(self, question: str) -> str:
        """
        Get an answer and verify its consistency with sources.
        
        Args:
            question: The question to answer
            
        Returns:
            Verified answer with consistency check
        """
        try:
            # Get documents using fusion retrieval
            docs = self.retrieve_with_fusion([question])
            
            if not docs:
                return "I couldn't find any relevant information to answer your question."
            
            # Format context from documents
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # Generate initial answer
            answer_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant that provides accurate answers based on the given context."),
                ("human", """Please answer the following question using only the provided context. If you cannot answer the question from the context, say so.

Context:
{context}

Question: {question}""")
            ])
            
            chain = answer_prompt | self.llm | StrOutputParser()
            initial_answer = chain.invoke({
                "context": context,
                "question": question
            })
            
            # Verify answer consistency
            verification = self.verify_answer_consistency(initial_answer, question, context)
            
            if verification["verified"]:
                response = f"""Answer: {initial_answer}

Fact-Check Results:
{verification['verification_result']}

Sources:
{self.format_references(docs)}"""
            else:
                response = f"""Answer: {initial_answer}

Note: Unable to perform fact-checking due to error: {verification.get('error')}

Sources:
{self.format_references(docs)}"""
            
            return response
            
        except Exception as e:
            logging.error(f"Error in get_answer_with_verification: {str(e)}")
            if "429" in str(e) or "quota" in str(e).lower():
                return "The service is currently experiencing high demand. Please try again in a few minutes."
            return f"Error: {str(e)}"
