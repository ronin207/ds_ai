"""
Web interface for the RAG system using Streamlit.
"""
import os
import sys
from pathlib import Path
import streamlit as st
from datetime import datetime
import time
import logging
import warnings

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from rag.src.rag_advanced import AdvancedRAG
from config import RAW_DATA_DIR

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='Convert_system_message_to_human will be deprecated!')

# Get Gemini API key from environment
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GEMINI_API_KEY:
    st.error("Please set the GOOGLE_API_KEY environment variable")
    st.stop()

# Initialize RAG system
@st.cache_resource
def init_rag():
    pdf_paths = [str(p) for p in Path(RAW_DATA_DIR).glob("*.pdf")]
    return AdvancedRAG(pdf_paths=pdf_paths, gemini_api_key=GEMINI_API_KEY)

# Page config
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stTextInput, .stTextArea {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("🤖 RAG Document Assistant")
    
    # Initialize session state
    if 'rag' not in st.session_state:
        st.session_state.rag = init_rag()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md', 'doc', 'docx']
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save uploaded file
                save_path = Path(RAW_DATA_DIR) / uploaded_file.name
                os.makedirs(RAW_DATA_DIR, exist_ok=True)
                
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                st.success(f"Uploaded: {uploaded_file.name}")
            
            if st.button("Update Knowledge Base"):
                with st.spinner("Updating knowledge base..."):
                    st.session_state.rag.setup_knowledge_base()
                st.success("Knowledge base updated!")
        
        # Display current documents
        st.header("📚 Current Documents")
        supported_extensions = {'.pdf', '.txt', '.md', '.doc', '.docx'}
        docs = [
            doc for doc in Path(RAW_DATA_DIR).glob("*.*")
            if doc.suffix.lower() in supported_extensions
            and not doc.name.endswith('_state.json')
            and not doc.name.startswith('.')
        ]
        
        if docs:
            for doc in sorted(docs):
                st.text(f"• {doc.name}")
        else:
            st.text("No documents uploaded yet")
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Display chat history
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        
        with st.container():
            st.markdown(
                f"""<div class="chat-message {'user-message' if role == 'user' else 'assistant-message'}">
                    <div><strong>{'You' if role == 'user' else '🤖 Assistant'}</strong></div>
                    <div>{content}</div>
                </div>""",
                unsafe_allow_html=True
            )
    
    # Query input
    query = st.text_area("Ask a question about your documents:", height=100)
    
    if st.button("Send"):
        if not query.strip():
            st.warning("Please enter a question")
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": query
        })
        
        # Get response from RAG
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag.get_answer_rag_token(query)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Use the new rerun method
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logging.error(f"Error processing query: {str(e)}")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        logging.error(f"Application Error: {str(e)}", exc_info=True)
