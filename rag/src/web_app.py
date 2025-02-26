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
import re
import threading
from typing import Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from rag.src.cli import initialize_rag, get_pdf_files
from rag.src.watch_docs import DocumentWatcher
from config import RAW_DATA_DIR

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='Convert_system_message_to_human will be deprecated!')

# Get Gemini API key from environment
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GEMINI_API_KEY:
    st.error("Please set the GOOGLE_API_KEY environment variable")
    st.stop()

# Initialize document watcher
doc_watcher: Optional[DocumentWatcher] = None
watcher_thread: Optional[threading.Thread] = None

def start_doc_watcher():
    """Start the document watcher in a background thread."""
    global doc_watcher, watcher_thread
    
    if watcher_thread and watcher_thread.is_alive():
        return  # Already running
    
    doc_watcher = DocumentWatcher(RAW_DATA_DIR)
    
    def watcher_loop():
        try:
            doc_watcher.watch(interval_seconds=30)  # Check every 30 seconds
        except Exception as e:
            logging.error(f"Document watcher error: {e}")
    
    watcher_thread = threading.Thread(target=watcher_loop, daemon=True)
    watcher_thread.start()
    logging.info("Document watcher started")

def show_document_monitor():
    """Show the document monitoring interface."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Document Monitor")
    
    if st.sidebar.button("🔄 Rebuild Vector Store"):
        with st.spinner("Rebuilding vector store..."):
            try:
                rag = st.session_state.rag
                chunks = rag.rebuild_vector_store()
                st.sidebar.success(f"✅ Vector store rebuilt with {chunks} chunks!")
            except Exception as e:
                st.sidebar.error(f"❌ Error rebuilding vector store: {str(e)}")
    
    if doc_watcher:
        # Show current files
        current_files = doc_watcher.get_current_files()
        if current_files:
            st.sidebar.markdown("**📚 Current Documents:**")
            for f in current_files:
                last_modified = datetime.fromtimestamp(f.stat().st_mtime)
                st.sidebar.markdown(
                    f"- {f.name}\n  *Last modified: {last_modified.strftime('%Y-%m-%d %H:%M')}*"
                )
        else:
            st.sidebar.warning("No documents found in watch directory")
        
        # Show recent changes
        if hasattr(doc_watcher, 'recent_changes'):
            changes = doc_watcher.recent_changes
            if any(changes.values()):
                st.sidebar.markdown("**📝 Recent Changes:**")
                for change_type, files in changes.items():
                    if files:
                        files_str = "\n  ".join(f.split('/')[-1] for f in files)
                        st.sidebar.markdown(f"*{change_type.title()}:*\n  {files_str}")

# Initialize RAG system
@st.cache_resource
def init_rag():
    """Initialize RAG system with error handling."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY environment variable not set. Please set it and restart the app.")
        st.stop()
    
    try:
        st.info("🔄 Initializing RAG system...")
        rag = initialize_rag(api_key)
        st.success("✅ RAG system initialized successfully!")
        logging.info("RAG system initialized successfully!")
        return rag
    except Exception as e:
        st.error(f"⚠️ Error initializing RAG system: {str(e)}")
        logging.error(f"Error initializing RAG system: {str(e)}")
        st.stop()

# Page config
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# Update the CSS section with better LaTeX styling
st.markdown("""
    <style>
    /* Base styles */
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    
    /* Enhanced LaTeX styling */
    .katex {
        font-size: 1.2em !important;
        line-height: 1.5 !important;
    }
    
    /* Display math (block equations) */
    .katex-display {
        margin: 1.5rem 0 !important;
        padding: 1.5rem !important;
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        border-left: 4px solid #0366d6;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Inline math */
    .katex-inline {
        padding: 0.2em 0.4em;
        border-radius: 4px;
        background-color: rgba(3, 102, 214, 0.05);
        margin: 0 0.2em;
    }
    
    /* Chat message containers */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .user-message {
        background-color: #f8f9fa;
        border-left: 4px solid #0366d6;
    }
    
    .assistant-message {
        background-color: #ffffff;
        border-left: 4px solid #28a745;
    }
    
    /* Math content within messages */
    .assistant-message .katex-display {
        background: linear-gradient(to right, #f0f7ff, #ffffff);
    }
    </style>
""", unsafe_allow_html=True)

def format_math_response(response: str) -> str:
    """Format response to properly display mathematical expressions with better spacing."""
    # Handle display math (block equations)
    response = re.sub(
        r'\$\$(.*?)\$\$',
        lambda m: f'\n<div class="math-block">\n$$\n{m.group(1).strip()}\n$$\n</div>\n',
        response,
        flags=re.DOTALL
    )
    
    # Handle inline math with proper spacing
    response = re.sub(
        r'([^\$])\$([^\$]+?)\$([^\$])',
        r'\1<span class="math-inline">$\2$</span>\3',
        response
    )
    
    # Clean up extra newlines
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    return response

def display_message(role: str, content: str):
    """Display a chat message with properly formatted math."""
    css_class = "assistant-message" if role == "assistant" else "user-message"
    formatted_content = format_math_response(content)
    
    st.markdown(
        f"""<div class="chat-message {css_class}">
            <div style="margin-bottom: 0.8rem;">
                <strong>{'🤖 Assistant' if role == 'assistant' else '👤 You'}</strong>
            </div>
            <div style="line-height: 1.6;">
                {formatted_content}
            </div>
        </div>""",
        unsafe_allow_html=True
    )

def main():
    """Main function for the Streamlit app."""
    # Start document watcher
    start_doc_watcher()
    
    # Initialize session state
    if 'rag' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag = init_rag()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Show document monitor in sidebar
    show_document_monitor()
    
    st.title("🤖 RAG Document Assistant")
    
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
            display_message(role, content)
    
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
                
                # Handle rate limit messages
                if "Rate limit cooldown" in response:
                    cooldown_time = int(response.split("wait")[1].split("seconds")[0].strip())
                    
                    # Show countdown
                    for remaining in range(cooldown_time, 0, -1):
                        st.warning(f"⏳ Rate limit reached. Please wait {remaining} seconds...")
                        time.sleep(1)
                    
                    # Retry after cooldown
                    with st.spinner("Retrying..."):
                        response = st.session_state.rag.get_answer_rag_token(query)
                
                # Handle other errors
                if response.startswith("Error:"):
                    if "quota" in response.lower() or "429" in response:
                        st.error("🚫 Service is busy. Please try again in a few minutes.")
                    else:
                        st.error(f"❌ {response}")
                else:
                    st.markdown(response)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                
                # Use the new rerun method
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
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
