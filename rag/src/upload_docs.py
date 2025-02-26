#!/usr/bin/env python3
"""
Document uploader script for the RAG system.
This script allows users to upload documents through the command line.
"""

import argparse
import os
import shutil
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from config import RAW_DATA_DIR
from rag.src.cli import initialize_rag, get_pdf_files
from rag.models.rag_advanced import AdvancedRAG

def validate_file(file_path: str) -> bool:
    """Validate if the file exists and has a supported extension."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File {file_path} does not exist")
        return False
        
    supported_extensions = {'.pdf', '.txt', '.md', '.doc', '.docx'}
    if path.suffix.lower() not in supported_extensions:
        print(f"Error: Unsupported file type {path.suffix}. Supported types: {', '.join(supported_extensions)}")
        return False
    
    return True

def copy_file_to_raw_data(file_path: str) -> Path:
    """Copy the file to the raw data directory."""
    source_path = Path(file_path)
    dest_path = Path(RAW_DATA_DIR) / source_path.name
    
    # Create raw data directory if it doesn't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Copy the file
    shutil.copy2(source_path, dest_path)
    print(f"Copied {source_path.name} to {RAW_DATA_DIR}")
    return dest_path

def main():
    parser = argparse.ArgumentParser(description='Upload documents to the RAG system')
    parser.add_argument('files', nargs='+', help='Path to one or more files to upload')
    parser.add_argument('--rebuild-index', action='store_true', 
                       help='Rebuild the vector store index after uploading')
    
    args = parser.parse_args()
    
    # Validate all files first
    valid_files = []
    for file_path in args.files:
        if validate_file(file_path):
            valid_files.append(file_path)
    
    if not valid_files:
        print("No valid files to upload")
        return
    
    # Copy valid files to raw data directory
    copied_files = []
    for file_path in valid_files:
        try:
            dest_path = copy_file_to_raw_data(file_path)
            copied_files.append(dest_path)
        except Exception as e:
            print(f"Error copying {file_path}: {str(e)}")
    
    if copied_files and args.rebuild_index:
        print("\nRebuilding vector store index...")
        try:
            rag = AdvancedRAG()
            rag.setup_knowledge_base()
            print("Successfully rebuilt vector store index")
        except Exception as e:
            print(f"Error rebuilding index: {str(e)}")
    
    print("\nSummary:")
    print(f"- Files processed: {len(args.files)}")
    print(f"- Files successfully uploaded: {len(copied_files)}")
    print(f"- Failed uploads: {len(args.files) - len(copied_files)}")
    
    if copied_files:
        print("\nUploaded files:")
        for file in copied_files:
            print(f"- {file.name}")

if __name__ == '__main__':
    main()
