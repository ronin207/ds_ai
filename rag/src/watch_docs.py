#!/usr/bin/env python3
"""
Documentation watcher script for the RAG system.
This script monitors the raw data directory for changes and updates the knowledge base accordingly.
"""

import sys
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, Set
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from config import RAW_DATA_DIR
from rag.src.rag_advanced import AdvancedRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('doc_watcher.log')
    ]
)

class DocumentWatcher:
    def __init__(self, watch_dir: str, state_file: str = 'doc_watcher_state.json'):
        self.watch_dir = Path(watch_dir)
        self.state_file = Path(self.watch_dir) / state_file
        self.file_states: Dict[str, str] = {}
        self.supported_extensions = {'.pdf', '.txt', '.md', '.doc', '.docx'}
        self.load_state()
        
    def load_state(self):
        """Load the previous state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    self.file_states = json.load(f)
                logging.info(f"Loaded previous state with {len(self.file_states)} files")
            except Exception as e:
                logging.error(f"Error loading state file: {e}")
                self.file_states = {}
    
    def save_state(self):
        """Save the current state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.file_states, f, indent=2)
            logging.info("Saved current state")
        except Exception as e:
            logging.error(f"Error saving state file: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file."""
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logging.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def get_current_files(self) -> Set[Path]:
        """Get all supported files in the watch directory."""
        return {
            f for f in self.watch_dir.glob('**/*')
            if f.is_file() and f.suffix.lower() in self.supported_extensions
        }
    
    def check_for_changes(self) -> bool:
        """Check for any changes in the documentation files."""
        current_files = self.get_current_files()
        changes_detected = False
        
        # Check for new or modified files
        for file_path in current_files:
            rel_path = str(file_path.relative_to(self.watch_dir))
            current_hash = self.get_file_hash(file_path)
            
            if rel_path not in self.file_states:
                logging.info(f"New file detected: {rel_path}")
                changes_detected = True
            elif self.file_states[rel_path] != current_hash:
                logging.info(f"Modified file detected: {rel_path}")
                changes_detected = True
            
            self.file_states[rel_path] = current_hash
        
        # Check for deleted files
        stored_files = set(self.file_states.keys())
        current_rel_paths = {str(f.relative_to(self.watch_dir)) for f in current_files}
        deleted_files = stored_files - current_rel_paths
        
        if deleted_files:
            logging.info(f"Deleted files detected: {deleted_files}")
            for file in deleted_files:
                del self.file_states[file]
            changes_detected = True
        
        return changes_detected
    
    def update_knowledge_base(self):
        """Update the RAG system's knowledge base."""
        try:
            rag = AdvancedRAG()
            rag.setup_knowledge_base()
            logging.info("Successfully updated knowledge base")
        except Exception as e:
            logging.error(f"Error updating knowledge base: {e}")
    
    def watch(self, interval_seconds: int = 300):
        """
        Watch for changes in documentation and update knowledge base when needed.
        
        Args:
            interval_seconds: How often to check for changes (default: 5 minutes)
        """
        logging.info(f"Starting document watcher for directory: {self.watch_dir}")
        logging.info(f"Check interval: {interval_seconds} seconds")
        
        try:
            while True:
                if self.check_for_changes():
                    logging.info("Changes detected, updating knowledge base...")
                    self.update_knowledge_base()
                    self.save_state()
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logging.info("Stopping document watcher...")
            self.save_state()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Watch documentation directory for changes')
    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval in seconds (default: 300)')
    parser.add_argument('--dir', type=str, default=RAW_DATA_DIR,
                       help='Directory to watch (default: raw data directory)')
    
    args = parser.parse_args()
    
    watcher = DocumentWatcher(args.dir)
    watcher.watch(args.interval)

if __name__ == '__main__':
    main()
