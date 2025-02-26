from typing import Dict, List, Tuple, Optional, Any
import faiss
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import time

@dataclass
class CacheEntry:
    query: str
    answer: str
    embedding: np.ndarray
    frequency: int = 1
    last_used: datetime = None
    quality_score: float = 1.0
    metadata: Dict[str, Any] = None
    generation_time: float = None  # Time taken to generate the answer
    token_count: int = None  # Number of tokens in the answer
    source_docs: List[str] = None  # Source document references

class EnhancedCacheStore:
    def __init__(self, dimension: int, cache_size: int = 1000):
        self.dimension = dimension
        self.cache_size = cache_size
        
        # L1 (frequent) and L2 (less frequent) cache
        self.l1_index = faiss.IndexFlatL2(dimension)
        self.l2_index = faiss.IndexFlatL2(dimension)
        
        self.l1_cache: Dict[int, CacheEntry] = {}
        self.l2_cache: Dict[int, CacheEntry] = {}
        
        # Enhanced thresholds and metrics
        self.l1_size = cache_size // 3  # Keep top 1/3 in L1
        self.quality_threshold = 0.7
        self.frequency_threshold = 3
        self.time_decay_factor = 0.1  # For time-based score adjustment
        self.cache_hits = 0
        self.cache_misses = 0
        
    def add_to_cache(self, query: str, answer: str, embedding: np.ndarray, 
                     quality_score: float = 1.0, metadata: Dict[str, Any] = None) -> None:
        start_time = time.time()
        query_hash = hash(query)
        
        # Calculate additional metrics
        token_count = len(query.split()) + len(answer.split())  # Simple approximation
        generation_time = time.time() - start_time
        
        entry = CacheEntry(
            query=query,
            answer=answer,
            embedding=embedding,
            last_used=datetime.now(),
            quality_score=quality_score,
            metadata=metadata,
            generation_time=generation_time,
            token_count=token_count,
            source_docs=metadata.get('source_docs') if metadata else None
        )
        
        # Add to L2 cache initially
        self.l2_cache[query_hash] = entry
        self.l2_index.add(embedding.reshape(1, -1))
        
        # Promote to L1 if meets criteria
        if self._should_promote_to_l1(entry):
            self._promote_to_l1(query_hash)
        
        self._maintain_cache_size()
        
    def search_cache(self, query_embedding: np.ndarray, threshold: float = 0.8) -> Optional[Tuple[str, float]]:
        # Apply time-based decay to threshold
        current_time = datetime.now()
        
        # Search L1 cache first
        D1, I1 = self.l1_index.search(query_embedding.reshape(1, -1), 1)
        if D1[0][0] < threshold:
            query_hash = list(self.l1_cache.keys())[I1[0][0]]
            entry = self.l1_cache[query_hash]
            
            # Apply time decay to quality score
            time_diff = (current_time - entry.last_used).total_seconds() / 3600  # Hours
            adjusted_score = entry.quality_score * np.exp(-self.time_decay_factor * time_diff)
            
            if adjusted_score > threshold:
                self._update_stats(query_hash, True)
                self.cache_hits += 1
                return entry.answer, adjusted_score
        
        # Search L2 cache if L1 misses
        D2, I2 = self.l2_index.search(query_embedding.reshape(1, -1), 1)
        if D2[0][0] < threshold:
            query_hash = list(self.l2_cache.keys())[I2[0][0]]
            entry = self.l2_cache[query_hash]
            
            # Apply time decay to quality score
            time_diff = (current_time - entry.last_used).total_seconds() / 3600
            adjusted_score = entry.quality_score * np.exp(-self.time_decay_factor * time_diff)
            
            if adjusted_score > threshold:
                self._update_stats(query_hash, False)
                self.cache_hits += 1
                return entry.answer, adjusted_score
        
        self.cache_misses += 1
        return None
    
    def _should_promote_to_l1(self, entry: CacheEntry) -> bool:
        """Enhanced promotion criteria based on multiple factors."""
        if entry.quality_score <= self.quality_threshold:
            return False
            
        # Calculate composite score
        time_factor = 1.0  # Recent entries get higher score
        if entry.last_used:
            hours_old = (datetime.now() - entry.last_used).total_seconds() / 3600
            time_factor = np.exp(-self.time_decay_factor * hours_old)
        
        speed_factor = 1.0
        if entry.generation_time:
            # Normalize generation time (faster = better)
            speed_factor = 1.0 / (1.0 + entry.generation_time)
        
        composite_score = (
            0.4 * entry.quality_score +
            0.3 * time_factor +
            0.2 * (entry.frequency / self.frequency_threshold) +
            0.1 * speed_factor
        )
        
        return composite_score > 0.8
    
    def _promote_to_l1(self, query_hash: int) -> None:
        if query_hash in self.l2_cache and len(self.l1_cache) < self.l1_size:
            entry = self.l2_cache.pop(query_hash)
            self.l1_cache[query_hash] = entry
            self.l1_index.add(entry.embedding.reshape(1, -1))
    
    def _maintain_cache_size(self) -> None:
        # Remove least frequently used entries if cache is full
        while len(self.l1_cache) + len(self.l2_cache) > self.cache_size:
            if len(self.l2_cache) > 0:
                oldest_hash = min(
                    self.l2_cache.keys(),
                    key=lambda x: (self.l2_cache[x].frequency, self.l2_cache[x].last_used)
                )
                del self.l2_cache[oldest_hash]
    
    def _update_stats(self, query_hash: int, is_l1: bool) -> None:
        cache = self.l1_cache if is_l1 else self.l2_cache
        entry = cache[query_hash]
        entry.frequency += 1
        entry.last_used = datetime.now()
        
        # Consider promotion to L1 if frequently used
        if not is_l1 and entry.frequency >= self.frequency_threshold:
            self._promote_to_l1(query_hash)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'l1_size': len(self.l1_cache),
            'l2_size': len(self.l2_cache)
        }