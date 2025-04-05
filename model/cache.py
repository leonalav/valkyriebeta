import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

class ResponseCache(nn.Module):
    """Efficient response cache with LRU eviction and semantic similarity search"""
    def __init__(
        self,
        cache_size: int = 1000,
        hidden_size: int = 2048,
        similarity_threshold: float = 0.8,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.cache_size = cache_size
        self.hidden_size = hidden_size
        self.similarity_threshold = similarity_threshold
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize cache as OrderedDict for LRU functionality
        self.cache: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        
        # Projection for semantic similarity
        self.key_projection = nn.Linear(hidden_size, hidden_size // 4).to(self.device)
        
        # Statistics
        self.hits = 0
        self.misses = 0
        
    def compute_cache_key(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute semantic key for cache lookup"""
        # Project to lower dimension
        projected = self.key_projection(hidden_states)
        
        # Normalize for cosine similarity
        return F.normalize(projected.mean(dim=1), dim=-1)
        
    def find_similar_key(self, query_key: torch.Tensor) -> Optional[str]:
        """Find most similar cached key using cosine similarity"""
        if not self.cache:
            return None
            
        # Stack all cache keys
        cache_keys = torch.stack([
            self._string_to_tensor(k) for k in self.cache.keys()
        ])
        
        # Compute similarities
        similarities = torch.matmul(query_key, cache_keys.t())
        
        # Find most similar key
        max_sim, idx = similarities.max(dim=-1)
        if max_sim > self.similarity_threshold:
            return list(self.cache.keys())[idx.item()]
        return None
        
    def _tensor_to_string(self, tensor: torch.Tensor) -> str:
        """Convert tensor to string representation"""
        return tensor.cpu().numpy().tobytes().hex()
        
    def _string_to_tensor(self, key: str) -> torch.Tensor:
        """Convert string back to tensor"""
        array = np.frombuffer(bytes.fromhex(key), dtype=np.float32)
        return torch.from_numpy(array).to(self.device)
        
    def get(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """Try to retrieve response from cache"""
        query_key = self.compute_cache_key(hidden_states)
        cache_key = self.find_similar_key(query_key)
        
        if cache_key is not None:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key][1]  # Return cached output
        
        self.misses += 1
        return None
        
    def update(self, hidden_states: torch.Tensor, output: torch.Tensor):
        """Update cache with new response"""
        hidden_states = hidden_states.to(self.device)
        output = output.to(self.device)
        cache_key = self._tensor_to_string(self.compute_cache_key(hidden_states))
        
        # Add to cache
        self.cache[cache_key] = (hidden_states.detach(), output.detach())
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
            
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'memory_usage': sum(
                sum(x.nelement() * x.element_size() for x in tensors)
                for tensors in self.cache.values()
            ) / 1024**2  # MB
        }
        
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0