import torch
import logging
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import threading

class TensorCache:
    """Thread-safe LRU cache for tensors"""
    
    def __init__(self, max_size_gb: float = 4.0):
        self.max_size = max_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.current_size = 0
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def add(self, key: str, tensor: torch.Tensor) -> bool:
        """Add tensor to cache"""
        with self.lock:
            tensor_size = tensor.numel() * tensor.element_size()
            
            # Check if tensor can fit
            if tensor_size > self.max_size:
                return False
                
            # Make space if needed
            while self.current_size + tensor_size > self.max_size and self.cache:
                _, evicted_tensor = self.cache.popitem(last=False)
                self.current_size -= evicted_tensor.numel() * evicted_tensor.element_size()
                
            # Add to cache
            self.cache[key] = tensor.detach().clone()
            self.current_size += tensor_size
            return True
            
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve tensor from cache"""
        with self.lock:
            if key in self.cache:
                tensor = self.cache.pop(key)  # Remove and re-add for LRU
                self.cache[key] = tensor
                return tensor.clone()
            return None
            
    def clear(self):
        """Clear the cache"""
        with self.lock:
            self.cache.clear()
            self.current_size = 0
            
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        with self.lock:
            return {
                'size_gb': self.current_size / 1024**3,
                'max_size_gb': self.max_size / 1024**3,
                'utilization': self.current_size / self.max_size,
                'num_tensors': len(self.cache)
            }
