import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class ResponseCache:
    def __init__(self, hidden_size: int, max_size: int, threshold: float = 0.9):
        self.hidden_size = hidden_size
        self.max_size = max_size
        self.threshold = threshold
        self.cache = {}
        
    def lookup(self, hidden_states: torch.Tensor) -> Tuple[bool, Optional[torch.Tensor]]:
        cache_key = self._compute_cache_key(hidden_states)
        if cache_key in self.cache:
            return True, self.cache[cache_key]
        return False, None
        
    def update(self, hidden_states: torch.Tensor):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
            
        cache_key = self._compute_cache_key(hidden_states)
        self.cache[cache_key] = hidden_states
        
    def _compute_cache_key(self, hidden_states: torch.Tensor) -> str:
        # Simple hashing function for demonstration
        # In practice, you might want a more sophisticated hashing mechanism
        return str(hidden_states.sum().item())
