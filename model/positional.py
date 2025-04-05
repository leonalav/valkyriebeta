import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    """Memory-efficient rotary position embeddings"""
    def __init__(self,
                 dim: int,
                 max_seq_length: int = 2048,
                 base: int = 10000,
                 device=None):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        
        # Generate and cache position embeddings
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache sin/cos values for fast lookup
        t = torch.arange(max_seq_length, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_length:
            # Compute on the fly if sequence length exceeds cache
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()[None, None, :, :]
            sin = emb.sin()[None, None, :, :]
        else:
            # Use cached values
            cos = self.cos_cached[:, :, :seq_len]
            sin = self.sin_cached[:, :, :seq_len]
            
        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor,
                        k: torch.Tensor,
                        cos: torch.Tensor,
                        sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    # Reshape for broadcasting
    cos = cos[:, :, :q.shape[2]]  # [1, 1, seq_len, dim]
    sin = sin[:, :, :q.shape[2]]  # [1, 1, seq_len, dim]
    
    # Apply rotation using einsum for efficiency
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

class ALiBi(nn.Module):
    """Attention with Linear Biases - Memory efficient implementation"""
    def __init__(self,
                 num_heads: int,
                 max_seq_length: int = 2048,
                 device=None):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        
        # Generate slopes for each head
        def get_slopes(n: int):
            def get_slopes_power_of_2(n: int):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                return [start * 2 ** (i) for i in range(n)]
            
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                slopes_power_2 = get_slopes_power_of_2(closest_power_of_2)
                slopes_remaining = get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
                return slopes_power_2 + slopes_remaining
                
        slopes = torch.tensor(get_slopes(num_heads), device=device)
        self.register_buffer('slopes', slopes)
        
        # Cache position biases
        positions = torch.arange(max_seq_length, device=device)
        bias = positions[None, :].expand(max_seq_length, -1)
        bias = bias.tril()  # Lower triangular part
        bias = -bias * slopes[:, None, None]  # [num_heads, max_seq_len, max_seq_len]
        self.register_buffer('cached_bias', bias)
        
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        if seq_len is None:
            seq_len = x.size(1)
            
        if seq_len <= self.max_seq_length:
            # Use cached bias
            bias = self.cached_bias[:, :seq_len, :seq_len]
        else:
            # Compute bias on the fly
            positions = torch.arange(seq_len, device=x.device)
            bias = positions[None, :].expand(seq_len, -1)
            bias = bias.tril()
            bias = -bias * self.slopes[:, None, None]
            
        return bias

class CompositionalPositionalEncoding(nn.Module):
    """Combines multiple positional encoding methods efficiently"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.rotary = RotaryEmbedding(
            config.hidden_size // config.num_heads,
            max_seq_length=config.max_seq_length
        )
        self.alibi = ALiBi(
            config.num_heads,
            max_seq_length=config.max_seq_length
        )
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_scores: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get rotary embeddings
        cos, sin = self.rotary(hidden_states, hidden_states.size(1))
        
        # Get ALiBi bias
        alibi_bias = self.alibi(hidden_states, hidden_states.size(1))
        
        return cos, sin, alibi_bias 