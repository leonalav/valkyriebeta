import torch
import torch.nn as nn
import math

def precompute_rotary_embeddings(seq_length: int, dim: int, base: int = 10000):
    """Precompute rotary embeddings for efficient reuse"""
    angles = torch.ones(dim) / (base ** (torch.arange(0, dim, 2) / dim))
    pos = torch.arange(seq_length)
    pos_emb = torch.outer(pos, angles).float()
    return torch.cat([pos_emb, pos_emb], dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size // config.num_heads
        self.base = config.rotary_embedding_base
        self.register_buffer("max_seq_len_cached", torch.tensor(0))
        self.register_buffer("cos_cached", torch.empty(0))
        self.register_buffer("sin_cached", torch.empty(0))
        
    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            pos_emb = precompute_rotary_embeddings(seq_len, self.dim, self.base)
            self.cos_cached = pos_emb.cos()
            self.sin_cached = pos_emb.sin()
            
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]
