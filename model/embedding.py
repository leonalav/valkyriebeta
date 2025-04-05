import torch
import torch.nn as nn
from typing import Optional
import math
import torch.nn.functional as F

class EfficientEmbeddingLayer(nn.Module):
    """Memory-efficient embedding layer with optional factorization"""
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        factorize: bool = False,
        rank: Optional[int] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        if factorize:
            # Use factorized embedding for large vocabulary sizes
            rank = rank or embedding_dim // 4
            self.weight_A = nn.Parameter(torch.empty((num_embeddings, rank)))
            self.weight_B = nn.Parameter(torch.empty((rank, embedding_dim)))
            self.use_factorized = True
            # Initialize factorized weights
            nn.init.normal_(self.weight_A, mean=0.0, std=0.02)
            nn.init.normal_(self.weight_B, mean=0.0, std=0.02)
        else:
            # Regular embedding
            self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
            self.use_factorized = False
            # Initialize regular weights
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
        
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        if self.padding_idx is not None:
            if self.use_factorized:
                with torch.no_grad():
                    self.weight_A[self.padding_idx].fill_(0)
            else:
                with torch.no_grad():
                    self.weight[self.padding_idx].fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_factorized:
            # Compute embedding through matrix factorization
            full_weight = self.weight_A @ self.weight_B
            return F.embedding(
                x, full_weight,
                self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
        else:
            # Regular embedding lookup
            return F.embedding(
                x, self.weight,
                self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )

    def extra_repr(self) -> str:
        return f'num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, factorized={self.use_factorized}'
