"""
Attention mechanisms for transformer models.

This module provides various attention implementations:
- Standard multi-head attention
- Enhanced attention with additional capabilities
- Linear attention for improved efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Any

class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention implementation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize multi-head attention.
        
        Args:
            hidden_size: Size of hidden layer
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        # Check if hidden size is divisible by number of heads
        assert self.head_size * num_heads == hidden_size, \
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_size)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, hidden_size]
            key: Key tensor [batch_size, seq_len_k, hidden_size]
            value: Value tensor [batch_size, seq_len_v, hidden_size]
            attention_mask: Mask tensor [batch_size, seq_len_q, seq_len_k]
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape
        
        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_size).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_size)
        
        # Apply output projection
        output = self.o_proj(context)
        
        if output_attentions:
            return output, attn_weights
        else:
            return output, None

class EnhancedAttention(MultiHeadAttention):
    """
    Enhanced attention with additional capabilities.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        use_rotary: bool = False,
        use_gating: bool = False,
        use_kv_cache: bool = False
    ):
        """
        Initialize enhanced attention.
        
        Args:
            hidden_size: Size of hidden layer
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias
            use_rotary: Whether to use rotary positional embeddings
            use_gating: Whether to use gating mechanism
            use_kv_cache: Whether to use key-value cache
        """
        super().__init__(hidden_size, num_heads, dropout, bias)
        
        self.use_rotary = use_rotary
        self.use_gating = use_gating
        self.use_kv_cache = use_kv_cache
        
        # Additional components for enhanced attention
        if use_gating:
            self.gate = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.gate_activation = nn.Sigmoid()

class LinearAttention(nn.Module):
    """
    Linear attention for improved efficiency.
    
    This implementation uses linear attention which has O(N) complexity
    instead of O(N^2) for standard attention.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        feature_dim: int = 16,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize linear attention.
        
        Args:
            hidden_size: Size of hidden layer
            num_heads: Number of attention heads
            feature_dim: Dimension of feature map
            dropout: Dropout probability
            bias: Whether to use bias
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.feature_dim = feature_dim
        
        # Check if hidden size is divisible by number of heads
        assert self.head_size * num_heads == hidden_size, \
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Feature map projections
        self.q_feature = nn.Linear(self.head_size, feature_dim, bias=False)
        self.k_feature = nn.Linear(self.head_size, feature_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for linear attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, hidden_size]
            key: Key tensor [batch_size, seq_len_k, hidden_size]
            value: Value tensor [batch_size, seq_len_v, hidden_size]
            attention_mask: Mask tensor [batch_size, seq_len_q, seq_len_k]
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape
        
        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_size).transpose(1, 2)
        
        # Apply feature map to query and key
        q_prime = torch.relu(self.q_feature(q))
        k_prime = torch.relu(self.k_feature(k))
        
        # Apply mask to key if needed
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(-1)
            k_prime = k_prime * mask
        
        # Linear attention calculation
        kv = torch.einsum("bhnd,bhne->bhde", k_prime, v)
        qkv = torch.einsum("bhmd,bhde->bhme", q_prime, kv)
        
        # Normalize by sum of key features
        if attention_mask is not None:
            k_sum = torch.sum(k_prime, dim=2, keepdim=True)
            qk_sum = torch.einsum("bhmd,bhd->bhm", q_prime, k_sum.squeeze(2))
            qk_sum = qk_sum.unsqueeze(-1)
            output = qkv / (qk_sum + 1e-8)
        else:
            output = qkv / seq_len_k
        
        # Reshape back to original dimensions
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_size)
        
        # Apply output projection
        output = self.o_proj(output)
        
        if output_attentions:
            # Approximation of attention weights for visualization
            if seq_len_k <= 128 and seq_len_q <= 128:
                attn_weights = torch.matmul(q_prime, k_prime.transpose(-1, -2))
                attn_weights = F.softmax(attn_weights, dim=-1)
            else:
                attn_weights = None
            return output, attn_weights
        else:
            return output, None