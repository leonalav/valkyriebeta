"""
Utility module for managing attention-related imports and functionality.
This centralizes imports of optional dependencies like flash-attention.
"""

import torch
import logging

logger = logging.getLogger(__name__)

# Flash Attention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTENTION = True
    logger.info("Flash Attention is available")
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTENTION = False
    logger.warning("Flash Attention is not available, falling back to standard attention")

# Helper function for applying flash attention when available
def apply_flash_attention(q, k, v, dropout_p=0.0, causal=False):
    """
    Apply flash attention if available, otherwise fall back to standard attention.
    
    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_heads, head_dim]
        v: Value tensor [batch, seq_len, num_heads, head_dim]
        dropout_p: Dropout probability
        causal: Whether to apply causal mask
        
    Returns:
        attention_output: Attention output [batch, seq_len, num_heads, head_dim]
    """
    if HAS_FLASH_ATTENTION:
        # Flash attention expects inputs in format [batch, seq, heads, dim]
        return flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
    else:
        # Fall back to standard attention
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Reshape for batched attention
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scale query
        q = q * (head_dim ** -0.5)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        
        # Apply causal mask if needed
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = torch.softmax(attn_weights, dim=-1)
        if dropout_p > 0.0:
            attn_weights = torch.dropout(attn_weights, p=dropout_p, train=True)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2)  # [batch, seq, heads, dim]
        
        return attn_output 