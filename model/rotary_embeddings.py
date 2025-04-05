import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding implementation.
    
    Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    
    Extended with support for very long contexts through interpolation techniques.
    """
    def __init__(
        self, 
        dim: int, 
        base: int = 10000, 
        max_position_embeddings: int = 32768, 
        interpolation_factor: Optional[float] = None
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.interpolation_factor = interpolation_factor
        
        # Create inv_freq for sinusoidal functions with appropriate dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for efficient forward pass
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        
    def _update_cos_sin_tables(self, x: torch.Tensor, seq_len: int):
        """
        Update cached cos and sin tables for efficiency if needed.
        
        For very long sequence lengths, we use cached computation and interpolation
        to improve efficiency and handle context lengths beyond what the model was
        originally trained on.
        """
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            
            # Check if interpolation is needed for very long sequences
            needs_interpolation = (self.interpolation_factor is not None and 
                                  seq_len > self.max_position_embeddings // self.interpolation_factor)
            
            if needs_interpolation:
                # Use interpolation for extending to very long contexts
                # This creates frequency coefficients that extrapolate beyond the original training length
                base_seq_len = self.max_position_embeddings // self.interpolation_factor
                t_short = torch.arange(base_seq_len, device=x.device).type_as(self.inv_freq)
                freqs_short = torch.einsum("i,j->ij", t_short, self.inv_freq)
                
                # Interpolate to the full sequence length
                t_full = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
                t_full = t_full.float() * (base_seq_len / seq_len)
                
                # Compute interpolated frequencies
                freqs = torch.zeros(seq_len, self.dim // 2, device=x.device).type_as(self.inv_freq)
                for i in range(seq_len):
                    # Find the two closest indices in the short sequence
                    idx_low = int(t_full[i])
                    idx_high = min(idx_low + 1, base_seq_len - 1)
                    weight_high = t_full[i] - idx_low
                    weight_low = 1.0 - weight_high
                    
                    # Interpolate frequencies
                    if idx_low == idx_high:  # Handle edge case
                        freqs[i] = freqs_short[idx_low]
                    else:
                        freqs[i] = weight_low * freqs_short[idx_low] + weight_high * freqs_short[idx_high]
            else:
                # Standard computation for normal/shorter sequences
                t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
                freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            
            # Create cosine and sine embeddings
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]
            
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors.
        
        Args:
            q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim] or [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor of shape [batch_size, seq_len, num_heads, head_dim] or [batch_size, num_heads, seq_len, head_dim]
            seq_len: Optional sequence length for caching efficiency
            
        Returns:
            q_rotated: Query tensor with rotary embeddings applied
            k_rotated: Key tensor with rotary embeddings applied
        """
        # Determine sequence length
        if seq_len is None:
            if q.dim() == 4:
                if q.shape[1] == q.shape[2]:  # Ambiguous case
                    raise ValueError("For ambiguous shapes, seq_len must be provided")
                seq_len = q.shape[1] if q.shape[1] != q.shape[3] else q.shape[2]
            else:
                seq_len = q.shape[1]
                
        # Update cached sin and cos tables if needed
        self._update_cos_sin_tables(q, seq_len)
        
        # Standard format: [batch, seq_len, heads, dim]
        if q.dim() == 4 and q.shape[1] != self.dim:
            # Get the appropriate cos and sin values for sequence length
            cos = self._cos_cached[:, :, :seq_len, :]
            sin = self._sin_cached[:, :, :seq_len, :]
            
            # If input is in [batch, heads, seq_len, dim] format
            if q.shape[2] == seq_len:
                # Ensure cos and sin match input shape
                cos = cos.permute(0, 1, 2, 3)
                sin = sin.permute(0, 1, 2, 3)
                
            # Apply rotary embeddings
            q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin)
            return q_rotated, k_rotated
        
        # Return input tensors if dimensions don't match expected format
        return q, k
    
    @classmethod
    def from_pretrained(cls, pretrained_rotary: "RotaryEmbedding", max_position_embeddings: int = 32768) -> "RotaryEmbedding":
        """
        Create a new RotaryEmbedding with extended context length from a pretrained one.
        
        Args:
            pretrained_rotary: Pretrained rotary embedding module
            max_position_embeddings: New maximum context length
            
        Returns:
            Extended rotary embedding module with interpolation support
        """
        interpolation_factor = max_position_embeddings / pretrained_rotary.max_position_embeddings
        
        new_rotary = cls(
            dim=pretrained_rotary.dim,
            base=pretrained_rotary.base,
            max_position_embeddings=max_position_embeddings,
            interpolation_factor=interpolation_factor
        )
        
        # Copy over the pretrained weights
        new_rotary.inv_freq.copy_(pretrained_rotary.inv_freq)
        
        return new_rotary

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    Args:
        q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim] or [batch_size, num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch_size, seq_len, num_heads, head_dim] or [batch_size, num_heads, seq_len, head_dim]
        cos: Cosine part of rotary embeddings
        sin: Sine part of rotary embeddings
        
    Returns:
        q_rotated: Query tensor with rotary embeddings applied
        k_rotated: Key tensor with rotary embeddings applied
    """
    # Handle both [batch, seq, heads, dim] and [batch, heads, seq, dim] formats
    if q.shape[1] == k.shape[1] == q.shape[2] == k.shape[2]:  # Ambiguous case
        # Check which dimension is likely the sequence dimension based on cos shape
        if cos.shape[2] == q.shape[1]:  # [batch, seq, heads, dim] format
            seq_dim = 1
        else:  # [batch, heads, seq, dim] format
            seq_dim = 2
    else:
        # Determine format based on which dimension matches the sequence length in cos
        seq_dim = 1 if cos.shape[2] == q.shape[1] else 2
    
    # Apply different handling based on determined format
    if seq_dim == 1:  # [batch, seq, heads, dim] format
        return _apply_rotary_pos_emb_bshd(q, k, cos, sin)
    else:  # [batch, heads, seq, dim] format
        return _apply_rotary_pos_emb_bhsd(q, k, cos, sin)

def _apply_rotary_pos_emb_bshd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to inputs in [batch, seq_len, heads, dim] format"""
    # Split features into even and odd dimensions
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    
    # Rotate even and odd dimensions
    q_rotated_even = q_even * cos - q_odd * sin
    q_rotated_odd = q_odd * cos + q_even * sin
    k_rotated_even = k_even * cos - k_odd * sin
    k_rotated_odd = k_odd * cos + k_even * sin
    
    # Interleave the rotated dimensions
    q_rotated = torch.zeros_like(q)
    k_rotated = torch.zeros_like(k)
    
    q_rotated[..., 0::2] = q_rotated_even
    q_rotated[..., 1::2] = q_rotated_odd
    k_rotated[..., 0::2] = k_rotated_even
    k_rotated[..., 1::2] = k_rotated_odd
    
    return q_rotated, k_rotated

def _apply_rotary_pos_emb_bhsd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to inputs in [batch, heads, seq_len, dim] format"""
    # Adjust cos and sin for this format if needed
    if cos.shape[1] != q.shape[1]:
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)
    
    # Split features into even and odd dimensions
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    
    # Rotate even and odd dimensions
    q_rotated_even = q_even * cos - q_odd * sin
    q_rotated_odd = q_odd * cos + q_even * sin
    k_rotated_even = k_even * cos - k_odd * sin
    k_rotated_odd = k_odd * cos + k_even * sin
    
    # Interleave the rotated dimensions
    q_rotated = torch.zeros_like(q)
    k_rotated = torch.zeros_like(k)
    
    q_rotated[..., 0::2] = q_rotated_even
    q_rotated[..., 1::2] = q_rotated_odd
    k_rotated[..., 0::2] = k_rotated_even
    k_rotated[..., 1::2] = k_rotated_odd
    
    return q_rotated, k_rotated
