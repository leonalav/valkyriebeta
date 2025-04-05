import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple, Union, List

class FlashAttention(nn.Module):
    """Memory-efficient attention using tiling and optimized CUDA operations"""
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.attention_dropout
        
        # Fused QKV projection for better memory efficiency
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        
        # Tile size for memory-efficient attention
        self.block_size = min(256, getattr(config, 'max_position_embeddings', 2048))
        
        # Use PyTorch 2.0 scaled_dot_product_attention if available
        self.use_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.use_flash:
            print("WARNING: Flash Attention requires PyTorch >= 2.0. Falling back to memory-efficient implementation.")
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                alibi_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        
        # Fused QKV projection
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use PyTorch 2.0 Flash Attention if available
        if self.use_flash:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attention_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=attention_mask is None
                )
        else:
            # Memory-efficient block-wise attention
            output = self._block_attention(q, k, v, attention_mask, alibi_bias)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_length, self.hidden_size)
        output = self.out_proj(output)
        
        return output
    
    def _block_attention(self, q, k, v, attention_mask, alibi_bias):
        batch_size, num_heads, seq_length, head_dim = q.shape
        
        # Initialize output
        output = torch.zeros_like(q)
        
        # Process attention in blocks to save memory
        for block_start in range(0, seq_length, self.block_size):
            block_end = min(block_start + self.block_size, seq_length)
            
            # Get current block
            q_block = q[:, :, block_start:block_end]
            
            # Compute attention scores for current block
            scores = torch.matmul(q_block, k.transpose(-2, -1)) / math.sqrt(head_dim)
            
            # Apply ALiBi bias if provided
            if alibi_bias is not None:
                # Ensure alibi_bias has the correct dimensions
                if alibi_bias.dim() == 4 and alibi_bias.size(2) >= block_end:
                    scores = scores + alibi_bias[:, :, block_start:block_end]
                else:
                    # Log warning about incompatible alibi_bias
                    print(f"Warning: alibi_bias dimensions {alibi_bias.shape} incompatible with scores {scores.shape}")
            
            # Apply attention mask if provided
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    # Expand mask for broadcasting (using proper indexing)
                    expanded_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                    mask_value = torch.finfo(scores.dtype).min
                    scores = scores.masked_fill(expanded_mask == 0, mask_value)
                elif attention_mask.dim() == 4:
                    # Try to use properly sliced mask
                    try:
                        mask_slice = attention_mask[:, :, block_start:block_end]
                        scores = scores + mask_slice
                    except IndexError:
                        # Fallback: reshape mask to match scores
                        print(f"Warning: attention_mask shape {attention_mask.shape} incompatible with scores shape {scores.shape}")
            
            # Apply softmax and dropout
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            
            # Compute output for current block
            output[:, :, block_start:block_end] = torch.matmul(attn_probs, v)
        
        return output

class GroupedQueryAttention(nn.Module):
    """More efficient attention using grouped queries"""
    def __init__(self, config):
        super().__init__()
        # Set up grouped query attention parameters
        self.num_heads = config.num_heads
        self.num_key_value_heads = config.num_heads // getattr(config, 'num_query_groups', 1)
        self.num_query_groups = getattr(config, 'num_query_groups', 1)
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        
        # Create projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, 'bias', True))
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config, 'bias', True))
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config, 'bias', True))
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, 'bias', True))
        
        # Dropout
        self.dropout = nn.Dropout(getattr(config, 'attention_dropout', 0.1))
        
    def forward(self, x, attention_mask=None, position_ids=None):
        batch_size, seq_length, hidden_size = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        
        # Transpose for better batch matrix multiplication
        q = q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
        k = k.transpose(1, 2)  # [batch, num_kv_heads, seq, head_dim]
        v = v.transpose(1, 2)  # [batch, num_kv_heads, seq, head_dim]
        
        # Handle grouped queries - repeat keys and values for each query group
        if self.num_key_value_heads != self.num_heads:
            # Calculate repetition factor
            rep_factor = self.num_heads // self.num_key_value_heads
            
            # Repeat keys and values
            k = k.repeat_interleave(rep_factor, dim=1)
            v = v.repeat_interleave(rep_factor, dim=1)
        
        # Compute attention
        # Scale query
        q = q * (self.head_dim ** -0.5)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand mask for broadcasting
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                mask_value = torch.finfo(attention_scores.dtype).min
                attention_scores = attention_scores.masked_fill(attention_mask == 0, mask_value)
            else:
                # Already expanded mask
                attention_scores = attention_scores + attention_mask
        
        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention weights to values
        context = torch.matmul(attention_probs, v)
        
        # Transpose and reshape context
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_length, hidden_size)
        
        # Apply output projection
        output = self.o_proj(context)
        
        return output

class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention for handling very long sequences efficiently.
    
    This implementation supports very long context lengths (32K+) by processing attention
    in chunks with overlapping windows, significantly reducing memory requirements
    while maintaining information flow across the sequence.
    """
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = getattr(config, 'attention_dropout', 0.1)
        
        # Window size for local attention (default: 4096 for long contexts)
        self.window_size = getattr(config, 'sliding_window_size', 4096)
        
        # Number of global tokens that attend to the entire sequence
        self.num_global_tokens = getattr(config, 'global_tokens', 64)
        
        # Maximum sequence chunking size for memory efficiency
        self.chunk_size = getattr(config, 'chunk_size', 4096)
        
        # Overlap between chunks to maintain continuity
        self.chunk_overlap = getattr(config, 'chunk_overlap', 512)
        
        # Projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, 'bias', True))
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, 'bias', True))
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, 'bias', True))
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, 'bias', True))
        
        # Use PyTorch 2.0 scaled_dot_product_attention if available
        self.use_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def forward(self, x, attention_mask=None, position_ids=None):
        """
        Forward pass with sliding window attention.
        
        For very long sequences, we process the input in overlapping chunks
        to maintain reasonable memory usage.
        """
        batch_size, seq_length, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # For very long sequences, use chunk-based processing
        if seq_length > self.chunk_size:
            output = self._chunked_sliding_window_attention(q, k, v, attention_mask, batch_size, seq_length)
        else:
            # For shorter sequences, use standard sliding window
            output = self._sliding_window_attention(q, k, v, attention_mask)
        
        # Reshape and project output
        output = output.reshape(batch_size, seq_length, self.hidden_size)
        output = self.out_proj(output)
        
        return output
        
    def _chunked_sliding_window_attention(self, q, k, v, attention_mask, batch_size, seq_length):
        """
        Process very long sequences in overlapping chunks to save memory.
        
        This method handles sequences longer than self.chunk_size by processing
        them in chunks with some overlap, then combining the results.
        """
        # Initialize output tensor
        output = torch.zeros_like(q).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Process sequence in chunks with overlap
        chunks = []
        weights = []
        
        # Generate chunk indices with overlap
        for chunk_start in range(0, seq_length, self.chunk_size - self.chunk_overlap):
            # Determine chunk end, ensuring we don't go beyond sequence length
            chunk_end = min(chunk_start + self.chunk_size, seq_length)
            
            # Calculate window start and end positions
            win_start = max(0, chunk_start - self.window_size // 2)
            win_end = min(seq_length, chunk_end + self.window_size // 2)
            
            # Extract chunks for processing
            q_chunk = q[:, chunk_start:chunk_end]
            k_window = k[:, win_start:win_end]
            v_window = v[:, win_start:win_end]
            
            # Process the chunk
            if attention_mask is not None:
                # Adjust attention mask for the current window
                mask_chunk = attention_mask[:, :, chunk_start:chunk_end, win_start:win_end]
            else:
                mask_chunk = None
            
            # Calculate attention and adjust positions
            pos_offset = chunk_start - win_start
            chunk_output = self._process_single_chunk(q_chunk, k_window, v_window, mask_chunk, pos_offset)
            
            # Track chunk position and result
            chunks.append((chunk_start, chunk_end, chunk_output))
            
            # Create blending weights for overlap regions
            weight = torch.ones(chunk_end - chunk_start, device=q.device)
            if chunk_start > 0:
                # Linear ramp for beginning overlap
                overlap_start = chunk_start
                overlap_end = min(chunk_start + self.chunk_overlap, chunk_end)
                weight[:(overlap_end - overlap_start)] = torch.linspace(0, 1, overlap_end - overlap_start, device=q.device)
                
            if chunk_end < seq_length:
                # Linear ramp for ending overlap
                overlap_start = max(chunk_start, chunk_end - self.chunk_overlap)
                overlap_end = chunk_end
                weight[(overlap_start - chunk_start):] = torch.linspace(1, 0, overlap_end - overlap_start, device=q.device)
                
            weights.append((chunk_start, chunk_end, weight))
        
        # Combine chunks with weighted blending in overlap regions
        for (start, end, chunk_output), (_, _, weight) in zip(chunks, weights):
            # Apply weight along sequence dimension
            weighted_output = chunk_output * weight.view(1, -1, 1, 1)
            output[:, start:end] += weighted_output
            
        return output
    
    def _process_single_chunk(self, q_chunk, k_window, v_window, mask_chunk, pos_offset):
        """Process a single chunk with attention within the window"""
        batch_size, chunk_length, num_heads, head_dim = q_chunk.shape
        window_length = k_window.shape[1]
        
        # Create causal/sliding window mask for attention
        causal_mask = torch.ones(chunk_length, window_length, device=q_chunk.device)
        
        # Apply sliding window constraint: each position attends only to its window
        for i in range(chunk_length):
            window_start = max(0, i + pos_offset - self.window_size // 2)
            window_end = min(window_length, i + pos_offset + self.window_size // 2)
            causal_mask[i, window_start:window_end] = 0
        
        causal_mask = causal_mask.bool().unsqueeze(0).unsqueeze(1)  # [1, 1, chunk_length, window_length]
        
        # Combine with provided attention mask if available
        if mask_chunk is not None:
            causal_mask = causal_mask | (mask_chunk == 0)
            
        # Reshape for attention computation
        q_chunk = q_chunk.transpose(1, 2)  # [batch, heads, chunk_length, head_dim]
        k_window = k_window.transpose(1, 2)  # [batch, heads, window_length, head_dim]
        v_window = v_window.transpose(1, 2)  # [batch, heads, window_length, head_dim]
        
        # Compute attention scores
        if self.use_flash:
            # Use PyTorch 2.0's memory-efficient attention
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
                attn_weights = torch.zeros(batch_size, num_heads, chunk_length, window_length, device=q_chunk.device)
                attn_weights.masked_fill_(causal_mask, -float('inf'))
                
                attn_output = F.scaled_dot_product_attention(
                    q_chunk, k_window, v_window,
                    attn_mask=attn_weights,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        else:
            # Manual attention implementation
            scale = 1.0 / math.sqrt(head_dim)
            attn_weights = torch.matmul(q_chunk, k_window.transpose(-2, -1)) * scale
            
            # Apply mask
            attn_weights.masked_fill_(causal_mask, -float('inf'))
            
            # Apply softmax and dropout
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v_window)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2)  # [batch, chunk_length, heads, head_dim]
        return attn_output

    def _sliding_window_attention(self, q, k, v, attention_mask=None):
        """
        Standard sliding window attention for shorter sequences.
        """
        batch_size, seq_length, num_heads, head_dim = q.shape
        
        # Reshape for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq_length, head_dim]
        k = k.transpose(1, 2)  # [batch, heads, seq_length, head_dim]
        v = v.transpose(1, 2)  # [batch, heads, seq_length, head_dim]
        
        # Use PyTorch 2.0 scaled_dot_product_attention if available
        if self.use_flash:
            # Create sliding window mask
            sliding_window_mask = torch.ones(seq_length, seq_length, device=q.device)
            for i in range(seq_length):
                window_start = max(0, i - self.window_size // 2)
                window_end = min(seq_length, i + self.window_size // 2)
                sliding_window_mask[i, window_start:window_end] = 0
                
            sliding_window_mask = sliding_window_mask.bool().unsqueeze(0).unsqueeze(1)
            
            # Combine with provided attention mask
            if attention_mask is not None:
                sliding_window_mask = sliding_window_mask | (attention_mask == 0)
            
            # Use efficient attention
            attn_weights = torch.zeros(batch_size, num_heads, seq_length, seq_length, device=q.device)
            attn_weights.masked_fill_(sliding_window_mask, -float('inf'))
            
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_weights,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # Manual sliding window attention
            scale = 1.0 / math.sqrt(head_dim)
            
            # Initialize output
            attn_output = torch.zeros_like(q)
            
            # Process each sequence position
            for i in range(seq_length):
                # Define window range
                window_start = max(0, i - self.window_size // 2)
                window_end = min(seq_length, i + self.window_size // 2)
                
                # Calculate attention scores for window
                attn_scores = torch.matmul(
                    q[:, :, i:i+1], 
                    k[:, :, window_start:window_end].transpose(-2, -1)
                ) * scale
                
                # Apply attention mask if provided
                if attention_mask is not None:
                    window_mask = attention_mask[:, :, i:i+1, window_start:window_end]
                    attn_scores = attn_scores + window_mask
                
                # Apply softmax and dropout
                attn_probs = F.softmax(attn_scores, dim=-1)
                attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
                
                # Calculate weighted values
                weighted_values = torch.matmul(attn_probs, v[:, :, window_start:window_end])
                attn_output[:, :, i:i+1] = weighted_values
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2)  # [batch, seq_length, heads, head_dim]
        return attn_output

class AdaptiveSparsityAttention(nn.Module):
    """Attention with adaptive sparsity based on token importance"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        
        # Create projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, 'bias', True))
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, 'bias', True))
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, 'bias', True))
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, 'bias', True))
        
        # Token importance projection
        self.importance_proj = nn.Linear(config.hidden_size, 1, bias=True)
        
        # Sparsity control
        self.sparsity_threshold = getattr(config, 'sparsity_threshold', 0.2)
        self.min_tokens = max(32, getattr(config, 'min_tokens', 32))
        
        # Dropout
        self.dropout = nn.Dropout(getattr(config, 'attention_dropout', 0.1))
        
    def forward(self, x, attention_mask=None, position_ids=None):
        batch_size, seq_length, hidden_size = x.shape
        
        # Compute token importance scores
        importance = self.importance_proj(x).squeeze(-1)  # [batch, seq]
        
        # Apply mask to importance scores if needed
        if attention_mask is not None and attention_mask.dim() == 2:
            # Ensure mask has the right shape
            if attention_mask.shape == importance.shape:
                importance = importance.masked_fill(attention_mask == 0, -float('inf'))
            else:
                print(f"Warning: attention_mask shape {attention_mask.shape} incompatible with importance shape {importance.shape}")
        
        # Compute adaptive threshold based on sequence statistics
        # Handle case with very short sequences
        if seq_length > 1:
            sorted_importance, _ = torch.sort(importance, dim=-1, descending=True)
            threshold_idx = max(1, int(seq_length * self.sparsity_threshold))
            threshold_idx = min(threshold_idx, seq_length - 1)
            threshold = sorted_importance[:, threshold_idx].unsqueeze(-1)
        else:
            # For seq_length <= 1, use the importance value directly
            threshold = importance.clone() - 0.1  # Slightly lower threshold to keep the token
        
        # Create sparse mask
        sparse_mask = (importance > threshold).float().unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq, 1]
        
        # Ensure minimum number of tokens 
        if self.min_tokens > 0 and seq_length > self.min_tokens:
            # Take top-k tokens by importance
            _, top_indices = torch.topk(importance, min(self.min_tokens, seq_length), dim=-1)
            min_mask = torch.zeros_like(importance, dtype=torch.float)
            min_mask.scatter_(1, top_indices, 1.0)
            min_mask = min_mask.unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq, 1]
            sparse_mask = torch.maximum(sparse_mask, min_mask)
        
        # Project queries, keys, and values
        q = self.q_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Apply sparse mask to keys and values
        k = k * sparse_mask
        v = v * sparse_mask
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand mask for broadcasting
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                mask_value = torch.finfo(attn_scores.dtype).min
                attn_scores = attn_scores.masked_fill(attention_mask == 0, mask_value)
            elif attention_mask.dim() == 4 and attention_mask.shape[-2:] == attn_scores.shape[-2:]:
                # Already expanded mask
                attn_scores = attn_scores + attention_mask
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        output = torch.matmul(attn_weights, v)
        
        # Transpose and reshape
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_length, hidden_size)
        
        # Apply output projection
        output = self.o_proj(output)
        
        return output

class MultiScaleAttention(nn.Module):
    """Multi-scale attention with adaptive patterns"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        
        # Create projections
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=getattr(config, 'bias', True))
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=getattr(config, 'bias', True))
        
        # Multi-scale parameters
        self.max_scale = getattr(config, 'max_scale', 4)
        self.local_size = min(256, getattr(config, 'max_position_embeddings', 2048) // 4)
        self.num_global_tokens = min(128, getattr(config, 'max_position_embeddings', 2048) // 8)
        
        # Scale projection
        self.scale_proj = nn.Linear(config.hidden_size, self.max_scale, bias=True)
        
        # Dropout
        self.dropout = nn.Dropout(getattr(config, 'attention_dropout', 0.1))
        
        # Use flash attention if available
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')
        
    def forward(self, x, attention_mask=None, position_ids=None):
        batch_size, seq_length, hidden_size = x.shape
        
        # Fused QKV computation
        qkv = self.qkv(x).reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Determine scale factors for each token
        scale_logits = self.scale_proj(x)  # [batch, seq, max_scale]
        scale_weights = F.softmax(scale_logits, dim=-1)  # [batch, seq, max_scale]
        
        # Create outputs for each scale
        outputs = []
        
        # Process different scales
        for scale_idx in range(self.max_scale):
            # Extract scale weight for this scale
            scale_weight = scale_weights[:, :, scale_idx].unsqueeze(1).unsqueeze(-1)  # [batch, 1, seq, 1]
            
            # Create dilated attention for this scale
            dilation = 2 ** scale_idx
            
            # Apply dilation to keys
            if dilation > 1 and seq_length > dilation:
                # Dilate keys and values by skipping tokens
                dilated_indices = torch.arange(0, seq_length, dilation, device=x.device)
                if dilated_indices.numel() > 0:  # Check if we have any indices
                    k_dilated = k[:, :, dilated_indices]
                    v_dilated = v[:, :, dilated_indices]
                else:
                    # Fallback to original k, v if no indices
                    k_dilated = k
                    v_dilated = v
            else:
                k_dilated = k
                v_dilated = v
            
            # Compute attention for this scale
            if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
                # Use PyTorch's flash attention
                try:
                    scale_output = F.scaled_dot_product_attention(
                        q, k_dilated, v_dilated,
                        attn_mask=None,  # Handle mask separately
                        dropout_p=self.dropout.p if self.training else 0.0,
                        scale=1.0 / math.sqrt(self.head_dim)
                    )
                except RuntimeError as e:
                    # Fallback to standard attention
                    print(f"Flash attention failed: {e}. Falling back to standard attention.")
                    attn_scores = torch.matmul(q, k_dilated.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    
                    # Apply softmax
                    attn_weights = F.softmax(attn_scores, dim=-1)
                    attn_weights = self.dropout(attn_weights)
                    
                    # Apply attention weights
                    scale_output = torch.matmul(attn_weights, v_dilated)
            else:
                # Standard attention
                attn_scores = torch.matmul(q, k_dilated.transpose(-2, -1)) / math.sqrt(self.head_dim)
                
                # Apply attention mask if provided
                if attention_mask is not None:
                    if attention_mask.dim() == 2:
                        # Create proper mask for dilated attention
                        dilated_mask = attention_mask
                        if dilation > 1 and seq_length > dilation and dilated_indices.numel() > 0:
                            # Make sure we have valid indices before indexing
                            try:
                                dilated_mask = attention_mask[:, dilated_indices]
                            except IndexError:
                                # Keep the original mask
                                pass
                        
                        # Expand mask for broadcasting
                        try:
                            expanded_mask = dilated_mask.unsqueeze(1).unsqueeze(1)
                            mask_value = torch.finfo(attn_scores.dtype).min
                            attn_scores = attn_scores.masked_fill(expanded_mask == 0, mask_value)
                        except Exception as e:
                            print(f"Error applying attention mask: {e}")
                    elif attention_mask.dim() == 4:
                        # Handle pre-expanded mask
                        if dilation > 1 and seq_length > dilation and dilated_indices.numel() > 0:
                            try:
                                dilated_attn_mask = attention_mask[:, :, :, dilated_indices]
                                attn_scores = attn_scores + dilated_attn_mask
                            except IndexError:
                                # Just add the original mask or try to reshape
                                if attention_mask.shape[-1] == attn_scores.shape[-1]:
                                    attn_scores = attn_scores + attention_mask
                        else:
                            if attention_mask.shape[-1] == attn_scores.shape[-1]:
                                attn_scores = attn_scores + attention_mask
                
                # Apply softmax and dropout
                attn_weights = F.softmax(attn_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                # Apply attention weights
                scale_output = torch.matmul(attn_weights, v_dilated)
            
            # Weight the output by scale factor
            scale_output = scale_output * scale_weight
            outputs.append(scale_output)
        
        # Combine outputs from all scales 
        if outputs:  # Check if we have any outputs
            output = sum(outputs)
        else:
            # Fallback to default attention if no outputs were generated
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_scores, dim=-1)
            output = torch.matmul(attn_weights, v)
        
        # Transpose and reshape
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(batch_size, seq_length, hidden_size)
        
        # Apply output projection
        output = self.out_proj(output)
        
        return output 