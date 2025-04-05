import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any, List


class RWKVTimeFirst(nn.Module):
    """
    RWKV Time-mixing module - processes time dimension first.
    
    This module combines the benefits of both transformers and RNNs, allowing for efficient
    parallelized attention-like computation with linear complexity and good context tracking.
    """
    
    def __init__(self, config, layer_id):
        """
        Initialize the RWKV time-mixing module.
        
        Args:
            config: Model configuration
            layer_id: Layer identifier
        """
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = getattr(config, 'max_seq_len', 2048)
        self.n_embd = getattr(config, 'hidden_size', 768)
        
        # Get RWKV-specific parameters
        self.time_mix_ratio = getattr(config, 'rwkv_time_mix_ratio', 1.0)
        self.use_linear_att = getattr(config, 'rwkv_use_linear_att', True)
        self.att_scale = getattr(config, 'rwkv_att_scale', 1.0)
        
        # Layer parameters
        self.time_decay = nn.Parameter(torch.ones(self.n_embd))
        self.time_first = nn.Parameter(torch.ones(self.n_embd))
        
        # Time mixing parameters
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, self.n_embd))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, self.n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, self.n_embd))
        
        # Projections
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.output = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        # Layer normalization
        self.ln_x = nn.LayerNorm(self.n_embd)
        
        # Initialize time-decay related parameters
        self._init_time_parameters()
    
    def _init_time_parameters(self):
        """Initialize time-related parameters for effective context processing"""
        # Calculate initial decay rates based on layer position
        with torch.no_grad():
            # Varying decay rates for different positions in context
            max_decay = 0.7 + 0.3 * (self.layer_id / 8.0)
            min_decay = 0.1 + 0.2 * (self.layer_id / 8.0)
            decay_range = max_decay - min_decay
            
            # Set time decay parameters with more nuanced init
            decay_init = torch.linspace(min_decay, max_decay, self.n_embd)
            self.time_decay.data = torch.log(decay_init / (1 - decay_init))
            
            # Initialize time-mix parameters
            time_mix_k_range = 0.6 + 0.3 * (self.layer_id / 8.0)
            time_mix_v_range = 0.3 + 0.4 * (self.layer_id / 8.0)
            time_mix_r_range = 0.5 + 0.3 * (self.layer_id / 8.0)
            
            self.time_mix_k.data = torch.rand(1, 1, self.n_embd) * 0.5 * time_mix_k_range
            self.time_mix_v.data = torch.rand(1, 1, self.n_embd) * time_mix_v_range
            self.time_mix_r.data = torch.rand(1, 1, self.n_embd) * time_mix_r_range
    
    def forward(self, x, state=None):
        """
        Forward pass through the RWKV time-mixing layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            state: Optional state from previous forward pass
            
        Returns:
            x: Output tensor of shape [batch_size, seq_len, hidden_size]
            state: Optional state to pass to the next call
        """
        # Apply layer normalization
        x = self.ln_x(x)
        B, T, C = x.size()  # batch, time, channels
        
        # Use previous state if provided
        if state is not None and T == 1:
            last_x = state
        else:
            # Shift input for time-mixing
            last_x = F.pad(x[:, :-1], (0, 0, 1, 0))
        
        # Time-mixing
        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        v = self.value(x * self.time_mix_v + last_x * (1 - self.time_mix_v))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        
        # Apply time-weighting
        k = k * self.att_scale
        
        # RWKV attention mechanism with time-decay
        wkv = self._time_mixing_mechanism(k, v, T)
        
        # Apply receptance (gating mechanism)
        r = torch.sigmoid(r)
        
        # Apply output projection
        out = self.output(r * wkv)
        
        # Return current x as state for next token
        if T > 1:
            new_state = x[:, -1:]
        else:
            new_state = x
        
        return out, new_state
    
    def _time_mixing_mechanism(self, k, v, T):
        """Time-mixing mechanism with linear attention"""
        # Get time decay factor
        time_decay = torch.exp(self.time_decay)
        time_first = torch.exp(self.time_first)
        
        # Efficient implementation for sequence processing
        if self.use_linear_att:
            # Linear attention implementation (O(n) complexity)
            k = torch.exp(k)
            
            # Initialize states for time-mixing
            wkv = torch.zeros_like(k)
            denominator = torch.zeros(k.shape[0], 1, k.shape[2], device=k.device)
            
            # Initial state
            numerator = torch.zeros(k.shape[0], 1, k.shape[2], device=k.device)
            
            # Process sequence step by step
            for t in range(T):
                # Get decayed attention scores
                # Update with current token
                denominator = denominator * time_decay + k[:, t:t+1]
                numerator = numerator * time_decay + k[:, t:t+1] * v[:, t:t+1]
                
                # Calculate attention at this timestep
                wkv[:, t:t+1] = numerator / (denominator + 1e-8)
        else:
            # Manually unrolled implementation for better numerical stability
            wkv = torch.zeros_like(v)
            
            # Process sequence with explicit time-decay
            for t in range(T):
                for u in range(0, t+1):
                    # Calculate time-decayed attention
                    decay_factor = time_decay ** (t - u)
                    attn_score = k[:, u] * decay_factor
                    wkv[:, t] = wkv[:, t] + attn_score * v[:, u]
                
                # Rescale with first token bonus
                if t == 0:
                    wkv[:, t] = wkv[:, t] + time_first * v[:, t]
        
        return wkv


class RWKVChannelMixer(nn.Module):
    """RWKV Channel mixing module - processes features after time mixing"""
    
    def __init__(self, config, layer_id):
        """
        Initialize RWKV channel mixer.
        
        Args:
            config: Model configuration
            layer_id: Layer identifier
        """
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = getattr(config, 'hidden_size', 768)
        
        # Hidden layer is larger for better feature transformation
        hidden_ratio = getattr(config, 'rwkv_ffn_hidden_ratio', 4.0)
        self.hidden_size = int(self.n_embd * hidden_ratio)
        
        # Channel mixing parameters
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, self.n_embd))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, self.n_embd))
        
        # Channel mixer projections
        self.key = nn.Linear(self.n_embd, self.hidden_size, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.hidden_size, self.n_embd, bias=False)
        
        # Layer normalization
        self.ln_x = nn.LayerNorm(self.n_embd)
        
        # Initialize time parameters
        self._init_time_parameters()
    
    def _init_time_parameters(self):
        """Initialize time-related parameters for channel mixing"""
        with torch.no_grad():
            # Initialize time-mix parameters based on layer position
            layer_weight = 0.3 + (0.5 * self.layer_id / 12.0)
            
            self.time_mix_k.data = torch.ones(1, 1, self.n_embd) * layer_weight
            self.time_mix_r.data = torch.ones(1, 1, self.n_embd) * (1.0 - layer_weight)
    
    def forward(self, x, state=None):
        """
        Forward pass through the channel mixer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            state: Optional state from previous forward pass (unused here)
            
        Returns:
            x: Output tensor after channel mixing
            state: None (channel mixer doesn't need to pass state)
        """
        # Apply layer normalization
        x = self.ln_x(x)
        B, T, C = x.size()  # batch, time, channels
        
        # Use previous state for time mixing
        if state is not None and T == 1:
            last_x = state
        else:
            # Shift input for time-mixing
            last_x = F.pad(x[:, :-1], (0, 0, 1, 0))
        
        # Time-mixing for FFN
        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        
        # Apply channel mixing with gating (similar to GeLU + gating)
        k = torch.square(torch.relu(k))  # Square ReLU - faster than GeLU but similar effect
        
        # Apply receptance gating
        r = torch.sigmoid(r)
        
        # Apply value transformation and gating
        out = r * self.value(k)
        
        # Return current x as state for next token
        if T > 1:
            new_state = x[:, -1:]
        else:
            new_state = x
        
        return out, new_state


class RWKVBlock(nn.Module):
    """RWKV Block - combines time mixing and channel mixing"""
    
    def __init__(self, config, layer_id):
        """
        Initialize RWKV block with time-mixing and channel-mixing.
        
        Args:
            config: Model configuration
            layer_id: Layer identifier
        """
        super().__init__()
        self.layer_id = layer_id
        
        # Get layer-specific configuration if available
        if hasattr(config, 'get_rwkv_layer_config'):
            layer_config = config.get_rwkv_layer_config(layer_id)
        else:
            layer_config = config
        
        # Configuration parameters
        self.residual_scale = getattr(layer_config, 'residual_scale', 1.0)
        self.layer_time_mix_params = {
            'time_mix_ratio': getattr(layer_config, 'rwkv_time_mix_ratio', 1.0),
            'use_linear_att': getattr(layer_config, 'rwkv_use_linear_att', True),
            'att_scale': getattr(layer_config, 'rwkv_att_scale', 1.0),
        }
        
        # Mixing components
        self.att = RWKVTimeFirst(
            config, 
            layer_id,
            head_qk_dim=getattr(config, 'rwkv_head_qk_dim', 0)
        )
        
        self.ffn = RWKVChannelMixer(
            config,
            layer_id
        )
        
        # Gradient checkpointing for memory efficiency
        self.use_checkpoint = getattr(config, 'gradient_checkpointing', False)
    
    def forward(self, x, state=None):
        """
        Forward pass through RWKV block.
        
        Args:
            x: Input tensor
            state: Optional state from previous forward pass
            
        Returns:
            x: Output tensor
            new_state: State to pass to the next call
        """
        # Split state if provided
        if state is not None:
            state_att, state_ffn = state
        else:
            state_att, state_ffn = None, None
        
        # Residual connection for time-mixing
        if self.use_checkpoint and x.requires_grad:
            # Use gradient checkpointing to save memory during training
            import torch.utils.checkpoint
            att_out, att_state = torch.utils.checkpoint.checkpoint(
                self.att, x, state_att)
        else:
            att_out, att_state = self.att(x, state_att)
        
        x = x + att_out * self.residual_scale
        
        # Residual connection for channel-mixing
        if self.use_checkpoint and x.requires_grad:
            ffn_out, ffn_state = torch.utils.checkpoint.checkpoint(
                self.ffn, x, state_ffn)
        else:
            ffn_out, ffn_state = self.ffn(x, state_ffn)
        
        x = x + ffn_out * self.residual_scale
        
        # Combine states for next token
        new_state = (att_state, ffn_state)
        
        return x, new_state


class RWKVModel(nn.Module):
    """RWKV Model implementation"""
    
    def __init__(self, config):
        """
        Initialize RWKV model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Configuration parameters
        self.vocab_size = getattr(config, 'vocab_size', 50257)
        self.n_embd = getattr(config, 'hidden_size', 768)
        self.n_layer = getattr(config, 'num_layers', 12)
        self.max_seq_len = getattr(config, 'max_seq_len', 2048)
        
        # Embeddings
        self.emb = nn.Embedding(self.vocab_size, self.n_embd)
        
        # RWKV Blocks
        self.blocks = nn.ModuleList()
        rwkv_layer_indices = getattr(config, 'rwkv_layer_indices', list(range(config.num_layers)))
        
        for i in range(self.n_layer):
            if i in rwkv_layer_indices:
                self.blocks.append(RWKVBlock(config, i))
            else:
                self.blocks.append(None)  # Placeholder for transformer blocks
        
        # Final layer norm and head
        self.ln_out = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        
        # Chunking for efficient processing
        self.chunk_size = getattr(config, 'rwkv_chunk_size', 1024)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = 0.02
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids=None, attention_mask=None, past_key_values=None,
                labels=None, use_cache=False, return_dict=True):
        """
        Forward pass through the RWKV model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (unused in RWKV but kept for API compatibility)
            past_key_values: Optional states from previous passes
            labels: Optional labels for computing loss
            use_cache: Whether to return states for future use
            return_dict: Whether to return a dict or tuple
            
        Returns:
            Dict or tuple containing loss, logits, and states
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        x = self.emb(input_ids)
        
        # Initialize states
        states = past_key_values if past_key_values is not None else {}
        new_states = {} if use_cache else None
        
        # Process through RWKV blocks
        for i, block in enumerate(self.blocks):
            if block is not None:  # RWKV block
                # Get state for this layer if available
                state = states.get(f"layer_{i}", None)
                
                # Apply block
                x, state_out = block(x, state)
                
                # Store state if using cache
                if use_cache:
                    new_states[f"layer_{i}"] = state_out
        
        # Final layer norm
        x = self.ln_out(x)
        
        # Compute logits
        logits = self.head(x)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, self.vocab_size), shift_labels.reshape(-1))
        
        # Return appropriate format
        if return_dict:
            return type('RWKVOutput', (), {'loss': loss, 'logits': logits, 'past_key_values': new_states})()
        else:
            return (loss, logits, new_states)
    
    def chunked_forward(self, input_ids, past_key_values=None, labels=None):
        """
        Process long sequences by chunking for memory efficiency.
        
        Args:
            input_ids: Input token IDs
            past_key_values: Optional states from previous passes
            labels: Optional labels for computing loss
            
        Returns:
            Dict containing loss, logits, and states
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        # Process in chunks to avoid OOM
        chunk_size = min(self.chunk_size, seq_len)
        
        # Outputs to accumulate
        all_logits = []
        total_loss = 0.0
        
        # Initialize states
        states = past_key_values if past_key_values is not None else None
        
        # Process sequence chunk by chunk
        for i in range(0, seq_len, chunk_size):
            # Get chunk bounds
            chunk_start = i
            chunk_end = min(i + chunk_size, seq_len)
            
            # Get chunk inputs and labels
            chunk_input_ids = input_ids[:, chunk_start:chunk_end]
            
            # Prepare chunk labels if provided
            chunk_labels = None
            if labels is not None:
                chunk_labels = labels[:, chunk_start:chunk_end]
            
            # Forward pass for this chunk
            outputs = self.forward(
                input_ids=chunk_input_ids,
                past_key_values=states,
                labels=chunk_labels,
                use_cache=True,
                return_dict=True
            )
            
            # Update states for next chunk
            states = outputs.past_key_values
            
            # Accumulate outputs
            all_logits.append(outputs.logits)
            if outputs.loss is not None:
                total_loss += outputs.loss.item() * (chunk_end - chunk_start)
        
        # Combine results
        logits = torch.cat(all_logits, dim=1)
        loss = total_loss / seq_len if labels is not None else None
        
        return type('RWKVOutput', (), {'loss': loss, 'logits': logits, 'past_key_values': states})()


class TransformerBlock(nn.Module):
    """Transformer block with sparse attention for hybrid RWKV-Transformer architecture"""
    
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = getattr(config, 'hidden_size', 768)
        self.num_attention_heads = getattr(config, 'num_attention_heads', 12)
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Layer normalization
        self.ln_1 = nn.LayerNorm(self.hidden_size)
        self.ln_2 = nn.LayerNorm(self.hidden_size)
        
        # Grouped-Query Attention (GQA) settings
        self.use_gqa = getattr(config, 'use_grouped_query_attention', False)
        self.num_kv_groups = getattr(config, 'num_kv_groups', 8) if self.use_gqa else self.num_attention_heads
        
        # Sparse attention with windowing
        self.window_size = getattr(config, 'transformer_window_size', 512)
        self.use_flash_attn = getattr(config, 'use_flash_attention', True)
        
        # Attention mechanism - implementing sparse attention
        if self.use_flash_attn and is_flash_attn_available():
            from flash_attn import flash_attn_qkvpacked_func
            self.flash_attn_func = flash_attn_qkvpacked_func
            # Flash attention with GQA support
            self.qkv_proj = nn.Linear(self.hidden_size, self.head_dim * (self.num_attention_heads + 2 * self.num_kv_groups))
            self.use_flash = True
        else:
            # Regular attention with GQA support
            self.query = nn.Linear(self.hidden_size, self.hidden_size)
            self.key = nn.Linear(self.hidden_size, self.hidden_size * self.num_kv_groups // self.num_attention_heads)
            self.value = nn.Linear(self.hidden_size, self.hidden_size * self.num_kv_groups // self.num_attention_heads)
            self.use_flash = False
        
        self.attn_dropout = nn.Dropout(getattr(config, 'attention_dropout', 0.1))
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.proj_dropout = nn.Dropout(getattr(config, 'hidden_dropout_prob', 0.1))
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, 4 * self.hidden_size),
            nn.GELU(),
            nn.Linear(4 * self.hidden_size, self.hidden_size),
            nn.Dropout(getattr(config, 'hidden_dropout_prob', 0.1))
        )
        
        # Gradient checkpointing for memory efficiency
        self.use_checkpoint = getattr(config, 'use_gradient_checkpointing', False)
    
    def _sparse_attention(self, q, k, v, attention_mask=None):
        """Compute sparse attention with local windowing"""
        batch_size, seq_len, _ = q.shape
        
        # For very long sequences, use windowed attention
        if seq_len > self.window_size:
            # Implement sliding window attention
            output = self._windowed_attention(q, k, v, self.window_size, attention_mask)
        else:
            # Regular attention for shorter sequences
            scaling = float(self.head_dim) ** -0.5
            q = q * scaling
            
            # Attention scores
            attn = torch.matmul(q, k.transpose(-2, -1))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn = attn + attention_mask
            
            # Softmax and dropout
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            
            # Apply attention to values
            output = torch.matmul(attn, v)
        
        return output
    
    def _windowed_attention(self, q, k, v, window_size, attention_mask=None):
        """Compute attention with a sliding window for long sequences"""
        batch_size, seq_len, _ = q.shape
        
        # Create output tensor
        output = torch.zeros_like(q)
        
        # Process sequence in windows with overlap
        for i in range(0, seq_len, window_size // 2):
            # Define window range with proper boundary handling
            start_idx = max(0, i - window_size // 4)
            end_idx = min(seq_len, i + window_size - window_size // 4)
            
            # Extract window tensors
            q_window = q[:, start_idx:end_idx]
            k_window = k[:, start_idx:end_idx]
            v_window = v[:, start_idx:end_idx]
            
            # Compute attention for this window
            scaling = float(self.head_dim) ** -0.5
            q_window = q_window * scaling
            
            # Attention scores for window
            attn = torch.matmul(q_window, k_window.transpose(-2, -1))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Extract relevant portion of the mask
                mask_window = attention_mask[:, start_idx:end_idx, start_idx:end_idx]
                attn = attn + mask_window
            
            # Softmax and dropout
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            
            # Apply attention to values
            window_output = torch.matmul(attn, v_window)
            
            # Add to output with overlap handling
            if i == 0:
                output[:, start_idx:end_idx] = window_output
            else:
                # Blend with linear weights in overlap region
                overlap_start = start_idx
                overlap_end = min(i + window_size // 4, seq_len)
                
                # Calculate linear blending weights
                alphas = torch.linspace(0, 1, overlap_end - overlap_start, device=q.device)
                alphas = alphas.view(1, -1, 1)  # Reshape for broadcasting
                
                # Blend previous and current outputs in overlap region
                output[:, overlap_start:overlap_end] = (
                    (1 - alphas) * output[:, overlap_start:overlap_end] +
                    alphas * window_output[:, :overlap_end-overlap_start]
                )
                
                # Copy non-overlapping part
                output[:, overlap_end:end_idx] = window_output[:, overlap_end-overlap_start:end_idx-overlap_start]
        
        return output
    
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # Use flash attention if available, otherwise use our sparse attention
        if self.use_flash:
            # Flash attention with GQA
            qkv = self.qkv_proj(hidden_states)
            batch_size, seq_len, _ = qkv.shape
            qkv = qkv.reshape(batch_size, seq_len, -1, self.head_dim)
            
            # Handle GQA - reshape qkv appropriately
            q, k, v = torch.split(qkv, [self.num_attention_heads, self.num_kv_groups, self.num_kv_groups], dim=2)
            
            # For kv_groups < num_heads, repeat k,v heads to match query heads
            if self.num_kv_groups < self.num_attention_heads:
                repeat_factor = self.num_attention_heads // self.num_kv_groups
                k = k.repeat_interleave(repeat_factor, dim=2)
                v = v.repeat_interleave(repeat_factor, dim=2)
            
            # Recombine for flash attention
            qkv_combined = torch.cat([q, k, v], dim=2)
            
            # Flash attention call
            context_layer = self.flash_attn_func(
                qkv_combined, 
                causal=True, 
                softmax_scale=1.0/math.sqrt(self.head_dim)
            )
            
            # Reshape output
            context_layer = context_layer.reshape(batch_size, seq_len, self.hidden_size)
        else:
            # Regular attention with GQA implementation
            batch_size, seq_len = hidden_states.shape[:2]
            
            # Project q, k, v
            q = self.query(hidden_states).reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            k = self.key(hidden_states).reshape(batch_size, seq_len, self.num_kv_groups, self.head_dim)
            v = self.value(hidden_states).reshape(batch_size, seq_len, self.num_kv_groups, self.head_dim)
            
            # For GQA, expand k, v to match number of heads
            if self.num_kv_groups < self.num_attention_heads:
                repeat_factor = self.num_attention_heads // self.num_kv_groups
                k = k.repeat_interleave(repeat_factor, dim=2)
                v = v.repeat_interleave(repeat_factor, dim=2)
            
            # Reshape for attention computation
            q = q.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            
            # Compute sparse attention
            context_layer = self._sparse_attention(q, k, v, attention_mask)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            context_layer = context_layer.reshape(batch_size, seq_len, self.hidden_size)
        
        # Output projection and dropout
        output = self.proj(context_layer)
        output = self.proj_dropout(output)
        
        # Residual connection
        output = output + residual
        
        # FFN
        residual = output
        output = self.ln_2(output)
        output = self.mlp(output)
        output = output + residual
        
        if use_cache:
            # For inference with KV caching
            return output, (k, v)
        
        return output 