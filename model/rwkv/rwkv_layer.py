import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any
from contextlib import nullcontext

@dataclass
class RWKVConfig:
    """Configuration for RWKV model components"""
    hidden_size: int = 768
    intermediate_size: Optional[int] = None
    num_layers: int = 12
    time_mix_factor: float = 1.0
    key_value_mixing: bool = True
    att_scale: float = 1.0
    use_linear_attn: bool = False
    use_gating: bool = True
    use_shifting: bool = True
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Advanced configuration
    use_state_compression: bool = False
    track_channel_state: bool = False
    use_mixed_precision: bool = False
    mixed_precision_dtype: Any = None
    use_gate_res: bool = False
    gate_init: float = 1e-3
    
    # Integration settings
    use_chunking: bool = False
    chunk_size: int = 1024
    chunk_overlap: int = 128
    
    # For hybrid models
    rwkv_layer_indices: List[int] = None
    
    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
            
        if self.rwkv_layer_indices is None:
            self.rwkv_layer_indices = list(range(self.num_layers))
            
        if self.mixed_precision_dtype is None and self.use_mixed_precision:
            import torch
            self.mixed_precision_dtype = torch.float16

class RWKVTimeFirst(nn.Module):
    """RWKV Time-mixing module, handles sequence relationships"""
    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        
        # Time-mixing parameters
        self.time_decay = nn.Parameter(torch.ones(hidden_size) * -1.0)
        self.time_first = nn.Parameter(torch.ones(hidden_size) * config.time_mix_factor)
        
        # Key and value projections
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        
        # Additional parameters for shifting
        if config.use_shifting:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            
        # Add state compression for memory efficiency
        self.use_state_compression = getattr(config, 'use_state_compression', False)
        if self.use_state_compression:
            self.state_compressor = nn.Linear(hidden_size * 2, hidden_size)
            self.state_expander = nn.Linear(hidden_size, hidden_size * 2)
    
    def forward(self, x, state=None):
        """Forward pass with optional state for recurrent processing"""
        B, T, C = x.size()  # batch, time, channels
        
        # Apply layer normalization
        x_ln = self.layer_norm(x)
        
        # Apply optional time-shifting
        if self.config.use_shifting:
            x_s = self.time_shift(x_ln)
        else:
            x_s = x_ln
            
        # Mix current time step with previous step
        k = self.key(x_ln * self.time_first + x_s * (1 - self.time_first))
        v = self.value(x_ln)
        r = torch.sigmoid(self.receptance(x_ln * self.time_first + x_s * (1 - self.time_first)))
        
        # Handle recurrent state if provided
        if state is not None:
            # Expand state if using compression
            if self.use_state_compression and hasattr(self, 'state_expander'):
                # Decompress the state
                expanded = self.state_expander(state)
                mid = expanded.shape[-1] // 2
                k_state = expanded[..., :mid]
                v_state = expanded[..., mid:]
            else:
                # State is provided as a tuple
                if isinstance(state, tuple):
                    k_state, v_state = state
                else:
                    # Handle single tensor state (assumes first half is k, second half is v)
                    mid = state.shape[-1] // 2
                    k_state = state[..., :mid]
                    v_state = state[..., mid:]
            
            # Apply recurrent calculation
            k_mix = k_state * torch.exp(self.time_decay.unsqueeze(0).unsqueeze(0)) + k
            v_mix = v_state * torch.exp(self.time_decay.unsqueeze(0).unsqueeze(0)) + v
        else:
            # Initialize with zeros for first token
            k_mix = torch.zeros_like(k)
            v_mix = torch.zeros_like(v)
            
            # Compute recurrent relationship
            for t in range(T):
                kt, vt = k[:, t, :], v[:, t, :]
                k_mix = k_mix * torch.exp(self.time_decay.unsqueeze(0)) + kt.unsqueeze(1)
                v_mix = v_mix * torch.exp(self.time_decay.unsqueeze(0)) + vt.unsqueeze(1)
        
        # Prepare new state
        new_k_state = k[:, -1, :].detach()
        new_v_state = v[:, -1, :].detach()
        
        # Compress state if configured
        if self.use_state_compression and hasattr(self, 'state_compressor'):
            combined = torch.cat([new_k_state, new_v_state], dim=-1)
            new_state = self.state_compressor(combined)
        else:
            new_state = (new_k_state, new_v_state)
        
        # Apply attention scaling if configured
        if self.config.att_scale != 1.0:
            k_mix = k_mix * self.config.att_scale
        
        # Mix the values with receptance gating
        y = r * self.output(v_mix)
        
        return y, new_state

class RWKVChannelMixer(nn.Module):
    """RWKV Channel-mixing module, processes information across channels"""
    def __init__(self, config: RWKVConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        
        # Channel mixing projections
        self.key = nn.Linear(hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        
        # Additional shifting if enabled
        if config.use_shifting:
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            
        # For optional state tracking
        self.track_state = getattr(config, 'track_channel_state', False)
    
    def forward(self, x, state=None):
        """Forward pass for channel mixing with optional state"""
        # Apply layer normalization
        x_ln = self.layer_norm(x)
        
        # Apply time shifting if configured
        if self.config.use_shifting:
            x_s = self.time_shift(x_ln)
        else:
            x_s = x_ln
            
        # Apply channel mixing
        k = self.key(x_s)
        k = torch.square(torch.relu(k))  # squared ReLU
        v = self.value(k)
        
        # Prepare new state if tracking
        new_state = x[:, -1].detach() if self.track_state else None
        
        # Apply gating if configured
        if self.config.use_gating:
            r = torch.sigmoid(self.receptance(x_ln))
            return r * v, new_state
        else:
            return v, new_state

class RWKVBlock(nn.Module):
    """Complete RWKV block with time and channel mixing"""
    def __init__(self, config: RWKVConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_rwkv_block = True  # Flag to identify as RWKV block for hybrid models
        
        # Time mixing and channel mixing layers
        self.time_mixer = RWKVTimeFirst(config)
        self.channel_mixer = RWKVChannelMixer(config)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Mixed precision support
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', False)
        self.mixed_precision_dtype = getattr(config, 'mixed_precision_dtype', torch.float16)
        
        # Optional gates for gated residual connections
        self.use_gate_res = getattr(config, 'use_gate_res', False)
        if self.use_gate_res:
            gate_init = getattr(config, 'gate_init', 1e-3)
            self.att_gate = nn.Parameter(torch.ones(1, 1, config.hidden_size) * gate_init)
            self.ffn_gate = nn.Parameter(torch.ones(1, 1, config.hidden_size) * gate_init)
    
    def forward(self, x, state=None):
        """Forward pass for a complete RWKV block"""
        # Handle state
        if state is not None:
            if isinstance(state, tuple) and len(state) == 2:
                time_state, channel_state = state
            else:
                time_state, channel_state = state, None
        else:
            time_state, channel_state = None, None
        
        # Mixed precision context
        if self.use_mixed_precision and torch.cuda.is_available():
            mp_ctx = torch.cuda.amp.autocast(dtype=self.mixed_precision_dtype)
        else:
            mp_ctx = nullcontext()
        
        # Apply block with mixed precision
        with mp_ctx:
            # Layer norm before time mixing
            residual = x
            x_ln1 = self.ln1(x)
            
            # Time mixing with residual connection
            time_out, new_time_state = self.time_mixer(x_ln1, time_state)
            
            # Apply gated residual if configured
            if self.use_gate_res:
                x = residual + self.dropout(time_out) * torch.sigmoid(self.att_gate)
            else:
                x = residual + self.dropout(time_out)
            
            # Layer norm before channel mixing
            residual = x
            x_ln2 = self.ln2(x)
            
            # Channel mixing with residual connection
            channel_out, new_channel_state = self.channel_mixer(x_ln2, channel_state)
            
            # Apply gated residual if configured
            if self.use_gate_res:
                x = residual + self.dropout(channel_out) * torch.sigmoid(self.ffn_gate)
            else:
                x = residual + self.dropout(channel_out)
        
        # Return output and new state
        new_state = (new_time_state, new_channel_state)
        return x, new_state 