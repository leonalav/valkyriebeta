"""
RWKV model and layer implementations

This module provides the RWKV model architecture, which combines aspects of
RNNs and Transformers, offering efficient handling of sequential data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from contextlib import nullcontext
import logging

class StructuralRepresentationLayer(nn.Module):
    """
    Layer that enhances RWKV with structural and hierarchical representations
    for better mathematical and logical reasoning
    """
    
    def __init__(self, hidden_size, graph_dim=64, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.graph_dim = graph_dim
        self.num_heads = num_heads
        
        # Structure-aware projections
        self.structure_proj = nn.Linear(hidden_size, graph_dim * num_heads)
        self.structure_nodes = nn.Linear(hidden_size, graph_dim * num_heads)
        self.structure_edges = nn.Linear(hidden_size, graph_dim * num_heads)
        
        # Graph neural network components
        self.graph_update = nn.Sequential(
            nn.Linear(graph_dim * 2, graph_dim),
            nn.GELU(),
            nn.Linear(graph_dim, graph_dim)
        )
        
        # Integration with main representation
        self.structure_gate = nn.Linear(hidden_size, hidden_size)
        self.structure_output = nn.Linear(graph_dim * num_heads, hidden_size)
        
    def forward(self, x):
        """Forward pass with structure extraction
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            enhanced_x: Structure-enhanced representation
            structure_info: Extracted structural information
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to structure space
        structure_feats = self.structure_proj(x)  # [batch_size, seq_len, graph_dim * num_heads]
        nodes = self.structure_nodes(x).view(batch_size, seq_len, self.num_heads, self.graph_dim)
        edges = self.structure_edges(x).view(batch_size, seq_len, self.num_heads, self.graph_dim)
        
        # Multi-head graph reasoning
        graph_outputs = []
        for h in range(self.num_heads):
            # Node features for this head
            head_nodes = nodes[:, :, h]  # [batch_size, seq_len, graph_dim]
            head_edges = edges[:, :, h]  # [batch_size, seq_len, graph_dim]
            
            # Create graph representation using attention mechanism
            # This simulates message passing in a graph
            node_sim = torch.bmm(head_nodes, head_nodes.transpose(1, 2)) / math.sqrt(self.graph_dim)
            node_sim = node_sim.masked_fill(
                torch.ones_like(node_sim).triu(diagonal=1).bool(), float('-inf')
            )  # Causal mask
            edge_weights = F.softmax(node_sim, dim=2)
            
            # Message passing - aggregate features from preceding nodes
            node_updates = torch.bmm(edge_weights, head_nodes)
            
            # Combine node features with edge weights (simulating graph update)
            updated_nodes = torch.cat([head_nodes, node_updates], dim=2)
            updated_nodes = self.graph_update(updated_nodes)
            
            graph_outputs.append(updated_nodes)
        
        # Combine heads and integrate with main representation
        graph_out = torch.cat(graph_outputs, dim=2)
        structure_out = self.structure_output(graph_out)
        
        # Gated integration with original features
        gate = torch.sigmoid(self.structure_gate(x))
        enhanced_x = x + gate * structure_out
        
        return enhanced_x, structure_out

class SymbolicIntegrationModule(nn.Module):
    """
    Module for integrating symbolic reasoning capabilities into RWKV
    """
    
    def __init__(self, hidden_size, symbolic_dim=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.symbolic_dim = symbolic_dim
        
        # Symbolic extractors
        self.extract_symbols = nn.Linear(hidden_size, symbolic_dim)
        self.extract_operators = nn.Linear(hidden_size, symbolic_dim)
        self.extract_relationships = nn.Linear(hidden_size, symbolic_dim)
        
        # Symbolic reasoner
        self.symbolic_reasoner = nn.Sequential(
            nn.Linear(symbolic_dim * 3, symbolic_dim * 2),
            nn.GELU(),
            nn.Linear(symbolic_dim * 2, symbolic_dim)
        )
        
        # Integration back to main representation
        self.symbolic_gate = nn.Linear(hidden_size, hidden_size)
        self.symbolic_output = nn.Linear(symbolic_dim, hidden_size)
        
    def forward(self, x):
        """Forward pass with symbolic reasoning
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            enhanced_x: Symbolically enhanced representation
            symbolic_info: Extracted symbolic information
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract symbolic components
        symbols = self.extract_symbols(x)
        operators = self.extract_operators(x)
        relationships = self.extract_relationships(x)
        
        # Combine symbolic components
        symbolic_feats = torch.cat([symbols, operators, relationships], dim=2)
        symbolic_reasoned = self.symbolic_reasoner(symbolic_feats)
        
        # Integrate back to main representation
        symbolic_out = self.symbolic_output(symbolic_reasoned)
        gate = torch.sigmoid(self.symbolic_gate(x))
        enhanced_x = x + gate * symbolic_out
        
        return enhanced_x, symbolic_reasoned

class HierarchicalTimeAttention(nn.Module):
    """
    Advanced time-mixing layer that adds hierarchical structure awareness
    to the standard RWKV time-mixing mechanism
    """
    
    def __init__(self, hidden_size, max_hierarchical_depth=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_depth = max_hierarchical_depth
        
        # Multi-scale time mixing
        self.time_decay_multi = nn.Parameter(torch.ones(self.max_depth, hidden_size))
        
        # Level selectors determine which hierarchical level to use
        self.level_selector = nn.Linear(hidden_size, self.max_depth)
        
        # Value, key, receptance for time mixing
        self.time_value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.time_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.time_receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.time_output = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Hierarchical state compression
        self.state_compress = nn.Linear(hidden_size * self.max_depth, hidden_size)
        
        # Initialize with decreasing time decay for deeper hierarchical levels
        with torch.no_grad():
            for d in range(self.max_depth):
                decay_base = 0.9 ** (d + 1)  # Deeper levels have longer memory
                self.time_decay_multi.data[d] = torch.ones(hidden_size) * decay_base
    
    def forward(self, x, state=None):
        """Forward pass with hierarchical time mixing
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            state: Optional previous state [batch_size, max_depth, hidden_size]
            
        Returns:
            output: Processed tensor
            new_state: Updated hierarchical state
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hierarchical state if none
        if state is None:
            state = torch.zeros(batch_size, self.max_depth, self.hidden_size, device=x.device)
        
        # Project inputs
        v = self.time_value(x)
        k = self.time_key(x)
        r = torch.sigmoid(self.time_receptance(x))
        
        # Level selection weights - determine how much each hierarchical level contributes
        level_weights = F.softmax(self.level_selector(x), dim=-1)  # [batch_size, seq_len, max_depth]
        
        # Initialize output and new state
        output = torch.zeros_like(x)
        new_state = state.clone()
        
        # Process sequence with hierarchical time decay
        for t in range(seq_len):
            # Current token
            vt = v[:, t:t+1]  # [batch_size, 1, hidden_size]
            kt = k[:, t:t+1]  # [batch_size, 1, hidden_size]
            rt = r[:, t:t+1]  # [batch_size, 1, hidden_size]
            level_wt = level_weights[:, t]  # [batch_size, max_depth]
            
            # Apply hierarchical time decay and update state
            for d in range(self.max_depth):
                time_decay = torch.exp(-torch.exp(self.time_decay_multi[d]))
                state_d = state[:, d]  # [batch_size, hidden_size]
                
                # Update state with decay
                new_state_d = state_d * time_decay + kt.squeeze(1) * vt.squeeze(1)
                new_state[:, d] = new_state_d
                
            # Combine states using level weights for hierarchical representation
            weighted_state = torch.zeros(batch_size, self.hidden_size, device=x.device)
            for d in range(self.max_depth):
                weighted_state += level_wt[:, d:d+1] * new_state[:, d]
            
            # Generate output with receptance gating
            ot = rt[:, 0] * weighted_state
            output[:, t] = ot
        
        # Final projection
        output = self.time_output(output)
        
        return output, new_state

class EnhancedRWKVBlock(nn.Module):
    """
    Enhanced RWKV block with improved hierarchical and symbolic capabilities
    """
    
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        
        # Enhanced hierarchical time mixing
        use_enhanced_time_mix = getattr(config, 'use_enhanced_time_mix', True)
        if use_enhanced_time_mix:
            self.time_mixer = HierarchicalTimeAttention(
                config.hidden_size,
                max_hierarchical_depth=getattr(config, 'hierarchical_depth', 4)
            )
        else:
            self.time_mixer = RWKVTimeFirst(config, layer_id)
        
        # Channel mixing (feed-forward equivalent)
        self.channel_mixer = RWKVChannelMix(config.hidden_size, layer_id)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=getattr(config, 'layer_norm_epsilon', 1e-5))
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=getattr(config, 'layer_norm_epsilon', 1e-5))
        
        # Structure enhancements
        use_structure_enhance = getattr(config, 'use_structure_enhance', True)
        if use_structure_enhance and layer_id > 0:  # Apply in higher layers
            self.structure_enhance = StructuralRepresentationLayer(
                config.hidden_size,
                graph_dim=getattr(config, 'graph_dim', 64),
                num_heads=getattr(config, 'structure_heads', 8)
            )
        else:
            self.structure_enhance = None
        
        # Symbolic reasoning
        use_symbolic = getattr(config, 'use_symbolic', True)
        if use_symbolic and layer_id > 0:  # Apply in higher layers
            self.symbolic_module = SymbolicIntegrationModule(
                config.hidden_size,
                symbolic_dim=getattr(config, 'symbolic_dim', 128)
            )
        else:
            self.symbolic_module = None
        
        # Memory compression for reducing state size
        self.enable_state_compression = getattr(config, 'enable_state_compression', False)
        if self.enable_state_compression:
            self.state_compressor = nn.Linear(config.hidden_size * 2, config.hidden_size)
    
    def forward(self, x, state=None):
        """
        Forward pass with enhanced hierarchical processing
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            state: Previous state tuple (time_state, channel_state, structure_state)
            
        Returns:
            output: Processed tensor
            new_state: Updated state tuple
        """
        batch_size, seq_len, _ = x.shape
        
        # Extract states
        if state is None:
            time_state = None
            channel_state = None
            structure_state = None
        else:
            time_state, channel_state, structure_state = state
        
        # Apply structural enhancement if available
        if self.structure_enhance is not None:
            x_structure, structure_info = self.structure_enhance(x)
            # Use weighted combination of original and structure-enhanced representation
            x = 0.7 * x + 0.3 * x_structure
            new_structure_state = structure_info
        else:
            new_structure_state = structure_state
        
        # Apply time mixing with residual connection
        residual = x
        x = self.ln1(x)
        x_time, new_time_state = self.time_mixer(x, time_state)
        x = x_time + residual
        
        # Apply symbolic reasoning if available
        if self.symbolic_module is not None:
            x_symbolic, _ = self.symbolic_module(x)
            # Use weighted combination of original and symbolic-enhanced representation
            x = 0.8 * x + 0.2 * x_symbolic
        
        # Apply channel mixing with residual connection
        residual = x
        x = self.ln2(x)
        x_channel, new_channel_state = self.channel_mixer(x, channel_state)
        x = x_channel + residual
        
        # Compress state if enabled
        if self.enable_state_compression and new_time_state is not None:
            if isinstance(new_time_state, tuple):
                # Handle hierarchical time states
                compressed_states = []
                for s in new_time_state:
                    if s is not None:
                        s_mean = s.mean(dim=1, keepdim=True)
                        s_compressed = s_mean.expand_as(s)
                        compressed_states.append(s_compressed)
                    else:
                        compressed_states.append(None)
                new_time_state = tuple(compressed_states)
            else:
                # Standard state compression
                state_mean = new_time_state.mean(dim=1, keepdim=True)
                new_time_state = state_mean.expand_as(new_time_state)
        
        # Return output and combined state
        new_state = (new_time_state, new_channel_state, new_structure_state)
        return x, new_state


class RWKVTimeFirst(nn.Module):
    """RWKV Time-mixing module - processes time dimension first"""
    
    def __init__(self, hidden_size, layer_id, att_scale=1.0, time_mix_ratio=1.0, use_linear_att=True, head_qk_dim=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        self.time_mix_ratio = time_mix_ratio
        self.use_linear_att = use_linear_att
        self.att_scale = att_scale
        
        # Time mixing parameters
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, hidden_size) * time_mix_ratio)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, hidden_size) * time_mix_ratio)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, hidden_size) * time_mix_ratio)
        
        # Time first projections
        self.time_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.time_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.time_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.time_output = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Time decay (learned)
        decay_base = 0.9 ** (layer_id + 1)
        self.time_decay = nn.Parameter(torch.ones(hidden_size) * decay_base)
        
        # Optional head-based QK dimensionality
        self.head_qk_dim = head_qk_dim if head_qk_dim > 0 else hidden_size
        if head_qk_dim > 0 and head_qk_dim != hidden_size:
            self.time_qk_proj = nn.Linear(hidden_size, self.head_qk_dim, bias=False)
            self.time_qk_output = nn.Linear(self.head_qk_dim, hidden_size, bias=False)
        
        # Initialize parameters properly
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize time projections
        std = 0.02 * (self.layer_id + 1)
        nn.init.normal_(self.time_r.weight, mean=0.0, std=std)
        nn.init.normal_(self.time_k.weight, mean=0.0, std=std)
        nn.init.normal_(self.time_v.weight, mean=0.0, std=std)
        nn.init.normal_(self.time_output.weight, mean=0.0, std=std)
        
        # Initialize QK projections if used
        if hasattr(self, 'time_qk_proj'):
            nn.init.normal_(self.time_qk_proj.weight, mean=0.0, std=std)
            nn.init.normal_(self.time_qk_output.weight, mean=0.0, std=std)
        
    def forward(self, x, state=None):
        """
        Forward pass with optional state for recurrent processing
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            state: Optional state tensor from previous computation
            
        Returns:
            output: Output tensor of same shape as x
            new_state: Updated state for recurrent processing
        """
        B, T, C = x.size()  # batch, time, channels
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        
        # Time mixing - split
        x_mix = torch.cat([state.unsqueeze(1), x[:, :-1]], dim=1)
        
        # Apply time mixing
        r = self.time_r(x * self.time_mix_r + x_mix * (1 - self.time_mix_r))
        k = self.time_k(x * self.time_mix_k + x_mix * (1 - self.time_mix_k))
        v = self.time_v(x * self.time_mix_v + x_mix * (1 - self.time_mix_v))
        
        # Use optional QK projection for head-based computation
        if hasattr(self, 'time_qk_proj'):
            r = self.time_qk_proj(r)
            k = self.time_qk_proj(k)
        
        # Apply non-linearities
        r = torch.sigmoid(r)
        k = torch.exp(k)  # exponential key
        
        # Linear attention approximation if enabled
        if self.use_linear_att:
            output = self._linear_attention(r, k, v, state)
            new_state = x[:, -1].clone()  # Update state with last token
        else:
            output = self._full_attention(r, k, v, state)
            new_state = x[:, -1].clone()  # Update state with last token
        
        # Return to original dimension if needed
        if hasattr(self, 'time_qk_output'):
            output = self.time_qk_output(output)
        
        # Project output
        output = self.time_output(output)
        
        return output, new_state
        
    def _linear_attention(self, r, k, v, state):
        """Efficient linear attention - O(n) scaling"""
        time_decay = torch.exp(self.time_decay).unsqueeze(0).unsqueeze(0)
        
        # Cumulative computation
        output = torch.zeros_like(v)
        weighted_k = k * v
        
        # Efficient computation of attention with state
        for t in range(v.size(1)):
            # State-based attention update - linear complexity
            state = state * time_decay + weighted_k[:, t]
            output[:, t] = r[:, t] * state
            
        return output * self.att_scale
    
    def _full_attention(self, r, k, v, state):
        """Full attention computation - used when linear approximation is disabled"""
        B, T, C = v.size()
        time_decay = torch.exp(self.time_decay).unsqueeze(0).unsqueeze(0)
        
        # Compute attention weights explicitly
        att_weights = torch.zeros(B, T, T, device=v.device)
        
        # Fill the attention matrix
        for t1 in range(T):
            for t2 in range(t1 + 1):
                decay = time_decay ** (t1 - t2)
                att_weights[:, t1, t2] = decay.squeeze(1)
        
        # Apply attention
        att_weights = att_weights * k.unsqueeze(1)
        output = torch.bmm(att_weights, v) * r
        
        return output * self.att_scale


class RWKVChannelMix(nn.Module):
    """RWKV Channel mixing module"""
    
    def __init__(self, hidden_size, layer_id, ffn_scale=1.0, use_glu=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        self.ffn_scale = ffn_scale
        self.use_glu = use_glu
        
        # Channel mixing ratio
        self.time_mix_ratio = 1.0
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, hidden_size) * self.time_mix_ratio)
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, hidden_size) * self.time_mix_ratio)
        
        # Channel mixing projections
        self.key = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        
        # Channel output
        if use_glu:
            self.value = nn.Linear(hidden_size * 4, hidden_size, bias=False)
            self.gate = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        else:
            self.value = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize with scaled values based on layer depth
        std = 0.02 * (self.layer_id + 1)
        nn.init.normal_(self.key.weight, mean=0.0, std=std)
        nn.init.normal_(self.value.weight, mean=0.0, std=std)
        if self.use_glu:
            nn.init.normal_(self.gate.weight, mean=0.0, std=std)
    
    def forward(self, x, state=None):
        """
        Forward pass with optional state
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            state: Optional state from previous computation
            
        Returns:
            output: Output tensor same shape as x
            new_state: Last token as state for recurrent processing
        """
        B, T, C = x.size()
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        
        # Time mixing - using previous state
        x_mix = torch.cat([state.unsqueeze(1), x[:, :-1]], dim=1)
        
        # Apply time mixing
        k = self.key(x * self.time_mix_k + x_mix * (1 - self.time_mix_k))
        
        # Apply activation
        k = torch.square(torch.relu(k))
        
        # GLU or standard output
        if self.use_glu:
            v = self.value(k)
            g = self.gate(k)
            output = v * torch.sigmoid(g)
        else:
            output = self.value(k)
        
        # Scale output
        output = output * self.ffn_scale
        
        # Update state with last token
        new_state = x[:, -1].clone()
        
        return output, new_state


class RWKVBlock(nn.Module):
    """
    RWKV block with time and channel mixing
    
    This is the core building block of the RWKV architecture,
    combining time-based processing (attention-like) with channel
    mixing (feedforward-like) capabilities.
    """
    
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        
        # Flag to identify as RWKV block for hybrid models
        self.is_rwkv_block = True
        
        # Time mixing (attention-like)
        if getattr(config, 'use_enhanced_rwkv', False):
            self.time_mixer = HierarchicalTimeAttention(
                config.hidden_size,
                max_hierarchical_depth=getattr(config, 'hierarchical_depth', 4)
            )
        else:
            self.time_mixer = RWKVTimeFirst(config, layer_id)
        
        # Channel mixing (feedforward-like)
        self.channel_mixer = RWKVChannelMix(config.hidden_size, layer_id)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
        # Output gates for residual connections
        self.use_gate_res = getattr(config, 'use_gate_res', True)
        if self.use_gate_res:
            gate_init = getattr(config, 'gate_init', 1e-3)
            self.att_gate = nn.Parameter(torch.ones(1, 1, config.hidden_size) * gate_init)
            self.ffn_gate = nn.Parameter(torch.ones(1, 1, config.hidden_size) * gate_init)
    
    def forward(self, x, att_state=None, ffn_state=None):
        """
        Forward pass with optional recurrent state
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            att_state: Previous attention state
            ffn_state: Previous FFN state
            
        Returns:
            output: Processed tensor
            new_att_state: Updated attention state
            new_ffn_state: Updated FFN state
        """
        # Time mixing with residual
        att_out, new_att_state = self.time_mixer(self.ln1(x), att_state)
        if self.use_gate_res:
            x = x + att_out * torch.sigmoid(self.att_gate)
        else:
            x = x + att_out
        
        # Channel mixing with residual
        ffn_out, new_ffn_state = self.channel_mixer(self.ln2(x), ffn_state)
        if self.use_gate_res:
            x = x + ffn_out * torch.sigmoid(self.ffn_gate)
        else:
            x = x + ffn_out
        
        return x, new_att_state, new_ffn_state


class RWKVModel(nn.Module):
    """RWKV Model implementation with advanced features for training and inference"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        
        # Word embedding
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings (optional)
        self.use_position_embeddings = getattr(config, 'use_position_embeddings', False)
        if self.use_position_embeddings:
            self.max_position_embeddings = getattr(config, 'max_position_embeddings', 2048)
            self.position_embeddings = nn.Embedding(self.max_position_embeddings, config.hidden_size)
            self.position_dropout = nn.Dropout(getattr(config, 'position_embedding_dropout', 0.1))
        
        # RWKV Blocks
        self.blocks = nn.ModuleList()
        rwkv_layer_indices = getattr(config, 'rwkv_layer_indices', list(range(config.num_layers)))
        
        for i in range(config.num_layers):
            if i in rwkv_layer_indices:
                self.blocks.append(RWKVBlock(config, i))
            else:
                # Add transformer block for hybrid architecture
                self.blocks.append(TransformerBlock(config, i))
        
        # Final layer norm
        self.ln_out = nn.LayerNorm(config.hidden_size)
        
        # Output projection
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if specified
        if getattr(config, 'tie_word_embeddings', False):
            self.head.weight = self.emb.weight
        
        # State for recurrent processing
        self.state = None
        self.chunk_size = getattr(config, 'rwkv_chunk_size', 1024)
        
        # Chunk overlap for continuity between chunks
        self.chunk_overlap = getattr(config, 'rwkv_chunk_overlap', self.chunk_size // 8)
        
        # State compression
        self.use_state_compression = getattr(config, 'rwkv_state_compression', False)
        if self.use_state_compression:
            self.state_compressors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                    nn.Tanh()
                ) for _ in range(self.num_layers)
            ])
        
        # Learnable initial states
        self.use_learnable_state = getattr(config, 'use_learnable_state', False)
        if self.use_learnable_state:
            self.learnable_att_states = nn.ParameterList([
                nn.Parameter(torch.zeros(1, self.hidden_size))
                for _ in range(self.num_layers)
            ])
            self.learnable_ffn_states = nn.ParameterList([
                nn.Parameter(torch.zeros(1, self.hidden_size))
                for _ in range(self.num_layers)
            ])
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
        
        # Mixed precision support
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', False)
        self.mixed_precision_dtype = getattr(config, 'mixed_precision_dtype', torch.float16)
    
    def reset_state(self, batch_size=1):
        """Reset recurrent state with optional learnable initialization"""
        device = next(self.parameters()).device
        
        if self.use_learnable_state:
            # Initialize with learned parameters
            self.state = [
                (
                    self.learnable_att_states[i].expand(batch_size, -1).clone(),
                    self.learnable_ffn_states[i].expand(batch_size, -1).clone()
                )
                for i in range(self.num_layers)
            ]
        else:
            # Initialize with zeros
            self.state = [
                (
                    torch.zeros(batch_size, self.hidden_size, device=device),
                    torch.zeros(batch_size, self.hidden_size, device=device)
                )
                for _ in range(self.num_layers)
            ]
    
    def set_chunk_size(self, chunk_size, chunk_overlap=None):
        """Set chunk size for processing and optional overlap"""
        self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
    
    def _compress_state(self, states):
        """Compress state for memory efficiency"""
        if not self.use_state_compression:
            return states
            
        compressed_states = []
        for i, (att_state, ffn_state) in enumerate(states):
            # Concatenate states
            combined = torch.cat([att_state, ffn_state], dim=-1)
            
            # Compress using layer-specific compressor
            compressed = self.state_compressors[i](combined)
            
            # Split back into attention and FFN states
            mid = compressed.shape[-1] // 2
            compressed_att = compressed[..., :mid]
            compressed_ffn = compressed[..., mid:]
            
            compressed_states.append((compressed_att, compressed_ffn))
            
        return compressed_states
    
    def _decompress_state(self, compressed_states):
        """Decompress state"""
        if not self.use_state_compression:
            return compressed_states
            
        # Implementation would depend on compression technique
        # For simple linear compression, no explicit decompression needed
        return compressed_states
    
    def process_with_state(self, input_ids, state=None):
        """
        Process input with optional state
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            state: Optional previous state
            
        Returns:
            outputs: Model outputs
            new_state: Updated state
        """
        # Initialize state if None
        if state is None:
            batch_size = input_ids.size(0)
            self.reset_state(batch_size)
            state = self.state
        else:
            # Decompress state if using compression
            state = self._decompress_state(state)
        
        # Get embeddings
        x = self.emb(input_ids)
        
        # Add position embeddings if enabled
        if self.use_position_embeddings:
            seq_len = input_ids.size(1)
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            position_embeddings = self.position_embeddings(position_ids)
            x = x + position_embeddings
            x = self.position_dropout(x)
        
        # Process through blocks
        new_states = []
        for i, block in enumerate(self.blocks):
            block_state = state[i] if state is not None else None
            if hasattr(block, 'is_rwkv_block') and block.is_rwkv_block:
                # RWKV block
                att_state, ffn_state = block_state if block_state is not None else (None, None)
                x, new_att_state, new_ffn_state = block(x, att_state, ffn_state)
                new_states.append((new_att_state, new_ffn_state))
            else:
                # Transformer block (requires compatible interface)
                x, _, _ = block(x, None, None)
                # Placeholder state for transformer blocks to maintain state structure
                new_states.append((None, None))
        
        # Output projection
        x = self.ln_out(x)
        logits = self.head(x)
        
        # Compress new state if needed
        if self.use_state_compression:
            new_states = self._compress_state(new_states)
            
        return logits, new_states
    
    def forward(self, input_ids, attention_mask=None, labels=None, use_chunking=False, position_ids=None):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            labels: Optional labels for computing loss
            use_chunking: Whether to process in chunks for memory efficiency
            position_ids: Optional position IDs
            
        Returns:
            outputs: Model outputs with loss if labels provided
        """
        batch_size, seq_len = input_ids.size()
        
        # Use chunking for long sequences
        if use_chunking and seq_len > self.chunk_size:
            return self.forward_chunked(input_ids, attention_mask, labels, position_ids)
        
        # Mixed precision context
        mp_context = torch.autocast(device_type="cuda", dtype=self.mixed_precision_dtype) if self.use_mixed_precision else nullcontext()
        
        with mp_context:
            # Get embeddings
            x = self.emb(input_ids)
            
            # Add position embeddings if enabled
            if self.use_position_embeddings and position_ids is None:
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                position_embeddings = self.position_embeddings(position_ids)
                x = x + position_embeddings
                x = self.position_dropout(x)
            elif self.use_position_embeddings and position_ids is not None:
                position_embeddings = self.position_embeddings(position_ids)
                x = x + position_embeddings
                x = self.position_dropout(x)
            
            # Process through blocks with gradient checkpointing if enabled
            for i, block in enumerate(self.blocks):
                if self.gradient_checkpointing and self.training:
                    if hasattr(block, 'is_rwkv_block') and block.is_rwkv_block:
                        # Custom checkpoint function for RWKV blocks
                        x, _, _ = torch.utils.checkpoint.checkpoint(
                            lambda x_in: block(x_in, None, None),
                            x
                        )
                    else:
                        # Standard checkpoint for transformer blocks
                        x, _, _ = torch.utils.checkpoint.checkpoint(block, x, None, None)
                else:
                    # Standard forward pass
                    x, _, _ = block(x, None, None)
            
            # Output projection
            x = self.ln_out(x)
            logits = self.head(x)
            
            # Compute loss if labels provided
            loss = None
            if labels is not None:
                # Shift logits and labels for next token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
            # Return outputs
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": x
            }
    
    def forward_chunked(self, input_ids, attention_mask=None, labels=None, position_ids=None):
        """
        Forward pass with chunking for memory efficiency with improved overlap handling
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            labels: Optional labels for computing loss
            position_ids: Optional position IDs
            
        Returns:
            outputs: Model outputs with loss if labels provided
        """
        batch_size, seq_len = input_ids.size()
        chunk_size = self.chunk_size
        overlap = min(self.chunk_overlap, chunk_size // 2)  # Ensure overlap isn't too large
        
        # Mixed precision context
        mp_context = torch.autocast(device_type="cuda", dtype=self.mixed_precision_dtype) if self.use_mixed_precision else nullcontext()
        
        with mp_context:
            # Reset state
            self.reset_state(batch_size)
            
            # Process in chunks with overlap
            all_logits = []
            total_loss = 0.0
            effective_tokens = 0
            
            # Calculate positions if using position embeddings
            if self.use_position_embeddings and position_ids is None:
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            
            # Process each chunk
            i = 0
            while i < seq_len:
                # Get current chunk with overlap
                end_idx = min(i + chunk_size, seq_len)
                chunk_ids = input_ids[:, i:end_idx]
                
                # Get position IDs for this chunk if needed
                chunk_positions = None
                if self.use_position_embeddings:
                    chunk_positions = position_ids[:, i:end_idx]
                
                # Process chunk with state
                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module, chunk_input, chunk_pos):
                        def custom_forward(*inputs):
                            # inputs will be (None, None) to ensure checkpoint works
                            # but we use our real inputs
                            states = module.state  # Save current state
                            with torch.no_grad():  # No need to track state computation
                                logits, module.state = module.process_with_state(chunk_input, states)
                            return logits
                        return custom_forward
                    
                    # Use gradient checkpointing
                    logits = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self, chunk_ids, chunk_positions),
                        None, None  # Dummy inputs
                    )
                else:
                    # Normal forward pass with state
                    logits, self.state = self.process_with_state(chunk_ids, self.state)
                
                # Compute loss if labels provided
                if labels is not None:
                    chunk_labels = labels[:, i:end_idx]
                    
                    # For all except last chunk, ignore the overlapped region for loss
                    if end_idx < seq_len:
                        effective_len = end_idx - i - overlap
                        shift_logits = logits[:, :effective_len-1, :].contiguous()
                        shift_labels = chunk_labels[:, 1:effective_len].contiguous()
                    else:
                        # For last chunk, use all tokens
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = chunk_labels[:, 1:].contiguous()
                    
                    # Compute loss
                    if shift_labels.numel() > 0:  # Only if we have valid tokens
                        loss_fct = nn.CrossEntropyLoss()
                        chunk_loss = loss_fct(shift_logits.view(-1, self.vocab_size), 
                                            shift_labels.view(-1))
                        total_loss += chunk_loss.item() * shift_labels.numel()
                        effective_tokens += shift_labels.numel()
                
                # For next chunks, skip tokens that were overlapped except for state computation
                if end_idx < seq_len:
                    # Only add non-overlapping part to the output logits
                    if i == 0:  # First chunk
                        all_logits.append(logits[:, :-overlap] if overlap > 0 else logits)
                    else:
                        all_logits.append(logits[:, overlap:-overlap] if i + chunk_size < seq_len else logits[:, overlap:])
                else:
                    # Last chunk - add all remaining after overlap
                    if i > 0:  # Not the only chunk
                        all_logits.append(logits[:, overlap:])
                    else:  # Only one chunk
                        all_logits.append(logits)
                
                # Move to next chunk with overlap
                i += chunk_size - overlap
            
            # Concatenate logits from all chunks
            if len(all_logits) > 1:
                logits = torch.cat(all_logits, dim=1)
            else:
                logits = all_logits[0]
            
            # Compute overall loss
            loss = total_loss / max(1, effective_tokens) if labels is not None else None
            
            # Return outputs
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": None  # Full hidden states not available in chunked mode
            }
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
        for m in self.modules():
            if hasattr(m, 'gradient_checkpointing'):
                m.gradient_checkpointing = True
        return self
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False
        for m in self.modules():
            if hasattr(m, 'gradient_checkpointing'):
                m.gradient_checkpointing = False
        return self
    
    def enable_state_compression(self):
        """Enable compression of states for memory efficiency"""
        self.use_state_compression = True
        return self
    
    def disable_state_compression(self):
        """Disable state compression"""
        self.use_state_compression = False
        return self
    
    def enable_mixed_precision(self, dtype=torch.float16):
        """Enable mixed precision for faster computation"""
        self.use_mixed_precision = True
        self.mixed_precision_dtype = dtype
        return self
    
    def disable_mixed_precision(self):
        """Disable mixed precision"""
        self.use_mixed_precision = False
        return self
    
    def optimize_for_inference(self):
        """Apply optimizations for inference time"""
        # Disable dropout for inference
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0
        
        # Disable gradient computation
        for param in self.parameters():
            param.requires_grad = False
        
        # Enable state compression for memory efficiency
        self.enable_state_compression()
        
        # Fuse operations where possible
        # This would require custom kernels or ONNX/TensorRT integration
        
        # Set flag
        self.optimized_for_inference = True
        return self

class TransformerBlock(nn.Module):
    """Transformer block for hybrid RWKV-Transformer architecture"""
    
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        
        # Flag to identify as non-RWKV block for hybrid models
        self.is_rwkv_block = False
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
        # Multi-head attention
        self.n_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Ensure head dimension works with number of heads
        assert self.head_dim * config.num_attention_heads == config.hidden_size, \
            f"Hidden size {config.hidden_size} not divisible by num_attention_heads {config.num_attention_heads}"
        
        # QKV projections
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.use_bias)
        
        # Output projection
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout if hasattr(config, 'attention_dropout') else 0.1)
        self.resid_dropout = nn.Dropout(config.hidden_dropout if hasattr(config, 'hidden_dropout') else 0.1)
        
        # MLP/FFN
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=config.use_bias),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size, bias=config.use_bias),
            nn.Dropout(config.hidden_dropout if hasattr(config, 'hidden_dropout') else 0.1)
        )
        
        # Initialize weights with scaled values based on layer depth
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize with scalable values based on layer depth
        scale_factor = 0.02 * (1.0 - 0.1 * min(self.layer_id / 24, 1.0))
        
        # QKV projection
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=scale_factor)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
            
        # Output projection
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=scale_factor)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
            
        # MLP layers
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=scale_factor)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, att_state=None, ffn_state=None):
        # Layer normalization before attention
        ln1_x = self.ln1(x)
        
        # QKV projection
        qkv = self.qkv_proj(ln1_x)
        batch_size, seq_len, _ = qkv.shape
        
        # Reshape for multi-head attention
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, n_head, seq_len, head_dim]
        
        # Split into query, key, value
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scale query
        q = q * (self.head_dim ** -0.5)
        
        # Compute attention scores
        att = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, n_head, seq_len, seq_len]
        
        # Causal mask - lower triangular matrix of ones
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        # Attention weights
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention to values
        y = torch.matmul(att, v)  # [batch_size, n_head, seq_len, head_dim]
        y = y.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, n_head, head_dim]
        y = y.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        
        # Output projection
        attn_output = self.out_proj(y)
        attn_output = self.resid_dropout(attn_output)
        
        # Residual connection
        x = x + attn_output
        
        # Layer normalization before MLP
        ln2_x = self.ln2(x)
        
        # MLP
        mlp_output = self.mlp(ln2_x)
        
        # Residual connection
        output = x + mlp_output
        
        return output, None, None  # Return None for states to match RWKV interface 