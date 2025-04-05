import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math

@dataclass
class MemoryConfig:
    """Configuration for memory modules"""
    memory_size: int = 1024  # Number of memory slots
    memory_dim: int = 768    # Dimension of each memory slot
    num_memory_heads: int = 8  # Number of memory attention heads
    memory_update_rate: float = 0.1  # Rate at which memory is updated
    use_gated_memory: bool = True  # Whether to use gated memory updates
    use_hierarchical_memory: bool = True  # Whether to use hierarchical memory
    num_memory_hierarchies: int = 3  # Number of memory hierarchies
    memory_dropout: float = 0.1  # Dropout rate for memory operations
    use_memory_compression: bool = True  # Whether to compress memory
    compression_ratio: float = 0.5  # Ratio for memory compression
    use_learnable_memory: bool = True  # Whether to use learnable memory initialization

class MemoryAttention(nn.Module):
    """Multi-head attention for memory access"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_memory_heads if hasattr(config, 'num_memory_heads') else 8
        self.head_dim = self.hidden_size // self.num_heads
        
        # Query, key, value projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.dropout = nn.Dropout(config.memory_dropout if hasattr(config, 'memory_dropout') else 0.1)
        
    def forward(self, query, key_value, attention_mask=None):
        batch_size = query.size(0)
        
        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.o_proj(context)
        
        return output, attn_weights

class GatedMemoryUpdate(nn.Module):
    """Gated memory update mechanism"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Gate networks
        self.update_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.reset_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.memory_content = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
    def forward(self, memory, update_candidates):
        # Concatenate memory and update candidates
        combined = torch.cat([memory, update_candidates], dim=-1)
        
        # Compute gates
        update_gate = torch.sigmoid(self.update_gate(combined))
        reset_gate = torch.sigmoid(self.reset_gate(combined))
        
        # Compute new memory content
        reset_memory = reset_gate * memory
        memory_input = torch.cat([reset_memory, update_candidates], dim=-1)
        new_memory_content = torch.tanh(self.memory_content(memory_input))
        
        # Update memory
        updated_memory = (1 - update_gate) * memory + update_gate * new_memory_content
        
        return updated_memory

class MemoryCompressor(nn.Module):
    """Compresses memory to reduce size while preserving important information"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.compression_ratio = config.compression_ratio if hasattr(config, 'compression_ratio') else 0.5
        
        # Compression networks
        self.importance_scorer = nn.Linear(self.hidden_size, 1)
        self.compressor = nn.Linear(self.hidden_size, int(self.hidden_size * self.compression_ratio))
        self.decompressor = nn.Linear(int(self.hidden_size * self.compression_ratio), self.hidden_size)
        
    def forward(self, memory):
        # Score memory slots by importance
        importance = self.importance_scorer(memory)
        
        # Sort memory slots by importance
        _, indices = torch.sort(importance, dim=1, descending=True)
        sorted_memory = torch.gather(memory, 1, indices.expand(-1, -1, memory.size(-1)))
        
        # Compress memory
        compressed = self.compressor(sorted_memory)
        
        return compressed, indices
    
    def decompress(self, compressed_memory, indices=None):
        # Decompress memory
        decompressed = self.decompressor(compressed_memory)
        
        # Restore original order if indices provided
        if indices is not None:
            # Create reverse indices
            batch_size, seq_len = indices.size()
            reverse_indices = torch.zeros_like(indices)
            batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, seq_len)
            reverse_indices[batch_indices, indices] = torch.arange(seq_len).expand(batch_size, -1)
            
            # Reorder memory
            decompressed = torch.gather(decompressed, 1, reverse_indices.expand(-1, -1, decompressed.size(-1)))
        
        return decompressed

class HierarchicalMemory(nn.Module):
    """Hierarchical memory with multiple levels of abstraction"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_hierarchies = config.num_memory_hierarchies if hasattr(config, 'num_memory_hierarchies') else 3
        
        # Memory hierarchies
        self.hierarchies = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_hierarchies)
        ])
        
        # Cross-hierarchy attention
        self.cross_attn = nn.ModuleList([
            MemoryAttention(config) for _ in range(self.num_hierarchies)
        ])
        
        # Hierarchy gates
        self.hierarchy_gates = nn.ModuleList([
            nn.Linear(self.hidden_size, 1) for _ in range(self.num_hierarchies)
        ])
        
    def forward(self, memory):
        batch_size, seq_len, hidden_size = memory.shape
        hierarchy_outputs = []
        
        # Process each hierarchy
        current_memory = memory
        for i in range(self.num_hierarchies):
            # Transform memory for this hierarchy
            hierarchy_memory = self.hierarchies[i](current_memory)
            
            # Attend to previous hierarchy if not the first
            if i > 0:
                cross_output, _ = self.cross_attn[i](hierarchy_memory, hierarchy_outputs[i-1])
                hierarchy_memory = hierarchy_memory + cross_output
            
            # Compute importance gate for this hierarchy
            gate = torch.sigmoid(self.hierarchy_gates[i](hierarchy_memory))
            
            # Store hierarchy output
            hierarchy_outputs.append(hierarchy_memory)
            
            # Update current memory with gated hierarchy memory
            current_memory = current_memory * (1 - gate) + hierarchy_memory * gate
        
        return current_memory, hierarchy_outputs

class EnhancedMemoryLayer(nn.Module):
    """Enhanced memory layer with hierarchical structure and compression"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Memory size
        self.memory_size = config.memory_size if hasattr(config, 'memory_size') else 1024
        
        # Initialize memory
        if hasattr(config, 'use_learnable_memory') and config.use_learnable_memory:
            self.memory = nn.Parameter(torch.randn(1, self.memory_size, self.hidden_size))
            nn.init.normal_(self.memory, mean=0.0, std=0.02)
        else:
            self.register_buffer('memory', torch.zeros(1, self.memory_size, self.hidden_size))
        
        # Memory attention
        self.memory_attention = MemoryAttention(config)
        
        # Gated memory update
        if hasattr(config, 'use_gated_memory') and config.use_gated_memory:
            self.memory_update = GatedMemoryUpdate(config)
        else:
            self.memory_update = None
        
        # Memory compression
        if hasattr(config, 'use_memory_compression') and config.use_memory_compression:
            self.memory_compressor = MemoryCompressor(config)
        else:
            self.memory_compressor = None
            
        # Hierarchical memory
        if hasattr(config, 'use_hierarchical_memory') and config.use_hierarchical_memory:
            self.hierarchical_memory = HierarchicalMemory(config)
        else:
            self.hierarchical_memory = None
            
        # Output projection
        self.output_proj = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Memory update rate
        self.update_rate = config.memory_update_rate if hasattr(config, 'memory_update_rate') else 0.1
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Expand memory to batch size
        current_memory = self.memory.expand(batch_size, -1, -1)
        
        # Apply hierarchical memory if enabled
        if self.hierarchical_memory is not None:
            hierarchical_memory, _ = self.hierarchical_memory(current_memory)
            current_memory = hierarchical_memory
        
        # Compress memory if enabled
        if self.memory_compressor is not None and self.training:
            compressed_memory, indices = self.memory_compressor(current_memory)
            # We'll decompress later when updating the memory
        
        # Attend to memory
        memory_context, memory_weights = self.memory_attention(hidden_states, current_memory)
        
        # Combine with input
        combined = torch.cat([hidden_states, memory_context], dim=-1)
        output = self.output_proj(combined)
        output = self.layer_norm(output + hidden_states)
        
        # Update memory if training
        if self.training:
            # Compute memory update candidates
            update_candidates = hidden_states
            
            # Apply gated update if enabled
            if self.memory_update is not None:
                updated_memory = self.memory_update(current_memory, update_candidates)
            else:
                # Simple interpolation update
                updated_memory = (1 - self.update_rate) * current_memory + self.update_rate * update_candidates.mean(dim=1, keepdim=True).expand(-1, self.memory_size, -1)
            
            # Update the memory parameter (for the next batch)
            if not hasattr(self.config, 'use_learnable_memory') or not self.config.use_learnable_memory:
                # Only update non-learnable memory
                self.memory = updated_memory[0:1].detach()  # Take first batch and detach
        
        return output
    
    def reset(self):
        """Reset memory to initial state"""
        if not hasattr(self.config, 'use_learnable_memory') or not self.config.use_learnable_memory:
            # Only reset non-learnable memory
            self.memory.zero_()

class WorkingMemory(nn.Module):
    """Working memory for complex reasoning tasks"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_slots = config.working_memory_slots if hasattr(config, 'working_memory_slots') else 8
        
        # Initialize memory slots
        self.memory_slots = nn.Parameter(torch.randn(1, self.num_slots, self.hidden_size))
        nn.init.normal_(self.memory_slots, mean=0.0, std=0.02)
        
        # Memory controllers
        self.read_controller = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        self.write_controller = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        # Memory update gate
        self.update_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Expand memory slots to batch size
        memory = self.memory_slots.expand(batch_size, -1, -1)
        
        # Read from memory
        read_output, read_weights = self.read_controller(
            query=hidden_states,
            key=memory,
            value=memory
        )
        
        # Write to memory
        write_output, write_weights = self.write_controller(
            query=memory,
            key=hidden_states,
            value=hidden_states
        )
        
        # Update memory with gate
        gate_input = torch.cat([memory, write_output], dim=-1)
        update_gate = torch.sigmoid(self.update_gate(gate_input))
        updated_memory = (1 - update_gate) * memory + update_gate * write_output
        
        # Update memory slots
        self.memory_slots = updated_memory[0:1].detach()  # Take first batch and detach
        
        # Combine read output with input
        output = hidden_states + read_output
        
        return output

class MemoryBank(nn.Module):
    """
    Memory bank for storing and retrieving information during model execution.
    Provides episodic, working, and long-term memory capabilities.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        memory_size: int = 1024,
        num_memory_heads: int = 8,
        dropout: float = 0.1,
        use_episodic_memory: bool = True,
        episodic_memory_size: int = 512,
        use_working_memory: bool = True,
        working_memory_size: int = 256,
        use_long_term_memory: bool = True,
        long_term_memory_size: int = 2048
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_memory_heads = num_memory_heads
        
        # Memory types
        self.use_episodic_memory = use_episodic_memory
        self.episodic_memory_size = episodic_memory_size
        self.use_working_memory = use_working_memory
        self.working_memory_size = working_memory_size
        self.use_long_term_memory = use_long_term_memory
        self.long_term_memory_size = long_term_memory_size
        
        # Initialize memory stores
        if use_episodic_memory:
            self.episodic_memory = nn.Parameter(
                torch.zeros(episodic_memory_size, hidden_size)
            )
            self.episodic_memory_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_memory_heads,
                dropout=dropout,
                batch_first=True
            )
            
        if use_working_memory:
            self.working_memory = nn.Parameter(
                torch.zeros(working_memory_size, hidden_size)
            )
            self.working_memory_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_memory_heads,
                dropout=dropout,
                batch_first=True
            )
            
        if use_long_term_memory:
            self.long_term_memory = nn.Parameter(
                torch.zeros(long_term_memory_size, hidden_size)
            )
            self.long_term_memory_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_memory_heads,
                dropout=dropout,
                batch_first=True
            )
            
        # Memory controllers
        self.memory_controller = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Memory update gate
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Memory router
        self.memory_router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 3 if use_episodic_memory and use_working_memory and use_long_term_memory else 
                      2 if (use_episodic_memory and use_working_memory) or 
                           (use_episodic_memory and use_long_term_memory) or 
                           (use_working_memory and use_long_term_memory) else 1)
        )
        
        # Initialize memory state
        self.is_initialized = False
        
    def initialize(self):
        """Initialize memory states"""
        if not self.is_initialized:
            # Initialize with small random values
            if self.use_episodic_memory:
                nn.init.normal_(self.episodic_memory, mean=0.0, std=0.02)
                
            if self.use_working_memory:
                nn.init.normal_(self.working_memory, mean=0.0, std=0.02)
                
            if self.use_long_term_memory:
                nn.init.normal_(self.long_term_memory, mean=0.0, std=0.02)
                
            self.is_initialized = True
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Access and update memory based on current hidden states
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            memory_output: Hidden states enhanced with memory
            memory_info: Dictionary containing memory access information
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Ensure memory is initialized
        if not self.is_initialized:
            self.initialize()
            
        # Prepare memory info dictionary
        memory_info = {}
        
        # Route query to appropriate memory stores
        if hasattr(self, 'memory_router') and self.memory_router is not None:
            memory_routing = self.memory_router(hidden_states.mean(dim=1))
            memory_routing = F.softmax(memory_routing, dim=-1)
            memory_info['memory_routing'] = memory_routing
        else:
            # Default routing weights if no router
            if self.use_episodic_memory and self.use_working_memory and self.use_long_term_memory:
                memory_routing = torch.tensor([0.4, 0.4, 0.2], device=device).expand(batch_size, 3)
            elif (self.use_episodic_memory and self.use_working_memory) or \
                 (self.use_episodic_memory and self.use_long_term_memory) or \
                 (self.use_working_memory and self.use_long_term_memory):
                memory_routing = torch.tensor([0.5, 0.5], device=device).expand(batch_size, 2)
            else:
                memory_routing = torch.tensor([1.0], device=device).expand(batch_size, 1)
        
        # Access episodic memory
        episodic_output = None
        if self.use_episodic_memory:
            # Expand memory for batch size
            episodic_memory = self.episodic_memory.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Query episodic memory
            episodic_output, episodic_attn = self.episodic_memory_attention(
                hidden_states, episodic_memory, episodic_memory,
                key_padding_mask=None
            )
            memory_info['episodic_attention'] = episodic_attn
            
        # Access working memory
        working_output = None
        if self.use_working_memory:
            # Expand memory for batch size
            working_memory = self.working_memory.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Query working memory
            working_output, working_attn = self.working_memory_attention(
                hidden_states, working_memory, working_memory,
                key_padding_mask=None
            )
            memory_info['working_attention'] = working_attn
            
        # Access long-term memory
        long_term_output = None
        if self.use_long_term_memory:
            # Expand memory for batch size
            long_term_memory = self.long_term_memory.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Query long-term memory
            long_term_output, long_term_attn = self.long_term_memory_attention(
                hidden_states, long_term_memory, long_term_memory,
                key_padding_mask=None
            )
            memory_info['long_term_attention'] = long_term_attn
            
        # Combine memory outputs based on routing
        memory_output = hidden_states.clone()
        routing_idx = 0
        
        if self.use_episodic_memory and episodic_output is not None:
            if memory_routing.size(1) > routing_idx:
                episodic_weight = memory_routing[:, routing_idx].view(batch_size, 1, 1)
                memory_output = memory_output + episodic_output * episodic_weight
                routing_idx += 1
            
        if self.use_working_memory and working_output is not None:
            if memory_routing.size(1) > routing_idx:
                working_weight = memory_routing[:, routing_idx].view(batch_size, 1, 1)
                memory_output = memory_output + working_output * working_weight
                routing_idx += 1
            
        if self.use_long_term_memory and long_term_output is not None:
            if memory_routing.size(1) > routing_idx:
                long_term_weight = memory_routing[:, routing_idx].view(batch_size, 1, 1)
                memory_output = memory_output + long_term_output * long_term_weight
                
        # Apply memory controller
        memory_output = self.memory_controller(memory_output)
        
        # Update memories if in training mode
        if self.training:
            # Compute update gate values
            update_input = torch.cat([hidden_states, memory_output], dim=-1)
            update_gate_values = self.update_gate(update_input)
            memory_info['update_gate'] = update_gate_values.mean().item()
            
            # Update episodic memory
            if self.use_episodic_memory:
                # Compute attention weights for memory update
                update_weights = torch.matmul(
                    hidden_states.mean(dim=1, keepdim=True), 
                    self.episodic_memory.transpose(0, 1)
                )
                update_weights = F.softmax(update_weights / math.sqrt(self.hidden_size), dim=-1)
                
                # Compute memory updates
                memory_updates = torch.matmul(
                    update_weights.transpose(1, 2),
                    hidden_states.mean(dim=1, keepdim=True)
                ).squeeze(1)
                
                # Apply updates with gate
                self.episodic_memory.data = self.episodic_memory.data * (1 - update_gate_values.mean().item()) + \
                                           memory_updates.mean(dim=0) * update_gate_values.mean().item()
                
            # Similar updates for working and long-term memory would go here
            # ...
            
        return memory_output, memory_info
        
    def reset_episodic_memory(self):
        """Reset episodic memory to initial state"""
        if self.use_episodic_memory:
            nn.init.normal_(self.episodic_memory, mean=0.0, std=0.02)
            
    def reset_working_memory(self):
        """Reset working memory to initial state"""
        if self.use_working_memory:
            nn.init.normal_(self.working_memory, mean=0.0, std=0.02)
            
    def reset_all_memory(self):
        """Reset all memory stores to initial state"""
        self.reset_episodic_memory()
        self.reset_working_memory()
        if self.use_long_term_memory:
            nn.init.normal_(self.long_term_memory, mean=0.0, std=0.02)

class CacheManager(nn.Module):
    """
    Manages key-value caches for efficient inference and attention computation.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_cache_size: int = 2048,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_cache_size = max_cache_size
        self.use_flash_attention = use_flash_attention
        
        # Cache state
        self.key_cache = None
        self.value_cache = None
        self.cache_length = 0
        
        # Cache management
        self.cache_enabled = True
        
        # Cache pruning parameters
        self.prune_threshold = 0.1
        self.importance_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize state
        self.is_initialized = False
        
    def initialize(self):
        """Initialize cache manager"""
        if not self.is_initialized:
            # Nothing specific to initialize here
            self.is_initialized = True
        
    def forward(self, hidden_states: torch.Tensor, layer_idx: int, is_self_attn: bool = True):
        """
        Process hidden states through cache
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            layer_idx: Index of the current layer
            is_self_attn: Whether this is for self-attention (vs cross-attention)
            
        Returns:
            cached_states: Processed hidden states with cache information
        """
        if not self.cache_enabled:
            return hidden_states
            
        batch_size, seq_len, _ = hidden_states.shape
        
        # Initialize cache if needed
        if self.key_cache is None or self.value_cache is None:
            head_dim = self.hidden_size // self.num_heads
            self.key_cache = torch.zeros(
                batch_size, self.num_layers, self.max_cache_size, self.num_heads, head_dim,
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            self.value_cache = torch.zeros(
                batch_size, self.num_layers, self.max_cache_size, self.num_heads, head_dim,
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            self.cache_length = 0
            
        # Check if cache needs pruning
        if self.cache_length + seq_len > self.max_cache_size:
            self._prune_cache(hidden_states)
            
        # Update cache with new hidden states
        # (This is a placeholder - actual implementation would depend on how keys/values are computed)
        
        # Return processed hidden states
        return hidden_states
        
    def _prune_cache(self, hidden_states: torch.Tensor):
        """
        Prune cache when it exceeds maximum size
        
        Args:
            hidden_states: Current hidden states
        """
        if self.cache_length == 0:
            return
            
        # Compute importance scores for cached items
        # (This is a placeholder - actual implementation would compute importance)
        importance = torch.rand(self.cache_length, device=hidden_states.device)
        
        # Keep most important items
        keep_size = int(self.max_cache_size * 0.8)  # Keep 80% of max size
        _, keep_indices = torch.topk(importance, keep_size)
        
        # Create new cache with only important items
        new_key_cache = torch.zeros_like(self.key_cache)
        new_value_cache = torch.zeros_like(self.value_cache)
        
        for i, idx in enumerate(keep_indices):
            new_key_cache[:, :, i] = self.key_cache[:, :, idx]
            new_value_cache[:, :, i] = self.value_cache[:, :, idx]
            
        # Update cache
        self.key_cache = new_key_cache
        self.value_cache = new_value_cache
        self.cache_length = keep_size
        
    def reset_cache(self):
        """Reset the KV cache"""
        self.key_cache = None
        self.value_cache = None
        self.cache_length = 0
        
    def enable_cache(self):
        """Enable caching"""
        self.cache_enabled = True
        
    def disable_cache(self):
        """Disable caching"""
        self.cache_enabled = False
        self.reset_cache()

# Add other memory-related classes here
# ...