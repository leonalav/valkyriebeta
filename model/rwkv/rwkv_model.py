import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any, Union

from .rwkv_layer import RWKVConfig, RWKVBlock
from model.transformer import TransformerConfig, TransformerLayer

class RWKVModel(nn.Module):
    """Complete RWKV model implementation with advanced features"""
    def __init__(
        self, 
        config: RWKVConfig,
        vocab_size: int = 50000,
        max_seq_length: int = 2048
    ):
        super().__init__()
        self.config = config
        
        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, config.hidden_size)
        
        # Position embeddings (optional)
        self.use_position_embeddings = getattr(config, 'use_position_embeddings', False)
        if self.use_position_embeddings:
            self.position_embedding = nn.Embedding(max_seq_length, config.hidden_size)
            self.position_dropout = nn.Dropout(config.dropout)
        
        # RWKV layers
        self.layers = nn.ModuleList([
            RWKVBlock(config, i) for i in range(config.num_layers)
        ])
        
        # Output layer normalization
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output head
        self.output = nn.Linear(config.hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Optional state for recurrent processing
        self.states = None
        
        # Chunking for memory efficiency
        self.use_chunking = getattr(config, 'use_chunking', False)
        self.chunk_size = getattr(config, 'chunk_size', 1024)
        self.chunk_overlap = getattr(config, 'chunk_overlap', self.chunk_size // 8)
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
    
    def _init_weights(self, module):
        """Initialize weights with scaled initialization"""
        if isinstance(module, nn.Linear):
            # Apply scaled initialization
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def init_states(self, batch_size: int = 1, device = None):
        """Initialize recurrent states"""
        if device is None:
            device = next(self.parameters()).device
            
        self.states = []
        for i, layer in enumerate(self.layers):
            # Initialize states for each layer
            k_state = torch.zeros(batch_size, self.config.hidden_size, device=device)
            v_state = torch.zeros(batch_size, self.config.hidden_size, device=device)
            self.states.append((k_state, v_state))
    
    def clear_states(self):
        """Clear recurrent states"""
        self.states = None
    
    def forward(
        self, 
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_states: bool = False,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for the RWKV model"""
        # Handle chunking for long sequences
        if self.use_chunking and input_ids.size(1) > self.chunk_size:
            return self.forward_chunked(input_ids, attention_mask, position_ids, use_states, return_dict)
            
        # Get embedding
        x = self.word_embedding(input_ids)
        
        # Add position embeddings if enabled
        if self.use_position_embeddings:
            if position_ids is None:
                position_ids = torch.arange(
                    input_ids.size(1), device=input_ids.device
                ).unsqueeze(0).expand_as(input_ids)
            position_embeddings = self.position_embedding(position_ids)
            x = x + position_embeddings
            x = self.position_dropout(x)
        
        # Initialize states if needed
        batch_size = input_ids.size(0)
        if use_states and self.states is None:
            self.init_states(batch_size, input_ids.device)
        
        # Process through RWKV layers
        new_states = []
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory
                def create_custom_forward(module, layer_idx):
                    def custom_forward(*inputs):
                        # Get the first input which is x
                        layer_x = inputs[0]
                        # Get state for this layer if using states
                        layer_state = self.states[layer_idx] if use_states and self.states else None
                        return module(layer_x, layer_state)
                    return custom_forward
                
                x, layer_state = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer, i),
                    x
                )
            else:
                # Standard forward pass
                if use_states:
                    x, layer_state = layer(x, self.states[i] if self.states else None)
                else:
                    x, layer_state = layer(x, None)
                    
            new_states.append(layer_state)
        
        # Update states if using recurrence
        if use_states:
            self.states = new_states
        
        # Apply final layer norm
        x = self.final_layer_norm(x)
        
        # Get logits
        logits = self.output(x)
        
        # Return as dict if requested
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": x,
                "states": new_states if not use_states else None
            }
        
        return logits, new_states
    
    def forward_chunked(
        self, 
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_states: bool = False,
        return_dict: bool = True,
        past_key_values: Optional[List[Any]] = None
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, List]]:
        """Process long sequences in chunks with optimized overlap handling"""
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        chunk_size = self.chunk_size
        overlap = min(self.chunk_overlap, chunk_size // 2)
        
        # Reset state if using states and not continuing from past
        if use_states and (past_key_values is None or not any(past_key_values)):
            self.clear_states()
            self._ensure_states_initialized(batch_size, device)
        
        # Process in chunks with shared state
        all_logits = []
        all_hidden_states = [] if return_dict else None
        
        # Prepare position ids if needed
        if self.use_position_embeddings and position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Use tensor ops for efficient chunking when possible
        if self.use_fused_ops and hasattr(torch, 'chunk'):
            # Try to use more efficient chunk processing
            chunks = seq_len // (chunk_size - overlap) + (1 if seq_len % (chunk_size - overlap) > 0 else 0)
            
            # Process each chunk with state propagation
            for i in range(chunks):
                chunk_start = i * (chunk_size - overlap)
                chunk_end = min(chunk_start + chunk_size, seq_len)
                
                # Get current chunk data
                chunk_input_ids = input_ids[:, chunk_start:chunk_end]
                
                # Get position ids for this chunk if needed
                chunk_position_ids = None
                if self.use_position_embeddings and position_ids is not None:
                    chunk_position_ids = position_ids[:, chunk_start:chunk_end]
                
                # Process chunk
                chunk_attention_mask = None
                if attention_mask is not None:
                    chunk_attention_mask = attention_mask[:, chunk_start:chunk_end]
                
                # Forward through model
                if return_dict:
                    chunk_output = self.forward(
                        chunk_input_ids, 
                        attention_mask=chunk_attention_mask,
                        position_ids=chunk_position_ids,
                        use_states=use_states,  # Use propagated state
                        return_dict=True,
                        past_key_values=None  # No past key values in chunked mode
                    )
                    chunk_logits = chunk_output["logits"]
                    if all_hidden_states is not None:
                        all_hidden_states.append(chunk_output.get("hidden_states"))
                else:
                    chunk_logits, _ = self.forward(
                        chunk_input_ids, 
                        attention_mask=chunk_attention_mask,
                        position_ids=chunk_position_ids,
                        use_states=use_states,  # Use propagated state
                        return_dict=False
                    )
                
                # For chunks after the first one, remove the overlapped tokens from output
                # but keep state influence from the full chunk
                if i > 0 and overlap > 0:
                    chunk_logits = chunk_logits[:, overlap:]
                
                # For all except the last chunk, trim the overlapping tokens at the end
                if i < chunks - 1 and overlap > 0:
                    chunk_logits = chunk_logits[:, :-(overlap)]
                
                all_logits.append(chunk_logits)
        else:
            # Fallback to original implementation for compatibility
            for chunk_start in range(0, seq_len, chunk_size - overlap):
                chunk_end = min(chunk_start + chunk_size, seq_len)
                chunk_input_ids = input_ids[:, chunk_start:chunk_end]
                
                # Get position ids for this chunk if needed
                chunk_position_ids = None
                if self.use_position_embeddings and position_ids is not None:
                    chunk_position_ids = position_ids[:, chunk_start:chunk_end]
                
                # Process chunk
                chunk_attention_mask = None
                if attention_mask is not None:
                    chunk_attention_mask = attention_mask[:, chunk_start:chunk_end]
                    
                if return_dict:
                    chunk_output = self.forward(
                        chunk_input_ids, 
                        attention_mask=chunk_attention_mask,
                        position_ids=chunk_position_ids,
                        use_states=use_states,
                        return_dict=True
                    )
                    chunk_logits = chunk_output["logits"]
                    if all_hidden_states is not None:
                        all_hidden_states.append(chunk_output.get("hidden_states"))
                else:
                    chunk_logits, _ = self.forward(
                        chunk_input_ids, 
                        attention_mask=chunk_attention_mask,
                        position_ids=chunk_position_ids,
                        use_states=use_states,
                        return_dict=False
                    )
                
                # Handle overlap for output logits
                if chunk_start > 0 and overlap > 0:
                    chunk_logits = chunk_logits[:, overlap:]
                
                if chunk_end < seq_len and overlap > 0:
                    chunk_logits = chunk_logits[:, :-overlap]
                
                all_logits.append(chunk_logits)
        
        # Concatenate all chunk outputs
        if len(all_logits) > 1:
            logits = torch.cat(all_logits, dim=1)
        else:
            logits = all_logits[0]
        
        # Concatenate hidden states if needed
        hidden_states = None
        if all_hidden_states and all(h is not None for h in all_hidden_states):
            hidden_states = torch.cat(all_hidden_states, dim=1)
        
        # Return as dict if requested
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": hidden_states,
                "past_key_values": self.kv_cache if self.kv_cache_enabled else None,
                "states": None  # States are maintained internally
            }
        
        return logits, None
    
    def enable_kv_cache(self):
        """Enable KV caching for efficient autoregressive generation"""
        self.kv_cache_enabled = True
        self.kv_cache = None
        return self
    
    def disable_kv_cache(self):
        """Disable KV caching and clear cache"""
        self.kv_cache_enabled = False
        self.kv_cache = None
        return self
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.gradient_checkpointing = True
        return self
        
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False
        return self
    
    def enable_fused_operations(self):
        """Enable fused operations for better performance"""
        self.use_fused_ops = True
        self._setup_fused_operations()
        return self
    
    def optimize_for_inference(self):
        """Apply optimizations for inference time"""
        # Set evaluation mode
        self.eval()
        
        # Disable dropout
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0
        
        # Enable KV caching for efficient autoregressive generation
        self.enable_kv_cache()
        
        # Enable fused operations
        self.enable_fused_operations()
        
        # Disable gradient computation
        for p in self.parameters():
            p.requires_grad = False
            
        # Enable memory optimizations
        torch.backends.cudnn.benchmark = True
        
        return self

class HybridRWKVTransformerModel(nn.Module):
    """Hybrid model combining RWKV and Transformer layers with enhanced performance"""
    def __init__(
        self,
        transformer_config: TransformerConfig,
        rwkv_config: RWKVConfig,
        vocab_size: int = 50000,
        max_seq_length: int = 2048,
        rwkv_layer_indices: List[int] = None
    ):
        super().__init__()
        self.transformer_config = transformer_config
        self.rwkv_config = rwkv_config
        self.rwkv_layer_indices = rwkv_layer_indices or []
        self.hybrid_model = True  # Flag to identify as hybrid
        
        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, transformer_config.hidden_size)
        
        # Position embedding 
        self.use_position_embeddings = getattr(transformer_config, 'use_position_embeddings', True)
        if self.use_position_embeddings:
            self.position_embedding = nn.Embedding(max_seq_length, transformer_config.hidden_size)
            self.position_dropout = nn.Dropout(transformer_config.dropout)
        
        # Create mixed layers
        self.layers = nn.ModuleList()
        for i in range(transformer_config.num_layers):
            if i in self.rwkv_layer_indices:
                # Add RWKV layer
                self.layers.append(RWKVBlock(rwkv_config, i))
            else:
                # Add Transformer layer
                self.layers.append(TransformerLayer(transformer_config))
        
        # Output layer normalization
        self.final_layer_norm = nn.LayerNorm(transformer_config.hidden_size, eps=transformer_config.layer_norm_eps)
        
        # Output head
        self.output = nn.Linear(transformer_config.hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # States for RWKV layers with proper dimensions
        self.rwkv_states = {}
        self.state_dimensions = self._determine_state_dimensions()
        
        # Gradient checkpointing
        self.gradient_checkpointing = False
        
        # Mixed precision support
        self.use_mixed_precision = getattr(rwkv_config, 'use_mixed_precision', False)
        self.mixed_precision_dtype = getattr(rwkv_config, 'mixed_precision_dtype', torch.float16)
        
        # Performance tracking
        self.metadata = {
            "total_tokens_processed": 0,
            "rwkv_layers_processed": 0,
            "transformer_layers_processed": 0
        }
    
    def _determine_state_dimensions(self):
        """Determine the expected state dimensions for each RWKV layer"""
        x = torch.zeros(1, 1, self.transformer_config.hidden_size)
        dims = {}
        
        # Process through each RWKV layer to get expected state shapes
        for i in self.rwkv_layer_indices:
            layer = self.layers[i]
            _, state = layer(x, None)
            
            # Store type and shape information
            if isinstance(state, tuple):
                dims[i] = [(s.shape if s is not None else None) for s in state]
            else:
                dims[i] = state.shape if state is not None else None
        
        return dims
    
    def _init_weights(self, module):
        """Initialize weights with scaled initialization based on layer type"""
        if isinstance(module, nn.Linear):
            # Use different initialization scale for RWKV vs Transformer
            if hasattr(module, 'is_rwkv_layer') and module.is_rwkv_layer:
                module.weight.data.normal_(mean=0.0, std=0.01)
            else:
                module.weight.data.normal_(mean=0.0, std=0.02)
                
            if module.bias is not None:
                module.bias.data.zero_()
                
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def init_rwkv_states(self, batch_size: int = 1, device = None):
        """Initialize RWKV states with correct dimensions for each layer"""
        if device is None:
            device = next(self.parameters()).device
        
        # Clear existing states
        self.rwkv_states = {}
        
        # Initialize states based on determined dimensions
        for i in self.rwkv_layer_indices:
            # Get dimensions for this layer
            if i in self.state_dimensions and self.state_dimensions[i] is not None:
                dims = self.state_dimensions[i]
                
                if isinstance(dims, list):
                    # Create tuple of states with correct shapes
                    state = tuple(
                        torch.zeros(batch_size, *dim[1:], device=device) 
                        if dim is not None else None 
                        for dim in dims
                    )
                else:
                    # Single state tensor
                    state = torch.zeros(batch_size, *dims[1:], device=device)
            else:
                # Default to standard k,v state initialization
                k_state = torch.zeros(batch_size, self.rwkv_config.hidden_size, device=device)
                v_state = torch.zeros(batch_size, self.rwkv_config.hidden_size, device=device)
                state = (k_state, v_state)
            
            self.rwkv_states[i] = state
    
    def clear_rwkv_states(self):
        """Clear RWKV states"""
        self.rwkv_states = {i: None for i in self.rwkv_layer_indices}
    
    def _ensure_rwkv_states_initialized(self, batch_size, device):
        """Ensure RWKV states are properly initialized"""
        if not self.rwkv_states or any(self.rwkv_states.get(i) is None for i in self.rwkv_layer_indices):
            self.init_rwkv_states(batch_size, device)
    
    def forward(
        self, 
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_rwkv_states: bool = False,
        return_dict: bool = True,
        past_key_values: Optional[List[Any]] = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass for the hybrid model with enhanced performance"""
        # Extract batch dimensions
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Update tracking
        self.metadata["total_tokens_processed"] += batch_size * seq_len
        
        # Mixed precision context
        if self.use_mixed_precision and torch.cuda.is_available():
            mp_ctx = torch.cuda.amp.autocast(dtype=self.mixed_precision_dtype)
        else:
            from contextlib import nullcontext
            mp_ctx = nullcontext()
        
        with mp_ctx:
            # Get word embeddings
            x = self.word_embedding(input_ids)
            
            # Add position embeddings if enabled
            if self.use_position_embeddings:
                if position_ids is None:
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                position_embeddings = self.position_embedding(position_ids)
                x = x + position_embeddings
                x = self.position_dropout(x)
            
            # Initialize RWKV states if needed
            if use_rwkv_states:
                self._ensure_rwkv_states_initialized(batch_size, device)
            
            # Process through layers
            new_rwkv_states = {} if use_rwkv_states else None
            
            for i, layer in enumerate(self.layers):
                if i in self.rwkv_layer_indices:
                    # Process through RWKV layer with proper state handling
                    self.metadata["rwkv_layers_processed"] += 1
                    
                    # Use gradient checkpointing if enabled and training
                    if self.gradient_checkpointing and self.training:
                        def create_custom_forward(module, layer_idx):
                            def custom_forward(*inputs):
                                # Get the first input which is x
                                layer_x = inputs[0]
                                # Get state for this layer
                                layer_state = self.rwkv_states.get(layer_idx) if use_rwkv_states else None
                                return module(layer_x, layer_state)
                            return custom_forward
                        
                        x, new_state = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer, i),
                            x
                        )
                    else:
                        # Standard forward pass
                        state = self.rwkv_states.get(i) if use_rwkv_states else None
                        x, new_state = layer(x, state)
                    
                    # Store new state for this layer
                    if use_rwkv_states:
                        new_rwkv_states[i] = new_state
                
                else:
                    # Process through Transformer layer
                    self.metadata["transformer_layers_processed"] += 1
                    
                    # Use gradient checkpointing if enabled and training
                    if self.gradient_checkpointing and self.training:
                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                # Get the first input which is x
                                layer_x = inputs[0]
                                return module(layer_x, attention_mask)
                            return custom_forward
                        
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(layer),
                            x
                        )
                    else:
                        # All transformer layers should have a forward method
                        # The check was redundant and has been removed
                        x = layer(x, attention_mask)
            
            # Update RWKV states if using
            if use_rwkv_states:
                self.rwkv_states = new_rwkv_states
            
            # Apply final layer norm
            x = self.final_layer_norm(x)
            
            # Get logits
            logits = self.output(x)
        
        # Return as dict if requested
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": x,
                "rwkv_states": new_rwkv_states if not use_rwkv_states else None
            }
        
        return logits
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for better memory efficiency"""
        self.gradient_checkpointing = True
        return self
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        self.gradient_checkpointing = False
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
        # Set evaluation mode
        self.eval()
        
        # Disable dropout
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0
        
        # Enable mixed precision for faster computation
        if torch.cuda.is_available():
            self.enable_mixed_precision()
        
        # Disable gradient computation
        for p in self.parameters():
            p.requires_grad = False
            
        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        return self
    
    def forward_sliding_window(
        self, 
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_rwkv_states: bool = True,
        return_dict: bool = True,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward pass using sliding window for long sequences"""
        # Get dimensions and settings
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Use config defaults if not specified
        chunk_size = chunk_size or self.rwkv_config.rwkv_chunk_size
        chunk_overlap = chunk_overlap or self.rwkv_config.rwkv_chunk_overlap
        
        # Track states for each layer if using RWKV states
        if use_rwkv_states and not hasattr(self, 'rwkv_states'):
            self.reset_rwkv_states(batch_size)
        
        # Prepare result containers
        all_logits = []
        
        # Check if sequence fits in one chunk
        if seq_len <= chunk_size:
            return self.forward(
                input_ids, 
                attention_mask, 
                use_rwkv_states=use_rwkv_states, 
                return_dict=return_dict
            )
        
        # Split sequence into overlapping chunks
        for chunk_start in range(0, seq_len, chunk_size - chunk_overlap):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            
            # Extract current chunk
            chunk_input_ids = input_ids[:, chunk_start:chunk_end]
            
            # Extract corresponding attention mask if provided
            chunk_attention_mask = None
            if attention_mask is not None:
                chunk_attention_mask = attention_mask[:, chunk_start:chunk_end]
            
            # Process chunk
            chunk_outputs = self.forward(
                chunk_input_ids, 
                chunk_attention_mask, 
                use_rwkv_states=use_rwkv_states,
                return_dict=True  # Always return dict for chunks
            )
            
            # Store useful part of logits (excluding overlap with next chunk)
            if chunk_start == 0:
                # For first chunk, keep everything
                useful_end = chunk_end
                all_logits.append(chunk_outputs["logits"])
            else:
                # For subsequent chunks, discard the overlapping beginning
                useful_start = chunk_overlap
                all_logits.append(chunk_outputs["logits"][:, useful_start:])
        
        # Concatenate logits
        logits = torch.cat(all_logits, dim=1)
        
        # Return as dict if requested
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": None,  # Not meaningful for chunked processing
                "rwkv_states": self.rwkv_states if use_rwkv_states else None
            }
        
        return logits
    
    def reset_rwkv_states(self, batch_size=1):
        """Reset RWKV states for a new sequence"""
        device = next(self.parameters()).device
        self.rwkv_states = {}
        
        # Initialize states for each RWKV layer
        for i in self.rwkv_layer_indices:
            if i in self.state_dimensions:
                dims = self.state_dimensions[i]
                if isinstance(dims, list):
                    # Handle tuple states
                    self.rwkv_states[i] = tuple(
                        torch.zeros(batch_size, *dim[1:], device=device) if dim is not None else None
                        for dim in dims
                    )
                else:
                    # Handle tensor states
                    self.rwkv_states[i] = torch.zeros(batch_size, *dims[1:], device=device)
        
        return self
    
    def enable_kv_cache_compression(self):
        """Enable 8-bit quantization for KV cache to save memory"""
        if not hasattr(self, "using_compressed_cache"):
            self.using_compressed_cache = True
            
            # Import quantization utilities if available
            try:
                from bitsandbytes.nn import Linear8bitLt
                have_bnb = True
            except ImportError:
                have_bnb = False
                
            # Apply KV cache compression to transformer layers
            for i, layer in enumerate(self.layers):
                if i not in self.rwkv_layer_indices:  # Is a transformer layer
                    if hasattr(layer, 'enable_8bit_kv_cache'):
                        layer.enable_8bit_kv_cache()
            
            print("Enabled KV cache compression")
        return self 