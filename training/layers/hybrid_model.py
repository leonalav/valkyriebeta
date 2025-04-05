import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import GPT2Block, GPT2Config

from .rwkv_layer import RWKVBlock, RWKVModel

class LayerRouter(nn.Module):
    """
    Dynamic router that determines which layer type (RWKV or Transformer) to prioritize
    based on input characteristics. This version includes safeguards for error handling.
    """
    
    def __init__(self, hidden_size, num_layers, adaptation_factor=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.adaptation_factor = adaptation_factor
        
        # Learnable mixing coefficients for each layer
        self.layer_coefficients = nn.Parameter(torch.ones(num_layers, 2))  # [RWKV, Transformer] weights
        
        # Task complexity analyzer
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_layers * 2)  # Predict weights for each layer
        )
        
        # Sparsity controller
        self.sparsity_controller = nn.Parameter(torch.ones(1) * 0.5)  # Initial 50% sparsity
        
        # Initialize with bias toward RWKV for efficiency
        with torch.no_grad():
            self.layer_coefficients[:, 0] = 0.6  # Bias toward RWKV
            self.layer_coefficients[:, 1] = 0.4  # Lower initial weight for transformer
            
        # Fallback active layers (used if analysis fails)
        self.fallback_active_layers = torch.ones(num_layers, dtype=torch.bool)
    
    def forward(self, x):
        """
        Analyze input and determine layer routing coefficients with safeguards
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            coefficients: Layer coefficients [num_layers, 2]
            active_layers: Boolean mask of which layers to activate [num_layers]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Sanity check for input dimensions
        if hidden_dim != self.hidden_size:
            raise ValueError(f"Input hidden dimension ({hidden_dim}) doesn't match expected ({self.hidden_size})")
        
        try:
            # Analyze input complexity from global representation
            # Use mean pooling to get sequence representation (safely handle NaN/Inf values)
            pooled = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).mean(dim=1)  # [batch_size, hidden_size]
            
            # Check for NaN or Inf values in pooled representation
            if torch.isnan(pooled).any() or torch.isinf(pooled).any():
                # Fall back to simple averaging across all layers
                return self._get_fallback_coefficients()
            
            # Predict task complexity factors
            task_factors = self.complexity_analyzer(pooled)  # [batch_size, num_layers * 2]
            
            # Check for numerical issues
            if torch.isnan(task_factors).any() or torch.isinf(task_factors).any():
                return self._get_fallback_coefficients()
                
            task_factors = task_factors.mean(dim=0)  # Average over batch
            task_factors = task_factors.view(self.num_layers, 2)
            
            # Get adaptive coefficients by combining learned weights with task factors
            adaptive_coefficients = self.layer_coefficients + self.adaptation_factor * task_factors
            
            # Apply softmax to normalize weights between RWKV and Transformer per layer
            coefficients = F.softmax(adaptive_coefficients, dim=-1)
            
            # Create sparse activation mask based on importance
            importance = coefficients.sum(dim=-1)  # Sum of coefficients per layer
            
            # Dynamically set sparsity threshold based on controller
            sparsity = torch.sigmoid(self.sparsity_controller)
            
            # Always keep at least 25% of layers active (minimum layers = max(1, 25% of num_layers))
            min_active_layers = max(1, int(0.25 * self.num_layers))
            k = max(min_active_layers, int((1 - sparsity) * self.num_layers))
            
            # Get indices of top-k important layers
            _, indices = torch.topk(importance, k)
            
            # Create sparse activation mask - only top-k important layers are active
            active_layers = torch.zeros(self.num_layers, dtype=torch.bool, device=coefficients.device)
            active_layers[indices] = True
            
            # Always keep the first and last layers active for stability
            active_layers[0] = True
            active_layers[-1] = True
            
            return coefficients, active_layers
            
        except Exception as e:
            # If any error occurs, fall back to a simple activation pattern
            return self._get_fallback_coefficients()
    
    def _get_fallback_coefficients(self):
        """Provide fallback coefficients if dynamic analysis fails"""
        device = self.layer_coefficients.device
        
        # Generate constant coefficients with slight bias toward RWKV for efficiency
        coefficients = torch.ones(self.num_layers, 2, device=device)
        coefficients[:, 0] = 0.6  # RWKV
        coefficients[:, 1] = 0.4  # Transformer
        coefficients = F.softmax(coefficients, dim=-1)
        
        # Simple alternating pattern keeping only every other layer active, but always keep first and last
        active_layers = torch.zeros(self.num_layers, dtype=torch.bool, device=device)
        active_layers[0::2] = True  # Activate every other layer
        active_layers[0] = True     # Always activate first layer
        active_layers[-1] = True    # Always activate last layer
        
        return coefficients, active_layers

class TransformerBlock(nn.Module):
    """Standard transformer block with self-attention"""
    
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        
        # Create GPT-2 style block
        gpt_config = GPT2Config(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=getattr(config, 'ffn_hidden_size', 4 * config.hidden_size),
            activation_function=getattr(config, 'activation_function', 'gelu_new'),
            resid_dropout=getattr(config, 'dropout', 0.1),
            attn_dropout=getattr(config, 'attention_dropout', 0.1),
            layer_norm_epsilon=getattr(config, 'layer_norm_epsilon', 1e-5),
            use_cache=True,
        )
        
        # Use transformer block from Hugging Face
        self.block = GPT2Block(gpt_config)
    
    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        """
        Forward pass for transformer block
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            layer_past: Key-value cache for attention
            attention_mask: Optional attention mask
            head_mask: Optional head mask
            
        Returns:
            output: Processed tensor
            present: Updated key-value cache
        """
        # Prepare attention mask if not None
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Call transformer block
        outputs = self.block(
            x,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=True,
        )
        
        # Unpack outputs
        hidden_states, present = outputs[:2]
        
        return hidden_states, present


class HybridRWKVTransformerModel(nn.Module):
    """
    Hybrid model using both RWKV and Transformer layers with dynamic routing
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.hidden_size))
        
        # Layer config
        self.num_layers = config.num_layers
        
        # Create both RWKV and Transformer layers for each position
        self.rwkv_layers = nn.ModuleList([
            RWKVBlock(config, i) for i in range(config.num_layers)
        ])
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])
        
        # Dynamic router
        self.layer_router = LayerRouter(
            config.hidden_size, 
            config.num_layers,
            adaptation_factor=getattr(config, 'router_adaptation_factor', 0.1)
        )
        
        # Layer mixing coefficients - how to combine RWKV and Transformer outputs
        self.layer_mixer = nn.Parameter(torch.ones(config.num_layers, 2))  # [RWKV, Transformer] weights
        
        # Layer norm and output
        self.ln_out = nn.LayerNorm(config.hidden_size, eps=getattr(config, 'layer_norm_epsilon', 1e-5))
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Default indices for static routing (used if dynamic routing is disabled)
        self.rwkv_layer_indices = list(range(0, config.num_layers, 2))  # Even layers
        self.transformer_layer_indices = list(range(1, config.num_layers, 2))  # Odd layers
        
        # State for recurrent processing
        self.rwkv_states = None
        self.transformer_past_keys = None
        self.chunk_size = getattr(config, 'rwkv_chunk_size', 1024)
        
        # Initialize with learned positional embeddings
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly"""
        # Initialize embedding
        nn.init.normal_(self.emb.weight, std=0.02)
        
        # Initialize positional embeddings with sinusoidal pattern
        position = torch.arange(0, self.config.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2) * -(math.log(10000.0) / self.hidden_size))
        pos_emb = torch.zeros(1, self.config.max_seq_len, self.hidden_size)
        pos_emb[0, :, 0::2] = torch.sin(position * div_term)
        pos_emb[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_emb.data = pos_emb
        
        # Initialize layer mixer with slight bias toward RWKV layers
        nn.init.ones_(self.layer_mixer)
        self.layer_mixer.data[:, 0] = 0.6  # RWKV gets 60% weight by default
        self.layer_mixer.data[:, 1] = 0.4  # Transformer gets 40% weight by default
    
    def reset_state(self, batch_size=1):
        """Reset recurrent state"""
        device = next(self.parameters()).device
        
        # Reset RWKV states
        self.rwkv_states = {}
        for i in range(self.num_layers):
            self.rwkv_states[i] = (
                torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device)
            )
        
        # Reset transformer KV cache
        self.transformer_past_keys = {}
    
    def set_chunk_size(self, chunk_size):
        """Set chunk size for processing"""
        self.chunk_size = chunk_size
    
    def process_with_state(self, input_ids, states=None, use_dynamic_routing=True):
        """
        Process input with optional state
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            states: Optional previous states (rwkv_states, transformer_past_keys)
            use_dynamic_routing: Whether to use dynamic routing or static routing
            
        Returns:
            outputs: Model outputs
            new_states: Updated states
        """
        batch_size, seq_len = input_ids.size()
        
        # Initialize states if None
        if states is None:
            self.reset_state(batch_size)
            rwkv_states = self.rwkv_states
            transformer_past_keys = self.transformer_past_keys
        else:
            rwkv_states, transformer_past_keys = states
        
        # Get embeddings and add position embeddings
        x = self.emb(input_ids)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_emb[:, :seq_len]
        x = x + pos_emb
        
        # Process through blocks using dynamic or static routing
        new_rwkv_states = {}
        new_transformer_past_keys = {}
        
        # Create attention mask for transformer layers
        attention_mask = None
        if seq_len > 1:
            # Causal mask
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device) * -10000.0,
                diagonal=1
            )
            attention_mask = attention_mask.unsqueeze(0)
        
        # Determine routing based on input content if using dynamic routing
        if use_dynamic_routing:
            layer_coefficients, active_layers = self.layer_router(x)
        else:
            # Use static routing with alternating layers
            layer_coefficients = F.softmax(self.layer_mixer, dim=-1)
            active_layers = torch.ones(self.num_layers, dtype=torch.bool, device=x.device)
        
        # Process through layers
        for i in range(self.num_layers):
            # Skip inactive layers
            if not active_layers[i]:
                # Propagate previous state
                if i in rwkv_states:
                    new_rwkv_states[i] = rwkv_states[i]
                if i in transformer_past_keys:
                    new_transformer_past_keys[i] = transformer_past_keys[i]
                continue
                
            # Get layer coefficients
            rwkv_coef, transformer_coef = layer_coefficients[i]
            
            # Process with both layer types and combine with learned coefficients
            rwkv_out = None
            transformer_out = None
            
            # Only compute RWKV if coefficient is significant
            if rwkv_coef > 0.01:  # Threshold to save computation
                rwkv_state = rwkv_states.get(i, None)
                rwkv_out, new_rwkv_state = self.rwkv_layers[i](x, rwkv_state)
                new_rwkv_states[i] = new_rwkv_state
            
            # Only compute Transformer if coefficient is significant
            if transformer_coef > 0.01:  # Threshold to save computation
                layer_past = transformer_past_keys.get(i, None)
                transformer_out, present = self.transformer_layers[i](
                    x, layer_past, attention_mask
                )
                new_transformer_past_keys[i] = present
            
            # Combine outputs with learned coefficients
            if rwkv_out is not None and transformer_out is not None:
                x = rwkv_coef * rwkv_out + transformer_coef * transformer_out
            elif rwkv_out is not None:
                x = rwkv_out
            elif transformer_out is not None:
                x = transformer_out
        
        # Output projection
        x = self.ln_out(x)
        logits = self.head(x)
        
        # Return logits and new states
        new_states = (new_rwkv_states, new_transformer_past_keys)
        return logits, new_states
    
    def forward(self, input_ids, labels=None, attention_mask=None, use_chunking=False, use_dynamic_routing=True):
        """
        Forward pass
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Optional labels for computing loss
            attention_mask: Optional attention mask (for transformer layers)
            use_chunking: Whether to process in chunks for memory efficiency
            use_dynamic_routing: Whether to use dynamic layer routing
            
        Returns:
            outputs: Model outputs with loss if labels provided
        """
        batch_size, seq_len = input_ids.size()
        
        if use_chunking and seq_len > self.chunk_size:
            return self.forward_chunked(input_ids, labels, attention_mask, use_dynamic_routing)
        
        # Process normally
        logits, _ = self.process_with_state(input_ids, None, use_dynamic_routing)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        return type('HybridOutput', (), {'loss': loss, 'logits': logits})()
    
    def forward_chunked(self, input_ids, labels=None, attention_mask=None, use_dynamic_routing=True):
        """
        Forward pass with chunking for memory efficiency
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Optional labels for computing loss
            attention_mask: Optional attention mask (for transformer layers)
            use_dynamic_routing: Whether to use dynamic layer routing
            
        Returns:
            outputs: Model outputs with loss if labels provided
        """
        batch_size, seq_len = input_ids.size()
        chunk_size = self.chunk_size
        
        # Reset state
        self.reset_state(batch_size)
        
        # Process in chunks
        all_logits = []
        total_loss = 0.0
        
        # Store cross-layer feedback for enhancing RWKV with transformer hierarchical awareness
        hierarchical_feedback = None
        
        # Process sequence in chunks
        for i in range(0, seq_len, chunk_size):
            # Get current chunk
            end_idx = min(i + chunk_size, seq_len)
            chunk_ids = input_ids[:, i:end_idx]
            
            # Get chunk labels if provided
            chunk_labels = None
            if labels is not None:
                chunk_labels = labels[:, i:end_idx]
            
            # Get chunk attention mask if provided
            chunk_attention_mask = None
            if attention_mask is not None:
                chunk_attention_mask = attention_mask[:, i:end_idx]
            
            # Process chunk with state
            states = (self.rwkv_states, self.transformer_past_keys)
            chunk_out = self._process_chunk_with_feedback(
                chunk_ids, 
                states, 
                hierarchical_feedback,
                use_dynamic_routing
            )
            
            # Update states and extract outputs
            logits = chunk_out.logits
            self.rwkv_states = chunk_out.rwkv_states
            self.transformer_past_keys = chunk_out.transformer_past_keys
            hierarchical_feedback = chunk_out.hierarchical_feedback
            
            # Compute loss if labels provided
            if chunk_labels is not None:
                # Shift logits and labels for next token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = chunk_labels[:, 1:].contiguous()
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss()
                chunk_loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
                total_loss += chunk_loss.item() * (end_idx - i - 1)  # Weight by sequence length
            
            # Collect logits
            all_logits.append(logits)
        
        # Concatenate all logits
        if all_logits:
            all_logits = torch.cat(all_logits, dim=1)
        else:
            all_logits = torch.zeros(batch_size, 0, self.vocab_size, device=input_ids.device)
        
        # Compute average loss if labels provided
        loss = None
        if labels is not None:
            loss = total_loss / (seq_len - 1)  # Average loss over sequence
        
        return type('HybridOutput', (), {'loss': loss, 'logits': all_logits})()
    
    def _process_chunk_with_feedback(self, input_ids, states, hierarchical_feedback=None, use_dynamic_routing=True):
        """
        Process a chunk with hierarchical feedback for enhanced RWKV performance
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            states: Previous states (rwkv_states, transformer_past_keys)
            hierarchical_feedback: Optional hierarchical feedback from transformer layers
            use_dynamic_routing: Whether to use dynamic routing
            
        Returns:
            output: Object containing logits, states, and hierarchical feedback
        """
        rwkv_states, transformer_past_keys = states
        batch_size, seq_len = input_ids.size()
        
        # Get embeddings and add position embeddings
        x = self.emb(input_ids)
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_emb[:, :seq_len]
        x = x + pos_emb
        
        # Apply hierarchical feedback if available (enhances RWKV's hierarchical understanding)
        if hierarchical_feedback is not None:
            # Scale down feedback to avoid dominating the signal
            x = x + 0.1 * hierarchical_feedback
        
        # Create attention mask for transformer layers
        attention_mask = None
        if seq_len > 1:
            # Causal mask
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device) * -10000.0,
                diagonal=1
            )
            attention_mask = attention_mask.unsqueeze(0)
        
        # Determine routing based on input content if using dynamic routing
        if use_dynamic_routing:
            layer_coefficients, active_layers = self.layer_router(x)
        else:
            # Use static routing with alternating layers
            layer_coefficients = F.softmax(self.layer_mixer, dim=-1)
            active_layers = torch.ones(self.num_layers, dtype=torch.bool, device=x.device)
        
        # Store intermediate outputs for hierarchical feedback
        transformer_intermediates = []
        rwkv_intermediates = []
        
        # Process through layers
        new_rwkv_states = {}
        new_transformer_past_keys = {}
        
        # Process through layers
        for i in range(self.num_layers):
            # Skip inactive layers
            if not active_layers[i]:
                # Propagate previous state
                if i in rwkv_states:
                    new_rwkv_states[i] = rwkv_states[i]
                if i in transformer_past_keys:
                    new_transformer_past_keys[i] = transformer_past_keys[i]
                continue
                
            # Get layer coefficients
            rwkv_coef, transformer_coef = layer_coefficients[i]
            
            # Process with both layer types and combine with learned coefficients
            rwkv_out = None
            transformer_out = None
            
            # Only compute RWKV if coefficient is significant
            if rwkv_coef > 0.01:  # Threshold to save computation
                rwkv_state = rwkv_states.get(i, None)
                
                # Apply hierarchical information from previous transformer layer if available
                if i > 0 and len(transformer_intermediates) > 0:
                    # Enhanced input with hierarchical information from transformer
                    enhanced_x = x
                    if len(transformer_intermediates) > 0:
                        # Get last transformer output and use it to enhance RWKV's hierarchical understanding
                        last_transformer = transformer_intermediates[-1]
                        # Add a residual connection with a small weight to avoid dominating the signal
                        enhanced_x = x + 0.1 * last_transformer
                        
                    rwkv_out, new_rwkv_state = self.rwkv_layers[i](enhanced_x, rwkv_state)
                else:
                    # Standard processing without enhancement
                    rwkv_out, new_rwkv_state = self.rwkv_layers[i](x, rwkv_state)
                    
                new_rwkv_states[i] = new_rwkv_state
                rwkv_intermediates.append(rwkv_out)
            
            # Only compute Transformer if coefficient is significant
            if transformer_coef > 0.01:  # Threshold to save computation
                layer_past = transformer_past_keys.get(i, None)
                
                # Enhanced transformer input with RWKV temporal information if available
                enhanced_x = x
                if i > 0 and len(rwkv_intermediates) > 0:
                    # Get weighted sum of previous RWKV outputs for temporal context
                    last_rwkv = rwkv_intermediates[-1]
                    # Add as a parallel input stream
                    enhanced_x = x + 0.05 * last_rwkv
                
                transformer_out, present = self.transformer_layers[i](
                    enhanced_x, layer_past, attention_mask
                )
                new_transformer_past_keys[i] = present
                transformer_intermediates.append(transformer_out)
            
            # Combine outputs with learned coefficients
            if rwkv_out is not None and transformer_out is not None:
                x = rwkv_coef * rwkv_out + transformer_coef * transformer_out
            elif rwkv_out is not None:
                x = rwkv_out
            elif transformer_out is not None:
                x = transformer_out
        
        # Calculate hierarchical feedback for next chunk based on transformer layers
        # This helps RWKV layers learn more hierarchical representations
        new_hierarchical_feedback = None
        if len(transformer_intermediates) > 0:
            new_hierarchical_feedback = transformer_intermediates[-1].detach()
        
        # Output projection
        x = self.ln_out(x)
        logits = self.head(x)
        
        # Return combined output
        return type('ChunkOutput', (), {
            'logits': logits, 
            'rwkv_states': new_rwkv_states, 
            'transformer_past_keys': new_transformer_past_keys,
            'hierarchical_feedback': new_hierarchical_feedback
        })()
    
    def enable_state_compression(self):
        """Enable compression of states for memory efficiency"""
        # Implement state compression techniques here
        self.using_state_compression = True
    
    def optimize_for_inference(self):
        """Apply optimizations for inference time"""
        # Can implement various inference optimizations here
        # e.g., fusing operations, etc.
        self.optimized_for_inference = True
    
    def quantize_time_mixing(self):
        """Quantize time mixing parameters for more efficient inference"""
        for i, (block_type, block) in enumerate(self.blocks):
            if block_type == 'rwkv':
                # Quantize RWKV parameters
                if hasattr(block.att, 'time_decay'):
                    block.att.time_decay.data = torch.round(block.att.time_decay.data * 100) / 100
                if hasattr(block.att, 'time_mix_r'):
                    block.att.time_mix_r.data = torch.round(block.att.time_mix_r.data * 100) / 100
                if hasattr(block.att, 'time_mix_k'):
                    block.att.time_mix_k.data = torch.round(block.att.time_mix_k.data * 100) / 100
                if hasattr(block.att, 'time_mix_v'):
                    block.att.time_mix_v.data = torch.round(block.att.time_mix_v.data * 100) / 100 