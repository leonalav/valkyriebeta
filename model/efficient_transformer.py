import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

from .attention import FlashAttention, MultiScaleAttention, AdaptiveSparsityAttention
from .layers import MemoryEfficientLinear, EfficientFeedForward, ParallelFeedForward
from .embedding import EfficientEmbeddingLayer  # Updated import path
from .lora import LoRALinear  # Updated import path
from .normalization import MemoryEfficientLayerNorm
from .memory_bank import MemoryBank

class EfficientAttention(nn.Module):
    """Efficient attention implementation with linear complexity."""
    def __init__(self, config):
        super().__init__() 
        # Use optimized attention implementations
        if hasattr(F, 'scaled_dot_product_attention'):
            self.attention = FlashAttention(config)
        else:
            self.attention = MultiScaleAttention(config)
            
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.attention(x, mask)

class EfficientFeedForwardBlock(nn.Module):
    """Lightweight feed-forward block with optimized activations."""
    def __init__(self, config):
        super().__init__()
        self.ffn = EfficientFeedForward(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class EfficientTransformerBlock(nn.Module):
    """Optimized transformer block with efficient connectivity."""
    def __init__(self, config, linear_class=None):
        super().__init__()
        
        linear_class = linear_class or MemoryEfficientLinear
        
        # Efficient layer normalization
        self.norm1 = MemoryEfficientLayerNorm(config.hidden_size)
        self.norm2 = MemoryEfficientLayerNorm(config.hidden_size)
        
        # Efficient attention with LoRA support
        if config.use_flash_attention:
            self.attention = FlashAttention(config)
        else:
            self.attention = MultiScaleAttention(config)
            
        # Convert attention projections to LoRA if needed
        if linear_class is not LoRALinear:
            self.attention_proj = linear_class(config.hidden_size, config.hidden_size, bias=config.bias)
        else:
            self.attention_proj = linear_class(
                config.hidden_size,
                config.hidden_size,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                bias=config.bias
            )
        
        # Feed-forward with optional LoRA
        if config.use_parallel_ffn:
            self.ffn = ParallelFeedForward(config, linear_class)
        else:
            self.ffn = EfficientFeedForward(config, linear_class)
        
        # Memory-efficient residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1))
        
        # Optional memory bank for enhanced capabilities
        self.use_memory = config.use_memory if hasattr(config, 'use_memory') else False
        if self.use_memory:
            self.memory_bank = MemoryBank(
                memory_size=config.memory_size,
                hidden_size=config.hidden_size,
                num_heads=config.num_heads
            )
            
        # Gradient checkpointing
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(x, attention_mask)
            
        # Efficient residual connections with pre-normalization
        residual = x
        x = self.norm1(x)
        
        # Attention with projection
        attn_out = self.attention(x, attention_mask)
        x = self.attention_proj(attn_out)
        x = residual + x * self.residual_scale
        
        # Optional memory bank processing
        if self.use_memory:
            x = x + self.memory_bank(x)
        
        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x * self.residual_scale
        
        return x
        
    def _forward_with_checkpointing(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
            
        # Checkpointed attention
        residual = x
        x = self.norm1(x)
        x = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.attention),
            x, attention_mask
        )
        x = self.attention_proj(x)
        x = residual + x * self.residual_scale
        
        # Optional memory bank
        if self.use_memory:
            x = x + self.memory_bank(x)
        
        # Checkpointed feed-forward
        residual = x
        x = self.norm2(x)
        x = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.ffn),
            x
        )
        x = residual + x * self.residual_scale
        
        return x

class EfficientTransformer(nn.Module):
    """Memory-efficient transformer with optimized architecture"""
    def __init__(self, config):
        super().__init__()
        
        # Initialize memory manager
        from utils.memory_manager import MemoryManager
        self.memory_manager = MemoryManager(config)
        
        # Initialize LoRA if enabled
        self.use_lora = getattr(config, 'use_lora', False)
        if self.use_lora:
            # Use the already imported LoRALinear class from the top of the file
            linear_class = LoRALinear
        else:
            linear_class = MemoryEfficientLinear
            
        # Embeddings with weight sharing
        self.embedding_layer = EfficientEmbeddingLayer(config)
        
        # Transformer layers with optimized patterns
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(config, linear_class=linear_class)
            for _ in range(config.num_layers)
        ])
        
        # Output head with weight sharing
        self.output_head = linear_class(config.hidden_size, config.vocab_size, bias=config.bias)
        if config.tie_word_embeddings:
            self.embedding_layer.set_output_layer(self.output_head)
            
        # Response cache
        self.use_response_cache = getattr(config, 'use_response_cache', False)
        if self.use_response_cache:
            from model.cache import ResponseCache
            self.response_cache = ResponseCache(
                cache_size=config.cache_size,
                hidden_size=config.hidden_size,
                device=self.device
            )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply memory optimizations
        self._setup_memory_optimizations()
        
        # Initialize QLoRA if enabled
        if getattr(config, 'use_qlora', False):
            self._setup_qlora()

    def _setup_qlora(self):
        """Setup QLoRA (Quantized LoRA) for efficient fine-tuning"""
        try:
            import bitsandbytes as bnb
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                print("Warning: BitsAndBytesConfig not found in transformers, QLoRA disabled")
                return
            
            # Quantize base model to 4 bits
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Keep LoRA adapters in 16-bit precision
            for name, module in self.named_modules():
                if isinstance(module, LoRALinear):
                    # Check that lora_A and lora_B exist before converting
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        module.lora_A = module.lora_A.to(torch.float16)
                        module.lora_B = module.lora_B.to(torch.float16)
                    else:
                        print(f"Warning: LoRA module {name} missing lora_A or lora_B attributes")
        except ImportError:
            print("Warning: bitsandbytes not found, QLoRA disabled")
        except Exception as e:
            print(f"Warning: Error setting up QLoRA: {e}, QLoRA disabled")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        # Initialize memory tracking
        memory_manager = self.memory_manager
        memory_manager.clear_memory()
        
        # Check response cache first
        if self.use_response_cache and not self.training:
            cached_output = self.response_cache.get(input_ids)
            if cached_output is not None:
                return {'logits': cached_output} if return_dict else (None, cached_output, None)
        
        # Get current memory stats
        stats = memory_manager.get_memory_stats()
        current_memory_usage = stats.gpu_allocated / (torch.cuda.get_device_properties(0).total_memory)
        
        # Optimize dtype based on memory usage
        dtype = memory_manager.get_optimal_dtype(current_memory_usage)
        
        # Process in chunks if sequence is long
        seq_length = input_ids.size(1)
        attention_pattern = memory_manager.optimize_attention_pattern(seq_length)
        
        # Forward pass with memory optimizations
        with torch.cuda.amp.autocast(enabled=True):
            # Get embeddings
            hidden_states = self.embedding_layer(input_ids)
            
            # Process through transformer layers
            for i, layer in enumerate(self.layers):
                # Check if we should use CPU offloading
                if memory_manager.should_offload_to_cpu(hidden_states.numel() * hidden_states.element_size()):
                    hidden_states = hidden_states.cpu()
                    hidden_states = hidden_states.to(self.device)
                    memory_manager.clear_memory()
                
                # Process through layer
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask
                )
            
            # Project to vocabulary
            logits = self.output_head(hidden_states)
            
            # Update response cache
            if self.use_response_cache and not self.training:
                self.response_cache.update(input_ids, logits)
            
            # Clear memory after forward pass
            memory_manager.clear_memory()
            
            # Compute loss if labels provided
            loss = None
            if labels is not None:
                # Reshape logits and labels for loss computation
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Compute loss
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            
            if return_dict:
                return {
                    'loss': loss,
                    'logits': logits,
                    'hidden_states': hidden_states
                }
            else:
                return (loss, logits, hidden_states)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Efficient autoregressive generation."""
        batch_size = input_ids.shape[0]
        generated = input_ids
        
        for _ in range(max_length - input_ids.shape[1]):
            # Create attention mask for autoregressive generation
            mask = torch.ones(batch_size, 1, generated.shape[1]).to(input_ids.device)
            mask = torch.triu(mask, diagonal=1).bool()
            
            # Forward pass
            outputs = self.forward(generated, mask)
            next_token_logits = outputs['logits'][:, -1, :] @ self.embedding_layer.embedding.weight.t()
            next_token_logits = next_token_logits / temperature
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Concatenate with previous tokens
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated