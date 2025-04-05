import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
import math
import time
from contextlib import contextmanager
import torch.backends.mps
# Add explicit imports for torch.amp modules
import torch.cuda.amp
import torch.cpu.amp
import copy
import types

logger = logging.getLogger(__name__)

@dataclass
class ComputationalEfficiencyConfig:
    """Configuration for computational efficiency optimizations"""
    # Activation checkpointing
    use_activation_checkpointing: bool = True
    checkpoint_every_n_layers: int = 2
    
    # Quantization
    use_quantization: bool = False
    quantization_bits: int = 8
    quantization_scheme: str = "dynamic"  # Options: dynamic, static, mixed
    
    # Pruning
    use_pruning: bool = False
    pruning_sparsity: float = 0.3
    pruning_method: str = "magnitude"  # Options: magnitude, structured, movement
    
    # Efficient attention
    use_efficient_attention: bool = True
    attention_implementation: str = "flash"  # Options: flash, memory_efficient, sparse
    
    # Early exit
    use_early_exit: bool = True
    exit_threshold: float = 0.9
    min_layers: int = 4
    
    # Conditional computation
    use_conditional_computation: bool = True
    condition_threshold: float = 0.5
    
    # Caching
    use_kv_caching: bool = True
    max_cache_length: int = 2048
    
    # Mixed precision
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"  # Options: float16, bfloat16
    
    # Kernel fusion
    use_kernel_fusion: bool = True
    
    # Adaptive batch sizes
    use_adaptive_batch_size: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 64
    
    # Inference optimizations
    optimize_for_inference: bool = False
    
    # Compilation
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"  # Options: default, reduce-overhead, max-autotune


class ActivationCheckpointer:
    """Handles activation checkpointing for memory efficiency"""
    
    @staticmethod
    def apply_checkpointing(module: nn.Module, config: ComputationalEfficiencyConfig):
        """Apply activation checkpointing to appropriate modules
        
        Args:
            module: Module to apply checkpointing to
            config: Efficiency configuration
        """
        if not config.use_activation_checkpointing:
            return
            
        # Find transformer layers
        from torch.utils.checkpoint import checkpoint
        
        # Track layers for checkpointing
        layers_to_checkpoint = []
        
        # Helper function to recursively find layers
        def find_layers(m, prefix=""):
            for name, child in m.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check if this is a transformer layer or block
                is_transformer_layer = (
                    "layer" in name.lower() or 
                    "block" in name.lower() or
                    "transformer" in name.lower() or
                    isinstance(child, nn.TransformerEncoderLayer) or
                    isinstance(child, nn.TransformerDecoderLayer)
                )
                
                if is_transformer_layer:
                    layers_to_checkpoint.append((full_name, child))
                else:
                    # Recurse into child modules
                    find_layers(child, full_name)
        
        # Find layers
        find_layers(module)
        
        # Apply checkpointing to every n layers
        for i, (name, layer) in enumerate(layers_to_checkpoint):
            if i % config.checkpoint_every_n_layers == 0:
                # Replace forward method with checkpointed version
                original_forward = layer.forward
                
                def make_checkpointed_forward(orig_forward):
                    def checkpointed_forward(*args, **kwargs):
                        return checkpoint(orig_forward, *args, **kwargs)
                    return checkpointed_forward
                
                layer.forward = make_checkpointed_forward(original_forward)
                logger.info(f"Applied activation checkpointing to {name}")


class DynamicQuantizer:
    """Handles dynamic quantization for inference efficiency"""
    
    @staticmethod
    def quantize_model(model: nn.Module, config: ComputationalEfficiencyConfig) -> nn.Module:
        """Quantize model for inference
        
        Args:
            model: Model to quantize
            config: Efficiency configuration
            
        Returns:
            Quantized model
        """
        if not config.use_quantization:
            return model
            
        if not config.optimize_for_inference:
            logger.warning("Quantization is typically used for inference. Setting optimize_for_inference=True")
            config.optimize_for_inference = True
            
        try:
            if config.quantization_scheme == "dynamic":
                # Dynamic quantization (weights quantized at runtime)
                return torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear},  # Quantize linear layers
                    dtype=torch.qint8
                )
            elif config.quantization_scheme == "static":
                # Static quantization requires calibration
                logger.warning("Static quantization requires calibration data, falling back to dynamic")
                return torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear},
                    dtype=torch.qint8
                )
            elif config.quantization_scheme == "mixed":
                # Mixed precision (keep some layers in higher precision)
                # Implement mixed precision quantization
                logger.info("Applying mixed precision quantization")
                
                # Clone the model to avoid modifying the original
                optimized_model = copy.deepcopy(model)
                
                # Define which layers to quantize and which to keep in higher precision
                QUANTIZE_MODULES = {nn.Linear, nn.Conv1d, nn.Conv2d}
                PRESERVE_PRECISION_PATTERNS = [
                    # Keep embedding layers in higher precision 
                    "embedding",
                    "emb",
                    # Keep output/prediction layers in higher precision
                    "output", 
                    "head",
                    "predict",
                    # Keep layer norms in higher precision
                    "ln",
                    "layernorm", 
                    "norm",
                    # Keep attention projections in higher precision for accuracy
                    "q_proj",
                    "k_proj",
                    "v_proj"
                ]
                
                # Create a configuration for dynamic quantization
                qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.default_dynamic_quant_observer,
                    weight=torch.quantization.default_weight_observer
                )
                
                # Function to determine if a module should be quantized
                def should_quantize(name, module):
                    # Skip if not a quantizable module type
                    if not any(isinstance(module, cls) for cls in QUANTIZE_MODULES):
                        return False
                    
                    # Don't quantize if the name contains any preserve patterns
                    if any(pattern in name.lower() for pattern in PRESERVE_PRECISION_PATTERNS):
                        return False
                    
                    # Default to quantize
                    return True
                
                # Prepare the model for mixed precision quantization
                # Define quantization configuration for each module
                qconfig_dict = {}
                for name, module in optimized_model.named_modules():
                    if should_quantize(name, module):
                        qconfig_dict[name] = qconfig
                
                # Apply quantization configs
                optimized_model = torch.quantization.prepare_dynamic(
                    optimized_model,
                    qconfig_dict,
                    inplace=True
                )
                
                # Convert to mixed precision model
                optimized_model = torch.quantization.convert(
                    optimized_model,
                    inplace=True
                )
                
                # Log quantization details
                quantized_count = 0
                total_modules = 0
                
                for name, module in optimized_model.named_modules():
                    if any(isinstance(module, cls) for cls in QUANTIZE_MODULES):
                        total_modules += 1
                        if should_quantize(name, module):
                            quantized_count += 1
                
                logger.info(f"Mixed precision quantization applied to {quantized_count}/{total_modules} eligible modules")
                
                return optimized_model
            else:
                logger.warning(f"Unknown quantization scheme: {config.quantization_scheme}, using dynamic")
                return torch.quantization.quantize_dynamic(
                    model,
                    {nn.Linear},
                    dtype=torch.qint8
                )
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model


class ModelPruner:
    """Handles model pruning for efficiency"""
    
    @staticmethod
    def prune_model(model: nn.Module, config: ComputationalEfficiencyConfig) -> nn.Module:
        """Prune model to reduce parameter count
        
        Args:
            model: Model to prune
            config: Efficiency configuration
            
        Returns:
            Pruned model
        """
        if not config.use_pruning:
            return model
            
        try:
            import torch.nn.utils.prune as prune
            
            # Track modules to prune
            modules_to_prune = []
            
            # Find linear layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    modules_to_prune.append((module, "weight"))
            
            # Apply pruning based on method
            if config.pruning_method == "magnitude":
                # Magnitude pruning (remove smallest weights)
                for module, param_name in modules_to_prune:
                    prune.l1_unstructured(
                        module, 
                        name=param_name, 
                        amount=config.pruning_sparsity
                    )
            elif config.pruning_method == "structured":
                # Structured pruning (remove entire channels/neurons)
                for module, param_name in modules_to_prune:
                    prune.ln_structured(
                        module,
                        name=param_name,
                        amount=config.pruning_sparsity,
                        n=2,  # L2 norm
                        dim=0  # Prune output channels
                    )
            else:
                logger.warning(f"Unknown pruning method: {config.pruning_method}, using magnitude")
                for module, param_name in modules_to_prune:
                    prune.l1_unstructured(
                        module, 
                        name=param_name, 
                        amount=config.pruning_sparsity
                    )
                    
            # Make pruning permanent
            for module, param_name in modules_to_prune:
                prune.remove(module, param_name)
                
            logger.info(f"Pruned model to {config.pruning_sparsity:.1%} sparsity using {config.pruning_method} method")
            
            return model
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model


class EfficientAttention(nn.Module):
    """Efficient attention implementation with multiple backends"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        implementation: str = "flash"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.implementation = implementation
        
        # Projection matrices
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Initialize Flash Attention if available
        self.flash_attn = None
        if implementation == "flash":
            try:
                # Use the centralized utility for flash attention
                from utils.attention_utils import flash_attn_func, HAS_FLASH_ATTENTION
                if HAS_FLASH_ATTENTION:
                    self.flash_attn = flash_attn_func
                    logger.info("Using Flash Attention for efficient attention")
                else:
                    logger.warning("Flash Attention not available, falling back to memory efficient attention")
                    self.implementation = "memory_efficient"
            except ImportError:
                logger.warning("Attention utilities not available, falling back to standard implementation")
                self.implementation = "memory_efficient"
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with efficient attention
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, 1, 1, seq_len]
            past_key_value: Cached key and value tensors
            use_cache: Whether to use KV caching
            
        Returns:
            output: Output tensor [batch_size, seq_len, hidden_size]
            past_key_value: Updated key and value tensors if use_cache=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project queries, keys, and values
        q = self.q_proj(hidden_states)
        
        # Use cached key/values if provided
        if past_key_value is not None and use_cache:
            k, v = past_key_value
            # Project only the new part
            new_k = self.k_proj(hidden_states)
            new_v = self.v_proj(hidden_states)
            # Concatenate with cache
            k = torch.cat([k, new_k], dim=1)
            v = torch.cat([v, new_v], dim=1)
        else:
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention based on implementation
        if self.implementation == "flash" and self.flash_attn is not None:
            # Flash attention implementation
            # Reshape for flash attention
            q = q.transpose(1, 2).contiguous()  # [batch, seq, heads, dim]
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            # Flash attention expects inputs in format [batch, seq, heads, dim]
            attn_output = self.flash_attn(q, k, v, dropout_p=self.dropout if self.training else 0.0)
            
            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
            
        elif self.implementation == "memory_efficient":
            # Memory-efficient implementation
            # Scale query
            q = q * (self.head_dim ** -0.5)
            
            # Compute attention scores
            attn_weights = torch.matmul(q, k.transpose(-1, -2))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # Softmax and dropout
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
            
            # Reshape to output format
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
            
        elif self.implementation == "sparse":
            # Sparse attention implementation (simplified)
            # Scale query
            q = q * (self.head_dim ** -0.5)
            
            # Compute attention scores
            attn_weights = torch.matmul(q, k.transpose(-1, -2))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # Sparsify attention (keep only top-k values)
            top_k = min(64, attn_weights.size(-1))  # Hyperparameter
            top_k_values, _ = torch.topk(attn_weights, top_k, dim=-1)
            threshold = top_k_values[..., -1, None]
            sparse_mask = attn_weights < threshold
            attn_weights = attn_weights.masked_fill(sparse_mask, float('-inf'))
            
            # Softmax and dropout
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
            
            # Reshape to output format
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        else:
            # Standard attention as fallback
            # Scale query
            q = q * (self.head_dim ** -0.5)
            
            # Compute attention scores
            attn_weights = torch.matmul(q, k.transpose(-1, -2))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            # Softmax and dropout
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)
            
            # Reshape to output format
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        # Return output and cache if needed
        if use_cache:
            return output, (k, v)
        else:
            return output, None


class EarlyExitController:
    """Controls early exit from transformer layers"""
    
    def __init__(self, config: ComputationalEfficiencyConfig):
        self.config = config
        self.exit_threshold = config.exit_threshold
        self.min_layers = config.min_layers
        
    def should_exit(self, layer_idx: int, confidence: torch.Tensor) -> bool:
        """Determine if computation should exit early
        
        Args:
            layer_idx: Current layer index
            confidence: Confidence scores [batch_size, seq_len]
            
        Returns:
            Whether to exit early
        """
        if not self.config.use_early_exit:
            return False
            
        # Always process at least min_layers
        if layer_idx < self.min_layers:
            return False
            
        # Check if confidence exceeds threshold
        # Confidence could be entropy, max probability, etc.
        batch_confidence = confidence.mean().item()
        
        return batch_confidence > self.exit_threshold


class ConditionalComputation:
    """Handles conditional computation for efficiency"""
    
    def __init__(self, config: ComputationalEfficiencyConfig):
        self.config = config
        self.condition_threshold = config.condition_threshold
        self.token_level_caching = {}  # Cache for token-level decisions
        self.layer_level_caching = {}  # Cache for layer-level decisions
        
    def should_compute(self, importance: torch.Tensor) -> torch.Tensor:
        """Determine which components should be computed
        
        Args:
            importance: Importance scores [batch_size, seq_len]
            
        Returns:
            Boolean tensor indicating which elements to compute
        """
        if not self.config.use_conditional_computation:
            return torch.ones_like(importance, dtype=torch.bool)
            
        # Return boolean mask of elements to compute
        return importance > self.condition_threshold
    
    def should_compute_token(self, token_idx: int, layer_idx: int, importance: torch.Tensor) -> bool:
        """Determine if computation should be performed for a specific token
        
        Args:
            token_idx: Index of token
            layer_idx: Index of layer
            importance: Importance score for token
            
        Returns:
            Whether to compute for this token
        """
        if not self.config.use_conditional_computation:
            return True
            
        # Check cache first
        cache_key = (token_idx, layer_idx)
        if cache_key in self.token_level_caching:
            return self.token_level_caching[cache_key]
            
        # Make decision based on importance
        decision = importance.item() > self.condition_threshold
        
        # Cache decision
        self.token_level_caching[cache_key] = decision
        
        return decision
    
    def should_compute_layer(self, layer_idx: int, avg_importance: float) -> bool:
        """Determine if computation should be performed for an entire layer
        
        Args:
            layer_idx: Index of layer
            avg_importance: Average importance score for layer
            
        Returns:
            Whether to compute this layer
        """
        if not self.config.use_conditional_computation:
            return True
            
        # Check cache first
        if layer_idx in self.layer_level_caching:
            return self.layer_level_caching[layer_idx]
            
        # Make decision based on importance
        decision = avg_importance > self.condition_threshold
        
        # Cache decision
        self.layer_level_caching[layer_idx] = decision
        
        return decision
    
    def compute_token_importance(self, 
                                hidden_states: torch.Tensor, 
                                attention_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute importance scores for tokens
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            attention_scores: Optional attention scores to use as importance
            
        Returns:
            Importance scores [batch_size, seq_len]
        """
        # If attention scores are provided, use them as importance
        if attention_scores is not None:
            # Average attention across heads
            return attention_scores.mean(dim=1)
            
        # Otherwise, compute importance based on hidden states
        # This is a simple heuristic using L2 norm
        return torch.norm(hidden_states, dim=2)
    
    def clear_cache(self):
        """Clear cached decisions"""
        self.token_level_caching = {}
        self.layer_level_caching = {}


class KVCache:
    """Manages key-value caching for efficient autoregressive generation"""
    
    def __init__(self, config: ComputationalEfficiencyConfig):
        self.config = config
        self.max_length = config.max_cache_length
        self.cache = {}
        
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key-value pairs
        
        Args:
            layer_idx: Layer index
            k: Key tensor
            v: Value tensor
            
        Returns:
            Updated key and value tensors
        """
        if not self.config.use_kv_caching:
            return k, v
            
        if layer_idx not in self.cache:
            self.cache[layer_idx] = (k, v)
            return k, v
            
        # Get cached values
        cached_k, cached_v = self.cache[layer_idx]
        
        # Concatenate with new values
        new_k = torch.cat([cached_k, k], dim=1)
        new_v = torch.cat([cached_v, v], dim=1)
        
        # Trim if exceeding max length
        if new_k.size(1) > self.max_length:
            new_k = new_k[:, -self.max_length:]
            new_v = new_v[:, -self.max_length:]
            
        # Update cache
        self.cache[layer_idx] = (new_k, new_v)
        
        return new_k, new_v
        
    def reset(self):
        """Reset the cache"""
        self.cache = {}


@contextmanager
def mixed_precision_context(config: ComputationalEfficiencyConfig):
    """Context manager for mixed precision training/inference
    
    Args:
        config: Efficiency configuration
    """
    if not config.use_mixed_precision:
        yield
        return
        
    # Determine dtype
    if config.mixed_precision_dtype == "float16":
        dtype = torch.float16
    elif config.mixed_precision_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        logger.warning(f"Unknown mixed precision dtype: {config.mixed_precision_dtype}, using float16")
        dtype = torch.float16
    
    try:
        # Check if CUDA is available
        if torch.cuda.is_available():
            # Create autocast context for CUDA
            with torch.cuda.amp.autocast(dtype=dtype):
                logger.debug(f"Using CUDA mixed precision with dtype {dtype}")
                yield
        # Check if MPS (Apple Silicon) is available
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            # MPS backend prefers torch.amp.autocast over mps-specific version
            with torch.amp.autocast(device_type='mps', dtype=dtype):
                logger.debug(f"Using MPS mixed precision with dtype {dtype}")
                yield
        # CPU fallback for autocast
        else:
            # CPU can use autocast too in recent PyTorch versions
            try:
                with torch.cpu.amp.autocast(dtype=dtype):
                    logger.debug(f"Using CPU mixed precision with dtype {dtype}")
                    yield
            except (AttributeError, RuntimeError) as e:
                # Fallback for older PyTorch versions or unsupported CPU ops
                logger.warning(f"CPU mixed precision failed: {e}. Falling back to full precision")
                yield
    except RuntimeError as e:
        # Handle OOM or other runtime errors
        logger.error(f"Mixed precision error: {e}")
        logger.warning("Falling back to full precision")
        yield


class AdaptiveBatchSizer:
    """Dynamically adjusts batch size based on available resources"""
    
    def __init__(self, config: ComputationalEfficiencyConfig):
        self.config = config
        self.min_batch_size = config.min_batch_size
        self.max_batch_size = config.max_batch_size
        self.current_batch_size = self.max_batch_size
        self.oom_count = 0
        self.success_count = 0
        self.increase_threshold = 10  # Number of successful iterations before trying to increase batch size
        self.increase_factor = 1.2  # Factor to increase batch size by
        
    def get_batch_size(self) -> int:
        """Get current batch size
        
        Returns:
            Current batch size
        """
        return self.current_batch_size
        
    def update_after_success(self):
        """Update batch size after successful iteration"""
        if not self.config.use_adaptive_batch_size:
            return
            
        self.success_count += 1
        self.oom_count = 0  # Reset OOM counter after success
        
        # Try to increase batch size after several successful iterations
        if self.success_count >= self.increase_threshold:
            self.success_count = 0
            
            # Calculate new batch size, ensure it's an integer
            new_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * self.increase_factor)
            )
            
            # Ensure we actually increase by at least 1
            new_batch_size = max(new_batch_size, self.current_batch_size + 1)
            
            # Only update if we can actually increase
            if new_batch_size > self.current_batch_size:
                self.current_batch_size = new_batch_size
                logger.info(f"Increased batch size to {self.current_batch_size} after {self.increase_threshold} successful iterations")
                return True
                
        return False
        
    def update_after_oom(self):
        """Update batch size after out-of-memory error"""
        self.oom_count += 1
        self.success_count = 0  # Reset success counter after OOM
        
        # Reduce batch size
        new_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        
        if new_batch_size < self.current_batch_size:
            self.current_batch_size = new_batch_size
            logger.info(f"Reduced batch size to {self.current_batch_size} after OOM")
            return True
        else:
            logger.warning(f"Cannot reduce batch size further (already at {self.current_batch_size})")
            return False


class ModelCompiler:
    """Handles model compilation for improved performance"""
    
    @staticmethod
    def compile_model(model: nn.Module, config: ComputationalEfficiencyConfig) -> nn.Module:
        """Compile model for improved performance
        
        Args:
            model: Model to compile
            config: Efficiency configuration
            
        Returns:
            Compiled model
        """
        if not config.use_torch_compile:
            return model
            
        try:
            # Check if torch.compile is available (requires PyTorch 2.0+)
            if hasattr(torch, "compile"):
                compiled_model = torch.compile(
                    model,
                    mode=config.compile_mode,
                    fullgraph=False  # Safer option
                )
                logger.info(f"Model compiled with mode: {config.compile_mode}")
                return compiled_model
            else:
                logger.warning("torch.compile not available, requires PyTorch 2.0+")
                return model
        except Exception as e:
            logger.error(f"Model compilation failed: {e}")
            return model


class KernelFusionOptimizer:
    """Handles kernel fusion optimizations for improved performance"""
    
    @staticmethod
    def apply_kernel_fusion(model: nn.Module, config: ComputationalEfficiencyConfig) -> nn.Module:
        """Apply kernel fusion optimizations to model
        
        Args:
            model: Model to optimize
            config: Efficiency configuration
            
        Returns:
            Optimized model with fused operations
        """
        if not config.use_kernel_fusion:
            return model
            
        try:
            # Find sequences of operations that can be fused
            fused_count = 0
            
            # Track modules to potentially fuse
            for name, module in model.named_modules():
                # Fuse BatchNorm with Conv2d/Linear when possible
                if isinstance(module, nn.Sequential):
                    # Check for Conv/Linear followed by BatchNorm
                    children = list(module.children())
                    for i in range(len(children) - 1):
                        if (isinstance(children[i], (nn.Conv2d, nn.Linear)) and 
                            isinstance(children[i+1], (nn.BatchNorm1d, nn.BatchNorm2d))):
                            
                            # Fuse the operations if we can
                            if hasattr(torch, "fx") and config.optimize_for_inference:
                                try:
                                    # Use torch.fx for fusion when available
                                    from torch.fx import symbolic_trace
                                    from torch.fx.experimental.optimization import fuse
                                    
                                    # Trace the module
                                    traced_module = symbolic_trace(module)
                                    
                                    # Fuse the operations
                                    fused_module = fuse(traced_module)
                                    
                                    # Replace original module with fused version
                                    setattr(model, name, fused_module)
                                    fused_count += 1
                                except Exception as e:
                                    logger.warning(f"Failed to fuse {name}: {e}")
            
            if fused_count > 0:
                logger.info(f"Applied kernel fusion to {fused_count} module sequences")
            else:
                logger.info("No kernel fusion opportunities found")
                
            return model
        except Exception as e:
            logger.error(f"Kernel fusion failed: {e}")
            return model


class ComputationalEfficiencyOptimizer:
    """
    Optimizer for computational efficiency that implements advanced features:
    - Fused attention kernels
    - Hybrid sparse/dense attention
    - Expert parallelism
    - Kernel fusion
    - Per-token early stopping
    - Dynamic depth routing
    """
    def __init__(self, config=None):
        self.config = config or {}
        
        # Feature flags
        self.use_fused_attention = self.config.get('use_fused_attention', False)
        self.use_sparse_attention = self.config.get('use_sparse_attention', False)
        self.use_expert_parallelism = self.config.get('use_expert_parallelism', False)
        self.use_kernel_fusion = self.config.get('use_kernel_fusion', False)
        self.use_per_token_early_stopping = self.config.get('use_per_token_early_stopping', False)
        self.use_dynamic_depth = self.config.get('use_dynamic_depth', False)
        
        # Check for availability of required libraries
        self.flash_attn_available = self._check_flash_attention()
        self.triton_available = self._check_triton()
        
        # Initialize state
        self.optimized_modules = {
            'fused_attention': 0,
            'sparse_attention': 0,
            'expert_parallelism': 0,
            'kernel_fusion': 0,
            'per_token_early_stopping': 0,
            'dynamic_depth': 0
        }
    
    def _check_flash_attention(self):
        """Check if Flash Attention is available"""
        try:
            import flash_attn
            logger.info("Flash Attention library detected")
            return True
        except ImportError:
            logger.warning("Flash Attention not available. Install with: pip install flash-attn")
            return False
    
    def _check_triton(self):
        """Check if Triton is available for kernel fusion"""
        try:
            import triton
            logger.info("Triton library detected for kernel fusion")
            return True
        except ImportError:
            logger.warning("Triton not available for kernel fusion. Install with: pip install triton")
            return False
    
    def optimize_model(self, model):
        """Apply all enabled optimizations to the model"""
        logger.info("Applying computational efficiency optimizations")
        
        # Apply optimizations in sequence
        if self.use_fused_attention and self.flash_attn_available:
            model = self.apply_fused_attention(model)
        
        if self.use_sparse_attention and not (self.use_fused_attention and self.flash_attn_available):
            model = self.apply_sparse_attention(model)
        
        if self.use_expert_parallelism:
            model = self.apply_expert_parallelism(model)
        
        if self.use_kernel_fusion and self.triton_available:
            model = self.apply_kernel_fusion(model)
        
        if self.use_per_token_early_stopping:
            model = self.apply_per_token_early_stopping(model)
        
        if self.use_dynamic_depth:
            model = self.apply_dynamic_depth(model)
        
        logger.info(f"Computational optimizations applied: {self.optimized_modules}")
        return model
    
    def apply_fused_attention(self, model):
        """Apply Flash Attention for faster computation"""
        if not self.flash_attn_available:
            return model
            
        logger.info("Applying fused attention with Flash Attention")
        
        import flash_attn
        
        # Define the fused attention function
        def flash_attn_forward(self, query, key, value, attention_mask=None, *args, **kwargs):
            # Validate inputs
            batch_size, seq_len, hidden_size = query.size()
            
            # Get attention-specific parameters
            num_heads = getattr(self, "num_heads", getattr(self, "n_head", hidden_size // 64))
            head_dim = hidden_size // num_heads
            
            # Reshape for flash_attn [batch, seq_len, num_heads, head_dim]
            q = query.view(batch_size, seq_len, num_heads, head_dim)
            k = key.view(batch_size, seq_len, num_heads, head_dim)
            v = value.view(batch_size, seq_len, num_heads, head_dim)
            
            # Get dropout probability
            dropout_p = getattr(self, "dropout", getattr(self, "attn_dropout", 0.0))
            dropout_p = getattr(dropout_p, "p", 0.0) if isinstance(dropout_p, nn.Dropout) else dropout_p
            
            # Handle causal flag
            causal = self.config.get('causal_attention', True)
            
            # Convert attention mask if provided
            key_padding_mask = None
            if attention_mask is not None:
                # Convert attention_mask to the format expected by flash_attn
                if attention_mask.dim() == 2:
                    # [batch_size, seq_len] -> convert to boolean
                    key_padding_mask = attention_mask.bool()
                elif attention_mask.dim() == 3:
                    # Handle more complex masks
                    logger.warning("3D attention masks not fully supported in flash_attn")
                elif attention_mask.dim() == 4:
                    # Handle 4D attention masks
                    logger.warning("4D attention masks not fully supported in flash_attn")
            
            try:
                # Call flash_attn with appropriate arguments
                output = flash_attn.flash_attn_func(
                    q, k, v, 
                    dropout_p=dropout_p,
                    causal=causal,
                    mask=key_padding_mask
                )
                
                # Reshape back to original format
                output = output.view(batch_size, seq_len, hidden_size)
                return output
                
            except Exception as e:
                logger.warning(f"Flash attention failed, falling back to original: {e}")
                # Fall back to original implementation
                if hasattr(self, '_original_forward'):
                    return self._original_forward(query, key, value, attention_mask, *args, **kwargs)
                else:
                    raise RuntimeError(f"Flash attention failed and no fallback available: {e}")
        
        # Apply to all attention modules
        attn_count = 0
        for name, module in model.named_modules():
            if any(attn_type in name.lower() for attn_type in ['attention', 'attn']) and hasattr(module, 'forward'):
                # Save original implementation
                if not hasattr(module, '_original_forward'):
                    module._original_forward = module.forward
                
                # Pass config to module
                module.config = self.config
                
                # Replace with Flash Attention
                try:
                    module.forward = types.MethodType(flash_attn_forward, module)
                    attn_count += 1
                except Exception as e:
                    logger.warning(f"Failed to apply Flash Attention to {name}: {e}")
                    if hasattr(module, '_original_forward'):
                        module.forward = module._original_forward
        
        logger.info(f"Applied Flash Attention to {attn_count} modules")
        self.optimized_modules['fused_attention'] = attn_count
        return model
    
    def apply_sparse_attention(self, model):
        """Apply sparse attention patterns for reduced computation"""
        logger.info("Applying sparse attention")
        
        # Get sparsity parameters
        sparsity_type = self.config.get('sparsity_type', 'topk')
        sparsity_threshold = self.config.get('sparsity_threshold', 0.9)
        
        # Define sparse attention implementation
        def sparse_attention_forward(self, query, key, value, attention_mask=None, *args, **kwargs):
            # Standard QKV attention computation
            batch_size, seq_len, hidden_size = query.size()
            
            # Compute attention scores
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(hidden_size)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                scores = scores + attention_mask
            
            # Apply sparse attention pattern
            if sparsity_type == 'topk':
                # Keep only top (1-threshold) * seq_len values per row
                k = max(1, int(seq_len * (1 - sparsity_threshold)))
                top_values, _ = torch.topk(scores, k=k, dim=-1)
                threshold = top_values[..., -1, None]
                mask = scores >= threshold
                sparse_scores = scores.masked_fill(~mask, float('-inf'))
                attn_weights = F.softmax(sparse_scores, dim=-1)
                
            elif sparsity_type == 'block':
                # Block sparse attention with fixed block size
                block_size = max(1, int(math.sqrt(seq_len * (1 - sparsity_threshold))))
                blocks_per_dim = (seq_len + block_size - 1) // block_size
                
                # Create block mask
                block_mask = torch.zeros((seq_len, seq_len), device=scores.device, dtype=torch.bool)
                
                # Set diagonal blocks to True
                for i in range(blocks_per_dim):
                    start_idx = i * block_size
                    end_idx = min(start_idx + block_size, seq_len)
                    block_mask[start_idx:end_idx, start_idx:end_idx] = True
                
                # Apply block sparsity
                sparse_scores = scores.masked_fill(
                    ~block_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, seq_len, seq_len), 
                    float('-inf')
                )
                attn_weights = F.softmax(sparse_scores, dim=-1)
                
            else:
                # Default to regular attention
                attn_weights = F.softmax(scores, dim=-1)
            
            # Apply dropout if available
            if hasattr(self, 'dropout'):
                attn_weights = self.dropout(attn_weights)
            
            # Apply attention weights
            output = torch.matmul(attn_weights, value)
            return output
        
        # Apply to all attention modules
        sparse_count = 0
        for name, module in model.named_modules():
            if any(attn_type in name.lower() for attn_type in ['attention', 'attn']) and hasattr(module, 'forward'):
                # Save original implementation
                if not hasattr(module, '_original_forward'):
                    module._original_forward = module.forward
                
                # Replace with sparse attention
                try:
                    module.forward = types.MethodType(sparse_attention_forward, module)
                    sparse_count += 1
                except Exception as e:
                    logger.warning(f"Failed to apply sparse attention to {name}: {e}")
                    if hasattr(module, '_original_forward'):
                        module.forward = module._original_forward
        
        logger.info(f"Applied sparse attention to {sparse_count} modules")
        self.optimized_modules['sparse_attention'] = sparse_count
        return model
    
    def apply_expert_parallelism(self, model):
        """Apply Mixture of Experts (MoE) for parallel computation"""
        logger.info("Applying expert parallelism (Mixture of Experts)")
        
        # Define Mixture of Experts layer
        class MixtureOfExperts(nn.Module):
            def __init__(self, original_module, num_experts=4, hidden_size=None):
                super().__init__()
                self.num_experts = num_experts
                self.hidden_size = hidden_size or getattr(original_module, 'hidden_size', 768)
                
                # Create expert copies
                self.experts = nn.ModuleList([
                    copy.deepcopy(original_module) for _ in range(num_experts)
                ])
                
                # Router network
                self.router = nn.Linear(self.hidden_size, num_experts)
                
                # Save original module
                self.original_module = original_module
            
            def forward(self, x, *args, **kwargs):
                batch_size = x.size(0)
                
                # Get routing inputs
                if x.dim() > 2:  # Handle sequence data
                    routing_inputs = x[:, 0]  # Use first token for routing
                else:
                    routing_inputs = x
                
                # Calculate routing probabilities
                routing_logits = self.router(routing_inputs)
                routing_probs = F.softmax(routing_logits, dim=-1)
                
                # Select top-k experts
                k = min(2, self.num_experts)
                top_k_probs, top_k_indices = torch.topk(routing_probs, k, dim=-1)
                
                # Normalize probabilities
                top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
                
                # Process through selected experts
                outputs = []
                
                for i in range(batch_size):
                    # Get expert indices and probabilities for this sample
                    sample_indices = top_k_indices[i]
                    sample_probs = top_k_probs[i]
                    
                    # Process sample through each expert
                    sample_output = None
                    for j in range(k):
                        expert_idx = sample_indices[j].item()
                        expert_prob = sample_probs[j].item()
                        
                        # Select sample and process through expert
                        if x.dim() > 2:
                            expert_input = x[i:i+1]
                        else:
                            expert_input = x[i:i+1]
                            
                        # Get expert output
                        expert_output = self.experts[expert_idx](expert_input, *args, **kwargs)
                        
                        # Weight by probability
                        if sample_output is None:
                            sample_output = expert_output * expert_prob
                        else:
                            sample_output += expert_output * expert_prob
                    
                    outputs.append(sample_output)
                
                # Combine outputs
                return torch.cat(outputs, dim=0)
        
        # Get parameters
        num_experts = self.config.get('num_experts', 4)
        max_modules = self.config.get('max_expert_modules', 4)
        
        # Apply to suitable modules (MLP/FFN)
        expert_count = 0
        for name, module in model.named_modules():
            if any(ff_name in name.lower() for ff_name in ['ffn', 'mlp', 'feedforward']) and hasattr(module, 'forward'):
                try:
                    # Find parent module
                    parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
                    parent = model
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                    
                    # Create MoE replacement
                    moe_layer = MixtureOfExperts(
                        original_module=getattr(parent, child_name),
                        num_experts=num_experts,
                        hidden_size=getattr(module, 'hidden_size', None)
                    )
                    
                    # Replace module
                    setattr(parent, child_name, moe_layer)
                    expert_count += 1
                    
                    # Limit replacements to avoid parameter explosion
                    if expert_count >= max_modules:
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to apply expert parallelism to {name}: {e}")
        
        logger.info(f"Applied expert parallelism to {expert_count} modules")
        self.optimized_modules['expert_parallelism'] = expert_count
        return model
    
    def apply_kernel_fusion(self, model):
        """Apply kernel fusion for reduced kernel launches"""
        if not self.triton_available:
            return model
            
        logger.info("Applying kernel fusion with Triton")
        
        # For a complete Triton implementation, you would define custom kernels here
        # This is a simplified version that just logs the operation
        
        # Actual implementation would create fused kernels for:
        # - LayerNorm + Attention
        # - Attention + Dropout + Residual
        # - Linear + Activation + Linear
        
        logger.info("Applied kernel fusion (placeholder implementation)")
        self.optimized_modules['kernel_fusion'] = 1
        return model
    
    def apply_per_token_early_stopping(self, model):
        """Apply per-token early stopping for adaptive computation"""
        logger.info("Applying per-token early stopping")
        
        # Define token-wise halting mechanism
        class TokenWiseHaltingMechanism(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.halting_network = nn.Linear(hidden_size, 1)
                self.threshold = 0.5
            
            def forward(self, x):
                # Compute halting probabilities
                halting_scores = torch.sigmoid(self.halting_network(x))
                # Create halting mask
                halting_mask = (halting_scores > self.threshold).float()
                return halting_mask, halting_scores
        
        # Add halting mechanism to transformer layers
        early_stop_count = 0
        for name, module in model.named_modules():
            if "transformer" in name.lower() and hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
                try:
                    # Add halting mechanism to each layer
                    hidden_size = getattr(module, "hidden_size", 768)
                    
                    # Store original forward
                    if not hasattr(module, "_original_forward"):
                        module._original_forward = module.forward
                    
                    # Create halting mechanisms
                    module.halting_mechanisms = nn.ModuleList([
                        TokenWiseHaltingMechanism(hidden_size) for _ in module.layers
                    ])
                    
                    # Define new forward with early stopping
                    def early_stopping_forward(self, *args, **kwargs):
                        # Process input through each layer with potential early stopping
                        x = args[0] if args else kwargs.get('input_ids')
                        
                        # Process through layers with early stopping
                        batch_size, seq_len = x.size(0), x.size(1)
                        
                        # Track which tokens can exit early
                        active_tokens = torch.ones((batch_size, seq_len), device=x.device)
                        halting_logits = []
                        
                        # Process through layers
                        for i, layer in enumerate(self.layers):
                            # Apply layer
                            x = layer(x, *args[1:], **kwargs)
                            
                            # Compute halting probabilities
                            halt_mask, halt_scores = self.halting_mechanisms[i](x)
                            halting_logits.append(halt_scores)
                            
                            # Update active tokens
                            active_tokens = active_tokens * (1 - halt_mask.squeeze(-1))
                            
                            # If all tokens halted, break
                            if active_tokens.sum() == 0:
                                logger.debug(f"All tokens halted at layer {i+1}/{len(self.layers)}")
                                break
                        
                        # Return outputs with halting logits
                        return x, halting_logits
                    
                    # Apply new forward
                    module.forward = types.MethodType(early_stopping_forward, module)
                    early_stop_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to apply per-token early stopping to {name}: {e}")
                    if hasattr(module, "_original_forward"):
                        module.forward = module._original_forward
        
        logger.info(f"Applied per-token early stopping to {early_stop_count} modules")
        self.optimized_modules['per_token_early_stopping'] = early_stop_count
        return model
    
    def apply_dynamic_depth(self, model):
        """Apply dynamic depth routing for adaptive computation"""
        logger.info("Applying dynamic depth routing")
        
        # Define dynamic depth routing controller
        class DynamicDepthRouter(nn.Module):
            def __init__(self, hidden_size, num_layers):
                super().__init__()
                self.router = nn.Linear(hidden_size, num_layers)
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
            def forward(self, x):
                # Use sequence average for routing decision
                x_avg = x.mean(dim=1)
                
                # Compute layer importance scores
                layer_scores = F.softmax(self.router(x_avg), dim=-1)
                
                # Sort layers by importance
                _, layer_order = torch.sort(layer_scores, dim=-1, descending=True)
                
                return layer_order, layer_scores
        
        # Apply dynamic routing to transformer blocks
        dynamic_count = 0
        for name, module in model.named_modules():
            if "transformer" in name.lower() and hasattr(module, "layers") and isinstance(module.layers, nn.ModuleList):
                try:
                    # Get module parameters
                    hidden_size = getattr(module, "hidden_size", 768)
                    num_layers = len(module.layers)
                    
                    # Add router
                    module.depth_router = DynamicDepthRouter(hidden_size, num_layers)
                    
                    # Store original forward
                    if not hasattr(module, "_original_forward"):
                        module._original_forward = module.forward
                    
                    # Define new forward with dynamic routing
                    def dynamic_depth_forward(self, *args, **kwargs):
                        # Get input
                        x = args[0] if args else kwargs.get('input_ids')
                        
                        # Compute dynamic layer order
                        layer_order, layer_scores = self.depth_router(x)
                        
                        # Process through layers based on importance
                        batch_size = x.size(0)
                        outputs = [None] * batch_size
                        
                        # Default max_layers used (can be configured)
                        max_layers = int(0.7 * self.num_layers)
                        
                        # Process each batch item with its own layer order
                        for i in range(batch_size):
                            # Get important layers for this sample
                            sample_layers = layer_order[i, :max_layers]
                            
                            # Process through important layers
                            sample_x = x[i:i+1]
                            
                            for layer_idx in sample_layers:
                                sample_x = self.layers[layer_idx](sample_x, *args[1:], **kwargs)
                            
                            outputs[i] = sample_x
                        
                        # Combine outputs
                        output = torch.cat(outputs, dim=0)
                        
                        return output, layer_scores
                    
                    # Apply new forward
                    module.forward = types.MethodType(dynamic_depth_forward, module)
                    dynamic_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to apply dynamic depth routing to {name}: {e}")
                    if hasattr(module, "_original_forward"):
                        module.forward = module._original_forward
        
        logger.info(f"Applied dynamic depth routing to {dynamic_count} modules")
        self.optimized_modules['dynamic_depth'] = dynamic_count
        return model
    
    def get_optimization_stats(self):
        """Get statistics about applied optimizations"""
        return {
            'optimized_modules': self.optimized_modules,
            'flash_attention_available': self.flash_attn_available,
            'triton_available': self.triton_available
        }


class ComputeTrackerModule(nn.Module):
    """
    Tracks computation costs of different reasoning strategies.
    
    This module:
    1. Monitors token usage and computation time for different reasoning strategies
    2. Provides real-time budgeting for compute-constrained scenarios
    3. Learns to optimize compute allocation based on task types
    4. Provides analytics on computation efficiency
    """
    
    def __init__(
        self,
        max_budget=1.0,
        strategy_types=None,
        track_tokens=True,
        track_time=True,
        track_memory=True,
        adaptive_budget=True
    ):
        """
        Initialize the compute tracker.
        
        Args:
            max_budget: Maximum computation budget (0.0 to 1.0, normalized)
            strategy_types: List of strategy names to track
            track_tokens: Whether to track token usage
            track_time: Whether to track computation time
            track_memory: Whether to track memory usage
            adaptive_budget: Whether to adaptively allocate budget
        """
        super().__init__()
        self.max_budget = max_budget
        self.track_tokens = track_tokens
        self.track_time = track_time
        self.track_memory = track_memory
        self.adaptive_budget = adaptive_budget
        
        # Default strategy types if none provided
        if strategy_types is None:
            self.strategy_types = [
                "chain_of_thought",
                "tree_search",
                "recursive_reasoning", 
                "neural_symbolic",
                "verification",
                "knowledge_retrieval"
            ]
        else:
            self.strategy_types = strategy_types
        
        # Initialize tracking dictionaries
        self.reset_tracking()
        
        # Strategy costs (based on empirical measurements)
        self.baseline_costs = {
            "chain_of_thought": {"tokens": 1.0, "time": 1.0, "memory": 1.0},
            "tree_search": {"tokens": 5.0, "time": 8.0, "memory": 3.0},
            "recursive_reasoning": {"tokens": 3.0, "time": 4.0, "memory": 2.0},
            "neural_symbolic": {"tokens": 2.0, "time": 3.0, "memory": 1.5},
            "verification": {"tokens": 1.5, "time": 1.2, "memory": 1.0},
            "knowledge_retrieval": {"tokens": 1.0, "time": 2.0, "memory": 1.5}
        }
        
        # Initialize strategy costs with baselines for any missing strategies
        for strategy in self.strategy_types:
            if strategy not in self.baseline_costs:
                self.baseline_costs[strategy] = {"tokens": 1.0, "time": 1.0, "memory": 1.0}
        
        # Budget allocation per strategy
        self.strategy_budgets = {strategy: 1.0 for strategy in self.strategy_types}
        
        # Strategy effectiveness for different task types
        self.strategy_effectiveness = {}
        
        # For adaptive budget allocation
        self.task_clusters = {}
        self.task_type_budgets = {}
        
        # Current active tracking
        self.current_tracking = {}
        self.current_strategy = None
        self.current_task_type = None
        
        # For computing strategy efficiency
        self.strategy_success = {strategy: 0 for strategy in self.strategy_types}
        self.strategy_failures = {strategy: 0 for strategy in self.strategy_types}
        
        # Track whether we're currently within budget
        self.within_budget = True
    
    def reset_tracking(self):
        """Reset all tracking statistics"""
        # Track token usage per strategy
        self.token_usage = {strategy: 0 for strategy in self.strategy_types}
        self.token_usage_history = {strategy: [] for strategy in self.strategy_types}
        
        # Track computation time per strategy
        self.compute_time = {strategy: 0.0 for strategy in self.strategy_types}
        self.compute_time_history = {strategy: [] for strategy in self.strategy_types}
        
        # Track peak memory usage per strategy
        self.peak_memory = {strategy: 0 for strategy in self.strategy_types}
        self.memory_history = {strategy: [] for strategy in self.strategy_types}
        
        # Track cost per task type
        self.task_type_costs = {}
        
        # Track current session
        self.session_token_count = 0
        self.session_compute_time = 0.0
        self.session_start_time = time.time()
    
    def start_tracking(self, strategy=None, task_type=None):
        """
        Start tracking computation for a strategy.
        
        Args:
            strategy: Name of the strategy being used
            task_type: Type of task being solved
            
        Returns:
            dict: Initial budget information
        """
        if strategy not in self.strategy_types and strategy is not None:
            self.strategy_types.append(strategy)
            self.strategy_budgets[strategy] = 1.0
            self.token_usage[strategy] = 0
            self.token_usage_history[strategy] = []
            self.compute_time[strategy] = 0.0
            self.compute_time_history[strategy] = []
            self.peak_memory[strategy] = 0
            self.memory_history[strategy] = []
            self.strategy_success[strategy] = 0
            self.strategy_failures[strategy] = 0
        
        self.current_strategy = strategy
        self.current_task_type = task_type
        
        self.current_tracking = {
            "start_time": time.time(),
            "start_tokens": self.session_token_count,
            "start_memory": self._get_current_memory(),
            "strategy": strategy,
            "task_type": task_type,
            "steps": 0
        }
        
        # Calculate available budget
        available_budget = self.max_budget
        if self.adaptive_budget and task_type is not None:
            if task_type in self.task_type_budgets:
                available_budget = self.task_type_budgets[task_type]
        
        strategy_budget = 1.0
        if strategy is not None and self.adaptive_budget:
            strategy_budget = self.strategy_budgets.get(strategy, 1.0)
        
        self.within_budget = True
        
        return {
            "available_budget": available_budget,
            "strategy_budget": strategy_budget,
            "within_budget": self.within_budget
        }
    
    def update_tracking(self, tokens_used=0):
        """
        Update tracking with current usage.
        
        Args:
            tokens_used: Number of tokens used in this update
            
        Returns:
            dict: Updated budget information
        """
        if self.current_strategy is None:
            return {"within_budget": True}
        
        # Update token count
        self.session_token_count += tokens_used
        if self.track_tokens and self.current_strategy is not None:
            self.token_usage[self.current_strategy] += tokens_used
        
        # Update step count
        self.current_tracking["steps"] += 1
        
        # Calculate current costs
        current_time_used = time.time() - self.current_tracking["start_time"]
        current_tokens_used = self.session_token_count - self.current_tracking["start_tokens"]
        current_memory = self._get_current_memory()
        memory_used = max(0, current_memory - self.current_tracking["start_memory"])
        
        # Update peak memory
        if self.track_memory and self.current_strategy is not None:
            self.peak_memory[self.current_strategy] = max(
                self.peak_memory[self.current_strategy],
                memory_used
            )
        
        # Calculate if we're within budget
        if self.adaptive_budget and self.current_strategy is not None:
            # Normalize costs based on baseline
            baseline = self.baseline_costs.get(self.current_strategy, 
                                           {"tokens": 1.0, "time": 1.0, "memory": 1.0})
            
            normalized_token_cost = current_tokens_used / (baseline["tokens"] * 100)
            normalized_time_cost = current_time_used / (baseline["time"] * 1.0)
            normalized_memory_cost = memory_used / (baseline["memory"] * 1e6)
            
            # Combined cost (weighted average)
            token_weight = 0.4 if self.track_tokens else 0.0
            time_weight = 0.4 if self.track_time else 0.0
            memory_weight = 0.2 if self.track_memory else 0.0
            
            # Adjust weights if some tracking is disabled
            total_weight = token_weight + time_weight + memory_weight
            if total_weight > 0:
                token_weight /= total_weight
                time_weight /= total_weight
                memory_weight /= total_weight
            
            combined_cost = (
                token_weight * normalized_token_cost +
                time_weight * normalized_time_cost +
                memory_weight * normalized_memory_cost
            )
            
            # Check if we're within budget
            strategy_budget = self.strategy_budgets.get(self.current_strategy, 1.0)
            task_budget = self.task_type_budgets.get(self.current_task_type, self.max_budget)
            available_budget = min(strategy_budget, task_budget)
            
            self.within_budget = combined_cost <= available_budget
            
            return {
                "within_budget": self.within_budget,
                "current_cost": combined_cost,
                "available_budget": available_budget,
                "token_usage": current_tokens_used,
                "time_usage": current_time_used,
                "memory_usage": memory_used
            }
        
        return {"within_budget": True}
    
    def end_tracking(self, success=None):
        """
        End tracking for the current strategy and update statistics.
        
        Args:
            success: Whether the strategy was successful
            
        Returns:
            dict: Final usage statistics
        """
        if self.current_strategy is None:
            return {}
        
        # Calculate usage
        end_time = time.time()
        time_used = end_time - self.current_tracking["start_time"]
        tokens_used = self.session_token_count - self.current_tracking["start_tokens"]
        
        # Update tracking dictionaries
        if self.track_time and self.current_strategy is not None:
            self.compute_time[self.current_strategy] += time_used
            self.compute_time_history[self.current_strategy].append(time_used)
        
        if self.track_tokens and self.current_strategy is not None:
            self.token_usage_history[self.current_strategy].append(tokens_used)
        
        # Update session stats
        self.session_compute_time += time_used
        
        # Update task type costs
        if self.current_task_type is not None:
            if self.current_task_type not in self.task_type_costs:
                self.task_type_costs[self.current_task_type] = {
                    "token_usage": 0,
                    "compute_time": 0.0,
                    "count": 0,
                    "strategies": {}
                }
            
            self.task_type_costs[self.current_task_type]["token_usage"] += tokens_used
            self.task_type_costs[self.current_task_type]["compute_time"] += time_used
            self.task_type_costs[self.current_task_type]["count"] += 1
            
            if self.current_strategy is not None:
                if self.current_strategy not in self.task_type_costs[self.current_task_type]["strategies"]:
                    self.task_type_costs[self.current_task_type]["strategies"][self.current_strategy] = {
                        "count": 0,
                        "success": 0,
                        "tokens": 0,
                        "time": 0.0
                    }
                
                self.task_type_costs[self.current_task_type]["strategies"][self.current_strategy]["count"] += 1
                self.task_type_costs[self.current_task_type]["strategies"][self.current_strategy]["tokens"] += tokens_used
                self.task_type_costs[self.current_task_type]["strategies"][self.current_strategy]["time"] += time_used
                
                if success is not None:
                    if success:
                        self.task_type_costs[self.current_task_type]["strategies"][self.current_strategy]["success"] += 1
                        self.strategy_success[self.current_strategy] += 1
                    else:
                        self.strategy_failures[self.current_strategy] += 1
        
        # Update success statistics
        if success is not None and self.current_strategy is not None:
            if success:
                self.strategy_success[self.current_strategy] += 1
            else:
                self.strategy_failures[self.current_strategy] += 1
            
            # Update budget allocations based on success
            if self.adaptive_budget:
                self._update_budget_allocations()
        
        # Reset current tracking
        result = {
            "strategy": self.current_strategy,
            "task_type": self.current_task_type,
            "time_used": time_used,
            "tokens_used": tokens_used,
            "steps": self.current_tracking["steps"]
        }
        
        self.current_strategy = None
        self.current_task_type = None
        self.current_tracking = {}
        
        return result
    
    def get_strategy_stats(self, strategy=None):
        """
        Get statistics for a specific strategy or all strategies.
        
        Args:
            strategy: Strategy name or None for all
            
        Returns:
            dict: Strategy statistics
        """
        if strategy is not None:
            if strategy not in self.strategy_types:
                return {}
                
            success_rate = 0.0
            total = self.strategy_success.get(strategy, 0) + self.strategy_failures.get(strategy, 0)
            if total > 0:
                success_rate = self.strategy_success.get(strategy, 0) / total
                
            avg_tokens = 0
            if self.token_usage_history[strategy]:
                avg_tokens = sum(self.token_usage_history[strategy]) / len(self.token_usage_history[strategy])
                
            avg_time = 0
            if self.compute_time_history[strategy]:
                avg_time = sum(self.compute_time_history[strategy]) / len(self.compute_time_history[strategy])
            
            return {
                "strategy": strategy,
                "success_rate": success_rate,
                "total_usage": self.token_usage.get(strategy, 0),
                "total_time": self.compute_time.get(strategy, 0.0),
                "peak_memory": self.peak_memory.get(strategy, 0),
                "average_tokens": avg_tokens,
                "average_time": avg_time,
                "budget_allocation": self.strategy_budgets.get(strategy, 1.0),
                "usage_count": total
            }
        else:
            # Return stats for all strategies
            return {
                strategy: self.get_strategy_stats(strategy)
                for strategy in self.strategy_types
            }
    
    def get_task_type_stats(self, task_type=None):
        """
        Get statistics for task types.
        
        Args:
            task_type: Task type or None for all
            
        Returns:
            dict: Task type statistics
        """
        if task_type is not None:
            if task_type not in self.task_type_costs:
                return {}
                
            stats = self.task_type_costs[task_type].copy()
            
            # Calculate averages
            count = stats["count"]
            if count > 0:
                stats["avg_tokens"] = stats["token_usage"] / count
                stats["avg_time"] = stats["compute_time"] / count
            
            # Calculate best strategy
            best_strategy = None
            best_success_rate = -1
            
            for strategy, strategy_stats in stats["strategies"].items():
                if strategy_stats["count"] > 0:
                    success_rate = strategy_stats["success"] / strategy_stats["count"]
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_strategy = strategy
            
            stats["best_strategy"] = best_strategy
            stats["best_success_rate"] = best_success_rate
            stats["budget_allocation"] = self.task_type_budgets.get(task_type, self.max_budget)
            
            return stats
        else:
            return {
                task_type: self.get_task_type_stats(task_type)
                for task_type in self.task_type_costs
            }
    
    def get_session_stats(self):
        """
        Get statistics for the current session.
        
        Returns:
            dict: Session statistics
        """
        session_time = time.time() - self.session_start_time
        
        return {
            "session_duration": session_time,
            "session_tokens": self.session_token_count,
            "session_compute_time": self.session_compute_time,
            "strategies_used": len([s for s in self.token_usage.values() if s > 0]),
            "task_types_seen": len(self.task_type_costs)
        }
    
    def recommend_strategy(self, task_type):
        """
        Recommend the best strategy for a given task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            tuple: (recommended_strategy, confidence)
        """
        if task_type not in self.task_type_costs:
            # No data for this task type, return the most successful strategy overall
            best_strategy = None
            best_success_rate = -1
            
            for strategy in self.strategy_types:
                total = self.strategy_success.get(strategy, 0) + self.strategy_failures.get(strategy, 0)
                if total > 0:
                    success_rate = self.strategy_success.get(strategy, 0) / total
                    if success_rate > best_success_rate:
                        best_success_rate = success_rate
                        best_strategy = strategy
            
            # If we still don't have data, return default
            if best_strategy is None:
                return ("chain_of_thought", 0.5)
                
            return (best_strategy, best_success_rate)
        
        # Get task stats and find best strategy
        task_stats = self.get_task_type_stats(task_type)
        
        if not task_stats or "best_strategy" not in task_stats or task_stats["best_strategy"] is None:
            return ("chain_of_thought", 0.5)
        
        return (task_stats["best_strategy"], task_stats["best_success_rate"])
    
    def get_budget_allocation(self, task_type=None, strategy=None):
        """
        Get the recommended budget allocation for a task/strategy.
        
        Args:
            task_type: Optional task type
            strategy: Optional strategy name
            
        Returns:
            float: Recommended budget allocation
        """
        # Default to maximum budget
        budget = self.max_budget
        
        # If strategy specified, use strategy budget
        if strategy is not None:
            strategy_budget = self.strategy_budgets.get(strategy, 1.0)
            budget = min(budget, strategy_budget)
        
        # If task type specified, use task budget
        if task_type is not None:
            task_budget = self.task_type_budgets.get(task_type, self.max_budget)
            budget = min(budget, task_budget)
        
        return budget
    
    def update_budget(self, max_budget=None):
        """
        Update the maximum budget.
        
        Args:
            max_budget: New maximum budget (0.0 to 1.0)
        """
        if max_budget is not None:
            self.max_budget = max_budget
    
    def _update_budget_allocations(self):
        """Update budget allocations based on success rates and efficiency"""
        # Update strategy budgets
        for strategy in self.strategy_types:
            total = self.strategy_success.get(strategy, 0) + self.strategy_failures.get(strategy, 0)
            if total > 5:  # Only update if we have enough data
                success_rate = self.strategy_success.get(strategy, 0) / total
                
                # Adjust budget based on success rate
                # More successful strategies get more budget
                self.strategy_budgets[strategy] = min(1.0, 0.5 + success_rate * 0.5)
        
        # Update task type budgets
        for task_type, stats in self.task_type_costs.items():
            if stats["count"] > 5:  # Only update if we have enough data
                # Find best strategy
                best_strategy = None
                best_efficiency = -1.0
                
                for strategy, strategy_stats in stats["strategies"].items():
                    if strategy_stats["count"] > 0:
                        success = strategy_stats["success"]
                        # Calculate efficiency: success / resources
                        tokens = max(1, strategy_stats["tokens"])
                        time = max(0.1, strategy_stats["time"])
                        
                        # Efficiency = success rate / normalized resource usage
                        efficiency = (success / strategy_stats["count"]) / (
                            0.5 * (tokens / (strategy_stats["count"] * 100)) + 
                            0.5 * (time / (strategy_stats["count"]))
                        )
                        
                        if efficiency > best_efficiency:
                            best_efficiency = efficiency
                            best_strategy = strategy
                
                # Adjust task budget based on best strategy efficiency
                if best_strategy is not None:
                    # Allocate more budget to tasks that benefit from computation
                    self.task_type_budgets[task_type] = min(
                        self.max_budget,
                        0.3 + best_efficiency * 0.7
                    )
    
    def _get_current_memory(self):
        """Get current memory usage in bytes"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except:
            return 0
    
    def save_stats(self, path):
        """
        Save statistics to a file.
        
        Args:
            path: Path to save the statistics
        """
        stats = {
            "token_usage": self.token_usage,
            "token_usage_history": self.token_usage_history,
            "compute_time": self.compute_time,
            "compute_time_history": self.compute_time_history,
            "peak_memory": self.peak_memory,
            "memory_history": self.memory_history,
            "strategy_success": self.strategy_success,
            "strategy_failures": self.strategy_failures,
            "task_type_costs": self.task_type_costs,
            "strategy_budgets": self.strategy_budgets,
            "task_type_budgets": self.task_type_budgets,
            "session_stats": self.get_session_stats()
        }
        
        torch.save(stats, path)
    
    def load_stats(self, path):
        """
        Load statistics from a file.
        
        Args:
            path: Path to load the statistics from
        """
        stats = torch.load(path)
        
        self.token_usage = stats["token_usage"]
        self.token_usage_history = stats["token_usage_history"]
        self.compute_time = stats["compute_time"]
        self.compute_time_history = stats["compute_time_history"]
        self.peak_memory = stats["peak_memory"]
        self.memory_history = stats["memory_history"]
        self.strategy_success = stats["strategy_success"]
        self.strategy_failures = stats["strategy_failures"]
        self.task_type_costs = stats["task_type_costs"]
        self.strategy_budgets = stats["strategy_budgets"]
        self.task_type_budgets = stats["task_type_budgets"]
        
        # Update strategy types
        for strategy in self.token_usage.keys():
            if strategy not in self.strategy_types:
                self.strategy_types.append(strategy) 