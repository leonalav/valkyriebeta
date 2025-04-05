import torch.nn as nn
from .attention import GroupedQueryAttention, MultiScaleAttention, AdaptiveSparsityAttention
try:
    from .moe import SparseMoE
except ImportError:
    # Create a mock SparseMoE class
    class SparseMoE(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("Using mock SparseMoE class as it could not be imported")
            
        def forward(self, x, *args, **kwargs):
            return x
from .activations import SwiGLU
from .lora import LoRALinear
import math
import torch.nn.functional as F
from typing import Dict, Optional
import torch

class MemoryEfficientLinear(nn.Module):
    """Memory-efficient linear layer with optional quantization and sparse operations"""
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 sparse: bool = False,
                 quantize: bool = False,
                 bits: int = 8):
        super().__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.sparse = sparse
        self.quantize = quantize
        self.bits = bits
        
        # Initialize weights with proper scaling
        scale = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * scale)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        if sparse:
            # Initialize sparsity mask
            self.register_buffer(
                'mask',
                torch.ones_like(self.weight, dtype=torch.bool)
            )
            
        if quantize:
            self.register_buffer('scale_factor', torch.ones(1))
            self.register_buffer('zero_point', torch.zeros(1))
            
    def _quantize_weights(self):
        if not self.quantize:
            return self.weight
            
        # Dynamic quantization
        qmin, qmax = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
        
        # Compute scale and zero point
        weight_min, weight_max = self.weight.min(), self.weight.max()
        self.scale_factor[0] = (weight_max - weight_min) / (qmax - qmin)
        self.zero_point[0] = qmin - weight_min / self.scale_factor[0]
        
        # Quantize weights
        quantized = torch.clamp(
            torch.round(self.weight / self.scale_factor[0] + self.zero_point[0]),
            qmin, qmax
        )
        
        # Dequantize for forward pass
        return (quantized - self.zero_point[0]) * self.scale_factor[0]
        
    def update_pattern(self, pattern: Dict[str, any]):
        """Update layer pattern dynamically based on memory usage"""
        # Update sparsity
        if 'use_sparse' in pattern:
            self.sparse = pattern['use_sparse']
            if self.sparse and 'sparsity_ratio' in pattern:
                # Update sparsity mask
                importance = torch.abs(self.weight).mean(dim=0)
                threshold = torch.quantile(importance, pattern['sparsity_ratio'])
                self.register_buffer(
                    'mask',
                    (importance > threshold).to(torch.bool)
                )
        
        # Update quantization
        if 'quantize' in pattern:
            self.quantize = pattern['quantize']
            if self.quantize and 'bits' in pattern:
                self.bits = pattern['bits']
                # Recompute quantization parameters
                weight_min, weight_max = self.weight.min(), self.weight.max()
                qmin, qmax = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
                self.scale_factor[0] = (weight_max - weight_min) / (qmax - qmin)
                self.zero_point[0] = qmin - weight_min / self.scale_factor[0]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Get effective weights (quantized and/or sparse)
        weight = self._quantize_weights()
        if self.sparse:
            weight = weight * self.mask
            
        # Efficient matrix multiplication with memory optimization
        if input.is_cuda and weight.is_cuda:
            # Use specialized CUDA kernels when available
            if hasattr(torch.nn.functional, 'linear_relu_forward'):
                output = torch.nn.functional.linear_relu_forward(
                    input, weight, self.bias
                )
            else:
                output = F.linear(input, weight, self.bias)
        else:
            # CPU fallback with chunked computation
            chunk_size = 1024
            chunks = []
            for i in range(0, input.size(0), chunk_size):
                chunk = F.linear(
                    input[i:i+chunk_size],
                    weight,
                    self.bias
                )
                chunks.append(chunk)
            output = torch.cat(chunks, dim=0)
        
        return output
        
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, '
                f'out_features={self.out_features}, '
                f'bias={self.bias is not None}, '
                f'sparse={self.sparse}, '
                f'quantize={self.quantize}, '
                f'bits={self.bits if self.quantize else None}')

class GatedLinearUnit(nn.Module):
    """Memory-efficient gated linear unit"""
    def __init__(self, dim: int, activation: nn.Module = nn.GELU()):
        super().__init__() 
        self.dim = dim
        self.activation = activation
        self.proj = MemoryEfficientLinear(dim, 2 * dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.activation(gate)

class EfficientFeedForward(nn.Module):
    """Memory-efficient feed-forward block with GLU activation"""
    def __init__(self, config, linear_class=None):
        super().__init__() 
        linear_class = linear_class or MemoryEfficientLinear
        
        # GLU activation
        self.glu = GatedLinearUnit(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        # Linear layer with optional LoRA
        if linear_class is not LoRALinear:
            self.linear = linear_class(
                config.hidden_size,
                config.hidden_size,
                sparse=config.use_sparse_linear,
                quantize=config.quantize_linear,
                bits=config.quantization_bits
            )
        else:
            self.linear = linear_class(
                config.hidden_size,
                config.hidden_size,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                bias=config.bias
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(self.glu(x)))

class ParallelFeedForward(nn.Module):
    """Parallel feed-forward block for efficient computation"""
    def __init__(self, config, linear_class=None):
        super().__init__() 
        linear_class = linear_class or MemoryEfficientLinear
        
        # Create linear layers with optional LoRA
        if linear_class is not LoRALinear:
            self.w1 = linear_class(config.hidden_size, config.intermediate_size)
            self.w2 = linear_class(config.hidden_size, config.intermediate_size)
            self.w3 = linear_class(config.intermediate_size, config.hidden_size)
        else:
            self.w1 = linear_class(
                config.hidden_size,
                config.intermediate_size,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                bias=config.bias
            )
            self.w2 = linear_class(
                config.hidden_size,
                config.intermediate_size,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                bias=config.bias
            )
            self.w3 = linear_class(
                config.intermediate_size,
                config.hidden_size,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                bias=config.bias
            )
            
        self.activation = nn.SiLU()  # More efficient than GELU
        self.dropout = nn.Dropout(config.hidden_dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Parallel computation of w1 and w2
        a1 = self.w1(x)
        a2 = self.w2(x)
        
        # Efficient activation and multiplication
        hidden = self.activation(a1) * a2
        
        return self.dropout(self.w3(hidden))

class EfficientTransformerLayer(nn.Module):
    """Memory-efficient transformer layer with optimized components"""
    def __init__(self, config):
        super().__init__() 
        # Choose attention mechanism based on config
        if config.use_flash_attention:
            self.attention = MultiScaleAttention(config)
        else:
            self.attention = AdaptiveSparsityAttention(config)
            
        self.norm1 = nn.LayerNorm(config.hidden_size)
        
        # Choose feed-forward type based on config
        if config.use_parallel_ffn:
            self.feed_forward = ParallelFeedForward(config)
        else:
            self.feed_forward = EfficientFeedForward(config)
            
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
        # Optional gradient checkpointing
        self.gradient_checkpointing = False
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(x, attention_mask)
        
        # Efficient residual connections with pre-norm
        normed = self.norm1(x)
        attention_output = self.attention(normed, attention_mask)
        x = x + attention_output
        
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + ff_output
        
        return x
        
    def _forward_with_checkpointing(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
            
        attention_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.attention),
            self.norm1(x), attention_mask
        )
        x = x + attention_output
        
        ff_output = torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.feed_forward),
            self.norm2(x)
        )
        x = x + ff_output
        
        return x