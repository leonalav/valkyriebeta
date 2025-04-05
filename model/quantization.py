"""
Quantization Support for Model Optimization

This module implements quantization techniques to reduce model size and improve inference
performance. It includes:
1. Post-training quantization for model weights
2. Activation quantization for reduced memory usage
3. Quantization-aware inference for optimized performance

Quantization is critical for deploying large models in resource-constrained environments
and can reduce memory usage by 4x or more with minimal quality degradation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import math
import copy

logger = logging.getLogger(__name__)

class QuantizationConfig:
    """Configuration settings for model quantization"""
    
    def __init__(
        self,
        bits: int = 8,
        group_size: int = 128,
        sym: bool = True,
        per_token: bool = False,
        per_channel: bool = True,
        quant_method: str = "absmax",  # "absmax", "zeropoint", "awq"
        dynamic_activations: bool = False,
        excluded_modules: List[str] = None,
        layer_wise_qconfig: Dict[str, Dict[str, Any]] = None,
    ):
        """
        Initialize quantization configuration.
        
        Args:
            bits: Bit precision for quantization (default: 8)
            group_size: Size of groups for group-wise quantization (default: 128)
            sym: Whether to use symmetric quantization (default: True)
            per_token: Whether to quantize per token (default: False)
            per_channel: Whether to quantize per output channel (default: True)
            quant_method: Quantization method to use (default: "absmax")
            dynamic_activations: Whether to dynamically quantize activations (default: False)
            excluded_modules: List of module names to exclude from quantization
            layer_wise_qconfig: Dict mapping layer names to specific quantization configs
        """
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.per_token = per_token
        self.per_channel = per_channel
        self.quant_method = quant_method.lower()
        self.dynamic_activations = dynamic_activations
        self.excluded_modules = excluded_modules or []
        self.layer_wise_qconfig = layer_wise_qconfig or {}
        
        # Validate settings
        self._validate()
    
    def _validate(self):
        """Validate configuration settings"""
        valid_bits = [2, 3, 4, 8, 16]
        if self.bits not in valid_bits:
            raise ValueError(f"Bits must be one of {valid_bits}, got {self.bits}")
        
        valid_methods = ["absmax", "zeropoint", "awq"]
        if self.quant_method not in valid_methods:
            raise ValueError(f"Quant method must be one of {valid_methods}, got {self.quant_method}")
        
        if self.group_size <= 0:
            raise ValueError(f"Group size must be positive, got {self.group_size}")
    
    def get_layer_config(self, layer_name: str) -> 'QuantizationConfig':
        """Get layer-specific configuration if it exists, otherwise return self"""
        if layer_name in self.layer_wise_qconfig:
            # Create a new config with layer-specific overrides
            config_dict = vars(self).copy()
            config_dict.update(self.layer_wise_qconfig[layer_name])
            
            # Don't copy these to avoid recursion/confusion
            config_dict.pop('layer_wise_qconfig', None)
            
            new_config = QuantizationConfig(bits=config_dict.pop('bits'))
            for key, value in config_dict.items():
                setattr(new_config, key, value)
            
            return new_config
        
        return self
    
    def is_excluded(self, module_name: str) -> bool:
        """Check if a module should be excluded from quantization"""
        for excluded in self.excluded_modules:
            if module_name == excluded or module_name.startswith(f"{excluded}."):
                return True
        return False
    
    def __repr__(self) -> str:
        return (f"QuantizationConfig(bits={self.bits}, group_size={self.group_size}, "
                f"sym={self.sym}, method={self.quant_method})")

class QuantizedLinear(nn.Module):
    """Linear layer with quantized weights for efficient inference"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: QuantizationConfig = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()
        
        # Quantized weight representation
        self.register_buffer('qweight', torch.zeros((out_features, in_features), dtype=torch.int8))
        
        # Scale factors for dequantization
        if self.config.per_channel:
            scale_shape = (out_features, 1)
        else:
            scale_shape = (1, 1)
            
        if self.config.group_size > 0:
            # Group-wise quantization requires scales per group
            num_groups = in_features // self.config.group_size
            if in_features % self.config.group_size != 0:
                num_groups += 1
            scale_shape = (out_features, num_groups)
            
        self.register_buffer('scales', torch.zeros(scale_shape, dtype=torch.float16))
        
        # Zero-points for asymmetric quantization
        if not self.config.sym:
            self.register_buffer('zero_points', torch.zeros(scale_shape, dtype=torch.int8))
        
        # Bias
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
        
        # For AWQ
        if self.config.quant_method == "awq":
            self.register_buffer('pre_scale', torch.ones((in_features), dtype=torch.float16))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using quantized weights"""
        # Apply pre-scaling for AWQ
        if self.config.quant_method == "awq" and hasattr(self, 'pre_scale'):
            x = x * self.pre_scale
        
        # Convert input if needed
        orig_dtype = x.dtype
        if orig_dtype != torch.float16:
            x = x.to(torch.float16)
        
        # Handle group-wise quantization
        if self.config.group_size > 0 and self.in_features > self.config.group_size:
            return self._forward_grouped(x)
        
        # Simple case: full quantization
        weight_deq = self._dequantize_weights()
        out = F.linear(x, weight_deq, None)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        # Restore original dtype
        if orig_dtype != torch.float16:
            out = out.to(orig_dtype)
            
        return out
    
    def _forward_grouped(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with group-wise quantization"""
        out_features, in_features = self.qweight.shape
        group_size = self.config.group_size
        num_groups = in_features // group_size
        remaining = in_features % group_size
        
        # Handle full groups
        if remaining == 0:
            # Reshape for batched matrix multiplication
            x_reshaped = x.reshape(-1, num_groups, group_size)
            qweight_reshaped = self.qweight.reshape(out_features, num_groups, group_size)
            
            # Apply scaling per group
            if self.config.sym:
                # Symmetric quantization
                scales_expanded = self.scales.unsqueeze(-1)
                weight_deq = qweight_reshaped.to(torch.float16) * scales_expanded
            else:
                # Asymmetric quantization
                scales_expanded = self.scales.unsqueeze(-1)
                zero_points_expanded = self.zero_points.unsqueeze(-1)
                weight_deq = (qweight_reshaped.to(torch.float16) - zero_points_expanded) * scales_expanded
            
            # Batched matmul across groups
            weight_deq = weight_deq.reshape(out_features, in_features)
            out = F.linear(x, weight_deq, None)
        else:
            # Handle uneven groups case
            results = []
            
            # Process full groups
            for g in range(num_groups):
                start_idx = g * group_size
                end_idx = start_idx + group_size
                
                x_part = x[:, start_idx:end_idx]
                qweight_part = self.qweight[:, start_idx:end_idx]
                
                if self.config.sym:
                    weight_deq = qweight_part.to(torch.float16) * self.scales[:, g:g+1]
                else:
                    weight_deq = (qweight_part.to(torch.float16) - self.zero_points[:, g:g+1]) * self.scales[:, g:g+1]
                
                results.append(F.linear(x_part, weight_deq, None))
            
            # Process remaining features
            if remaining > 0:
                start_idx = num_groups * group_size
                x_part = x[:, start_idx:]
                qweight_part = self.qweight[:, start_idx:]
                
                if self.config.sym:
                    weight_deq = qweight_part.to(torch.float16) * self.scales[:, -1:]
                else:
                    weight_deq = (qweight_part.to(torch.float16) - self.zero_points[:, -1:]) * self.scales[:, -1:]
                
                results.append(F.linear(x_part, weight_deq, None))
            
            # Sum results
            out = sum(results)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
            
        return out
    
    def _dequantize_weights(self) -> torch.Tensor:
        """Dequantize weights for inference"""
        if self.config.sym:
            return self.qweight.to(torch.float16) * self.scales
        else:
            return (self.qweight.to(torch.float16) - self.zero_points) * self.scales
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, config: QuantizationConfig = None) -> 'QuantizedLinear':
        """
        Create a quantized linear layer from a regular linear layer.
        
        Args:
            linear: Source linear layer
            config: Quantization configuration
            
        Returns:
            Quantized linear layer
        """
        config = config or QuantizationConfig()
        qlayer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            config=config
        )
        
        # Get weight and prepare for quantization
        weight = linear.weight.data.clone()
        
        # Process based on group size
        if config.group_size > 0 and weight.shape[1] > config.group_size:
            qlayer = cls._quantize_grouped(qlayer, weight, config)
        else:
            # Simple case: quantize the whole weight matrix
            if config.per_channel:
                # Per output channel quantization
                if config.sym:
                    scales = weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)
                    qweight = torch.round(weight / scales * (2**(config.bits-1) - 1))
                    qweight = qweight.clamp(-2**(config.bits-1), 2**(config.bits-1) - 1).to(torch.int8)
                    
                    qlayer.qweight.copy_(qweight)
                    qlayer.scales.copy_(scales / (2**(config.bits-1) - 1))
                else:
                    # Asymmetric quantization
                    weight_min = weight.min(dim=1, keepdim=True)[0]
                    weight_max = weight.max(dim=1, keepdim=True)[0]
                    scales = (weight_max - weight_min) / (2**config.bits - 1)
                    scales = scales.clamp(min=1e-5)
                    
                    zero_points = torch.round(-weight_min / scales)
                    qweight = torch.round(weight / scales + zero_points)
                    qweight = qweight.clamp(0, 2**config.bits - 1).to(torch.uint8)
                    
                    qlayer.qweight.copy_(qweight)
                    qlayer.scales.copy_(scales)
                    qlayer.zero_points.copy_(zero_points.to(torch.uint8))
            else:
                # Per tensor quantization
                if config.sym:
                    scale = weight.abs().max().clamp(min=1e-5)
                    qweight = torch.round(weight / scale * (2**(config.bits-1) - 1))
                    qweight = qweight.clamp(-2**(config.bits-1), 2**(config.bits-1) - 1).to(torch.int8)
                    
                    qlayer.qweight.copy_(qweight)
                    qlayer.scales.copy_(torch.tensor([scale / (2**(config.bits-1) - 1)], dtype=torch.float16))
                else:
                    # Asymmetric quantization
                    weight_min = weight.min()
                    weight_max = weight.max()
                    scale = (weight_max - weight_min) / (2**config.bits - 1)
                    scale = scale.clamp(min=1e-5)
                    
                    zero_point = torch.round(-weight_min / scale)
                    qweight = torch.round(weight / scale + zero_point)
                    qweight = qweight.clamp(0, 2**config.bits - 1).to(torch.uint8)
                    
                    qlayer.qweight.copy_(qweight)
                    qlayer.scales.copy_(torch.tensor([scale], dtype=torch.float16))
                    qlayer.zero_points.copy_(torch.tensor([zero_point], dtype=torch.uint8))
        
        # Copy bias if present
        if linear.bias is not None:
            qlayer.bias.copy_(linear.bias.data.to(torch.float16))
        
        return qlayer
    
    @staticmethod
    def _quantize_grouped(
        qlayer: 'QuantizedLinear', 
        weight: torch.Tensor, 
        config: QuantizationConfig
    ) -> 'QuantizedLinear':
        """Quantize weights using group-wise quantization"""
        out_features, in_features = weight.shape
        group_size = config.group_size
        num_groups = in_features // group_size
        remaining = in_features % group_size
        
        scales = []
        qweight = []
        zero_points = [] if not config.sym else None
        
        # Process full groups
        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = start_idx + group_size
            group_weight = weight[:, start_idx:end_idx]
            
            if config.per_channel:
                # Per output channel quantization
                if config.sym:
                    max_abs = group_weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)
                    g_scale = max_abs / (2**(config.bits-1) - 1)
                    g_qweight = torch.round(group_weight / max_abs * (2**(config.bits-1) - 1))
                    g_qweight = g_qweight.clamp(-2**(config.bits-1), 2**(config.bits-1) - 1).to(torch.int8)
                else:
                    g_min = group_weight.min(dim=1, keepdim=True)[0]
                    g_max = group_weight.max(dim=1, keepdim=True)[0]
                    g_scale = (g_max - g_min) / (2**config.bits - 1)
                    g_scale = g_scale.clamp(min=1e-5)
                    
                    g_zero_point = torch.round(-g_min / g_scale)
                    g_qweight = torch.round(group_weight / g_scale + g_zero_point)
                    g_qweight = g_qweight.clamp(0, 2**config.bits - 1).to(torch.uint8)
                    zero_points.append(g_zero_point)
            else:
                # Per group quantization
                if config.sym:
                    g_max_abs = group_weight.abs().max().clamp(min=1e-5)
                    g_scale = g_max_abs / (2**(config.bits-1) - 1)
                    g_scale = g_scale.expand(out_features, 1)
                    g_qweight = torch.round(group_weight / g_max_abs * (2**(config.bits-1) - 1))
                    g_qweight = g_qweight.clamp(-2**(config.bits-1), 2**(config.bits-1) - 1).to(torch.int8)
                else:
                    g_min = group_weight.min()
                    g_max = group_weight.max()
                    g_scale = (g_max - g_min) / (2**config.bits - 1)
                    g_scale = g_scale.clamp(min=1e-5)
                    g_scale = g_scale.expand(out_features, 1)
                    
                    g_zero_point = torch.round(-g_min / g_scale[0,0])
                    g_zero_point = g_zero_point.expand(out_features, 1)
                    g_qweight = torch.round(group_weight / g_scale[0,0] + g_zero_point[0,0])
                    g_qweight = g_qweight.clamp(0, 2**config.bits - 1).to(torch.uint8)
                    zero_points.append(g_zero_point)
            
            scales.append(g_scale)
            qweight.append(g_qweight)
        
        # Process remaining features if needed
        if remaining > 0:
            start_idx = num_groups * group_size
            group_weight = weight[:, start_idx:]
            
            if config.per_channel:
                # Per output channel quantization
                if config.sym:
                    max_abs = group_weight.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5)
                    g_scale = max_abs / (2**(config.bits-1) - 1)
                    g_qweight = torch.round(group_weight / max_abs * (2**(config.bits-1) - 1))
                    g_qweight = g_qweight.clamp(-2**(config.bits-1), 2**(config.bits-1) - 1).to(torch.int8)
                else:
                    g_min = group_weight.min(dim=1, keepdim=True)[0]
                    g_max = group_weight.max(dim=1, keepdim=True)[0]
                    g_scale = (g_max - g_min) / (2**config.bits - 1)
                    g_scale = g_scale.clamp(min=1e-5)
                    
                    g_zero_point = torch.round(-g_min / g_scale)
                    g_qweight = torch.round(group_weight / g_scale + g_zero_point)
                    g_qweight = g_qweight.clamp(0, 2**config.bits - 1).to(torch.uint8)
                    zero_points.append(g_zero_point)
            else:
                # Per group quantization
                if config.sym:
                    g_max_abs = group_weight.abs().max().clamp(min=1e-5)
                    g_scale = g_max_abs / (2**(config.bits-1) - 1)
                    g_scale = g_scale.expand(out_features, 1)
                    g_qweight = torch.round(group_weight / g_max_abs * (2**(config.bits-1) - 1))
                    g_qweight = g_qweight.clamp(-2**(config.bits-1), 2**(config.bits-1) - 1).to(torch.int8)
                else:
                    g_min = group_weight.min()
                    g_max = group_weight.max()
                    g_scale = (g_max - g_min) / (2**config.bits - 1)
                    g_scale = g_scale.clamp(min=1e-5)
                    g_scale = g_scale.expand(out_features, 1)
                    
                    g_zero_point = torch.round(-g_min / g_scale[0,0])
                    g_zero_point = g_zero_point.expand(out_features, 1)
                    g_qweight = torch.round(group_weight / g_scale[0,0] + g_zero_point[0,0])
                    g_qweight = g_qweight.clamp(0, 2**config.bits - 1).to(torch.uint8)
                    zero_points.append(g_zero_point)
            
            scales.append(g_scale)
            qweight.append(g_qweight)
        
        # Combine results
        qlayer.qweight = torch.cat(qweight, dim=1)
        qlayer.scales = torch.cat(scales, dim=1)
        
        if not config.sym:
            qlayer.zero_points = torch.cat(zero_points, dim=1)
        
        return qlayer

def quantize_model(
    model: nn.Module,
    config: QuantizationConfig = None
) -> nn.Module:
    """
    Quantize a model's weights to reduce size and improve inference performance.
    
    Args:
        model: The model to quantize
        config: Quantization configuration
        
    Returns:
        Quantized model
    """
    config = config or QuantizationConfig()
    
    # Create a copy to avoid modifying the original
    model_copy = copy.deepcopy(model)
    
    # Recursive quantization of linear layers
    for name, module in list(model_copy.named_modules()):
        if config.is_excluded(name):
            logger.info(f"Skipping quantization of excluded module: {name}")
            continue
        
        # Get layer-specific config if available
        layer_config = config.get_layer_config(name)
        
        # Replace linear layers with quantized versions
        if isinstance(module, nn.Linear):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model_copy if parent_name == "" else get_module_by_name(model_copy, parent_name)
            
            if parent is not None:
                logger.info(f"Quantizing {name} with config: {layer_config}")
                quantized_linear = QuantizedLinear.from_linear(module, layer_config)
                setattr(parent, child_name, quantized_linear)
    
    return model_copy

def get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Helper function to get a module by its name"""
    for n, m in model.named_modules():
        if n == name:
            return m
    return None

def calculate_quantization_stats(model: nn.Module, q_model: nn.Module) -> Dict[str, Any]:
    """
    Calculate statistics comparing original and quantized models.
    
    Args:
        model: Original model
        q_model: Quantized model
        
    Returns:
        Dictionary of statistics
    """
    orig_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quant_size = sum(p.numel() * p.element_size() for p in q_model.parameters())
    
    # Count parameters by type
    orig_params = {}
    quant_params = {}
    
    for name, param in model.named_parameters():
        dtype_name = str(param.dtype).split(".")[-1]
        orig_params[dtype_name] = orig_params.get(dtype_name, 0) + param.numel()
    
    for name, param in q_model.named_parameters():
        dtype_name = str(param.dtype).split(".")[-1]
        quant_params[dtype_name] = quant_params.get(dtype_name, 0) + param.numel()
    
    # Count buffers by type
    for name, buf in model.named_buffers():
        dtype_name = str(buf.dtype).split(".")[-1]
        orig_params[dtype_name] = orig_params.get(dtype_name, 0) + buf.numel()
    
    for name, buf in q_model.named_buffers():
        dtype_name = str(buf.dtype).split(".")[-1]
        quant_params[dtype_name] = quant_params.get(dtype_name, 0) + buf.numel()
    
    # Calculate compression ratio
    compression_ratio = orig_size / quant_size if quant_size > 0 else float('inf')
    
    return {
        "original_size_bytes": orig_size,
        "quantized_size_bytes": quant_size,
        "compression_ratio": compression_ratio,
        "original_parameters": orig_params,
        "quantized_parameters": quant_params
    }

def print_quantization_stats(stats: Dict[str, Any]) -> None:
    """Print quantization statistics in a readable format"""
    print("\n=== Quantization Statistics ===")
    
    # Size information
    orig_mb = stats["original_size_bytes"] / (1024 * 1024)
    quant_mb = stats["quantized_size_bytes"] / (1024 * 1024)
    
    print(f"Original model size: {orig_mb:.2f} MB")
    print(f"Quantized model size: {quant_mb:.2f} MB")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    
    # Parameter counts by data type
    print("\nOriginal model parameters by type:")
    for dtype, count in stats["original_parameters"].items():
        print(f"  {dtype}: {count:,} parameters")
    
    print("\nQuantized model parameters by type:")
    for dtype, count in stats["quantized_parameters"].items():
        print(f"  {dtype}: {count:,} parameters")

# Example usage:
"""
# Create quantization config
config = QuantizationConfig(
    bits=8,                  # Use 8-bit quantization
    group_size=128,          # Quantize in groups of 128 elements
    sym=True,                # Use symmetric quantization
    per_channel=True,        # Quantize per output channel
    quant_method="absmax",   # Use absmax quantization method
    excluded_modules=["model.embeddings"]  # Don't quantize embedding layers
)

# Apply quantization to model
quantized_model = quantize_model(original_model, config)

# Calculate and print statistics
stats = calculate_quantization_stats(original_model, quantized_model)
print_quantization_stats(stats)
""" 