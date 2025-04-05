import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class MemoryEfficientLayerNorm(nn.Module):
    """Memory-efficient layer normalization with fused operations"""
    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 device=None,
                 dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
    def _fused_normalization(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused computation of mean and variance"""
        # Compute mean and variance in a single pass
        # Shape: [batch_size, seq_length, 1]
        mean_and_variance = torch.stack(
            [x.mean(dim=-1, keepdim=True),
             x.var(dim=-1, keepdim=True, unbiased=False)],
            dim=-1
        )
        
        mean = mean_and_variance[..., 0]
        variance = mean_and_variance[..., 1]
        
        return mean, variance
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, variance = self._fused_normalization(x)
        
        # Normalize with fused operations
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        
        if self.elementwise_affine:
            x = x * self.weight + self.bias
            
        return x

class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalization with dynamic scaling"""
    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-5,
                 device=None,
                 dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters for adaptive scaling
        self.scale_factor = nn.Parameter(torch.ones(1, device=device, dtype=dtype))
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Apply adaptive scaling
        scale = self.scale_factor * torch.rsqrt(variance + self.eps)
        x = (x - mean) * scale
        
        return x * self.weight + self.bias

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-5,
                 partial: float = -1,
                 device=None,
                 dtype=None):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.partial = partial
        
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.partial < 0:
            # Standard RMS norm
            norm_x = x.norm(2, dim=-1, keepdim=True)
            rms_x = norm_x * x.size(-1) ** (-0.5)
            x_normed = x / (rms_x + self.eps)
        else:
            # Partial RMS norm
            partial_size = int(self.normalized_shape * self.partial)
            norm_x = x[..., :partial_size].norm(2, dim=-1, keepdim=True)
            rms_x = norm_x * (partial_size ** (-0.5))
            x_normed = x / (rms_x + self.eps)
            
        return self.weight * x_normed

def get_norm_layer(config) -> nn.Module:
    """Factory function to get the appropriate normalization layer"""
    if config.norm_type == 'layer':
        return MemoryEfficientLayerNorm(
            config.hidden_size,
            eps=config.norm_eps
        )
    elif config.norm_type == 'adaptive':
        return AdaptiveLayerNorm(
            config.hidden_size,
            eps=config.norm_eps
        )
    elif config.norm_type == 'rms':
        return RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            partial=config.partial_norm
        )
    else:
        raise ValueError(f"Unknown normalization type: {config.norm_type}") 