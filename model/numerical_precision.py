import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np

@dataclass
class NumericalPrecisionConfig:
    """Configuration for numerical precision in Valkyrie LLM"""
    precision_mode: str = "mixed"  # Options: "mixed", "float32", "float16", "bfloat16"
    use_numerical_precision: bool = True
    default_precision: str = "float16"
    quantization_aware: bool = False
    quantization_precision: int = 8  # Options: 4, 8, 16
    calibration_steps: int = 100
    dynamic_scaling: bool = True
    scale_factor: float = 1.0
    precision_by_layer: Dict[str, str] = field(default_factory=dict)  # Make this use default_factory
    mixed_precision_mapping: Dict[str, str] = field(default_factory=dict)  # Make this use default_factory
    
    # Epsilon values for different operations
    div_epsilon: float = 1e-12
    log_epsilon: float = 1e-12
    sqrt_epsilon: float = 1e-12
    
    # Clipping ranges
    exp_clip_max: float = 88.0  # prevent overflow in exp
    exp_clip_min: float = -88.0
    
    # Precision control
    use_double_precision: bool = False
    use_mixed_precision: bool = True
    
    # Specialized computation
    use_kahan_summation: bool = True
    use_compensated_dot_product: bool = True
    
    # Numerical verification
    verify_invertibility: bool = True
    verify_nan_inf: bool = True
    
    # Adaptive precision
    adaptive_precision_threshold: float = 1e-4


class NumericallyStableOperations(nn.Module):
    """Provides numerically stable versions of common mathematical operations"""
    
    def __init__(self, config: NumericalPrecisionConfig):
        super().__init__()
        self.config = config
    
    def safe_div(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """Numerically stable division operation"""
        # Add small epsilon to denominator to prevent division by zero
        safe_denominator = denominator + self.config.div_epsilon * (denominator == 0).float()
        return numerator / safe_denominator
    
    def safe_log(self, x: torch.Tensor) -> torch.Tensor:
        """Numerically stable logarithm"""
        # Ensure input is positive by adding small epsilon
        safe_x = x + self.config.log_epsilon * (x <= 0).float()
        return torch.log(safe_x)
    
    def safe_sqrt(self, x: torch.Tensor) -> torch.Tensor:
        """Numerically stable square root"""
        # Ensure input is non-negative
        safe_x = torch.clamp(x, min=self.config.sqrt_epsilon)
        return torch.sqrt(safe_x)
    
    def safe_exp(self, x: torch.Tensor) -> torch.Tensor:
        """Numerically stable exponential function"""
        # Clip to prevent overflow/underflow
        safe_x = torch.clamp(x, min=self.config.exp_clip_min, max=self.config.exp_clip_max)
        return torch.exp(safe_x)
    
    def safe_softmax(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Numerically stable softmax"""
        # Subtract max for stability
        x_max = torch.max(x, dim=dim, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    
    def kahan_sum(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Kahan summation algorithm for more accurate summation
        Reduces numerical errors in floating point addition
        """
        if not self.config.use_kahan_summation:
            return torch.sum(x, dim=dim)
            
        # Implementation of Kahan summation
        x_np = x.detach().cpu().numpy()
        sum_result = np.zeros(x_np.shape[:dim] + x_np.shape[dim+1:] if dim != -1 else x_np.shape[:-1])
        compensation = np.zeros_like(sum_result)
        
        # Perform Kahan summation
        for i in range(x_np.shape[dim]):
            # Extract slice along dimension
            if dim == -1:
                slice_idx = (..., i)
            else:
                slice_idx = tuple(slice(None) if j != dim else i for j in range(len(x_np.shape)))
                
            y = x_np[slice_idx] - compensation
            t = sum_result + y
            compensation = (t - sum_result) - y
            sum_result = t
            
        return torch.tensor(sum_result, device=x.device, dtype=x.dtype)
    
    def compensated_dot_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compensated dot product for higher precision
        Uses error compensation techniques
        """
        if not self.config.use_compensated_dot_product:
            return torch.matmul(a, b)
            
        # For higher precision, compute in double and convert back
        if a.dtype != torch.float64 and self.config.use_double_precision:
            a_double = a.double()
            b_double = b.double()
            result = torch.matmul(a_double, b_double)
            return result.to(a.dtype)
        
        return torch.matmul(a, b)
    
    def verify_tensor(self, x: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """Verify tensor doesn't contain NaN or Inf values"""
        if self.config.verify_nan_inf:
            has_nan = torch.isnan(x).any()
            has_inf = torch.isinf(x).any()
            
            if has_nan or has_inf:
                # Replace problematic values with zeros
                x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
                
        return x


class PrecisionAdaptiveLinear(nn.Linear):
    """Linear layer with adaptive precision based on input magnitude"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 config: NumericalPrecisionConfig = None):
        super().__init__(in_features, out_features, bias)
        self.config = config or NumericalPrecisionConfig()
        self.stable_ops = NumericallyStableOperations(self.config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive precision"""
        # Check if we need higher precision
        input_magnitude = torch.abs(x).mean()
        
        if (input_magnitude < self.config.adaptive_precision_threshold and 
            self.config.use_double_precision):
            # Use double precision for small magnitude inputs
            weight_double = self.weight.double()
            x_double = x.double()
            
            if self.bias is not None:
                bias_double = self.bias.double()
                output = F.linear(x_double, weight_double, bias_double)
            else:
                output = F.linear(x_double, weight_double)
                
            return output.to(x.dtype)
        
        # Regular computation with verification
        output = F.linear(x, self.weight, self.bias)
        return self.stable_ops.verify_tensor(output, "linear_output")


class HighPrecisionMathOperations(nn.Module):
    """Implements high precision mathematical operations for advanced reasoning"""
    
    def __init__(self, config: NumericalPrecisionConfig):
        super().__init__()
        self.config = config
        self.stable_ops = NumericallyStableOperations(config)
        
    def matrix_inverse(self, x: torch.Tensor, verify: bool = True) -> torch.Tensor:
        """Compute matrix inverse with numerical stability checks"""
        # Add small regularization to diagonal for stability
        batch_size, n, m = x.shape
        assert n == m, "Matrix must be square for inversion"
        
        # Add small epsilon to diagonal
        x_reg = x.clone()
        diag_indices = torch.arange(n, device=x.device)
        x_reg[:, diag_indices, diag_indices] += self.config.div_epsilon
        
        # Compute inverse
        try:
            x_inv = torch.linalg.inv(x_reg)
        except RuntimeError:
            # Fallback to pseudo-inverse
            x_inv = torch.linalg.pinv(x_reg)
        
        if verify and self.config.verify_invertibility:
            # Verify inverse by checking if A * A^-1 ≈ I
            identity = torch.matmul(x, x_inv)
            diag_ones = torch.eye(n, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
            error = torch.norm(identity - diag_ones) / (n * batch_size)
            
            if error > 0.1:  # High error threshold
                # Use more stable pseudo-inverse
                x_inv = torch.linalg.pinv(x)
        
        return self.stable_ops.verify_tensor(x_inv, "matrix_inverse")
    
    def determinant(self, x: torch.Tensor) -> torch.Tensor:
        """Compute determinant with numerical stability"""
        # For better numerical stability, use LU decomposition
        if self.config.use_double_precision:
            x_double = x.double()
            det = torch.linalg.det(x_double)
            return det.to(x.dtype)
        
        return torch.linalg.det(x)
    
    def eigendecomposition(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute eigenvalues and eigenvectors with numerical stability"""
        # For symmetric matrices, use more stable eigendecomposition
        if self.config.use_double_precision:
            x_double = x.double()
            eigenvalues, eigenvectors = torch.linalg.eigh(x_double)
            return eigenvalues.to(x.dtype), eigenvectors.to(x.dtype)
        
        return torch.linalg.eigh(x)
    
    def svd(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SVD with numerical stability"""
        if self.config.use_double_precision:
            x_double = x.double()
            u, s, v = torch.linalg.svd(x_double, full_matrices=False)
            return u.to(x.dtype), s.to(x.dtype), v.to(x.dtype)
        
        return torch.linalg.svd(x, full_matrices=False)
    
    def log_determinant(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log determinant with numerical stability"""
        # Use eigenvalues for more stable log determinant
        eigenvalues, _ = self.eigendecomposition(x)
        return torch.sum(self.stable_ops.safe_log(eigenvalues), dim=-1)
    
    def matrix_power(self, x: torch.Tensor, p: float) -> torch.Tensor:
        """Compute matrix power with numerical stability"""
        # Use eigendecomposition for stable matrix power
        eigenvalues, eigenvectors = self.eigendecomposition(x)
        powered_eigenvalues = torch.pow(self.stable_ops.safe_sqrt(eigenvalues), p)
        
        # Reconstruct matrix
        diag_powered = torch.diag_embed(powered_eigenvalues)
        return torch.matmul(torch.matmul(eigenvectors, diag_powered), eigenvectors.transpose(-2, -1)) 


class NumericalPrecisionModule:
    """
    Module for handling numerical precision operations in Valkyrie LLM.
    Provides utilities for precision casting, quantization, and other numerical operations.
    """
    def __init__(self, config: NumericalPrecisionConfig = None):
        self.config = config or NumericalPrecisionConfig()
        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float16
        }
        
    def get_dtype(self, layer_name: str = None) -> torch.dtype:
        """Get appropriate dtype for the specified layer"""
        if not layer_name or not self.config.precision_by_layer:
            return self.dtype_map.get(self.config.default_precision, torch.float32)
            
        # Check if specific precision is assigned to this layer
        layer_precision = self.config.precision_by_layer.get(layer_name)
        if layer_precision:
            return self.dtype_map.get(layer_precision, torch.float32)
            
        return self.dtype_map.get(self.config.default_precision, torch.float32)
        
    def apply_precision(self, tensor: torch.Tensor, layer_name: str = None) -> torch.Tensor:
        """Apply appropriate precision to the tensor"""
        if not self.config.use_numerical_precision:
            return tensor
            
        target_dtype = self.get_dtype(layer_name)
        return tensor.to(dtype=target_dtype)
        
    def apply_quantization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply quantization if enabled"""
        if not self.config.quantization_aware:
            return tensor
            
        # Simple mock implementation
        return tensor  # In a real implementation, this would apply actual quantization 