import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from torch import Tensor
import logging

logger = logging.getLogger(__name__)

class CoLT5Layer(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 num_experts: int = 4,
                 expert_capacity: int = 64,
                 dropout: float = 0.1,
                 expert_dropout: float = 0.1,
                 use_kernel_fusion: bool = True,
                 use_expert_parallelism: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        # Expert layers with enhanced initialization
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(expert_dropout)
            ) for _ in range(num_experts)
        ])
        
        # Enhanced routing mechanism
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, num_experts),
            nn.Dropout(dropout)
        )
        
        # Expert selection parameters
        self.top_k = min(2, num_experts)  # Dynamic top-k selection
        self.expert_dropout = nn.Dropout(dropout)
        
        # Performance optimizations
        self.use_kernel_fusion = use_kernel_fusion
        self.use_expert_parallelism = use_expert_parallelism
        
        # Initialize with kaiming_normal for better stability
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='gelu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        
        # Get routing scores with enhanced stability
        router_logits = self.router(x)  # [B, L, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        router_probs = self.expert_dropout(router_probs)
        
        # Expert selection with top-k routing
        expert_weights, expert_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts)
        
        # Process through selected experts with optional optimizations
        output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            # Get inputs for this expert
            expert_input = x * expert_mask[..., expert_idx].unsqueeze(-1)
            
            # Process through expert with optional kernel fusion
            if self.use_kernel_fusion and not self.training:
                expert_output = self._fused_expert_forward(expert_idx, expert_input)
            else:
                expert_output = self.experts[expert_idx](expert_input)
            
            # Weight by expert probability
            weighted_output = expert_output * expert_weights[..., expert_idx].unsqueeze(-1)
            output = output + weighted_output
            
        return output
    
    def _fused_expert_forward(self, expert_idx: int, x: Tensor) -> Tensor:
        """Optimized forward pass for inference with fused operations"""
        expert = self.experts[expert_idx]
        # Fused linear + GELU + linear operations
        x = F.linear(x, expert[0].weight, expert[0].bias)
        x = F.gelu(x)
        x = F.linear(x, expert[2].weight, expert[2].bias)
        return x

class CoLT5Block(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 num_layers: int = 4,
                 num_experts: int = 4,
                 expert_capacity: int = 64,
                 dropout: float = 0.1,
                 layer_dropout: float = 0.0,
                 use_residual: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            CoLT5Layer(
                hidden_size=hidden_size,
                num_experts=num_experts,
                expert_capacity=expert_capacity,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.layer_dropout = layer_dropout
        self.use_residual = use_residual
        
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        for layer in self.layers:
            if self.training and torch.rand(1).item() < self.layer_dropout:
                continue  # Skip layer during training with some probability
                
            x = layer(x)
            
        if self.use_residual:
            return self.norm(x + residual)
        return self.norm(x)

def replace_ffn_with_colt5(model: nn.Module, 
                          num_layers: int = 4,
                          num_experts: int = 4,
                          expert_capacity: int = 64,
                          layer_indices: Optional[List[int]] = None,
                          verbose: bool = True):
    """
    Enhanced version of replace_ffn_with_colt5 with more control and logging
    
    Args:
        model: The model to modify
        num_layers: Number of layers in each CoLT5 block
        num_experts: Number of experts per layer
        expert_capacity: Capacity of each expert
        layer_indices: Specific layer indices to replace (None for all)
        verbose: Whether to log replacement details
    """
    if layer_indices is None:
        # Default to replacing all eligible layers
        layer_indices = []
        for name, module in model.named_children():
            if isinstance(module, nn.Linear) and module.in_features == module.out_features:
                layer_indices.append(int(name))
    
    replacements = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            if not layer_indices or int(name) in layer_indices:
                if verbose:
                    logger.info(f"Replacing layer {name} with CoLT5Block")
                
                colt5_block = CoLT5Block(
                    hidden_size=module.in_features,
                    num_layers=num_layers,
                    num_experts=num_experts,
                    expert_capacity=expert_capacity
                )
                setattr(model, name, colt5_block)
                replacements += 1
        else:
            # Recursively apply to child modules
            replace_ffn_with_colt5(
                module, 
                num_layers, 
                num_experts, 
                expert_capacity,
                layer_indices,
                verbose
            )
    
    if verbose:
        logger.info(f"Total CoLT5 replacements: {replacements}")