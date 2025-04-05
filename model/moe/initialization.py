"""
Specialized initialization for Mixture of Experts models.
Provides optimized initialization strategies for expert networks.
"""

import torch
import torch.nn as nn
import logging
import math
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

def initialize_experts(model: nn.Module, init_strategy: str = "kaiming"):
    """
    Initialize MoE experts with specialized strategies.
    
    Args:
        model: Model with MoE components to initialize
        init_strategy: Initialization strategy ("kaiming", "xavier", "orthogonal")
    """
    # Count initialized modules
    count = 0
    
    # Find all expert modules in the model
    expert_modules = []
    
    # First pass: find MoE layers and expert modules
    for name, module in model.named_modules():
        # Check for common MoE module names
        if any(expert_name in name.lower() for expert_name in 
               ['expert', 'moe_layer', 'moe.experts', 'expert_layer']):
            expert_modules.append((name, module))
        
        # Check for ExpertGating modules
        if hasattr(module, 'experts') and isinstance(module.experts, (list, nn.ModuleList)):
            for i, expert in enumerate(module.experts):
                expert_modules.append((f"{name}.experts.{i}", expert))
    
    # Initialize each found expert
    for name, module in expert_modules:
        _init_expert_module(module, strategy=init_strategy)
        count += 1
    
    # Also initialize expert gates/routers
    gate_count = 0
    for name, module in model.named_modules():
        if any(gate_name in name.lower() for gate_name in 
               ['expert_gate', 'expert_router', 'moe_gate', 'moe_router', 'expert_gating']):
            _init_expert_gate(module, strategy=init_strategy)
            gate_count += 1
    
    logger.info(f"Initialized {count} expert modules and {gate_count} expert gates using {init_strategy} strategy")


def _init_expert_module(module: nn.Module, strategy: str = "kaiming"):
    """
    Initialize a single expert module with the specified strategy.
    
    Args:
        module: Expert module to initialize
        strategy: Initialization strategy
    """
    # Initialize linear layers with specified strategy
    for name, param in module.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            if strategy == "kaiming":
                # Use Kaiming initialization for ReLU-based experts
                nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
            elif strategy == "xavier":
                # Xavier is good for sigmoid/tanh activation functions
                nn.init.xavier_uniform_(param, gain=1.0)
            elif strategy == "orthogonal":
                # Orthogonal helps with gradient flow
                nn.init.orthogonal_(param, gain=1.0)
        
        elif 'bias' in name:
            # Initialize biases to small positive values for better expert utilization
            nn.init.constant_(param, 0.01)


def _init_expert_gate(module: nn.Module, strategy: str = "kaiming"):
    """
    Initialize expert gating/routing mechanism.
    
    Args:
        module: Expert gate module to initialize
        strategy: Initialization strategy
    """
    # Find gate parameters - typically these are the weights that route inputs to experts
    gate_weight = None
    gate_bias = None
    
    # Common parameter names for gates
    for name, param in module.named_parameters():
        if any(gate_name in name.lower() for gate_name in ['gate', 'router', 'w_gate']):
            if 'weight' in name:
                gate_weight = param
            elif 'bias' in name:
                gate_bias = param
    
    # Generic fallback if specific gate parameters not found
    if gate_weight is None:
        for name, param in module.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                gate_weight = param
            elif 'bias' in name:
                gate_bias = param
    
    # Initialize gate weights - special initialization to promote balanced routing
    if gate_weight is not None:
        # For gate weights, we want to initialize to small random values
        # to avoid initial expert specialization
        std_dev = 0.01  # Small standard deviation for initial balance
        
        if strategy == "orthogonal":
            nn.init.orthogonal_(gate_weight, gain=std_dev)
        else:
            nn.init.normal_(gate_weight, mean=0.0, std=std_dev)
    
    # Initialize gate biases - set to ensure roughly uniform expert usage
    if gate_bias is not None:
        # Specific bias initialization for load balancing
        nn.init.zeros_(gate_bias)


def init_baseline_expert(model: nn.Module):
    """
    Initialize a baseline expert that performs well on average.
    This helps with router training in early stages.
    
    Args:
        model: Model with MoE components
    """
    # Find all expert modules
    expert_modules = []
    
    for name, module in model.named_modules():
        if any(expert_name in name.lower() for expert_name in 
               ['expert', 'moe_layer', 'moe.experts', 'expert_layer']):
            expert_modules.append((name, module))
    
    # If experts were found
    if expert_modules:
        # Choose one expert as baseline (by convention, the first one)
        baseline_name, baseline_module = expert_modules[0]
        
        # Initialize baseline expert with conservative weights
        for name, param in baseline_module.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Small standard deviation for stable initial predictions
                nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        logger.info(f"Initialized baseline expert: {baseline_name}") 