import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from typing import Optional, Dict, Any, List, Tuple

def implement_gradient_checkpointing(self) -> nn.Module:
    """Implement gradient checkpointing for memory-efficient training"""
    if self.config.use_gradient_checkpointing:
        # Enable gradient checkpointing for transformer layers
        if hasattr(self.model, 'layers'):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            # Apply checkpointing to each transformer layer
            for layer in self.model.layers:
                if isinstance(layer, dict):
                    # For dict-style layers
                    for key in ['attention', 'feed_forward', 'logical_reasoning']:
                        if key in layer and layer[key] is not None:
                            layer[key].gradient_checkpointing = True
                else:
                    # For regular nn.Module layers
                    layer.gradient_checkpointing = True
                    
            # Modify the forward pass to use checkpointing
            def forward_with_checkpointing(self, *inputs):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                return torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.original_forward),
                    *inputs
                )
            
            # Store original forward
            self.model.original_forward = self.model.forward
            # Replace with checkpointed version
            self.model.forward = types.MethodType(forward_with_checkpointing, self.model)
            
    return self.model

def implement_activation_checkpointing(self) -> nn.Module:
    """Implement activation checkpointing for memory-efficient inference"""
    if self.config.use_activation_checkpointing:
        class ActivationCheckpointing(nn.Module):
            def __init__(self):
                super().__init__()
                self.checkpoint_activations = True
                
            def forward(self, x):
                if self.checkpoint_activations and self.training:
                    return torch.utils.checkpoint.checkpoint(
                        lambda x: F.gelu(x),
                        x
                    )
                return F.gelu(x)
        
        # Replace activation functions with checkpointed versions
        for name, module in self.model.named_modules():
            if isinstance(module, nn.GELU):
                parent = get_parent_module(self.model, name)
                setattr(parent, name.split('.')[-1], ActivationCheckpointing())
                
    return self.model

def get_parent_module(model, name):
    """Helper function to get parent module"""
    parent_name = '.'.join(name.split('.')[:-1])
    if parent_name:
        for n, m in model.named_modules():
            if n == parent_name:
                return m
    return model