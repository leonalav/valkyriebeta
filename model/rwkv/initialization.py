"""
Specialized initialization methods for RWKV layers.
Provides optimized initialization strategies for time-mixing and channel-mixing layers.
"""

import torch
import torch.nn as nn
import logging
import math
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

def initialize_rwkv_layers(model: nn.Module, init_strategy: str = "scaled"):
    """
    Initialize RWKV layers with specialized strategies.
    
    Args:
        model: Model with RWKV layers to initialize
        init_strategy: Initialization strategy ("scaled", "orthogonal", "xavier")
    """
    # Count initialized layers
    count = 0
    
    # Find all RWKV modules in model
    for name, module in model.named_modules():
        # Initialize time-mixing layers (WKV, time-decay, etc.)
        if hasattr(module, 'time_decay') or 'rwkv_time' in name.lower():
            _init_rwkv_time_layer(module, strategy=init_strategy)
            count += 1
        
        # Initialize channel-mixing layers
        elif hasattr(module, 'key_value_mixing') or 'rwkv_channel' in name.lower():
            _init_rwkv_channel_layer(module, strategy=init_strategy)
            count += 1
        
        # Initialize RWKV blocks
        elif hasattr(module, 'time_mixer') and hasattr(module, 'channel_mixer'):
            # Init time mixer
            _init_rwkv_time_layer(module.time_mixer, strategy=init_strategy)
            # Init channel mixer
            _init_rwkv_channel_layer(module.channel_mixer, strategy=init_strategy)
            count += 2
    
    logger.info(f"Initialized {count} RWKV layers using {init_strategy} strategy")


def _init_rwkv_time_layer(module: nn.Module, strategy: str = "scaled"):
    """
    Initialize time-mixing layer of RWKV.
    
    Args:
        module: RWKV time-mixing module
        strategy: Initialization strategy
    """
    # Find parameters to initialize
    time_decay = getattr(module, 'time_decay', None)
    time_first = getattr(module, 'time_first', None)
    time_mix_k = getattr(module, 'time_mix_k', None)
    time_mix_v = getattr(module, 'time_mix_v', None)
    time_mix_r = getattr(module, 'time_mix_r', None)
    
    # Set initialization scale based on hidden dimension
    hidden_size = 0
    for name, param in module.named_parameters():
        if param.dim() >= 2:
            hidden_size = max(hidden_size, param.shape[-1])
    
    scale = 1.0 / math.sqrt(hidden_size) if hidden_size > 0 else 0.01
    
    # Initialize time decay parameter
    if time_decay is not None:
        if strategy == "scaled":
            # Initialize with values between -2 and 0
            # This gives reasonable initial time-decay factors
            nn.init.uniform_(time_decay, -2.0, 0.0)
        else:
            # For other strategies, initialize to small negative values
            nn.init.uniform_(time_decay, -1.0, -0.1)
    
    # Initialize time_first parameter
    if time_first is not None:
        if strategy == "scaled":
            nn.init.normal_(time_first, mean=0.0, std=0.01)
        elif strategy == "orthogonal":
            nn.init.orthogonal_(time_first, gain=0.01)
        else:  # xavier
            nn.init.xavier_uniform_(time_first, gain=0.01)
    
    # Initialize time mixing factors
    for mix_param in [time_mix_k, time_mix_v, time_mix_r]:
        if mix_param is not None:
            if strategy == "scaled":
                # Initialize to values closer to 0.5 for good mixing
                nn.init.uniform_(mix_param, 0.3, 0.7)
            else:
                nn.init.uniform_(mix_param, 0.0, 1.0)
    
    # Initialize linear layers if present
    for name, param in module.named_parameters():
        # Skip already initialized parameters
        if param in [time_decay, time_first, time_mix_k, time_mix_v, time_mix_r]:
            continue
        
        if 'weight' in name and param.dim() >= 2:
            if strategy == "scaled":
                nn.init.normal_(param, mean=0.0, std=scale)
            elif strategy == "orthogonal":
                nn.init.orthogonal_(param, gain=1.0)
            else:  # xavier
                nn.init.xavier_uniform_(param, gain=1.0)
        
        elif 'bias' in name:
            nn.init.zeros_(param)


def _init_rwkv_channel_layer(module: nn.Module, strategy: str = "scaled"):
    """
    Initialize channel-mixing layer of RWKV.
    
    Args:
        module: RWKV channel-mixing module
        strategy: Initialization strategy
    """
    # Find parameters to initialize
    key_value_mixing = getattr(module, 'key_value_mixing', None)
    gating = getattr(module, 'gating', None)
    
    # Set initialization scale based on hidden dimension
    hidden_size = 0
    for name, param in module.named_parameters():
        if param.dim() >= 2:
            hidden_size = max(hidden_size, param.shape[-1])
    
    scale = 1.0 / math.sqrt(hidden_size) if hidden_size > 0 else 0.01
    
    # Initialize key-value mixing
    if key_value_mixing is not None:
        if strategy == "scaled":
            nn.init.normal_(key_value_mixing, mean=0.0, std=scale)
        elif strategy == "orthogonal":
            nn.init.orthogonal_(key_value_mixing)
        else:  # xavier
            nn.init.xavier_uniform_(key_value_mixing)
    
    # Initialize gating
    if gating is not None:
        # Initialize gating close to 1 for better initial behavior
        if strategy == "scaled":
            nn.init.normal_(gating, mean=1.0, std=0.1)
        else:
            nn.init.uniform_(gating, 0.9, 1.1)
    
    # Initialize linear layers if present
    for name, param in module.named_parameters():
        # Skip already initialized parameters
        if param in [key_value_mixing, gating]:
            continue
        
        if 'weight' in name and param.dim() >= 2:
            if strategy == "scaled":
                # Special initialization for key, value, receptance
                if 'key' in name:
                    nn.init.normal_(param, mean=0.0, std=scale * 0.8)
                elif 'value' in name:
                    nn.init.normal_(param, mean=0.0, std=scale)
                elif 'receptance' in name or 'gate' in name:
                    nn.init.normal_(param, mean=0.0, std=scale * 1.2)
                else:
                    nn.init.normal_(param, mean=0.0, std=scale)
            elif strategy == "orthogonal":
                nn.init.orthogonal_(param)
            else:  # xavier
                nn.init.xavier_uniform_(param)
        
        elif 'bias' in name:
            nn.init.zeros_(param)


def init_rwkv_layer_norm(model: nn.Module):
    """
    Initialize LayerNorm modules in RWKV model with optimized settings.
    
    Args:
        model: Model with LayerNorm modules to initialize
    """
    count = 0
    
    # Find all LayerNorm modules
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            # Initialize with ones and zeros
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
            count += 1
    
    logger.info(f"Initialized {count} layer norm modules in RWKV model") 