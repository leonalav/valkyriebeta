"""
Configuration for a 3B parameter model.

This file defines the configuration parameters for a model with approximately 3B parameters.
"""

import dataclasses
from typing import Optional

@dataclasses.dataclass
class Config3B:
    """Configuration for a 3B parameter transformer model."""
    
    # Core architecture
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    intermediate_size: int = 16384  # 4x hidden size
    vocab_size: int = 50257  # GPT-2 vocabulary size
    max_seq_length: int = 16384  # 16k context
    dropout: float = 0.1
    
    # Layer norm configuration
    layer_norm_eps: float = 1e-5
    
    # Activation function
    activation: str = "gelu"
    
    # Training configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Memory efficiency
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    
    # Mixed precision training
    use_mixed_precision: bool = True
    
    def __post_init__(self):
        """Ensure intermediate_size is 4x hidden_size if not specified."""
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

# Default 3B configuration
default_config_3b = Config3B() 