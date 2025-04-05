import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import os
import json

@dataclass
class ModelConfig:
    """
    Configuration class for model architecture and training parameters.
    
    This class contains all the configuration parameters for the model architecture,
    attention mechanisms, and tree reasoning components.
    """
    # Model architecture
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Model dimensions
    vocab_size: int = 50257
    n_positions: int = 2048
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    
    # Attention mechanisms
    num_heads: int = 12
    num_query_groups: int = 12
    use_flash_attention: bool = False
    use_xformers_attention: bool = False
    use_sliding_window: bool = False
    use_rotary_embeddings: bool = True
    sliding_window_size: int = 512
    global_tokens: int = 64
    
    # Performance and optimization
    use_sparse_linear: bool = False
    quantize_linear: bool = False
    quantization_bits: int = 8
    bias: bool = True
    use_checkpoint: bool = False
    
    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Tree reasoning configuration
    use_tree_reasoning: bool = False
    branching_factor: int = 4
    max_tree_depth: int = 5
    
    # MCTS configuration
    mcts_max_iterations: int = 100
    mcts_exploration_weight: float = 1.0
    mcts_rollout_depth: int = 3
    mcts_discount_factor: float = 0.95
    use_value_function: bool = True
    monte_carlo_samples: int = 8
    early_stopping_threshold: float = 0.95
    use_beam_search: bool = True
    beam_size: int = 4
    
    # Attention mechanism selection
    use_linear_attention: bool = False
    linear_attention_kernel_size: int = 4
    linear_attention_feature_dim: int = 16
    use_causal_mask: bool = True
    
    # Layer-specific attention configuration
    layer_specific_attention: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        """Load configuration from pretrained model path"""
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found at {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
            
        return cls(**config_dict)
    
    @classmethod
    def create_1_3b_config(cls, context_length: int = 32768) -> "ModelConfig":
        """
        Create a configuration for a ~1.3B parameter model with extended context length.
        
        Args:
            context_length: Maximum context length, defaults to 32K (32768)
            
        Returns:
            ModelConfig: Configuration for a ~1.3B parameter model
        """
        # Parameters sized for ~1.3B total parameters
        return cls(
            # Core model dimensions (these define the parameter count)
            hidden_size=2048,        # Increased from 768
            num_hidden_layers=24,    # Increased from 12
            num_attention_heads=16,  # Increased from 12
            intermediate_size=8192,  # Increased from 3072, 4x hidden size
            
            # Update all related parameters to match
            n_embd=2048,
            n_layer=24,
            n_head=16,
            num_heads=16,
            num_query_groups=16,
            
            # Extended context length
            max_position_embeddings=context_length,
            n_positions=context_length,
            
            # Performance optimizations needed for large context
            use_flash_attention=True,  # Enable fast attention
            use_checkpoint=True,       # Enable gradient checkpointing to save memory
            
            # Long context optimizations
            use_rotary_embeddings=True,
            sliding_window_size=4096,  # Increased sliding window size
            use_sliding_window=True,   # Enable sliding window attention for efficiency
            
            # Keep tree reasoning intact
            use_tree_reasoning=True,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save configuration to directory"""
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @property
    def head_dim(self) -> int:
        """Calculate head dimension based on hidden size and number of heads"""
        return self.hidden_size // self.num_heads   