"""
Configuration file for 3B RWKV model

This file defines the configuration for the 3B parameter RWKV model,
which combines RWKV layers with transformer layers for efficient
processing of long sequences.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import math

@dataclass
class ModelConfig3BRWKV:
    """Configuration for 3B RWKV model with extended context"""
    
    # General model architecture - reduced dimensions as suggested
    hidden_size: int = 2048  # Reduced from 2560 to save memory
    num_layers: int = 20     # Fewer but smarter layers (was 32)
    num_attention_heads: int = 32
    vocab_size: int = 50257  # GPT-2 vocabulary size
    max_position_embeddings: int = 49152  # Extended context support (~48K)
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # RWKV specific parameters
    use_rwkv: bool = True
    rwkv_time_mix_ratio: float = 1.0
    rwkv_use_linear_att: bool = True
    rwkv_att_scale: float = 1.0
    # Every 4th layer is transformer (indices not in this list)
    rwkv_layer_indices: List[int] = field(default_factory=lambda: [i for i in range(20) if i % 4 != 3])
    rwkv_chunk_size: int = 8192  # For sliding window inference
    rwkv_chunk_overlap: int = 1024  # Overlap between chunks
    rwkv_state_compression: bool = True
    
    # Transformer sparse attention parameters
    transformer_window_size: int = 512  # Local attention window size
    use_grouped_query_attention: bool = True  # GQA to reduce memory
    num_kv_groups: int = 8  # For GQA - share KV heads across queries
    
    # Memory optimizations
    use_flash_attention: bool = True
    use_8bit_kv_cache: bool = True  # KV cache compression
    use_gradient_checkpointing: bool = True
    
    # Hybrid configuration
    attention_mechanism: str = "hybrid_rwkv_transformer"
    use_sliding_window: bool = True
    use_adaptive_chunking: bool = True
    
    # Training parameters
    use_alibi: bool = True  # Better for extrapolating to longer sequences
    tie_word_embeddings: bool = False
    use_cache: bool = True
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bfloat16"  # More stable than fp16
    
    # Hardware adaptation
    target_vram_gb: int = 8
    optimize_for_inference: bool = True
    
    def get_rwkv_layer_config(self, layer_id: int) -> Dict[str, Any]:
        """
        Get layer-specific configuration for RWKV layer
        
        Args:
            layer_id: Layer ID
            
        Returns:
            Dictionary with layer configuration
        """
        # Scale certain parameters based on layer depth
        scale = 1.0 - 0.5 * (layer_id / max(1, self.num_layers - 1))
        
        return {
            'layer_index': layer_id,
            'hidden_size': self.hidden_size,
            'time_mix_ratio': self.rwkv_time_mix_ratio * scale,
            'use_linear_att': self.rwkv_use_linear_att,
            'att_scale': self.rwkv_att_scale,
            'use_gate_res': self.rwkv_use_gated_residual,
            'gate_init': self.rwkv_gate_init,
            'layer_scale': 1.0,
            'ffn_scale': self.rwkv_ffn_scale,
            'use_glu': self.rwkv_use_glu,
            'time_decay_base': self.rwkv_time_decay_base * (self.rwkv_recurrent_scaling ** layer_id)
        }

# Default configs
model_config = ModelConfig3BRWKV()

# Memory configuration for RWKV
@dataclass
class RWKVMemoryConfig:
    """Memory configuration optimized for RWKV models"""
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    activation_checkpointing: bool = True
    optimize_memory_use: bool = True
    mem_efficient_linear: bool = True
    cpu_offload: bool = False
    low_cpu_mem_usage: bool = True
    max_memory_MB: Optional[int] = None
    
    # RWKV-specific memory config
    rwkv_chunk_size: int = 1024
    rwkv_state_compression: bool = True
    rwkv_token_streaming: bool = True
    rwkv_cpu_offload_ratio: float = 0.0  # No CPU offloading by default
    
    # Advanced memory mechanisms
    use_episodic_memory: bool = False  # Disable by default for RWKV
    use_working_memory: bool = False   # Disable by default for RWKV

memory_config = RWKVMemoryConfig()

# Training efficiency config for RWKV
@dataclass
class RWKVTrainingEfficiencyConfig:
    """Training efficiency configuration optimized for RWKV"""
    use_mixed_precision: bool = True
    optimize_cuda_kernels: bool = True
    optimize_grouping: bool = True
    compile_model: bool = False
    dynamo_backend: Optional[str] = None
    use_fused_adam: bool = True
    use_fused_layer_norm: bool = True
    
    # RWKV specific optimizations
    rwkv_optimize_backward: bool = True
    rwkv_backward_pass_length: int = 1024
    
    # Advanced efficiency options
    activation_checkpointing: bool = True
    checkpoint_every_n_layers: int = 2
    use_sharded_ddp: bool = False
    use_fsdp: bool = False
    use_offload: bool = False
    use_cpu_offload: bool = False
    gradient_accumulation_steps: int = 1

training_config = RWKVTrainingEfficiencyConfig()

# RWKV-specific architecture parameters
@dataclass
class RWKVArchitectureParams:
    """RWKV-specific architecture parameters"""
    ffn_hidden_size: int = 4 * model_config.hidden_size
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_scaled_init: bool = True
    
    # RWKV learning rate parameters
    rwkv_lr_multiplier: float = 1.2
    rwkv_att_weight_decay: float = 0.0
    rwkv_ffn_weight_decay: float = 0.01
    
    # Hybrid model parameters
    hybrid_attention_ratio: float = 0.5
    hybrid_use_flash_attention: bool = True
    rwkv_time_shift: bool = True

architecture_params = RWKVArchitectureParams()

# Consolidate all parameters into a dictionary for easy access
training_params = {
    "rwkv_lr_multiplier": architecture_params.rwkv_lr_multiplier,
    "rwkv_att_weight_decay": architecture_params.rwkv_att_weight_decay,
    "rwkv_ffn_weight_decay": architecture_params.rwkv_ffn_weight_decay
} 