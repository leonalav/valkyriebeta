from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any


@dataclass
class ComputationalEfficiencyConfig:
    """
    Configuration for computational efficiency optimizations in the model.
    
    This config focuses on runtime optimization strategies such as mixed precision,
    efficient attention mechanisms, and memory optimizations.
    """
    # Mixed precision settings
    use_mixed_precision: bool = False
    mixed_precision_dtype: str = "float16"  # Options: "float16", "bfloat16"
    
    # Activation checkpointing
    use_activation_checkpointing: bool = False
    checkpoint_every_n_layers: int = 1
    
    # Memory optimization
    optimize_memory_use: bool = True
    use_memory_efficient_attention: bool = True
    
    # Attention optimizations
    use_efficient_attention: bool = True
    attention_implementation: str = "flash"  # Options: "flash", "xformers", "standard"
    
    # Flash attention specific settings
    flash_attention_dropout: float = 0.0
    use_flash_attention_2: bool = True  # Use Flash Attention 2 if available
    
    # Fused operations
    use_fused_operations: bool = True
    use_fused_layernorm: bool = True
    use_fused_softmax: bool = True
    
    # Early exit for inference
    use_early_exit: bool = False
    early_exit_thresholds: Optional[List[float]] = None
    
    # Quantization for inference
    use_quantization: bool = False
    quantization_config: Optional[Dict[str, Any]] = None
    
    # Dynamic batch sizing
    use_adaptive_batch_size: bool = False
    min_batch_size: int = 1
    max_batch_size: int = 64
    
    # Optimized kernels
    use_optimized_kernels: bool = True
    
    # For RWKV-specific optimizations
    rwkv_optimize_state_mem: bool = True
    rwkv_chunk_size: int = 1024
    rwkv_use_double_quant: bool = False
    
    # TPU-specific optimizations
    tpu_optimization: bool = False
    batch_process_with_pad: bool = True
    optimize_tpu_kernels: bool = True
    
    def __post_init__(self):
        # Ensure early exit thresholds are initialized if enabled
        if self.use_early_exit and not self.early_exit_thresholds:
            self.early_exit_thresholds = [0.9, 0.8, 0.7, 0.6]
        
        # Initialize quantization config if enabled but not provided
        if self.use_quantization and not self.quantization_config:
            self.quantization_config = {
                "method": "dynamic",
                "bits": 8,
                "compute_dtype": "float16"
            } 