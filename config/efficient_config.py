import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from .model_config import ModelConfig
from .training_config import TrainingConfig
from .memory_config import MemoryConfig
from .training_efficiency_config import TrainingEfficiencyConfig

logger = logging.getLogger(__name__)

@dataclass
class EfficientTransformerConfig:
    """Configuration for the efficient transformer that integrates with all other configs"""
    
    # Update vocabulary size for Gemma
    vocab_size: int = 256000  # Gemma's vocabulary size
    hidden_size: int = 2048  # Increased for 1.3B params
    num_layers: int = 24     # Increased depth
    num_heads: int = 16      # Increased heads
    max_seq_length: int = 2048
    intermediate_size: int = 8192  # 4x hidden_size
    dropout: float = 0.1
    num_classes: int = 2
    
    # Memory optimizations (enhanced for larger model)
    use_memory: bool = True
    memory_size: int = 2048
    use_quantization: bool = True
    pad_token_id: int = 0
    use_8bit_training: bool = True  # 8-bit training for memory efficiency
    use_4bit_quantization: bool = True  # More aggressive quantization
    
    # Attention optimizations
    use_flash_attention: bool = True
    attention_implementation: str = 'flash'  # 'flash' or 'efficient'
    use_grouped_query_attention: bool = True  # More efficient attention for large models
    num_query_groups: int = 4  # Reduces memory usage
    use_sliding_window: bool = True  # Use sliding window attention
    sliding_window_size: int = 256  # Size of attention window
    
    # Layer optimizations
    use_memory_efficient_linear: bool = True
    use_efficient_layer_norm: bool = True
    use_parallel_attention: bool = True
    use_fused_operations: bool = True
    
    # Training specific (adjusted for lower memory usage)
    batch_size: int = 1  # Minimal batch size
    gradient_accumulation_steps: int = 32  # Accumulate gradients instead of large batches
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    gradient_checkpointing: bool = True
    activation_checkpointing: bool = True
    selective_activation_checkpointing: bool = True  # Only checkpoint critical layers
    
    # Memory management
    max_memory_gb: float = 12.0  # Maximum GPU memory usage in GB
    cpu_offload: bool = True
    memory_efficient_fusion: bool = True
    dynamic_memory_allocation: bool = True
    layer_dropping: float = 0.0  # Can be increased if memory is still tight
    
    # Optimization flags
    use_amp: bool = True  # Automatic mixed precision
    amp_dtype: str = 'bfloat16'  # Better numerical stability than float16
    use_gradient_scaling: bool = True
    
    # Sequential layer loading
    sequential_loading: bool = True  # Load layers sequentially to save memory
    num_layers_per_load: int = 6  # Load 6 layers at a time
    
    # Add version tracking
    model_version: str = field(default="1.0.0")
    config_version: str = field(default="1.0.0")
    
    # Add Gemma-specific tokenizer config
    tokenizer_name: str = "google/gemma-2b-27b-it"
    tokenizer_padding_side: str = "left"
    tokenizer_truncation_side: str = "left"
    
    @classmethod
    def from_configs(
        cls,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
        efficiency_config: Optional[TrainingEfficiencyConfig] = None
    ) -> 'EfficientTransformerConfig':
        """Create config from other configuration objects"""
        try:
            config = cls()
            
            if model_config:
                # Model architecture
                config.vocab_size = model_config.vocab_size
                config.hidden_size = 2048  # Fixed for 1.3B
                config.num_layers = 24     # Fixed for 1.3B
                config.num_heads = 16      # Fixed for 1.3B
                config.max_seq_length = model_config.max_seq_length
                config.dropout = model_config.hidden_dropout
                config.num_classes = getattr(model_config, 'num_output_classes', 2)
                
                # Memory and optimization
                config.use_memory = True  # Always use memory optimizations for 1.3B
                config.memory_size = max(2048, getattr(model_config, 'memory_size', 2048))
                config.use_quantization = True  # Always use quantization for 1.3B
                config.use_4bit_quantization = True  # Enable 4-bit quantization
                
            if training_config:
                # Use gradient accumulation instead of large batches
                config.batch_size = 1
                config.gradient_accumulation_steps = 32
                config.learning_rate = training_config.learning_rate
                config.weight_decay = training_config.weight_decay
                config.warmup_steps = max(2000, training_config.warmup_steps)
                
            if memory_config:
                config.use_memory_efficient_linear = True  # Always use for 1.3B
                config.use_efficient_layer_norm = True     # Always use for 1.3B
                config.max_memory_gb = min(12.0, memory_config.max_cpu_memory_gb)  # Cap at 12GB
                config.cpu_offload = True  # Always enable CPU offloading
                config.sequential_loading = True  # Enable sequential layer loading
                
            if efficiency_config:
                config.use_flash_attention = True  # Always use flash attention
                config.attention_implementation = 'flash'
                config.use_amp = True  # Always use mixed precision
                config.amp_dtype = 'bfloat16'
                
            config.validate()
            return config
        except Exception as e:
            logger.error(f"Failed to create config: {str(e)}")
            raise
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict = {
            'model_config': {
                'vocab_size': self.vocab_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'max_seq_length': self.max_seq_length,
                'intermediate_size': self.intermediate_size,
                'dropout': self.dropout,
                'num_classes': self.num_classes,
            },
            'memory_config': {
                'use_memory': self.use_memory,
                'memory_size': self.memory_size,
                'use_quantization': self.use_quantization,
                'use_4bit_quantization': self.use_4bit_quantization,
                'use_8bit_training': self.use_8bit_training,
                'max_memory_gb': self.max_memory_gb,
                'cpu_offload': self.cpu_offload,
                'sequential_loading': self.sequential_loading,
                'num_layers_per_load': self.num_layers_per_load,
            },
            'optimization_config': {
                'use_flash_attention': self.use_flash_attention,
                'attention_implementation': self.attention_implementation,
                'use_grouped_query_attention': self.use_grouped_query_attention,
                'num_query_groups': self.num_query_groups,
                'use_sliding_window': self.use_sliding_window,
                'sliding_window_size': self.sliding_window_size,
                'use_memory_efficient_linear': self.use_memory_efficient_linear,
                'use_efficient_layer_norm': self.use_efficient_layer_norm,
                'use_parallel_attention': self.use_parallel_attention,
                'use_fused_operations': self.use_fused_operations,
                'layer_dropping': self.layer_dropping,
            },
            'training_config': {
                'batch_size': self.batch_size,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'warmup_steps': self.warmup_steps,
                'gradient_checkpointing': self.gradient_checkpointing,
                'activation_checkpointing': self.activation_checkpointing,
                'selective_activation_checkpointing': self.selective_activation_checkpointing,
                'use_amp': self.use_amp,
                'amp_dtype': self.amp_dtype,
            },
            "versions": {
                "model_version": self.model_version,
                "config_version": self.config_version
            }
        }
        return config_dict
        
    def get_parameter_count(self) -> int:
        """Calculate approximate parameter count"""
        embedding_params = self.vocab_size * self.hidden_size
        attention_params = self.num_layers * (
            3 * self.hidden_size * self.hidden_size +  # QKV projections
            self.hidden_size * self.hidden_size        # Output projection
        )
        ffn_params = self.num_layers * (
            self.hidden_size * self.intermediate_size +  # First FFN layer
            self.intermediate_size * self.hidden_size    # Second FFN layer
        )
        other_params = self.hidden_size * self.num_classes  # Output head
        
        total_params = embedding_params + attention_params + ffn_params + other_params
        return total_params
        
    def validate_memory_requirements(self) -> bool:
        """Check if configuration is feasible given memory constraints"""
        # Approximate memory requirements (in GB)
        params_memory = self.get_parameter_count() * 4 / (1024 ** 3)  # 4 bytes per parameter
        activation_memory = (
            self.batch_size * self.max_seq_length * self.hidden_size * 4 / (1024 ** 3)
        )
        
        total_memory = params_memory + activation_memory
        
        # Apply memory reduction factors
        if self.use_amp:
            total_memory *= 0.6  # Mixed precision savings
        if self.use_8bit_training:
            total_memory *= 0.5  # 8-bit training savings
        if self.use_4bit_quantization:
            total_memory *= 0.5  # Additional 4-bit quantization savings
        if self.sequential_loading:
            total_memory *= (self.num_layers_per_load / self.num_layers)  # Sequential loading savings
        if self.use_sliding_window:
            attention_memory_factor = self.sliding_window_size / self.max_seq_length
            total_memory *= (0.7 + 0.3 * attention_memory_factor)  # Sliding window attention savings
            
        # Account for gradient accumulation
        effective_batch_memory = total_memory * (1 / self.gradient_accumulation_steps)
        
        return effective_batch_memory <= self.max_memory_gb

    def validate(self) -> None:
        """Validate configuration values and relationships"""
        if not self.validate_memory_requirements():
            raise ValueError("Configuration exceeds memory constraints")
            
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
            
        if self.use_grouped_query_attention and self.num_heads % self.num_query_groups != 0:
            raise ValueError("num_heads must be divisible by num_query_groups")
        
        logger.info(f"Configuration validated successfully (version {self.config_version})")