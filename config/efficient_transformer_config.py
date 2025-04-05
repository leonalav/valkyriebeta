from dataclasses import dataclass, asdict

@dataclass
class EfficientTransformerConfig:
    """Configuration specific to EfficientTransformer"""
    ffn_hidden_size: int = 3072
    layer_norm_epsilon: float = 1e-5
    attention_dropout: float = 0.1
    use_bias: bool = True
    activation_function: str = "gelu_new"
    
    # Advanced configuration
    use_rmsnorm: bool = False
    use_parallel_attention: bool = True
    use_swiglu: bool = True
    use_flash_attention: bool = True
    use_efficient_qkv: bool = True
    use_fused_operations: bool = True
    use_rope_scaling: bool = True
    rope_scaling_factor: float = 1.0
    
    def __str__(self):
        return str(asdict(self)) 