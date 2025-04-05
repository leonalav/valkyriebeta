from dataclasses import dataclass, asdict
from typing import Dict, Any
from config.model_config import ModelConfig

@dataclass
class AdvancedModelConfig(ModelConfig):
    """Configuration for advanced model features"""
    # MoE configuration
    use_moe: bool = False
    num_experts: int = 8
    moe_capacity_factor: float = 1.25
    top_k_experts: int = 2
    expert_dropout: float = 0.1
    
    # Reasoning configuration
    use_tree_reasoning: bool = True
    reasoning_depth: int = 4
    use_neural_symbolic: bool = True
    use_formal_verification: bool = True
    use_mcts: bool = True
    mcts_simulations: int = 100
    use_recursive_reasoning: bool = True
    recursive_depth: int = 3
    use_knowledge_reasoning: bool = True
    knowledge_graph_size: int = 1000
    
    # Attention configuration
    use_enhanced_attention: bool = True
    attention_mechanism: str = "efficient"  # options: standard, flash, efficient, linear
    use_hierarchical_attention: bool = True
    use_sparse_attention: bool = False
    sparse_attention_pattern: str = "fixed"
    use_local_attention: bool = False
    local_window_size: int = 128
    
    # Memory configuration
    use_memory_augmentation: bool = True
    memory_size: int = 1024
    use_episodic_memory: bool = True
    use_working_memory: bool = True
    
    # Transformer configuration
    rotary_emb_base: int = 10000
    use_knowledge_integration: bool = True
    use_cache: bool = True
    
    # Numerical precision
    use_numerical_precision: bool = True
    numerical_precision_mode: str = "auto"
    use_fp8_matmul: bool = False
    use_stable_embedding: bool = True
    math_precision_enabled: bool = True
    
    # LoRA and Adapters
    use_lora: bool = False
    lora_rank: int = 8
    use_adapters: bool = False
    adapter_size: int = 64
    
    def __str__(self):
        base_str = super().__str__()
        return base_str 