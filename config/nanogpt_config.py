from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Any

@dataclass
class MemoryConfig:
    """Configuration for memory optimizations"""
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    activation_checkpointing: bool = True
    optimize_memory_use: bool = True
    mem_efficient_linear: bool = True
    cpu_offload: bool = False
    low_cpu_mem_usage: bool = True
    max_memory_MB: Optional[int] = None
    
    # Advanced memory mechanisms
    use_episodic_memory: bool = True
    episodic_memory_size: int = 1024
    use_working_memory: bool = True
    working_memory_size: int = 512
    use_long_term_memory: bool = True
    long_term_memory_size: int = 4096
    use_memory_router: bool = True
    memory_update_frequency: int = 10
    
    def __str__(self):
        return str(asdict(self))

@dataclass
class TrainingEfficiencyConfig:
    """Configuration for training optimizations"""
    use_mixed_precision: bool = True
    optimize_cuda_kernels: bool = True
    optimize_grouping: bool = True
    compile_model: bool = False  # PyTorch 2.0+ feature
    dynamo_backend: Optional[str] = None
    use_fused_adam: bool = True
    use_fused_layer_norm: bool = True
    
    # Advanced efficiency options
    activation_checkpointing: bool = True
    checkpoint_every_n_layers: int = 2
    use_sharded_ddp: bool = False
    use_fsdp: bool = False
    use_offload: bool = False
    use_cpu_offload: bool = False
    gradient_accumulation_steps: int = 1
    
    def __str__(self):
        return str(asdict(self))

@dataclass
class ModelConfig:
    """Base configuration for model architecture"""
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    vocab_size: int = 32000
    max_seq_len: int = 2048
    dropout: float = 0.1
    use_rotary_embeddings: bool = True
    tie_weights: bool = True
    use_gradient_checkpointing: bool = True
    
    def __str__(self):
        return str(asdict(self))
    
    def to_dict(self):
        return asdict(self)

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

class EnhancedTrainingConfig:
    def __init__(self, **kwargs):
        # Basic training parameters
        self.learning_rate = kwargs.get('learning_rate', 5e-5)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.adam_beta1 = kwargs.get('adam_beta1', 0.9)
        self.adam_beta2 = kwargs.get('adam_beta2', 0.999)
        self.adam_epsilon = kwargs.get('adam_epsilon', 1e-8)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.num_train_epochs = kwargs.get('num_train_epochs', 3)
        self.max_steps = kwargs.get('max_steps', -1)
        self.warmup_steps = kwargs.get('warmup_steps', 0)
        self.warmup_ratio = kwargs.get('warmup_ratio', 0.0)
        self.logging_steps = kwargs.get('logging_steps', 100)
        self.save_steps = kwargs.get('save_steps', 1000)
        self.save_total_limit = kwargs.get('save_total_limit', 3)
        self.evaluation_strategy = kwargs.get('evaluation_strategy', 'epoch')
        self.eval_steps = kwargs.get('eval_steps', 500)
        self.batch_size = kwargs.get('batch_size', 8)
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        
        # Advanced training parameters
        self.use_mixed_precision = kwargs.get('use_mixed_precision', True)
        self.use_distributed = kwargs.get('use_distributed', False)
        self.local_rank = kwargs.get('local_rank', -1)
        self.world_size = kwargs.get('world_size', 1)
        self.use_deepspeed = kwargs.get('use_deepspeed', False)
        self.use_fsdp = kwargs.get('use_fsdp', False)
        self.use_sharded_ddp = kwargs.get('use_sharded_ddp', False)
        self.use_gradient_checkpointing = kwargs.get('use_gradient_checkpointing', True)
        self.use_flash_attention = kwargs.get('use_flash_attention', True)
        self.use_fused_adam = kwargs.get('use_fused_adam', True)
        self.use_fused_layer_norm = kwargs.get('use_fused_layer_norm', True)
        self.use_compile = kwargs.get('use_compile', False)
        self.dynamo_backend = kwargs.get('dynamo_backend', None)
        
        # Domain-specific training
        self.use_domain_training = kwargs.get('use_domain_training', False)
        self.domain_weights = kwargs.get('domain_weights', None)
        self.domain_mixing_strategy = kwargs.get('domain_mixing_strategy', 'proportional')
        
        # Knowledge distillation
        self.use_distillation = kwargs.get('use_distillation', False)
        self.distillation_alpha = kwargs.get('distillation_alpha', 0.5)
        self.distillation_temperature = kwargs.get('distillation_temperature', 2.0)
        
        # RLHF parameters
        self.use_rlhf = kwargs.get('use_rlhf', False)
        self.rlhf_type = kwargs.get('rlhf_type', 'ppo')
        self.reward_model_path = kwargs.get('reward_model_path', None)
        self.reference_model_path = kwargs.get('reference_model_path', None)
        self.kl_penalty = kwargs.get('kl_penalty', 0.1)
        self.ppo_epochs = kwargs.get('ppo_epochs', 4)
        self.ppo_mini_batch_size = kwargs.get('ppo_mini_batch_size', 4)
        
        # Reasoning parameters
        self.use_reasoning = kwargs.get('use_reasoning', False)
        self.reasoning_type = kwargs.get('reasoning_type', 'adaptive')
        self.reasoning_depth = kwargs.get('reasoning_depth', 3)
        
        # Evaluation parameters
        self.evaluation_metrics = kwargs.get('evaluation_metrics', ['accuracy', 'perplexity'])
        self.comprehensive_evaluation = kwargs.get('comprehensive_evaluation', False)
        
    def __str__(self):
        return str(self.__dict__)
    
    def to_dict(self):
        return self.__dict__

class AdaptiveReasoningConfig:
    def __init__(self, **kwargs):
        # Adaptive reasoning strategy selection
        self.use_adaptive_reasoning = kwargs.get('use_adaptive_reasoning', True)
        self.reasoning_strategies = kwargs.get('reasoning_strategies', ['tree', 'recursive', 'neural_symbolic', 'knowledge', 'mcts'])
        self.strategy_selection_method = kwargs.get('strategy_selection_method', 'learned')  # options: learned, heuristic, random
        
        # Tree reasoning
        self.tree_reasoning_depth = kwargs.get('tree_reasoning_depth', 4)
        self.tree_branching_factor = kwargs.get('tree_branching_factor', 3)
        self.tree_pruning_threshold = kwargs.get('tree_pruning_threshold', 0.1)
        
        # Recursive reasoning
        self.recursive_depth = kwargs.get('recursive_depth', 3)
        self.recursive_aggregation = kwargs.get('recursive_aggregation', 'weighted_sum')  # options: weighted_sum, max, attention
        
        # Neural-symbolic reasoning
        self.use_symbolic_rules = kwargs.get('use_symbolic_rules', True)
        self.symbolic_rule_count = kwargs.get('symbolic_rule_count', 100)
        self.neural_symbolic_integration = kwargs.get('neural_symbolic_integration', 'deep')  # options: shallow, deep, hybrid
        
        # Knowledge reasoning
        self.knowledge_graph_size = kwargs.get('knowledge_graph_size', 1000)
        self.knowledge_retrieval_method = kwargs.get('knowledge_retrieval_method', 'attention')  # options: attention, similarity, hybrid
        self.knowledge_update_frequency = kwargs.get('knowledge_update_frequency', 100)
        
        # MCTS reasoning
        self.mcts_simulations = kwargs.get('mcts_simulations', 100)
        self.mcts_exploration_constant = kwargs.get('mcts_exploration_constant', 1.0)
        self.mcts_temperature = kwargs.get('mcts_temperature', 1.0)
        
        # Chain of thought
        self.use_chain_of_thought = kwargs.get('use_chain_of_thought', True)
        self.cot_max_steps = kwargs.get('cot_max_steps', 5)
        self.cot_verbosity = kwargs.get('cot_verbosity', 'medium')  # options: low, medium, high
        
    def __str__(self):
        return str(self.__dict__)
    
    def to_dict(self):
        return self.__dict__ 