from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Dict, Any, Tuple
import os
import torch
import logging
from pathlib import Path

# Fix relative imports
try:
    from utils.resource_manager import ResourceConfig
except ImportError:
    # Create mock ResourceConfig
    @dataclass
    class ResourceConfig:
        max_memory_gb: int = 24
        max_gpu_memory_gb: int = 16
        num_gpus: int = 1
        cpu_threads: int = 8

try:
    from utils.customization import CustomizationConfig
except ImportError:
    # Create mock CustomizationConfig
    @dataclass
    class CustomizationConfig:
        theme: str = "default"
        custom_modules: List[str] = field(default_factory=list)

try:
    from security.validator import SecurityConfig
except ImportError:
    # Create mock SecurityConfig
    @dataclass
    class SecurityConfig:
        enable_validation: bool = True
        max_token_limit: int = 100000
        allowed_modules: List[str] = field(default_factory=list)

try:
    from utils.compliance_checker import ComplianceConfig
except ImportError:
    # Create mock ComplianceConfig
    @dataclass
    class ComplianceConfig:
        check_compliance: bool = True
        compliance_level: str = "standard"

try:
    from training.distributed_handler import DistributedConfig
except ImportError:
    # Create mock DistributedConfig
    @dataclass
    class DistributedConfig:
        world_size: int = 1
        local_rank: int = 0
        distributed_backend: str = "nccl"
        use_distributed: bool = False

try:
    from config.base_config import BaseConfig
except ImportError:
    # Create mock BaseConfig
    @dataclass
    class BaseConfig:
        name: str = "default"
        version: str = "1.0.0"

logger = logging.getLogger(__name__)

@dataclass
class BaseModelConfig:
    """Base configuration for all model variants"""
    
    # Core architecture parameters
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 8192
    max_position_embeddings: int = 4096
    
    # Basic model parameters
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    
    # Activation and normalization
    activation_function: str = "gelu"  # ["gelu", "swiglu", "geglu"]
    norm_type: str = "layernorm"  # ["layernorm", "rmsnorm"]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration parameters"""
        valid = True
        errors = []
        
        # Validate numeric ranges
        if self.dropout < 0 or self.dropout > 1:
            valid = False
            errors.append(f"Dropout must be between 0 and 1, got {self.dropout}")
            
        if self.hidden_size % self.num_heads != 0:
            valid = False
            errors.append(f"Hidden size ({self.hidden_size}) must be divisible by number of heads ({self.num_heads})")
            
        # Validate enum values
        valid_activations = ["gelu", "swiglu", "geglu"]
        if self.activation_function not in valid_activations:
            valid = False
            errors.append(f"Activation function must be one of {valid_activations}, got {self.activation_function}")
            
        valid_norms = ["layernorm", "rmsnorm"]
        if self.norm_type not in valid_norms:
            valid = False
            errors.append(f"Norm type must be one of {valid_norms}, got {self.norm_type}")
            
        return valid, errors

@dataclass
class StandardModelConfig(BaseModelConfig):
    """Standard configuration for basic language models"""
    
    # Attention mechanism
    attention_type: str = "vanilla"  # ["vanilla", "multi_query", "grouped_query"]
    
    # Positional embeddings
    position_embedding_type: str = "learned"  # ["learned", "rotary", "alibi", "relative"]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate standard model configuration"""
        valid, errors = super().validate()
        
        # Validate enum values
        valid_attention_types = ["vanilla", "multi_query", "grouped_query"]
        if self.attention_type not in valid_attention_types:
            valid = False
            errors.append(f"Attention type must be one of {valid_attention_types}, got {self.attention_type}")
            
        valid_pos_embedding_types = ["learned", "rotary", "alibi", "relative"]
        if self.position_embedding_type not in valid_pos_embedding_types:
            valid = False
            errors.append(f"Position embedding type must be one of {valid_pos_embedding_types}, got {self.position_embedding_type}")
            
        return valid, errors

@dataclass
class AdvancedModelConfig(BaseModelConfig):
    """Enhanced configuration for production LLM"""
    
    # Advanced architecture parameters
    max_position_embeddings: int = 32768  # Override base value
    
    # Modern architecture components
    attention_type: str = "grouped_query"  # ["multi_query", "grouped_query", "flash_attention"]
    num_query_groups: int = 2
    use_rotary_embeddings: bool = True
    rotary_embedding_base: int = 10000
    activation_function: str = "swiglu"  # Override base value
    
    # Performance optimizations
    use_flash_attention: bool = True
    use_memory_efficient_attention: bool = True
    use_xformers: bool = False
    
    # Quantization options
    use_4bit_quantization: bool = False
    use_8bit_quantization: bool = False
    quantization_type: str = "nf4"  # ["nf4", "int8", "fp4"]
    
    # Mixture of Experts
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.0
    
    # Memory networks
    use_memory_networks: bool = False
    memory_size: int = 1024
    memory_key_dim: int = 64
    
    # Resource constraints
    max_memory_MB: int = 24000
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate advanced model configuration"""
        valid, errors = super().validate()
        
        # Validate enum values
        valid_attention_types = ["multi_query", "grouped_query", "flash_attention"]
        if self.attention_type not in valid_attention_types:
            valid = False
            errors.append(f"Attention type must be one of {valid_attention_types}, got {self.attention_type}")
            
        # Validate quantization settings
        if self.use_4bit_quantization and self.use_8bit_quantization:
            valid = False
            errors.append("Cannot use both 4-bit and 8-bit quantization simultaneously")
            
        valid_quantization_types = ["nf4", "int8", "fp4"]
        if self.quantization_type not in valid_quantization_types:
            valid = False
            errors.append(f"Quantization type must be one of {valid_quantization_types}, got {self.quantization_type}")
            
        # Validate MoE settings
        if self.use_moe:
            if self.num_experts < 2:
                valid = False
                errors.append(f"Number of experts must be at least 2, got {self.num_experts}")
                
            if self.num_experts_per_token > self.num_experts:
                valid = False
                errors.append(f"Number of experts per token ({self.num_experts_per_token}) cannot exceed total experts ({self.num_experts})")
                
        # Validate memory requirements
        if not self._validate_memory_requirements():
            valid = False
            errors.append("Estimated memory usage exceeds max_memory_MB limit")
            
        return valid, errors
    
    def _validate_memory_requirements(self) -> bool:
        """Validate that the model configuration fits within memory constraints"""
        try:
            total_memory_MB = self._estimate_memory_usage()
            logger.info(f"Estimated memory usage: {total_memory_MB}MB (limit: {self.max_memory_MB}MB)")
            return total_memory_MB <= self.max_memory_MB
        except Exception as e:
            logger.warning(f"Failed to estimate memory usage: {e}")
            return True  # Assume valid if estimation fails
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB based on model configuration"""
        # Base model parameters
        param_bytes = 4  # 32-bit float by default
        if self.use_8bit_quantization:
            param_bytes = 1
        elif self.use_4bit_quantization:
            param_bytes = 0.5
            
        # Estimate parameter count
        num_params = self.vocab_size * self.hidden_size  # Embeddings
        num_params += self.max_position_embeddings * self.hidden_size  # Position embeddings
        
        # Transformer layers
        params_per_layer = 4 * self.hidden_size * self.hidden_size  # Attention
        params_per_layer += 2 * self.hidden_size  # Layer norms
        params_per_layer += 2 * self.hidden_size * self.intermediate_size  # FF layers
        
        if self.use_moe:
            # MoE increases parameter count but not active memory
            num_params += params_per_layer * self.num_layers * self.num_experts
            # But we only use num_experts_per_token at a time
            active_params = num_params + params_per_layer * self.num_layers * self.num_experts_per_token
        else:
            num_params += params_per_layer * self.num_layers
            active_params = num_params
            
        # Convert to MB
        params_memory_MB = (num_params * param_bytes) / (1024 * 1024)
        
        # Activation memory (rough estimate)
        batch_size = 32  # Assume default batch size
        seq_length = self.max_position_embeddings
        activation_memory_MB = (batch_size * seq_length * self.hidden_size * 4) / (1024 * 1024) * self.num_layers
        
        # Optimizer states (Adam has 2 states per parameter)
        optimizer_memory_MB = params_memory_MB * 2
        
        # Additional memory for memory networks
        memory_networks_MB = 0
        if self.use_memory_networks:
            memory_networks_MB = (self.memory_size * self.memory_key_dim * 4) / (1024 * 1024)
            
        # Sum all memory components with a safety margin
        total_memory_MB = (params_memory_MB + activation_memory_MB + optimizer_memory_MB + memory_networks_MB) * 1.1
        
        return total_memory_MB

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
class TrainingConfig:
    batch_size: int = 32
    eval_batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    log_interval: int = 100
    save_interval: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_gradient_checkpointing: bool = True
    distributed: bool = False
    seed: int = 42
    model_name: str = "valkyrie-base"

@dataclass
class MemoryConfig:
    max_memory_usage: float = 0.9
    min_batch_size: int = 1
    use_cpu_offload: bool = False
    prefetch_factor: int = 2
    num_workers: int = 4

@dataclass
class TrainingEfficiencyConfig:
    use_amp: bool = True
    use_jit_compilation: bool = True
    use_cuda_graphs: bool = True
    num_cuda_streams: int = 2
    initial_lr: float = 1e-5
    max_lr: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000

@dataclass
class EfficientModelConfig:
    # Architecture
    vocab_size: int = 32000  # Reduced vocabulary size
    max_seq_length: int = 2048  # Shorter context for efficiency
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_query_groups: int = 2  # For grouped-query attention
    
    # MoE Configuration
    use_moe: bool = True
    num_experts: int = 4
    tokens_per_expert: int = 64
    expert_dropout: float = 0.1
    
    # Optimization
    use_grouped_query_attention: bool = True
    use_swiglu: bool = True
    intermediate_ratio: float = 2.0  # Reduced from standard 4.0
    
    # Quantization
    quantization_bits: int = 8
    quantization_scheme: str = "dynamic"  # or "static" 

@dataclass
class OptimizedModelConfig:
    # Core Architecture
    vocab_size: int = 32000
    max_seq_length: int = 2048
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    window_size: int = 256  # for sliding window attention
    
    # Memory Optimizations
    use_sliding_window: bool = True
    use_memory_efficient_attention: bool = True
    activation_checkpointing: bool = True
    
    # MoE Configuration (reduced experts for memory)
    use_moe: bool = True
    num_experts: int = 4
    num_active_experts: int = 2
    expert_capacity: int = 32
    
    # Quantization
    quantization_bits: int = 8
    quantization_strategy: str = "dynamic" 

@dataclass
class EnhancedReasoningConfig:
    # Previous configurations...
    
    # Enhanced reasoning settings
    memory_size: int = 1024
    use_tree_reasoning: bool = True
    use_knowledge_bank: bool = True
    max_reasoning_depth: int = 4
    reasoning_dropout: float = 0.1
    
    # Symbolic reasoning
    use_symbolic_reasoning: bool = True
    num_symbolic_rules: int = 64
    rule_embedding_size: int = 256
    
    # Knowledge integration
    knowledge_temperature: float = 0.8
    knowledge_update_steps: int = 2 

@dataclass
class EnhancedModelConfig:
    # Core Architecture - Reduced for memory
    vocab_size: int = 151936
    hidden_size: int = 1024  # Reduced from 2048
    num_layers: int = 16    # Reduced from 24
    num_heads: int = 8     # Reduced from 16
    max_seq_length: int = 2048  # Reduced from 48000 initially
    
    # Feed Forward Network - Simplified and explicit
    intermediate_size: int = 4096  # Fixed size for FFN
    
    # Aggressive Memory Optimizations
    use_flash_attention: bool = True
    use_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = True
    use_8bit_quantization: bool = True
    activation_checkpointing: bool = True
    empty_cache_between_chunks: bool = True
    use_cpu_offload: bool = True
    
    # Training Micro-batching
    batch_size: int = 1
    gradient_accumulation_steps: int = 128  # Increased for stable training
    chunk_size: int = 512  # Reduced chunk size
    
    # Memory Management
    max_memory_MB: int = 22000  # Reserve 2GB for system
    prefetch_factor: int = 2
    num_workers: int = 1
    pin_memory: bool = True
    
    # Model name and tokenizer
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    
    # Memory optimizations for long sequences (keep these for 24GB)
    use_rotary_embeddings: bool = True
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    # Attention optimizations
    attention_window_overlap: int = 256  # For sliding window
    
    # Memory management for long sequences
    max_position_embeddings: int = 48000  # Match max_seq_length
    
    # Reduce other parameters to compensate for memory
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # CRNN specific
    conv_channels: list = (64, 128, 256)
    conv_kernel_sizes: list = (3, 3, 3)
    gru_hidden_size: int = 512
    gru_num_layers: int = 2
    
    # Memory and Reasoning
    memory_size: int = 2048
    num_experts: int = 8
    reasoning_layers: int = 4
    tree_depth: int = 6
    num_reasoning_heads: int = 8
    use_tree_lstm: bool = True
    max_tree_depth: int = 8
    use_memory_networks: bool = True
    
    # Adapter specific
    adapter_size: int = 256
    adapter_dropout: float = 0.1
    adapter_init_scale: float = 0.001
    
    # Expert specific
    num_local_experts: int = 4
    expert_capacity: int = 32
    expert_dropout: float = 0.1
    expert_size: int = 1024
    
    # MoE specific
    moe_hidden_size: int = 1024
    moe_dropout: float = 0.1
    moe_capacity_factor: float = 1.25
    tokens_per_expert: int = 64  # Added: number of tokens each expert processes
    expert_routing_strategy: str = "top_k"  # Added: how to route tokens to experts
    routing_temperature: float = 0.1  # Added: temperature for expert routing
    min_expert_capacity: int = 32  # Added: minimum capacity per expert
    expert_overlap: float = 0.2  # Added: allowed expert overlap
    
    # Reasoning specific
    reasoning_dropout: float = 0.1
    reasoning_ffn_size: int = 1024
    num_reasoning_layers: int = 2
    max_reasoning_steps: int = 8
    
    # Knowledge distillation
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5
    
    # Meta learning
    meta_learning_rate: float = 0.001
    meta_hidden_size: int = 512
    num_strategies: int = 4  # Added: number of reasoning strategies
    strategy_dropout: float = 0.1  # Added: dropout for strategy selection
    strategy_temperature: float = 0.1  # Added: temperature for strategy sampling
    meta_update_steps: int = 5  # Added: steps between meta updates
    meta_warmup_steps: int = 100  # Added: warmup steps for meta learning
    use_strategy_mixing: bool = True  # Added: whether to mix strategies
    strategy_mixing_alpha: float = 0.2  # Added: mixing coefficient
    
    # Training specific
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Required for training
    tie_word_embeddings: bool = False
    
    # Chain of Thought Reasoning
    num_predicates: int = 32  # Number of logical predicates
    num_operators: int = 8    # Number of logical operators
    max_chain_length: int = 16  # Maximum reasoning chain length
    chain_dropout: float = 0.1  # Dropout for chain of thought
    
    # Logical Reasoning
    num_logical_rules: int = 64  # Number of logical rules
    max_rule_depth: int = 8  # Maximum depth of rule application
    rule_temperature: float = 0.1  # Temperature for rule selection
    use_rule_pruning: bool = True  # Whether to prune invalid rules
    
    # Predicate and Entity Handling
    max_predicates_per_step: int = 8  # Max predicates per reasoning step
    max_entities: int = 128  # Maximum number of entities to track
    entity_embedding_size: int = 256  # Size of entity embeddings
    use_entity_memory: bool = True  # Whether to use entity memory
    
    # Reasoning Graph
    max_graph_nodes: int = 64  # Maximum nodes in reasoning graph
    graph_attention_heads: int = 4  # Number of graph attention heads
    graph_hidden_size: int = 256  # Hidden size for graph processing
    use_graph_updates: bool = True  # Whether to update graph during reasoning
    
    # Consistency Checking
    consistency_threshold: float = 0.8  # Threshold for consistency checks
    max_consistency_steps: int = 5  # Maximum consistency verification steps
    use_consistency_verification: bool = True  # Whether to verify consistency
    
    # Inference Control
    max_inference_steps: int = 32  # Maximum inference steps
    inference_temperature: float = 0.7  # Temperature for inference
    use_beam_search: bool = True  # Whether to use beam search
    beam_size: int = 4  # Size of beam for search
    
    # Reasoning Memory
    reasoning_memory_size: int = 1024  # Size of reasoning memory
    memory_update_steps: int = 4  # Steps between memory updates
    use_episodic_memory: bool = True  # Whether to use episodic memory
    
    # Reasoning Steps Control
    num_reasoning_steps: int = 16  # Fixed number of reasoning steps
    max_reasoning_steps: int = 32  # Maximum allowed reasoning steps
    min_reasoning_steps: int = 4   # Minimum required reasoning steps
    adaptive_steps: bool = True    # Whether to use adaptive number of steps
    step_scaling_factor: float = 1.5  # Factor for adaptive step scaling
    
    # Dynamic Routing
    dynamic_routing_heads: int = 4  # Number of routing attention heads
    routing_hidden_size: int = 256  # Hidden size for routing computation
    routing_dropout: float = 0.1   # Dropout for routing layers
    use_adaptive_routing: bool = True  # Whether to use adaptive routing
    routing_update_steps: int = 2  # Steps between routing updates
    
    # Step Scheduling
    step_warmup_epochs: int = 2    # Epochs to warmup reasoning steps
    step_schedule: str = "linear"  # Step count scheduling strategy
    step_multiplier: float = 1.0   # Multiplier for step count
    
    # Output Classification
    num_output_classes: int = 2    # Number of output classes
    use_multi_class: bool = False  # Whether to use multi-class classification
    class_weights: List[float] = None  # Optional class weights
    output_dropout: float = 0.1    # Dropout for output layer
    
    # Output Head Configuration
    use_hierarchical_output: bool = False  # Whether to use hierarchical classification
    output_hidden_size: int = 512   # Hidden size for output projection
    output_activation: str = "gelu" # Activation function for output layer
    output_layer_norm: bool = True  # Whether to use layer norm in output
    
    # Loss Configuration
    label_smoothing: float = 0.1   # Label smoothing factor
    focal_loss_gamma: float = 2.0   # Focal loss gamma parameter
    use_weighted_loss: bool = False # Whether to use weighted loss

# Specify the default config to use
ModelConfig = AdvancedModelConfig