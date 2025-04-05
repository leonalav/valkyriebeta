from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import os
import json
import torch

@dataclass
class BasicTrainingConfig:
    """Basic training parameters"""
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

@dataclass
class CheckpointConfig:
    """Checkpointing parameters"""
    checkpoint_interval: int = 1
    save_total_limit: int = 3
    save_strategy: str = "epoch"  # Options: epoch, steps
    save_steps: int = 500

@dataclass
class EvaluationConfig:
    """Evaluation parameters"""
    eval_strategy: str = "epoch"  # Options: epoch, steps
    eval_steps: int = 500

@dataclass
class LoggingConfig:
    """Logging parameters"""
    logging_dir: str = "logs"
    logging_steps: int = 100

@dataclass
class OptimizerConfig:
    """Optimizer parameters"""
    optimizer_type: str = "adamw"  # Options: adamw, adafactor, sgd, lamb
    scheduler_type: str = "linear"  # Options: linear, cosine, constant, constant_with_warmup

@dataclass
class MixedPrecisionConfig:
    """Mixed precision parameters"""
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"  # Options: float16, bfloat16

@dataclass
class DistributedTrainingConfig:
    """Distributed training parameters"""
    use_distributed_training: bool = False
    local_rank: int = -1
    world_size: int = 1
    pipeline_parallel_size: int = 1
    use_fairscale_pipeline: bool = True
    fairscale_balance: Optional[List[int]] = None

@dataclass
class EnhancedRAGConfig:
    """Configuration for Retrieval-Augmented Generation"""
    hidden_size: int = 768
    retriever_dim: int = 768
    num_attention_heads: int = 8
    dropout: float = 0.1
    max_knowledge_items: int = 100
    use_approximate_search: bool = True
    index_type: str = "IVF"  # Options: IVF, HNSW, Flat
    num_partitions: int = 100
    num_probe: int = 10
    similarity_metric: str = "ip"  # ip (inner product) or l2
    normalize_embeddings: bool = True

@dataclass
class RWKVConfig:
    """Configuration for RWKV layers"""
    layer_indices: List[int] = field(default_factory=lambda: [])
    time_mix_factor: float = 1.0
    key_value_mixing: bool = True
    att_scale: float = 1.0
    use_linear_attn: bool = False
    use_gating: bool = True
    use_shifting: bool = True

@dataclass
class LongContextMoEConfig:
    """Configuration for Long Context Mixture of Experts"""
    hidden_size: int = 768
    num_experts: int = 8
    max_seq_length: int = 16384
    block_size: int = 2048
    token_routing_budget: float = 0.3
    use_rwkv_integration: bool = False
    use_gradient_checkpointing: bool = True
    use_state_compression: bool = False
    use_quantization: bool = False
    use_qat: bool = False

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation"""
    teacher_model_path: Optional[str] = None
    teacher_model_api: Optional[str] = None
    alpha: float = 0.5  # Weight for distillation loss vs task loss
    temperature: float = 2.0
    distill_logits: bool = True
    distill_hidden_states: bool = True
    distill_attention: bool = True
    use_progressive_distillation: bool = True
    progressive_stages: List[str] = field(default_factory=lambda: ["base", "reasoning", "domain_specific", "fine_tuning"])
    current_stage: str = "base"

@dataclass
class DomainSpecificConfig:
    """Configuration for domain-specific data handling"""
    domains: List[str] = field(default_factory=lambda: ["general", "math", "science", "logic", "coding"])
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "general": 1.0,
        "math": 1.5,
        "science": 1.2,
        "logic": 1.3,
        "coding": 1.4
    })
    domain_data_paths: Dict[str, str] = field(default_factory=dict)
    augment_vocabulary: bool = True
    domain_vocab_files: Dict[str, str] = field(default_factory=dict)
    use_curriculum: bool = True
    mixing_strategy: str = "proportional"  # Options: proportional, equal, curriculum
    augment_low_resource: bool = True
    min_domain_examples: int = 1000

@dataclass
class ComputationalEfficiencyConfig:
    """Configuration for computational efficiency optimizations"""
    use_memory_budgeting: bool = True
    max_memory_usage: float = 0.9  # Fraction of available memory to use
    attention_memory_limit: float = 0.4  # Fraction of budget for attention
    ffn_memory_limit: float = 0.3  # Fraction of budget for feed-forward
    residual_memory_limit: float = 0.2  # Fraction of budget for residuals
    safety_margin: float = 0.1  # Safety buffer
    auto_scale_components: bool = True
    min_attention_scale: float = 0.5
    min_ffn_scale: float = 0.5
    use_activation_checkpointing: bool = True
    checkpoint_every_n_layers: int = 2
    use_efficient_attention: bool = True
    attention_implementation: str = "flash"  # Options: flash, memory_efficient, sparse
    use_early_exit: bool = True
    exit_threshold: float = 0.9
    min_layers: int = 4
    use_conditional_computation: bool = True
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"  # Options: float16, bfloat16
    use_adaptive_batch_size: bool = True
    min_batch_size: int = 8
    max_batch_size: int = 64
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"  # Options: default, reduce-overhead, max-autotune

@dataclass
class AdaptiveReasoningConfig:
    """Configuration for adaptive reasoning"""
    low_complexity_threshold: float = 0.3
    medium_complexity_threshold: float = 0.7
    max_computation_budget: float = 1.0
    min_computation_budget: float = 0.2
    use_early_exit: bool = True
    early_exit_threshold: float = 0.9

@dataclass
class DatasetConfig:
    """Configuration for dataset and data loading"""
    dataset_path: str = "data/my_dataset"
    dataset_type: str = "text"  # Options: text, image, audio, etc.
    tokenizer_name: str = "my_tokenizer"
    max_seq_length: int = 512
    train_file: str = "train.txt"
    eval_file: str = "eval.txt"
    preprocessing_steps: Optional[List[str]] = field(default_factory=lambda: ["tokenize", "numericalize"])
    num_workers: int = 4
    prefetch_factor: int = 2

@dataclass
class EnhancedTrainingConfig:
    """Enhanced training configuration, composed of modular configs"""
    basic_config: BasicTrainingConfig = field(default_factory=BasicTrainingConfig)
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)
    eval_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    mixed_precision_config: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)
    distributed_config: DistributedTrainingConfig = field(default_factory=DistributedTrainingConfig)
    distillation_config: 'DistillationConfig' = field(default_factory=lambda: DistillationConfig())
    domain_config: 'DomainSpecificConfig' = field(default_factory=lambda: DomainSpecificConfig())
    efficiency_config: 'ComputationalEfficiencyConfig' = field(default_factory=lambda: ComputationalEfficiencyConfig())
    adaptive_reasoning_config: 'AdaptiveReasoningConfig' = field(default_factory=lambda: AdaptiveReasoningConfig())
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    
    # RAG configuration
    rag_config: EnhancedRAGConfig = field(default_factory=EnhancedRAGConfig)
    rwkv_config: Optional[RWKVConfig] = None
    moe_config: Optional[LongContextMoEConfig] = None
    
    # Training parameters from train_llm.py
    output_dir: str = "output"
    experiment_name: str = "llm_training"
    resume_from_checkpoint: Optional[str] = None
    model_type: str = "gpt2"
    max_seq_length: int = 2048
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    use_moe: bool = False
    use_rwkv: bool = False
    use_rag: bool = False
    use_distributed: bool = False
    local_rank: int = -1

    def __post_init__(self):
        """Initialize default values for paths if not provided"""
        if not self.domain_config.domain_data_paths:
            self.domain_config.domain_data_paths = {
                domain: f"data/{domain}" for domain in self.domain_config.domains
            }

        if not self.domain_config.domain_vocab_files:
            self.domain_config.domain_vocab_files = {
                domain: f"data/{domain}/vocab.json" for domain in self.domain_config.domains
            }

    def validate(self):
        """Validate configuration by delegating to sub-configs and performing cross-config validations"""
        errors = []
        warnings = []

        # Hardware capability checks
        if self.mixed_precision_config.use_mixed_precision and not torch.cuda.is_bf16_supported() and self.mixed_precision_config.mixed_precision_dtype == "bfloat16":
            errors.append("bfloat16 not supported on this hardware")

        if self.efficiency_config.attention_implementation == "flash" and not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            warnings.append("Flash attention requires PyTorch 2.0+, falling back to memory_efficient")
            self.efficiency_config.attention_implementation = "memory_efficient"

        # Memory constraints
        if self.efficiency_config.use_adaptive_batch_size and self.efficiency_config.max_batch_size > 64:
            warnings.append("Large max_batch_size may cause OOM, consider reducing")

        # Feature compatibility
        if self.efficiency_config.use_early_exit and self.efficiency_config.use_conditional_computation:
            warnings.append("Early exit and conditional computation may interact unpredictably")

        # Distributed training validation
        if self.distributed_config.use_distributed_training and not self.distributed_config.use_fairscale_pipeline:
            warnings.append("Using deprecated torch pipeline implementation, recommend switching to FairScale")

        if self.distributed_config.pipeline_parallel_size > 1 and not self.distributed_config.use_fairscale_pipeline:
            errors.append("Pipeline parallelism requires FairScale implementation")

        if errors:
            raise ValueError("\n".join(errors))

        return warnings

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, including nested configs"""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'): # Handle nested dataclasses
                config_dict[field_name] = field_value.to_dict()
            else:
                config_dict[field_name] = field_value
        return config_dict

    def save(self, path: str):
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedTrainingConfig':
        """Create config from dictionary, handling nested configs"""
        kwargs = {}
        for key, value in config_dict.items():
            if key == 'basic_config':
                kwargs[key] = BasicTrainingConfig.from_dict(value)
            elif key == 'checkpoint_config':
                kwargs[key] = CheckpointConfig.from_dict(value)
            elif key == 'eval_config':
                kwargs[key] = EvaluationConfig.from_dict(value)
            elif key == 'logging_config':
                kwargs[key] = LoggingConfig.from_dict(value)
            elif key == 'optimizer_config':
                kwargs[key] = OptimizerConfig.from_dict(value)
            elif key == 'mixed_precision_config':
                kwargs[key] = MixedPrecisionConfig.from_dict(value)
            elif key == 'distributed_config':
                kwargs[key] = DistributedTrainingConfig.from_dict(value)
            elif key == 'distillation_config':
                kwargs[key] = DistillationConfig.from_dict(value)
            elif key == 'domain_config':
                kwargs[key] = DomainSpecificConfig.from_dict(value)
            elif key == 'efficiency_config':
                kwargs[key] = ComputationalEfficiencyConfig.from_dict(value)
            elif key == 'adaptive_reasoning_config':
                kwargs[key] = AdaptiveReasoningConfig.from_dict(value)
            elif key == 'dataset_config':
                kwargs[key] = DatasetConfig.from_dict(value)
            elif key == 'rag_config':
                kwargs[key] = EnhancedRAGConfig.from_dict(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str) -> 'EnhancedTrainingConfig':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
