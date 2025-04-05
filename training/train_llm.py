import os
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import argparse
from dataclasses import dataclass, asdict, field
import types
import config
import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset as hf_load_dataset
import wandb
from sophia import SophiaG
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    default_auto_wrap_policy
)
from torch.distributed.pipelining import Pipe
# Core model components
from model.core_model import CoreModel
from model.transformer import TransformerConfig
from model.valkyrie_llm import ValkyrieLLM
# RAG components
from model.rag import EnhancedRAG, EnhancedRAGConfig
from model.generation.rag_generator import EnhancedRAGGenerator
from model.rag_utils import KnowledgeBankManager, KnowledgeEntry
# Memory and caching
from model.memory import CacheManager
from model.memory.memory_bank import StrategySequenceMemory
# Reasoning components
from model.reasoning import (
    TreeReasoning, RecursiveReasoner, NeuralSymbolicReasoner,
    KnowledgeReasoner
)
# GNN components
from model.gnn.integration import TransformerGNNIntegration
from model.gnn.gnn_model import GNNEncoder
from model.gnn.graph_encoder import GraphEncoder
from model.gnn.tree_gnn import TreeGNN
from model.gnn.contrastive import GraphCL, InfoGraph
from model.gnn.graph_reasoning import GraphReasoningModule
from model.gnn.layers import (
    GraphConvolution, GraphAttention, GraphSAGELayer,
    EdgeGAT, GraphTransformer, DiffPool, HGT
)
# Adaptive reasoning
from model.adaptive_reasoning import (
    ComponentSelector, AdaptiveReasoningConfig, ReasoningStrategy
)
# Attention and efficiency
from model.attention import EnhancedAttention
from model.attention_mechanisms import (
    FlashAttention, SlidingWindowAttention, GroupedQueryAttention
)
from model.computational_efficiency import (
    ComputationalEfficiencyConfig, DynamicQuantizer
)
# Advanced features
from model.moe import ExpertGating
from model.numerical_precision import (
    NumericalPrecisionModule, HighPrecisionMathOperations
)
from model.reinforcement.advanced_rlhf import AdvancedRLHFIntegration
from model.formal_verification import (
    FormalVerificationConfig, UncertaintyAwareVerifier
)
from model.constitutional_ai import ConstitutionalAIConfig, ConstitutionalAI
from model.tree_reasoning import (
    AdaptiveTreeReasoner, TreeReasoningModule, TreeReasoningConfig
)
from model.math_precision_integration import EnhancedMathematicalReasoning
from model.layers import MemoryEfficientLinear
# Generation and beam search
from model.generation.beam_search import BeamSearchGenerator
from model.generation.logical_beam_search import LogicalBeamSearch
# Additional reasoning
from model.logical_reasoning import LogicalReasoningLayer
from model.mcts_reasoner import MCTSReasoner, MCTSConfig
from model.neural_symbolic import NeuralSymbolicConfig, NeuralSymbolicIntegration
from model.neural_symbolic_reasoner import SymbolicReasoningLayer
from model.recursive_reasoning import (
    RecursiveReasoningConfig, RecurrentReasoningBlock
)
from model.rotary_embeddings import RotaryEmbedding, apply_rotary_pos_emb
from model.tree_lstm import TreeLSTM, TreeLSTMConfig, TreeLSTMIntegration
from model.sat import SATSolver
# Training utilities
from training.training_engine import TrainingEngine
from training.model_setup import setup_transformer_model
from training.data_loaders import create_dataloader
from training.validation import validate_model
from training.exceptions import TrainingError
from training.curriculum import CurriculumScheduler, CurriculumSampler
from training.distributed_handler import setup_distributed, cleanup_distributed
from training.scheduler import get_lr_scheduler
from training.initializer import initialize_model
from training.model_validator import validate_model_setup
from training.adaptive_reasoning import (
    ReasoningManager, ConfidencePredictor,
    AdaptiveMCTSReasoner, AdaptiveRecursiveReasoner,
    NeuralSymbolicReasoner
)
# Add these imports at the top of the file with other imports
from model.rwkv.rwkv_layer import (
    RWKVTimeFirst,
    RWKVChannelMixer,
    RWKVBlock,
    RWKVConfig
)
from model.rwkv.rwkv_model import (
    RWKVModel,
    HybridRWKVTransformerModel
)
from model.moe import LongContextMoEConfig, create_long_context_moe
from safetensors.torch import save_file # Added for safetensors export
import math
import numpy as np
import random
from pathlib import Path
# Add this import with the other imports:
from model.tree_reasoning_mcts import (
    MCTSConfig as TreeMCTSConfig,
    MonteCarloTreeSearch,
    MCTSEnhancedTreeReasoningModule,
    create_mcts_reasoning_module
)

logger = logging.getLogger(__name__)
@dataclass
class AdvancedTrainingConfig:
    """Advanced training configuration dataclass"""
    # Base training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    num_train_epochs: int = 3
    gradient_accumulation_steps: int = 1
    save_steps: int = 1000
    eval_steps: int = 500
    use_mixed_precision: bool = True
    use_curriculum: bool = False
    use_distributed: bool = False
    local_rank: int = -1
    
    # Data parameters
    batch_size: int = 8
    num_workers: int = 4
    
    # Output parameters
    output_dir: str = "output"
    experiment_name: str = "llm_training"
    resume_from_checkpoint: Optional[str] = None
    
    # Model parameters
    model_type: str = "gpt2"
    max_seq_length: int = 2048
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Advanced features
    use_linear_attention: bool = False
    linear_attention_feature_dim: int = 16
    
    # UL2 Denoising configuration
    use_ul2_denoising: bool = False
    ul2_denoising_ratio: float = 0.15
    ul2_denoising_schedule: str = "linear"  # linear, cosine, or fixed
    ul2_mode_switch_steps: int = 1000
    
    # Memory configuration
    use_memory_augmentation: bool = True
    memory_size: int = 1024
    use_episodic_memory: bool = True
    use_working_memory: bool = True
    use_long_term_memory: bool = True
    use_memory_compaction: bool = False  # Enable memory compaction for long sequences
    memory_compaction_threshold: int = 8192  # Sequence length threshold for compaction
    compaction_strategy: str = "block"  # "block", "sparse", "hybrid"
    compaction_interval: int = 1000
    
    # Reasoning configuration
    use_tree_reasoning: bool = True
    reasoning_depth: int = 4
    use_neural_symbolic: bool = True
    use_recursive_reasoning: bool = True
    recursive_depth: int = 3
    use_knowledge_reasoning: bool = True
    knowledge_graph_size: int = 1000
    use_mcts: bool = True
    mcts_simulations: int = 100
    
    # Adaptive reasoning
    use_adaptive_reasoning: bool = True
    default_reasoning_strategy: str = "DEFAULT"
    strategy_selection_threshold: float = 0.7
    max_reasoning_steps: int = 10
    min_reasoning_steps: int = 1
    use_reasoning_meta_learning: bool = False
    use_adaptive_computation: bool = True
    early_stopping: bool = True
    convergence_threshold: float = 0.01
    
    # Parallelism configuration
    use_tensor_parallelism: bool = False
    tensor_parallel_size: int = 2
    tensor_parallel_communication: str = "allreduce"  # allreduce, ring, hierarchical
    use_sequence_parallelism: bool = False
    sequence_parallel_size: int = 2
    sequence_parallel_chunk_size: int = 1024
    sequence_parallel_overlap: bool = True
    use_expert_parallelism: bool = False
    expert_parallel_size: int = 4
    expert_parallel_communication: str = "alltoall"  # alltoall, scatter_gather
    parallel_communication_backend: str = "nccl"  # "nccl", "gloo", "mpi"
    
    # MoE configuration
    use_moe: bool = True
    num_experts: int = 8
    moe_capacity_factor: float = 1.25
    top_k_experts: int = 2
    expert_dropout: float = 0.1
    use_expert_capacity_adaptation: bool = True
    expert_capacity_adjustment_interval: int = 500
    
    # Memory management for long contexts
    use_gradient_checkpointing: bool = True
    checkpointing_strategy: str = "selective"  # uniform, end_of_layer, selective
    checkpointing_layers: List[int] = None  # For selective checkpointing
    use_flash_attention_v2: bool = True
    flash_attention_v2_config: Dict[str, Any] = None
    use_memory_efficient_swa: bool = True  # Sliding window attention memory optimization
    
    # Dynamic architecture modifications
    use_progressive_layer_dropping: bool = False
    layer_dropping_rate: float = 0.1
    use_dynamic_width_adjustment: bool = False
    width_adjustment_interval: int = 1000
    use_dynamic_depth: bool = False
    
    # Numerical precision
    use_numerical_precision: bool = True
    numerical_precision_mode: str = "auto"
    use_fp8_matmul: bool = False
    use_stable_embedding: bool = True
    
    # Computational efficiency
    use_activation_checkpointing: bool = True
    checkpoint_every_n_layers: int = 2
    use_efficient_attention: bool = True
    attention_implementation: str = "flash"
    use_kv_caching: bool = True
    max_cache_length: int = 2048
    use_kernel_fusion: bool = True
    
    # Advanced optimization techniques
    use_lion_optimizer: bool = False
    use_adan_optimizer: bool = False
    use_gradient_centralization: bool = True
    use_layerwise_adaptive_scaling: bool = True
    layerwise_scaling_factors: Dict[str, float] = None
    
    # Training stability
    gradient_clipping_strategy: str = "global"  # global, layerwise, parameter_group
    layerwise_clip_thresholds: Dict[str, float] = None
    use_automatic_mixed_precision_scaling: bool = True
    loss_scaling_strategy: str = "dynamic"  # static, dynamic, logit
    nan_inf_detection: bool = True
    recovery_strategy: str = "rollback"  # rollback, skip, reduce_lr
    
    # Debugging and profiling
    enable_memory_profiling: bool = True
    memory_profile_interval: int = 100
    enable_communication_profiling: bool = True
    communication_profile_interval: int = 100
    enable_nan_inf_monitoring: bool = True
    nan_inf_check_interval: int = 50
    
    # Checkpointing enhancements
    use_differential_checkpointing: bool = True
    differential_checkpoint_interval: int = 500
    checkpoint_metadata: Dict[str, Any] = None
    checkpoint_compression: str = "zstd"  # none, zstd, lz4
    
    # Specialized initialization
    rwkv_initialization: str = "scaled"  # scaled, orthogonal, xavier
    expert_initialization: str = "kaiming"  # kaiming, xavier, orthogonal
    
    # Quantization
    use_quantization: bool = False
    quantization_bits: int = 8
    quantization_scheme: str = "dynamic"
    
    # Formal verification and uncertainty
    use_formal_verification: bool = False
    verification_threshold: float = 0.8
    uncertainty_threshold: float = 0.2
    verify_mathematical_reasoning: bool = True
    
    # Causal inference
    use_causal_inference: bool = False
    
    # Constitutional AI
    use_constitutional_ai: bool = True
    num_principles: int = 12
    max_revision_iterations: int = 2
    validation_threshold: float = 0.7
    
    # RLHF configuration
    use_rlhf: bool = True
    rlhf_type: str = "ppo"  # "ppo", "dpo", "constitutional_ai"
    use_multi_agent_debate: bool = True
    num_debate_agents: int = 3
    use_reward_ensemble: bool = True
    num_reward_models: int = 3
    dpo_beta: float = 0.1  # DPO beta parameter
    dpo_loss_type: str = "sigmoid"  # DPO loss type
    
    # Logical reasoning
    use_logical_reasoning: bool = True
    
    # MCTS reasoning with advanced config
    use_mcts_advanced: bool = True
    use_adaptive_simulation_count: bool = True
    confidence_threshold: float = 0.9
    use_dirichlet_noise: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    
    # Neural symbolic advanced features
    use_symbolic_verification: bool = True
    use_symbolic_abstraction: bool = True
    use_symbolic_composition: bool = True
    
    # Rotary embeddings
    use_rotary_embeddings: bool = True
    rotary_embedding_base: int = 10000
    max_position_embeddings: int = 32768
    
    # Tree LSTM for enhanced reasoning
    use_tree_lstm: bool = True
    tree_lstm_max_depth: int = 8
    
    # Satisfiability solving
    use_sat_solver: bool = False
    max_sat_iterations: int = 50
    
    # Beam search and generation
    use_logical_beam_search: bool = True
    beam_size: int = 4
    length_penalty: float = 1.0
    consistency_threshold: float = 0.7
    logical_reward_weight: float = 0.5
    
    # GNN parameters
    use_gnn: bool = False
    gnn_type: str = "gat"
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.1
    gnn_residual: bool = True
    gnn_layer_norm: bool = True
    gnn_num_heads: int = 8
    gnn_use_edge_features: bool = False
    gnn_edge_dim: Optional[int] = None
    gnn_readout_type: str = "mean"
    gnn_use_tree_structure: bool = False
    gnn_use_contrastive: bool = False
    gnn_contrastive_type: str = "graphcl"
    
    # RAG configuration
    use_rag: bool = True
    rag_config: Optional[EnhancedRAGConfig] = None
    rag_hidden_size: int = 768
    rag_retriever_dim: int = 768
    rag_num_attention_heads: int = 8
    rag_dropout: float = 0.1
    rag_max_knowledge_items: int = 100
    rag_use_approximate_search: bool = True
    rag_index_type: str = "IVF"  # IVF, HNSW, Flat
    rag_num_partitions: int = 100  # For IVF index
    rag_num_probe: int = 10  # Number of clusters to probe in IVF
    rag_similarity_metric: str = "ip"  # ip (inner product) or l2
    rag_normalize_embeddings: bool = True
    # RWKV configuration
    use_rwkv: bool = False
    
    # Hyena operator configuration
    use_hyena: bool = False # Renamed from use_hyena_operator for consistency
    hyena_layer_indices: List[int] = field(default_factory=list)  # Layers to use Hyena operator
    hyena_order: int = 2 # Order of the polynomial projection
    hyena_filter_order: int = 64 # Order of the filter generation function
    hyena_num_heads: int = 1 # Number of heads for filter projections
    hyena_num_blocks: int = 1 # Number of Hyena blocks (if applicable in operator)
    hyena_filter_dropout: float = 0.0 # Dropout for filter parameters
    hyena_use_short_conv: bool = True # Whether to use parallel short conv
    hyena_short_conv_size: int = 3 # Kernel size for short conv
    hyena_activation: str = "silu" # Activation function within Hyena

    # UL2-style mixture-of-denoisers configuration
    use_ul2_denoisers: bool = False
    ul2_num_denoisers: int = 3
    ul2_denoiser_types: List[str] = field(default_factory=lambda: ["prefix", "suffix", "mask"])
    
    # UL2 configuration
    use_ul2: bool = False
    ul2_objective_mix: List[str] = field(default_factory=lambda: ["R", "X", "S"])
    ul2_dynamic_schedule: bool = True
    ul2_causal_lm_weight: float = 0.5
    ul2_initial_task: str = "R"
    ul2_r_weight: float = 0.4
    ul2_x_weight: float = 0.3
    ul2_s_weight: float = 0.3
    
    # Retro-style retrieval configuration
    use_retro_retrieval: bool = False
    retro_num_neighbors: int = 2
    retro_chunk_size: int = 64
    
    # RWKV configuration
    rwkv_layer_indices: List[int] = field(default_factory=list)  # Layers to replace with RWKV
    rwkv_time_mix_factor: float = 1.0
    rwkv_key_value_mixing: bool = True
    rwkv_att_scale: float = 1.0
    rwkv_use_linear_attn: bool = False
    rwkv_use_gating: bool = True
    rwkv_use_shifting: bool = True
    use_hybrid_model: bool = False  # Use hybrid RWKV+Transformer model

    # CoLT5 conditional computation configuration
    use_colt5: bool = False
    colt5_threshold: float = 0.5  # Threshold for conditional computation
    colt5_layer_indices: List[int] = field(default_factory=list)  # Layers to apply CoLT5 to
    colt5_light_ffn_ratio: float = 0.25 # Ratio for light FFN intermediate size
    colt5_heavy_ffn_ratio: float = 1.0 # Ratio for heavy FFN intermediate size (relative to original FFN)
    colt5_routing_mechanism: str = 'norm' # 'norm' or 'max_abs'
    
    # Data parameters
    huggingface_dataset: Optional[str] = None
    huggingface_subset: Optional[str] = None
    use_streaming: bool = False
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    data_split: str = "train"
    val_split: str = "validation"
    val_size: float = 0.1  # If validation split doesn't exist
    
    def __post_init__(self):
        if self.rwkv_layer_indices is None:
            # By default, replace every other layer
            self.rwkv_layer_indices = list(range(0, self.num_layers, 2))
            
        if self.checkpointing_layers is None and self.checkpointing_strategy == "selective":
            # By default, checkpoint every other layer
            self.checkpointing_layers = list(range(1, self.num_layers, 2))
            
        if self.flash_attention_v2_config is None and self.use_flash_attention_v2:
            # Sensible defaults for FlashAttention-2
            self.flash_attention_v2_config = {
                "softmax_scale": 1.0/math.sqrt(self.hidden_size // self.num_attention_heads),
                "dropout_p": self.dropout,
                "causal": True,
                "return_attn_probs": False
            }
            
        # Initialize hyena_layer_indices if None
        if self.hyena_layer_indices is None and self.use_hyena:
            # By default, apply Hyena to every third layer
            self.hyena_layer_indices = list(range(2, self.num_layers, 3))
            
        # Initialize colt5_layer_indices if None
        if self.colt5_layer_indices is None and self.use_colt5:
            # By default, apply to middle layers
            mid_layer = self.num_layers // 2
            self.colt5_layer_indices = list(range(mid_layer - 1, mid_layer + 2))
            
        # Initialize ul2_objective_mix if None
        if self.ul2_objective_mix is None and self.use_ul2:
            self.ul2_objective_mix = ["R", "X", "S"]
            
        if self.layerwise_scaling_factors is None and self.use_layerwise_adaptive_scaling:
            # Initialize with default values based on layer depth
            self.layerwise_scaling_factors = {
                f"layer_{i}": 1.0 - (i / (2 * self.num_layers)) 
                for i in range(self.num_layers)
            }
            
        if self.layerwise_clip_thresholds is None and self.gradient_clipping_strategy == "layerwise":
            # Initialize with default values that decrease for deeper layers
            self.layerwise_clip_thresholds = {
                f"layer_{i}": self.max_grad_norm * (0.9 ** (i // 2)) 
                for i in range(self.num_layers)
            }
            
        if self.checkpoint_metadata is None:
            # Initialize with basic metadata tracking
            self.checkpoint_metadata = {
                "creation_time": None,  # Will be filled at checkpoint time
                "config_summary": {k: v for k, v in asdict(self).items() 
                                 if not isinstance(v, (dict, list)) and k != "checkpoint_metadata"},
                "training_metrics": {},
                "validation_metrics": {}
            }
            
        # Process CoLT5 layer indices if provided
        if self.colt5_layer_indices is None and self.use_colt5:
            # By default, apply to middle layers
            mid_layer = self.num_layers // 2
            self.colt5_layer_indices = list(range(mid_layer - 1, mid_layer + 2))
def parse_args() -> AdvancedTrainingConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a language model with advanced features")
    
    # Base training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between checkpoints")
    parser.add_argument("--eval_steps", type=int, default=500, help="Number of steps between evaluations")
    
    # Training features
    parser.add_argument("--use_mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--use_curriculum", action="store_true", help="Use curriculum learning")
    parser.add_argument("--use_distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="llm_training", help="Experiment name")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="gpt2", help="Model type")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="Intermediate size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12, help="Layer normalization epsilon")
    
    # Advanced features
    parser.add_argument("--use_linear_attention", action="store_true", help="Use linear attention")
    parser.add_argument("--linear_attention_feature_dim", type=int, default=16, help="Linear attention feature dimension")
    
    # Memory configuration
    parser.add_argument("--use_memory_augmentation", action="store_true", help="Use memory augmentation")
    parser.add_argument("--memory_size", type=int, default=1024, help="Memory size")
    parser.add_argument("--use_episodic_memory", action="store_true", help="Use episodic memory")
    parser.add_argument("--use_working_memory", action="store_true", help="Use working memory")
    parser.add_argument("--use_long_term_memory", action="store_true", help="Use long-term memory")
    
    # Reasoning configuration
    parser.add_argument("--use_tree_reasoning", action="store_true", help="Use tree reasoning")
    parser.add_argument("--reasoning_depth", type=int, default=4, help="Reasoning depth")
    parser.add_argument("--use_neural_symbolic", action="store_true", help="Use neural symbolic reasoning")
    parser.add_argument("--use_recursive_reasoning", action="store_true", help="Use recursive reasoning")
    parser.add_argument("--recursive_depth", type=int, default=3, help="Recursive depth")
    parser.add_argument("--use_knowledge_reasoning", action="store_true", help="Use knowledge reasoning")
    parser.add_argument("--knowledge_graph_size", type=int, default=1000, help="Knowledge graph size")
    parser.add_argument("--use_mcts", action="store_true", help="Use MCTS reasoning")
    parser.add_argument("--mcts_simulations", type=int, default=100, help="Number of MCTS simulations")
    
    # MoE configuration
    parser.add_argument("--use_moe", action="store_true", help="Use Mixture of Experts")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--moe_capacity_factor", type=float, default=1.25, help="MoE capacity factor")
    parser.add_argument("--top_k_experts", type=int, default=2, help="Number of top experts to use")
    parser.add_argument("--expert_dropout", type=float, default=0.1, help="Expert dropout")
    
    # CoLT5 configuration
    parser.add_argument("--use_colt5", action="store_true", help="Use CoLT5 conditional computation")
    parser.add_argument("--colt5_threshold", type=float, default=0.5,
                        help="Threshold for CoLT5 conditional computation")
    parser.add_argument("--colt5_layer_indices", type=str, default=None,
                        help="Comma-separated layer indices for CoLT5")
    # Note: Default heavy ratio changed to 1.0 as it's relative to original FFN size in the new implementation
    parser.add_argument("--colt5_light_ffn_ratio", type=float, default=0.25,
                        help="Ratio for light FFN intermediate size")
    parser.add_argument("--colt5_heavy_ffn_ratio", type=float, default=1.0,
                        help="Ratio for heavy FFN intermediate size (relative to original FFN)")
    parser.add_argument("--colt5_routing_mechanism", type=str, default='norm', choices=['norm', 'max_abs'],
                        help="Mechanism to calculate routing score ('norm' or 'max_abs')")

    # UL2 Mixture-of-Denoisers configuration
    parser.add_argument("--use_ul2", action="store_true", help="Use UL2 mixture-of-denoisers")
    parser.add_argument("--ul2_objective_mix", type=str, default="R,X,S",
                        help="Comma-separated denoising tasks (R,X,S)")
    parser.add_argument("--ul2_dynamic_schedule", action="store_true",
                        help="Dynamically schedule denoising tasks")
    parser.add_argument("--ul2_causal_lm_weight", type=float, default=0.3,
                        help="Weight for standard LM loss")
    parser.add_argument("--ul2_initial_task", type=str, default="R",
                        help="Starting denoising task (R,X,S)")
    parser.add_argument("--ul2_r_weight", type=float, default=0.4,
                        help="Weight for R-denoising")
    parser.add_argument("--ul2_x_weight", type=float, default=0.3,
                        help="Weight for X-denoising")
    parser.add_argument("--ul2_s_weight", type=float, default=0.3,
                        help="Weight for S-denoising")

    # Hyena Operator Configuration
    parser.add_argument("--use_hyena", action="store_true", help="Use Hyena operator")
    parser.add_argument("--hyena_layer_indices", type=str, default=None,
                        help="Comma-separated layer indices for Hyena")
    parser.add_argument("--hyena_num_blocks", type=int, default=2,
                       help="Number of Hyena blocks per layer")
    
    # Numerical precision
    parser.add_argument("--use_numerical_precision", action="store_true", help="Use numerical precision")
    parser.add_argument("--numerical_precision_mode", type=str, default="auto", help="Numerical precision mode")
    parser.add_argument("--use_fp8_matmul", action="store_true", help="Use FP8 matrix multiplication")
    parser.add_argument("--use_stable_embedding", action="store_true", help="Use stable embedding")
    
    # RLHF configuration
    parser.add_argument("--use_rlhf", action="store_true", help="Use RLHF")
    parser.add_argument("--rlhf_type", type=str, default="ppo", help="RLHF type")
    parser.add_argument("--use_multi_agent_debate", action="store_true", help="Use multi-agent debate")
    parser.add_argument("--num_debate_agents", type=int, default=3, help="Number of debate agents")
    parser.add_argument("--use_reward_ensemble", action="store_true", help="Use reward ensemble")
    parser.add_argument("--num_reward_models", type=int, default=3, help="Number of reward models")
    
    # Data sources - file or HuggingFace dataset
    parser.add_argument("--train_data_path", type=str, default=None, help="Path to training data (file based)")
    parser.add_argument("--val_data_path", type=str, default=None, help="Path to validation data (file based)")
    parser.add_argument("--huggingface_dataset", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--huggingface_subset", type=str, default=None, help="HuggingFace dataset configuration/subset")
    parser.add_argument("--use_streaming", action="store_true", help="Use streaming mode for HuggingFace datasets")
    parser.add_argument("--data_split", type=str, default="train", help="Data split to use for training")
    parser.add_argument("--val_split", type=str, default="validation", help="Data split to use for validation")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size if no validation split exists")
    
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    
    # Adaptive reasoning parameters
    parser.add_argument("--use_adaptive_reasoning", action="store_true", help="Use adaptive reasoning")
    parser.add_argument("--default_reasoning_strategy", type=str, default="DEFAULT", help="Default reasoning strategy")
    parser.add_argument("--strategy_selection_threshold", type=float, default=0.7, help="Strategy selection threshold")
    parser.add_argument("--max_reasoning_steps", type=int, default=10, help="Maximum reasoning steps")
    parser.add_argument("--use_reasoning_meta_learning", action="store_true", help="Use meta-learning for reasoning")
    parser.add_argument("--use_adaptive_computation", action="store_true", help="Use adaptive computation time")
    
    # Computational efficiency
    parser.add_argument("--use_activation_checkpointing", action="store_true", help="Use activation checkpointing")
    parser.add_argument("--checkpoint_every_n_layers", type=int, default=2, help="Checkpoint frequency")
    parser.add_argument("--use_efficient_attention", action="store_true", help="Use efficient attention")
    parser.add_argument("--attention_implementation", type=str, default="flash", help="Attention implementation")
    parser.add_argument("--use_kv_caching", action="store_true", help="Use KV caching")
    parser.add_argument("--max_cache_length", type=int, default=2048, help="Maximum cache length")
    parser.add_argument("--use_kernel_fusion", action="store_true", help="Use kernel fusion")
    
    # Quantization
    parser.add_argument("--use_quantization", action="store_true", help="Use quantization")
    parser.add_argument("--quantization_bits", type=int, default=8, help="Quantization bits")
    parser.add_argument("--quantization_scheme", type=str, default="dynamic", help="Quantization scheme")
    
    # Formal verification
    parser.add_argument("--use_formal_verification", action="store_true", help="Use formal verification")
    parser.add_argument("--verification_threshold", type=float, default=0.8, help="Verification threshold")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.2, help="Uncertainty threshold")
    parser.add_argument("--verify_mathematical_reasoning", action="store_true", help="Verify mathematical reasoning")
    
    # Causal inference
    parser.add_argument("--use_causal_inference", action="store_true", help="Enable causal inference engine for causal reasoning")
    
    # Constitutional AI
    parser.add_argument("--use_constitutional_ai", action="store_true", help="Use Constitutional AI")
    parser.add_argument("--num_principles", type=int, default=12, help="Number of constitutional principles")
    parser.add_argument("--max_revision_iterations", type=int, default=2, help="Maximum revision iterations")
    parser.add_argument("--validation_threshold", type=float, default=0.7, help="Validation threshold")
    
    # Logical reasoning
    parser.add_argument("--use_logical_reasoning", action="store_true", help="Use logical reasoning")
    
    # MCTS reasoning with advanced config
    parser.add_argument("--use_mcts_advanced", action="store_true", help="Use MCTS reasoning with advanced config")
    parser.add_argument("--use_adaptive_simulation_count", action="store_true", help="Use adaptive simulation count")
    parser.add_argument("--confidence_threshold", type=float, default=0.9, help="Confidence threshold")
    parser.add_argument("--use_dirichlet_noise", action="store_true", help="Use Dirichlet noise")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet alpha")
    parser.add_argument("--dirichlet_weight", type=float, default=0.25, help="Dirichlet weight")
    
    # Neural symbolic advanced features
    parser.add_argument("--use_symbolic_verification", action="store_true", help="Use symbolic verification")
    parser.add_argument("--use_symbolic_abstraction", action="store_true", help="Use symbolic abstraction")
    parser.add_argument("--use_symbolic_composition", action="store_true", help="Use symbolic composition")
    
    # Rotary embeddings
    parser.add_argument("--use_rotary_embeddings", action="store_true", help="Use rotary embeddings")
    parser.add_argument("--rotary_embedding_base", type=int, default=10000, help="Rotary embedding base")
    parser.add_argument("--max_position_embeddings", type=int, default=32768, help="Maximum position embeddings")
    
    # Tree LSTM for enhanced reasoning
    parser.add_argument("--use_tree_lstm", action="store_true", help="Use tree LSTM for enhanced reasoning")
    parser.add_argument("--tree_lstm_max_depth", type=int, default=8, help="Tree LSTM max depth")
    
    # Satisfiability solving
    parser.add_argument("--use_sat_solver", action="store_true", help="Use SAT solver")
    parser.add_argument("--max_sat_iterations", type=int, default=50, help="Maximum SAT iterations")
    
    # Recursive reasoning advanced options
    parser.add_argument("--use_recurrent_reasoning", action="store_true", help="Use recurrent reasoning")
    parser.add_argument("--min_reasoning_steps", type=int, default=1, help="Minimum reasoning steps")
    parser.add_argument("--max_reasoning_steps", type=int, default=10, help="Maximum reasoning steps")
    parser.add_argument("--early_stopping", action="store_true", help="Use early stopping")
    parser.add_argument("--convergence_threshold", type=float, default=0.01, help="Convergence threshold")
    
    # Beam search and generation
    parser.add_argument("--use_logical_beam_search", action="store_true", help="Use logical beam search")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty")
    parser.add_argument("--consistency_threshold", type=float, default=0.7, help="Consistency threshold")
    parser.add_argument("--logical_reward_weight", type=float, default=0.5, help="Logical reward weight")
    
    # GNN parameters
    parser.add_argument("--use_gnn", action="store_true", help="Use GNN")
    parser.add_argument("--gnn_type", type=str, default="gat", help="GNN type")
    parser.add_argument("--gnn_hidden_dim", type=int, default=256, help="GNN hidden dimension")
    parser.add_argument("--gnn_num_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--gnn_dropout", type=float, default=0.1, help="GNN dropout")
    parser.add_argument("--gnn_residual", action="store_true", help="Use GNN residual connections")
    parser.add_argument("--gnn_layer_norm", action="store_true", help="Use GNN layer normalization")
    parser.add_argument("--gnn_num_heads", type=int, default=8, help="Number of GNN heads")
    parser.add_argument("--gnn_use_edge_features", action="store_true", help="Use GNN edge features")
    parser.add_argument("--gnn_edge_dim", type=int, help="GNN edge dimension")
    parser.add_argument("--gnn_readout_type", type=str, default="mean", help="GNN readout type")
    parser.add_argument("--gnn_use_tree_structure", action="store_true", help="Use GNN tree structure")
    parser.add_argument("--gnn_use_contrastive", action="store_true", help="Use GNN contrastive learning")
    parser.add_argument("--gnn_contrastive_type", type=str, default="graphcl", help="GNN contrastive type")
    
    # RAG configuration
    parser.add_argument("--use_rag", action="store_true", help="Use Retrieval-Augmented Generation")
    parser.add_argument("--rag_hidden_size", type=int, default=768, help="RAG hidden size")
    parser.add_argument("--rag_retriever_dim", type=int, default=768, help="RAG retriever dimension") 
    parser.add_argument("--rag_num_attention_heads", type=int, default=8, help="RAG number of attention heads")
    parser.add_argument("--rag_dropout", type=float, default=0.1, help="RAG dropout rate")
    parser.add_argument("--rag_max_knowledge_items", type=int, default=100, help="Max knowledge items to retrieve")
    parser.add_argument("--rag_use_approximate_search", action="store_true", help="Use approximate search for RAG")
    parser.add_argument("--rag_index_type", type=str, default="IVF", help="RAG index type (IVF, HNSW, Flat)")
    parser.add_argument("--rag_num_partitions", type=int, default=100, help="Number of partitions for IVF index")
    parser.add_argument("--rag_num_probe", type=int, default=10, help="Number of clusters to probe in IVF")
    parser.add_argument("--rag_similarity_metric", type=str, default="ip", help="RAG similarity metric (ip or l2)")
    parser.add_argument("--rag_normalize_embeddings", action="store_true", help="Normalize RAG embeddings")
    # RWKV configuration
    parser.add_argument("--use_rwkv", action="store_true", help="Use RWKV layers")
    parser.add_argument("--rwkv_layer_indices", type=str, default=None,
                        help="Comma-separated list of layer indices to replace with RWKV")
    parser.add_argument("--rwkv_time_mix_factor", type=float, default=1.0,
                        help="RWKV time mixing factor")
    parser.add_argument("--rwkv_key_value_mixing", action="store_true",
                        help="Use key-value mixing in RWKV")
    parser.add_argument("--rwkv_att_scale", type=float, default=1.0,
                        help="RWKV attention scaling factor")
    parser.add_argument("--rwkv_use_linear_attn", action="store_true",
                        help="Use linear attention in RWKV")
    parser.add_argument("--rwkv_use_gating", action="store_true",
                        help="Use gating mechanism in RWKV")
    parser.add_argument("--rwkv_use_shifting", action="store_true",
                        help="Use time-shifting in RWKV")
    parser.add_argument("--use_hybrid_model", action="store_true",
                        help="Use hybrid RWKV+Transformer model")

    # Hyena operator configuration
    parser.add_argument("--use_hyena", action="store_true", help="Use Hyena operators")
    parser.add_argument("--hyena_layer_indices", type=str, default=None,
                        help="Comma-separated list of layer indices to replace with Hyena")
    parser.add_argument("--hyena_order", type=int, default=2, help="Order for Hyena polynomial projection")
    parser.add_argument("--hyena_filter_order", type=int, default=64, help="Order for Hyena filter generation function")
    parser.add_argument("--hyena_num_heads", type=int, default=1, help="Number of heads for Hyena filter projections")
    # hyena_num_blocks is already defined elsewhere, reusing that arg
    parser.add_argument("--hyena_filter_dropout", type=float, default=0.0, help="Dropout for Hyena filter parameters")
    parser.add_argument("--hyena_use_short_conv", action="store_true", default=True, help="Use short convolution in Hyena")
    parser.add_argument("--hyena_short_conv_size", type=int, default=3, help="Kernel size for Hyena short convolution")
    parser.add_argument("--hyena_activation", type=str, default="silu", help="Activation function for Hyena (e.g., silu, gelu)")

    # CoLT5 conditional computation configuration
    parser.add_argument("--use_colt5", action="store_true", help="Use CoLT5 conditional computation")
    parser.add_argument("--colt5_threshold", type=float, default=0.5,
                        help="Threshold for CoLT5 conditional computation")
    
    # UL2-style mixture-of-denoisers arguments
    parser.add_argument("--use_ul2", action="store_true", help="Use UL2-style mixture of denoisers")
    parser.add_argument("--ul2_objective_mix", type=str, default="r,x,s",
                        help="Comma-separated list of denoising objectives (r,x,s)")
    parser.add_argument("--ul2_dynamic_schedule", action="store_true",
                        help="Use dynamic schedule for UL2 objectives")
    parser.add_argument("--ul2_causal_lm_weight", type=float, default=0.5,
                        help="Weight for causal LM objective")
    parser.add_argument("--ul2_initial_task", type=str, default="r",
                        help="Initial task for UL2 (r,x,s)")
    parser.add_argument("--ul2_r_weight", type=float, default=0.4,
                        help="Weight for R objective (prefix LM)")
    parser.add_argument("--ul2_x_weight", type=float, default=0.3,
                        help="Weight for X objective (span corruption)")
    parser.add_argument("--ul2_s_weight", type=float, default=0.3,
                        help="Weight for S objective (sequential span)")
    
    # Hyena operator arguments
    parser.add_argument("--use_hyena", action="store_true", help="Use Hyena operators")
    parser.add_argument("--hyena_layer_indices", type=str, default=None,
                        help="Comma-separated list of layer indices to use Hyena")
    parser.add_argument("--hyena_num_blocks", type=int, default=4,
                        help="Number of blocks in Hyena operator")

    # UL2 Mixture-of-Denoisers configuration
    parser.add_argument("--use_ul2", action="store_true", help="Use UL2 mixture-of-denoisers")
    parser.add_argument("--ul2_objective_mix", type=str, default="R,X,S",
                        help="Comma-separated denoising tasks (R,X,S)")
    parser.add_argument("--ul2_dynamic_schedule", action="store_true",
                        help="Dynamically schedule denoising tasks")
    parser.add_argument("--ul2_causal_lm_weight", type=float, default=0.3,
                        help="Weight for standard LM loss")
    parser.add_argument("--ul2_initial_task", type=str, default="R",
                        help="Starting denoising task (R,X,S)")
    parser.add_argument("--ul2_r_weight", type=float, default=0.4,
                        help="Weight for R-denoising loss")
    parser.add_argument("--ul2_x_weight", type=float, default=0.3,
                        help="Weight for X-denoising loss")
    parser.add_argument("--ul2_s_weight", type=float, default=0.3,
                        help="Weight for S-denoising loss")

    # Hyena operator configuration
    parser.add_argument("--use_hyena", action="store_true", help="Use Hyena operator")
    parser.add_argument("--hyena_layer_indices", type=str, default=None,
                        help="Comma-separated list of layer indices to use Hyena")
    parser.add_argument("--hyena_num_blocks", type=int, default=2,
                        help="Number of Hyena blocks per layer")
    parser.add_argument("--hyena_use_short_conv", action="store_true",
                        help="Use short convolutions in Hyena")
    
    args = parser.parse_args()
    
    # Process RWKV layer indices if provided
    rwkv_layer_indices = None
    if args.rwkv_layer_indices:
        rwkv_layer_indices = [int(idx) for idx in args.rwkv_layer_indices.split(',')]
    
    # Process special configuration parameters
    # Handle ul2_objective_mix
    if args.ul2_objective_mix:
        ul2_objective_mix = args.ul2_objective_mix.split(',')
    else:
        ul2_objective_mix = ["R", "X", "S"]
    
    # Handle hyena_layer_indices
    if args.hyena_layer_indices:
        hyena_layer_indices = [int(idx) for idx in args.hyena_layer_indices.split(',')]
    else:
        if args.use_hyena:
            # Apply to every third layer by default
            hyena_layer_indices = list(range(2, args.num_layers, 3))
        else:
            hyena_layer_indices = []
            
    # Handle colt5_layer_indices
    if args.colt5_layer_indices:
        colt5_layer_indices = [int(idx) for idx in args.colt5_layer_indices.split(',')]
    else:
        if args.use_colt5:
            # Apply to middle layers by default
            mid_layer = args.num_layers // 2
            colt5_layer_indices = list(range(mid_layer - 1, mid_layer + 2))
        else:
            colt5_layer_indices = []
    
    # Create the config
    config = AdvancedTrainingConfig()
    
    # Set the processed parameters to the config
    config.ul2_objective_mix = ul2_objective_mix
    config.hyena_layer_indices = hyena_layer_indices
    config.colt5_layer_indices = colt5_layer_indices
    
    # Load datasets
    logger.info("Loading datasets")
    train_dataset, val_dataset = load_dataset_from_config(config)
    
    config.train_dataset = train_dataset 
    config.val_dataset = val_dataset
    
    return config
def load_dataset_from_config(config: AdvancedTrainingConfig):
    """Load datasets from either HuggingFace or local files based on config"""
    if config.huggingface_dataset is not None:
        # Using HuggingFace datasets
        return load_huggingface_dataset(config)
    elif config.train_data_path is not None:
        # Using local files
        train_dataset = load_file_dataset(config.train_data_path, config)
        val_dataset = load_file_dataset(config.val_data_path, config) if config.val_data_path else None
        return train_dataset, val_dataset
    else:
        raise ValueError("Either huggingface_dataset or train_data_path must be provided")
def load_huggingface_dataset(config: AdvancedTrainingConfig):
    """Load dataset from HuggingFace datasets hub with preprocessing for the model"""
    logger.info(f"Loading dataset '{config.huggingface_dataset}'{f' ({config.huggingface_subset})' if config.huggingface_subset else ''}")
    # Load the main dataset
    try:
        # Directly load fineweb dataset
        if config.huggingface_dataset == "HuggingFaceFW/fineweb":
            train_dataset = hf_load_dataset(
                "HuggingFaceFW/fineweb",
                name="sample-10BT",
                split="train",
                streaming=config.use_streaming
            )
            val_dataset = None # Fineweb does not have a validation split
        else:
            # Load other HuggingFace datasets as before
            # Load the training split
            train_dataset = hf_load_dataset(
                config.huggingface_dataset,
                name=config.huggingface_subset,
                split=config.data_split,
                streaming=config.use_streaming
            )
            # Check if validation split exists and use it
            try:
                val_dataset = hf_load_dataset(
                    config.huggingface_dataset,
                    name=config.huggingface_subset,
                    split=config.val_split,
                    streaming=config.use_streaming
                )
                logger.info(f"Loaded validation split: {config.val_split}")
            except Exception as e:
                logger.warning(f"No validation split found: {e}")
                logger.info(f"Creating validation set from training data with size {config.val_size}")
                if config.use_streaming:
                    # For streaming datasets, use a portion of the stream for validation
                    # This is an approximation - in streaming mode we can't easily split exactly
                    train_iter = iter(train_dataset)
                    train_buffer = []
                    val_buffer = []
                    # Collect a buffer of data for exact validation split
                    for i, example in enumerate(train_iter):
                        if i < 10000:  # Collect fixed number of examples
                            if i < int(10000 * config.val_size):
                                val_buffer.append(example)
                            else:
                                train_buffer.append(example)
                        else:
                            break
                    # Create validation dataset from buffer
                    from datasets import Dataset
                    val_dataset = Dataset.from_dict({k: [example[k] for example in val_buffer]
                                                   for k in val_buffer[0].keys()})
                    # Create a new streaming dataset that starts after our buffer
                    # Use a custom IterableDataset to handle this case
                    class BufferedStreamingDataset(IterableDataset):
                        def __init__(self, buffer, streaming_dataset):
                            self.buffer = buffer
                            self.streaming_dataset = streaming_dataset
                            self.streaming_iter = None
                        def __iter__(self):
                            # First yield from buffer
                            for example in self.buffer:
                                yield example
                            # Then continue with the stream, skipping what we've seen
                            self.streaming_iter = iter(self.streaming_dataset)
                            for _ in range(10000):  # Skip the examples we've already processed
                                try:
                                    next(self.streaming_iter)
                                except StopIteration:
                                    return
                            # Continue with the rest of the stream
                            for example in self.streaming_iter:
                                yield example
                    train_dataset = BufferedStreamingDataset(train_buffer, train_dataset)
                else:
                    # For non-streaming datasets, use train_test_split
                    from datasets import Dataset
                    if isinstance(train_dataset, Dataset):
                        train_val_dict = train_dataset.train_test_split(test_size=config.val_size)
                        train_dataset = train_val_dict['train']
                        val_dataset = train_val_dict['test']
                    else:
                        raise ValueError("Cannot split non-Dataset object and no validation split was found")
        # Apply preprocessing for model training
        tokenizer = config.tokenizer
        def tokenize_function(examples):
            # Get the column name that contains the text
            text_key = 'text' if 'text' in examples else next(iter(examples.keys()))
            # Tokenize the text
            tokenized = tokenizer(
                examples[text_key],
                truncation=True,
                max_length=config.max_seq_length,
                padding='max_length'
            )
            # Add labels for language modeling (same as input_ids)
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        # Apply tokenization
        if not config.use_streaming:
            # Process in batches for regular datasets
            train_dataset = train_dataset.map(
                tokenize_function,
                batched=True,
                num_proc=max(1, config.num_workers),
                remove_columns=[col for col in train_dataset.column_names if col != 'labels']
            )
            if val_dataset is not None: # Only map if val_dataset exists
                val_dataset = val_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=max(1, config.num_workers),
                    remove_columns=[col for col in val_dataset.column_names if col != 'labels']
                )
        else:
            # For streaming datasets, need to apply on-the-fly
            train_dataset = train_dataset.map(tokenize_function)
            if isinstance(val_dataset, IterableDataset) and val_dataset is not None: # Only map if val_dataset exists and is IterableDataset
                val_dataset = val_dataset.map(tokenize_function)
        logger.info(f"Dataset loaded and processed")
        return train_dataset, val_dataset
    except Exception as e:
        logger.error(f"Error loading HuggingFace dataset: {str(e)}")
        raise
    
def load_file_dataset(data_path: str, config: AdvancedTrainingConfig):
    """Load and prepare dataset from file path.
    
    Args:
        data_path: Path to the data
        config: Training configuration
        
    Returns:
        Dataset: The loaded dataset
    """
    try:
        from training.data_loaders import create_dataset, TextDataset, JsonDataset
        
        logger.info(f"Loading data from {data_path}")
        
        # Check if data path is a directory or file
        if os.path.isdir(data_path):
            # Handle directory with multiple files
            data_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                         if f.endswith('.txt') or f.endswith('.json')]
            
            if not data_files:
                raise ValueError(f"No data files found in {data_path}")
            
            # Load dataset
            dataset = create_dataset(data_files, config.max_seq_length)
        else:
            # Handle single file
            if not os.path.exists(data_path):
                raise ValueError(f"Data path {data_path} does not exist")
            
            dataset = create_dataset([data_path], config.max_seq_length)
        
        logger.info(f"Dataset size: {len(dataset)}")
        return dataset
        
    except ImportError:
        logger.error("Could not import dataset classes. Make sure training/data_loaders.py exists.")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise
def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    sampler=None,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
):
    """Create a dataloader for the given dataset"""
    if isinstance(dataset, IterableDataset):
        # For streaming datasets, we can't shuffle or use sampler
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    
    return dataloader
def setup_advanced_model(config: AdvancedTrainingConfig) -> ValkyrieLLM:
    """Set up model with advanced components including RWKV, enhanced MoE and causal inference if enabled."""
    logger.info("Setting up model with advanced components")
    
    # Create transformer configuration
    transformer_config = TransformerConfig(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_seq_length,
        layer_norm_eps=config.layer_norm_eps,
        dropout=config.dropout,
        # New config parameters
        use_ul2_denoising=config.use_ul2_denoising,
        denoising_modes=['regular', 'span_corruption', 'prefix_lm'],
        use_retro_retrieval=config.use_retro_retrieval,
        retro_num_neighbors=2
    )
    # Removed Hyena and CoLT5 params from TransformerConfig, will be handled by replacement functions

    # Create RWKV configuration if enabled
    rwkv_config = None
    if config.use_rwkv:
        rwkv_config = RWKVConfig(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            time_mix_factor=config.rwkv_time_mix_factor,
            key_value_mixing=config.rwkv_key_value_mixing,
            att_scale=config.rwkv_att_scale,
            use_linear_attn=config.rwkv_use_linear_attn,
            use_gating=config.rwkv_use_gating,
            use_shifting=config.rwkv_use_shifting
        )
        logger.info(f"RWKV configuration created with {len(config.rwkv_layer_indices)} RWKV layers")
    
    # Create core model first
    if config.use_hybrid_model and config.use_rwkv:
        logger.info("Creating hybrid RWKV-Transformer model")
        core_model = HybridRWKVTransformerModel(
            transformer_config=transformer_config,
            rwkv_config=rwkv_config,
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            rwkv_layer_indices=config.rwkv_layer_indices
        )
    elif config.use_rwkv:
        logger.info("Creating RWKV model")
        core_model = RWKVModel(
            config=rwkv_config,
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length
        )
    else:
        core_model = CoreModel(
            config=transformer_config,
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length
        )
    
    # Create base model with core model
    model = ValkyrieLLM(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout,
        core_model=core_model
    )
    
    # Apply CoLT5 if enabled
    if config.use_colt5:
        from model.colt5 import replace_ffn_with_colt5
        replace_ffn_with_colt5(
            model,
            num_layers=config.num_layers,
            num_experts=config.num_experts,
            expert_capacity=config.expert_capacity,
            light_ffn_ratio=config.colt5_light_ffn_ratio,
            heavy_ffn_ratio=config.colt5_heavy_ffn_ratio,
            routing_mechanism=config.colt5_routing_mechanism
        )
    
    # If using RWKV, store the configuration for later use
    if config.use_rwkv:
        model.rwkv_config = rwkv_config
        model.rwkv_layer_indices = config.rwkv_layer_indices
    
    # Set up GNN components if enabled
    if config.use_gnn:
        logger.info(f"Setting up GNN components with type: {config.gnn_type}")
        
        # Create graph encoder
        graph_encoder = GraphEncoder(
            input_dim=config.hidden_size,
            hidden_dim=config.gnn_hidden_dim,
            output_dim=config.hidden_size,
            num_layers=config.gnn_num_layers,
            dropout=config.gnn_dropout
        )
        
        # Create different GNN layers based on configuration
        gnn_layers = []
        
        if config.gnn_type == "gat":
            for _ in range(config.gnn_num_layers):
                gnn_layers.append(GraphAttention(
                    in_features=config.gnn_hidden_dim,
                    out_features=config.gnn_hidden_dim,
                    heads=config.gnn_num_heads,
                    dropout=config.gnn_dropout,
                    concat=True
                ))
        elif config.gnn_type == "gcn":
            for _ in range(config.gnn_num_layers):
                gnn_layers.append(GraphConvolution(
                    in_features=config.gnn_hidden_dim,
                    out_features=config.gnn_hidden_dim,
                    dropout=config.gnn_dropout,
                    bias=True,
                    activation="relu"
                ))
        elif config.gnn_type == "graphsage":
            for _ in range(config.gnn_num_layers):
                gnn_layers.append(GraphSAGELayer(
                    in_features=config.gnn_hidden_dim,
                    out_features=config.gnn_hidden_dim,
                    aggr="mean",
                    dropout=config.gnn_dropout
                ))
        elif config.gnn_type == "edgegat":
            for _ in range(config.gnn_num_layers):
                gnn_layers.append(EdgeGAT(
                    in_features=config.gnn_hidden_dim,
                    out_features=config.gnn_hidden_dim,
                    edge_dim=config.gnn_edge_dim or config.gnn_hidden_dim // 2,
                    heads=config.gnn_num_heads,
                    dropout=config.gnn_dropout
                ))
        elif config.gnn_type == "transformer":
            for _ in range(config.gnn_num_layers):
                gnn_layers.append(GraphTransformer(
                    in_features=config.gnn_hidden_dim,
                    out_features=config.gnn_hidden_dim,
                    num_heads=config.gnn_num_heads,
                    dropout=config.gnn_dropout
                ))
        elif config.gnn_type == "diffpool":
            gnn_layers.append(DiffPool(
                in_features=config.gnn_hidden_dim,
                hidden_dim=config.gnn_hidden_dim,
                embedding_dim=config.gnn_hidden_dim,
                num_clusters=10,  # Number of clusters to pool nodes into
                dropout=config.gnn_dropout
            ))
        elif config.gnn_type == "hgt":
            gnn_layers.append(HGT(
                in_dim=config.gnn_hidden_dim,
                hidden_dim=config.gnn_hidden_dim,
                out_dim=config.gnn_hidden_dim,
                num_types=3,       # Number of node types in heterogeneous graph
                num_relations=5,   # Number of relation types in heterogeneous graph
                num_heads=config.gnn_num_heads,
                dropout=config.gnn_dropout
            ))
        
        # Create GNN encoder
        gnn_encoder = GNNEncoder(
            input_dim=config.hidden_size,  # Match transformer hidden size
            hidden_dim=config.gnn_hidden_dim,
            output_dim=config.hidden_size,  # Match transformer hidden size for integration
            num_layers=config.gnn_num_layers,
            gnn_type=config.gnn_type,
            dropout=config.gnn_dropout,
            residual=config.gnn_residual,
            layer_norm=config.gnn_layer_norm,
            num_heads=config.gnn_num_heads,
            use_edge_features=config.gnn_use_edge_features,
            edge_dim=config.gnn_edge_dim,
            readout_type=config.gnn_readout_type,
            custom_layers=gnn_layers
        )
        
        # Create Tree GNN if using tree structure
        if config.gnn_use_tree_structure:
            tree_gnn = TreeGNN(
                input_dim=config.hidden_size,
                hidden_dim=config.gnn_hidden_dim,
                output_dim=config.hidden_size,
                num_layers=config.gnn_num_layers,
                dropout=config.gnn_dropout
            )
            model.tree_gnn = tree_gnn
        
        # Add contrastive learning if configured
        if config.gnn_use_contrastive:
            if config.gnn_contrastive_type == "graphcl":
                contrastive_model = GraphCL(
                    gnn_encoder=gnn_encoder,
                    hidden_dim=config.gnn_hidden_dim,
                    proj_dim=config.gnn_hidden_dim // 2
                )
                model.contrastive_model = contrastive_model
            elif config.gnn_contrastive_type == "infograph":
                infograph = InfoGraph(
                    gnn_encoder=gnn_encoder,
                    hidden_dim=config.gnn_hidden_dim,
                    proj_dim=config.gnn_hidden_dim // 2
                )
                model.contrastive_model = infograph
        
        # Create transformer-GNN integration
        model.gnn_integration = TransformerGNNIntegration(
            transformer_dim=config.hidden_size,
            graph_dim=config.hidden_size,
            hidden_dim=config.gnn_hidden_dim,
            num_graph_encoder_layers=config.gnn_num_layers,
            dropout=config.gnn_dropout,
            gnn_type=config.gnn_type,
            use_graph_attention=True,  # Enable graph attention for better integration
            use_tree_structure=config.gnn_use_tree_structure,
            use_contrastive=config.gnn_use_contrastive,
            contrastive_type=config.gnn_contrastive_type,
            gnn_kwargs={
                "num_heads": config.gnn_num_heads,
                "use_edge_features": config.gnn_use_edge_features,
                "edge_dim": config.gnn_edge_dim,
                "residual": config.gnn_residual,
                "layer_norm": config.gnn_layer_norm
            }
        )
        
        # Add graph reasoning module if needed
        if hasattr(model, 'reasoning_manager'):
            graph_reasoning = GraphReasoningModule(
                hidden_size=config.hidden_size,
                gnn_hidden_dim=config.gnn_hidden_dim,
                num_layers=config.gnn_num_layers,
                dropout=config.gnn_dropout,
                gnn_type=config.gnn_type,
                use_attention=True,
                num_heads=config.gnn_num_heads,
                use_edge_features=config.gnn_use_edge_features,
                gnn_encoder=gnn_encoder  # Connect to the created GNN encoder
            )
            model.reasoning_manager.graph_reasoning = graph_reasoning
            
            # Connect graph reasoning to other reasoning components if available
            if hasattr(model.reasoning_manager, 'tree_reasoning'):
                graph_reasoning.connect_tree_reasoning(model.reasoning_manager.tree_reasoning)
                
            logger.info(f"Added graph reasoning module with {config.gnn_type} encoder")
    
    # Replace standard linear layers with memory-efficient ones
    def replace_with_memory_efficient(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and child.in_features * child.out_features > 1000000:
                # Replace large linear layers with memory efficient version
                setattr(module, name, MemoryEfficientLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None
                ))
            else:
                # Recursively check child modules
                replace_with_memory_efficient(child)
    
    # Apply memory-efficient linear replacement
    replace_with_memory_efficient(model)
    
    # Enable memory components if configured
    if config.use_memory_augmentation:
        model.memory_bank = StrategySequenceMemory(
            hidden_size=config.hidden_size,
            memory_size=config.memory_size,
            strategy_embedding_size=config.hidden_size // 4,
            num_strategy_types=10,  # Default to 10 strategy types
            success_threshold=config.strategy_selection_threshold if hasattr(config, "strategy_selection_threshold") else 0.7,
            task_clustering_threshold=0.85
        )
        logger.info(f"Enabled memory augmentation with memory size {config.memory_size}")
        
        # Add CacheManager if it's used
        if hasattr(model, 'cache_manager') or 'CacheManager' in globals() or 'CacheManager' in locals():
            from model.cache import CacheManager
            model.cache_manager = CacheManager()
            logger.info(f"Initialized cache manager for memory operations")
    
    # Set up computational efficiency config
    comp_efficiency_config = ComputationalEfficiencyConfig(
        use_activation_checkpointing=config.use_activation_checkpointing,
        checkpoint_every_n_layers=config.checkpoint_every_n_layers,
        use_efficient_attention=config.use_efficient_attention,
        attention_implementation=config.attention_implementation,
        use_kv_caching=config.use_kv_caching,
        max_cache_length=config.max_cache_length,
        use_kernel_fusion=config.use_kernel_fusion,
        use_quantization=config.use_quantization,
        quantization_bits=config.quantization_bits,
        quantization_scheme=config.quantization_scheme,
        use_mixed_precision=config.use_mixed_precision
    )
    model.computational_efficiency = comp_efficiency_config
    
    # Add Rotary Embeddings for improved position encoding if configured
    if config.use_rotary_embeddings:
        model.rotary_embeddings = RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            base=config.rotary_embedding_base,
            max_position_embeddings=config.max_position_embeddings
        )
        # Monkey patch the attention mechanism to use rotary embeddings
        if hasattr(model, 'transformer'):
            for layer in model.transformer.layers:
                if hasattr(layer, 'self_attention'):
                    original_attention_forward = layer.self_attention.forward
                    
                    def attention_forward_with_rotary(self, q, k, v, mask=None):
                        # Apply rotary embeddings to q and k
                        seq_len = q.size(1)
                        q_rot, k_rot = apply_rotary_pos_emb(
                            q, k, 
                            model.rotary_embeddings.cos_cached,
                            model.rotary_embeddings.sin_cached,
                            seq_len=seq_len
                        )
                        return original_attention_forward(self, q_rot, k_rot, v, mask)
                    
                    layer.self_attention.forward = types.MethodType(
                        attention_forward_with_rotary, layer.self_attention
                    )
    
    # Enable logical reasoning if configured
    if config.use_logical_reasoning:
        logical_config = type('LogicalConfig', (), {
            'hidden_size': config.hidden_size,
            'intermediate_size': config.hidden_size * 4,
            'num_logical_layers': 2,
            'dropout': config.dropout,
            'use_predicate_groundings': True,
            'use_rule_applications': True,
            'max_rule_depth': 3,
            'use_constraint_validation': True,
            'use_attention': True,
            'num_heads': config.num_attention_heads // 2
        })
        
        # Create and add logical reasoning layer
        model.logical_reasoning = LogicalReasoningLayer(logical_config)
        
        # Register logical reasoning with the main model
        if hasattr(model, 'register_reasoning_module'):
            model.register_reasoning_module('logical', model.logical_reasoning)
        
        # Add integration with other components
        if hasattr(model, 'reasoning_manager'):
            # Configure logical reasoning integration with the reasoning manager
            model.reasoning_manager.register_logical_reasoner(model.logical_reasoning)
            
            # Connect logical reasoning to other reasoning components
            if hasattr(model, 'tree_reasoning'):
                model.logical_reasoning.connect_tree_reasoning(model.tree_reasoning)
                
            if hasattr(model, 'symbolic_reasoning'):
                model.logical_reasoning.connect_symbolic_reasoning(model.symbolic_reasoning)
            
        logger.info("Enabled logical reasoning capabilities with predicate grounding and rule application")

    # Add UL2 mixture-of-denoisers if configured
    if config.use_ul2:
        from model.ul2 import UL2Denoiser
        model.ul2_denoiser = UL2Denoiser(
            hidden_size=config.hidden_size,
            objective_mix=config.ul2_objective_mix.split(','),
            dynamic_schedule=config.ul2_dynamic_schedule,
            task_weights={
                'R': config.ul2_r_weight,
                'X': config.ul2_x_weight,
                'S': config.ul2_s_weight
            },
            causal_lm_weight=config.ul2_causal_lm_weight,
            initial_task=config.ul2_initial_task
        )
        logger.info(f"Enabled UL2 mixture-of-denoisers with objectives: {config.ul2_objective_mix}")

    # Add CoLT5 conditional computation if configured
    if config.use_colt5:
        from model.colt5 import CoLT5Layer
        # Process layer indices
        colt5_layer_indices = []
        if config.colt5_layer_indices:
            # Handle both List[int] and string types gracefully
            if isinstance(config.colt5_layer_indices, list):
                colt5_layer_indices = config.colt5_layer_indices
            elif isinstance(config.colt5_layer_indices, str):
                colt5_layer_indices = [int(idx) for idx in config.colt5_layer_indices.split(',')]
            else:
                # Convert single integer if necessary
                try:
                    colt5_layer_indices = [int(config.colt5_layer_indices)]
                except (ValueError, TypeError):
                    logger.warning(f"Invalid colt5_layer_indices format: {config.colt5_layer_indices}, using default")
                    colt5_layer_indices = list(range(0, config.num_layers, 2))
        else:
            # Default to every other layer
            colt5_layer_indices = list(range(0, config.num_layers, 2))
            
        # Replace selected layers with CoLT5 layers
        for i, layer in enumerate(model.transformer.layers):
            if i in colt5_layer_indices:
                model.transformer.layers[i] = CoLT5Layer(
                    original_layer=layer,
                    threshold=config.colt5_threshold,
                    light_ffn_ratio=config.colt5_light_ffn_ratio,
                    heavy_ffn_ratio=config.colt5_heavy_ffn_ratio
                )
        logger.info(f"Enabled CoLT5 conditional computation in layers: {colt5_layer_indices}")

    # Add Hyena operator if configured
    if config.use_hyena:
        from model.hyena import HyenaOperator
        # Process layer indices
        hyena_layer_indices = []
        if config.hyena_layer_indices:
            hyena_layer_indices = [int(idx) for idx in config.hyena_layer_indices.split(',')]
        else:
            # Default to first 25% of layers
            hyena_layer_indices = list(range(0, config.num_layers // 4))
            
        # Add Hyena operators to selected layers
        for i, layer in enumerate(model.transformer.layers):
            if i in hyena_layer_indices:
                layer.hyena = HyenaOperator(
                    dim=config.hidden_size,
                    num_blocks=config.hyena_num_blocks
                )
        logger.info(f"Enabled Hyena operator in layers: {hyena_layer_indices}")

    # Enable adaptive reasoning if configured
    if config.use_adaptive_reasoning:
        # Set up adaptive reasoning config
        adaptive_config = AdaptiveReasoningConfig(
            enabled=True,
            default_strategy=getattr(ReasoningStrategy, config.default_reasoning_strategy),
            strategy_selection_threshold=config.strategy_selection_threshold,
            max_reasoning_steps=config.max_reasoning_steps,
            use_meta_learning=config.use_reasoning_meta_learning
        )
        
        # Create component selector for adaptive reasoning
        model.reasoning_selector = ComponentSelector(adaptive_config, config.hidden_size)
        
        # Create reasoning manager
        model.reasoning_manager = ReasoningManager(
            hidden_size=config.hidden_size,
            max_recursive_depth=config.recursive_depth,
            max_mcts_simulations=config.mcts_simulations,
            use_mcts=config.use_mcts,
            use_recursive=config.use_recursive_reasoning,
            use_symbolic=config.use_neural_symbolic
        )
        
        # Add tree reasoning if configured
        if config.use_tree_reasoning:
            tree_reasoning = TreeReasoning(
                hidden_size=config.hidden_size,
                max_depth=config.reasoning_depth,
                dropout=config.dropout
            )
            model.reasoning_manager.register_tree_reasoner(tree_reasoning)
            
        # Add recursive reasoner if configured
        if config.use_recursive_reasoning:
            recursive_reasoner = RecursiveReasoner(
                hidden_size=config.hidden_size,
                max_depth=config.recursive_depth,
                dropout=config.dropout
            )
            model.reasoning_manager.register_recursive_reasoner(recursive_reasoner)
            
        # Add knowledge reasoner if configured
        if config.use_knowledge_reasoning:
            knowledge_reasoner = KnowledgeReasoner(
                hidden_size=config.hidden_size,
                knowledge_graph_size=config.knowledge_graph_size,
                dropout=config.dropout
            )
            model.reasoning_manager.register_knowledge_reasoner(knowledge_reasoner)
            
        # Set up adaptive tree reasoning if needed
        if config.use_tree_reasoning and config.use_adaptive_computation:
            tree_config = TreeReasoningConfig(
                hidden_size=config.hidden_size,
                use_adaptive_computation=True,
                max_computation_steps=config.max_reasoning_steps,
                reasoning_depth=config.reasoning_depth,
                dropout=config.dropout,
                use_attention=True
            )
            model.tree_reasoning = TreeReasoningModule(tree_config)
            
            # Create adaptive tree reasoner
            adaptive_tree_reasoner = AdaptiveTreeReasoner(
                config=tree_config,
                max_depth=config.reasoning_depth
            )
            model.reasoning_manager.register_tree_reasoner(adaptive_tree_reasoner)
    
    # Set up recurrent reasoning if configured
    if config.use_recurrent_reasoning:
        recursive_config = RecursiveReasoningConfig(
            hidden_size=config.hidden_size,
            max_steps=config.max_reasoning_steps,
            min_steps=config.min_reasoning_steps,
            early_stopping=config.early_stopping,
            convergence_threshold=config.convergence_threshold,
            use_residual_connection=True,
            use_gating=True,
            use_memory=True
        )
        
        model.recurrent_reasoning = RecurrentReasoningBlock(
            config=recursive_config
        )
    
    # Set up advanced MCTS reasoning if configured
    if config.use_mcts_advanced:
        # Set up standard MCTS reasoner
        mcts_config = MCTSConfig(
            hidden_size=config.hidden_size,
            max_simulations=config.mcts_simulations,
            use_adaptive_simulation_count=config.use_adaptive_simulation_count,
            confidence_threshold=config.confidence_threshold,
            use_dirichlet_noise=config.use_dirichlet_noise,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_weight=config.dirichlet_weight,
            store_reasoning_trace=True  # Store reasoning trace for explainability
        )
        model.mcts_reasoner = MCTSReasoner(mcts_config)
        
        # Set up tree-of-thought MCTS reasoner from tree_reasoning_mcts.py
        tree_mcts_config = TreeMCTSConfig(
            max_iterations=config.mcts_simulations,
            exploration_weight=1.0,
            max_depth=config.reasoning_depth,
            rollout_depth=3,
            top_k_candidates=4,
            use_value_network=True,
            early_stopping_threshold=0.95,
            use_beam_search=True,
            beam_size=4,
            enable_visualization=True
        )
        
        # Create and register the tree-based MCTS reasoning module
        model.tree_mcts_reasoning = create_mcts_reasoning_module(tree_mcts_config)
        logger.info("Initialized tree-based MCTS reasoning module for enhanced tree-of-thought reasoning")
        
        # Integrate MCTS with reasoning manager if available
        if hasattr(model, 'reasoning_manager'):
            model.reasoning_manager.register_mcts_reasoner(model.mcts_reasoner)
            # Add method to register tree MCTS if not already present
            if not hasattr(model.reasoning_manager, 'register_tree_mcts_reasoner'):
                def register_tree_mcts_reasoner(self, tree_mcts_reasoner):
                    self.tree_mcts_reasoner = tree_mcts_reasoner
                    return self
                model.reasoning_manager.register_tree_mcts_reasoner = types.MethodType(
                    register_tree_mcts_reasoner, model.reasoning_manager
                )
            # Register the tree MCTS reasoner
            model.reasoning_manager.register_tree_mcts_reasoner(model.tree_mcts_reasoning)
            
        # Set up MCTS optimization for inference
        if hasattr(model.mcts_reasoner, 'config'):
            model.mcts_reasoner.config.use_simulation_optimization = True
            model.mcts_reasoner.config.simulation_batch_size = 16
            
        logger.info("Enabled advanced MCTS reasoning with adaptive simulation count and reasoning trace")
    
    # Add Neural Symbolic Integration if configured
    if config.use_neural_symbolic:
        ns_config = NeuralSymbolicConfig(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            use_symbolic_verification=config.use_symbolic_verification,
            use_symbolic_abstraction=config.use_symbolic_abstraction,
            use_symbolic_composition=config.use_symbolic_composition
        )
        model.symbolic_reasoning = NeuralSymbolicIntegration(ns_config)
        
        # Add symbolic reasoning layer for more advanced reasoning
        model.symbolic_layer = SymbolicReasoningLayer(ns_config)
    
    # Add Tree LSTM for hierarchical reasoning if configured
    if config.use_tree_lstm:
        from model.tree_lstm import TreeLSTM, TreeLSTMConfig, TreeLSTMIntegration
        
        # Create Tree LSTM configuration
        tree_lstm_config = TreeLSTMConfig(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            cell_type="child_sum",  # Use child-sum variant by default
            n_ary_factor=2,         # Binary trees by default
            max_children=10,
            max_depth=config.tree_lstm_max_depth,
            dropout=config.dropout,
            layer_norm=True,
            residual_connections=True,
            recursive_mode=True if config.use_recursive_reasoning else False,
            share_weights=False
        )
        
        # Create the TreeLSTM module
        model.tree_lstm = TreeLSTM(tree_lstm_config)
        
        # Create the integration module for cleaner interface
        model.tree_lstm_integration = TreeLSTMIntegration(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            cell_type="child_sum",
            max_depth=config.tree_lstm_max_depth,
            dropout=config.dropout,
            predict_tree_structure=True  # Automatically infer tree structures
        )
        
        logger.info(f"Initialized enhanced TreeLSTM with max depth {config.tree_lstm_max_depth}")
        
        # Integrate with reasoning manager if available
        if hasattr(model, 'reasoning_manager'):
            # Add TreeLSTM as a reasoning component
            model.reasoning_manager.tree_lstm = model.tree_lstm_integration
            
            # Register a method to use TreeLSTM for reasoning
            def tree_lstm_reasoning(self, hidden_states, attention_mask=None, **kwargs):
                """Apply TreeLSTM reasoning to hidden states"""
                return self.tree_lstm(hidden_states, attention_mask=attention_mask)
                
            if not hasattr(model.reasoning_manager, 'apply_tree_lstm_reasoning'):
                import types
                model.reasoning_manager.apply_tree_lstm_reasoning = types.MethodType(
                    tree_lstm_reasoning, model.reasoning_manager
                )
    
    # Add SAT Solver for logical reasoning if configured
    if config.use_sat_solver:
        sat_config = type('SATConfig', (), {
            'hidden_size': config.hidden_size,
            'num_predicates': 100,  # Default number of predicates
            'sat_iterations': config.max_sat_iterations,
            'num_attention_heads': config.num_attention_heads,
            'dropout': config.dropout
        })
        model.sat_solver = SATSolver(sat_config)
    
    # Enable numerical precision if configured
    if config.use_numerical_precision:
        # Create and add numerical precision module
        model.numerical_precision = NumericalPrecisionModule(
            mode=config.numerical_precision_mode,
            use_fp8_matmul=config.use_fp8_matmul,
            use_stable_embedding=config.use_stable_embedding
        )
        
        # Add high precision math operations
        model.high_precision_math = HighPrecisionMathOperations(
            default_precision=config.numerical_precision_mode,
            guard_underflow=True,
            guard_overflow=True,
            validate_gradients=True
        )
        
        # Register hooks to automatically use high precision for mathematical operations
        def register_precision_hooks(module):
            """Register hooks for precision control in mathematical operations"""
            # Handle forward pre-hook to ensure math operations are high precision
            def math_precision_hook(m, inputs):
                if config.use_numerical_precision and hasattr(model, 'high_precision_math'):
                    return model.high_precision_math.prepare_inputs(inputs)
                return inputs
                
            # Register the hook on modules that do mathematical operations
            module.register_forward_pre_hook(math_precision_hook)
            
            # Recursively register for children
            for child in module.children():
                register_precision_hooks(child)
                
        # Apply precision hooks to model
        register_precision_hooks(model)
        
        logger.info(f"Enabled high-precision mathematical operations with mode: {config.numerical_precision_mode}")
        if config.use_fp8_matmul:
            logger.info("Enabled FP8 matrix multiplication for improved efficiency/precision balance")
    # Enable causal inference if configured
    if config.use_causal_inference:
        from model.reasoning.causal_inference import CausalInferenceEngine, CausalInferenceConfig
        causal_config = CausalInferenceConfig(
            hidden_size=config.hidden_size,
            use_do_calculus=True,
            use_counterfactual_reasoning=False,
            use_structural_causal_models=True
        )
        model.causal_engine = CausalInferenceEngine(causal_config)
        logger.info("Initialized causal inference engine")

    # Enable formal verification if configured
    if config.use_formal_verification:
        from model.reasoning.formal_verification import FormalVerificationConfig, UncertaintyAwareVerifier
        from model.reasoning.mathematical import EnhancedMathematicalReasoning
        
        verification_config = FormalVerificationConfig(
            hidden_size=config.hidden_size,
            verification_threshold=config.verification_threshold,
            uncertainty_threshold=config.uncertainty_threshold,
            quantify_uncertainty=True
        )
        model.verifier = UncertaintyAwareVerifier(verification_config)
        
        # Add enhanced mathematical reasoning if needed
        if config.verify_mathematical_reasoning:
            model.math_reasoning = EnhancedMathematicalReasoning(
                hidden_size=config.hidden_size,
                numerical_precision_config=config.numerical_precision_mode,
                verification_config=verification_config
            )
    
    # Enable RAG if configured
    if config.use_rag:
        # Create RAG config if not provided
        if config.rag_config is None:
            config.rag_config = EnhancedRAGConfig(
                hidden_size=config.rag_hidden_size,
                retriever_dim=config.rag_retriever_dim,
                num_attention_heads=config.rag_num_attention_heads,
                dropout=config.rag_dropout,
                max_knowledge_items=config.rag_max_knowledge_items,
                use_approximate_search=config.rag_use_approximate_search,
                index_type=config.rag_index_type,
                num_partitions=config.rag_num_partitions,
                num_probe=config.rag_num_probe,
                similarity_metric=config.rag_similarity_metric,
                normalize_embeddings=config.rag_normalize_embeddings
            )
        
        # Initialize RAG
        model.rag = EnhancedRAG(config.rag_config)
        
        # Initialize RAG generator
        model.rag_generator = EnhancedRAGGenerator(
            base_model=model,
            rag_config=config.rag_config
        )
        logger.info("Initialized RAG with enhanced retrieval and generation capabilities")
    # Enable constitutional AI if configured
    if config.use_constitutional_ai:
        constitutional_config = ConstitutionalAIConfig(
            hidden_size=config.hidden_size,
            num_principles=config.num_principles,
            max_revision_iterations=config.max_revision_iterations,
            validation_threshold=config.validation_threshold
        )
        model.constitutional_ai = ConstitutionalAI(constitutional_config)
    
    # Enable MoE if configured
    if config.use_moe:
        if config.max_seq_length >= 16384:
            # Use enhanced MoE for long contexts
            logger.info(f"Enabling Enhanced Memory-Efficient MoE with {config.num_experts} experts for long context ({config.max_seq_length} tokens)")
            
            # Create config for enhanced MoE
            moe_config = LongContextMoEConfig(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                max_seq_length=config.max_seq_length,
                block_size=min(2048, config.max_seq_length // 64),  # Adaptive block size
                token_routing_budget=min(0.3, 32768 / config.max_seq_length * 0.3),  # Scale with sequence length
                use_rwkv_integration=config.use_rwkv,
                use_gradient_checkpointing=config.use_activation_checkpointing,
                use_state_compression=config.max_seq_length >= 32768,
                use_quantization=config.use_quantization,
                use_qat=config.use_quantization
            )
            
            # Create and attach enhanced MoE
            model.moe = create_long_context_moe(moe_config)
            
            # Configure ValkyrieLLM to use the advanced MoE implementation
            model.use_enhanced_moe = True
            model.moe_config = moe_config
            
        else:
            # Use standard MoE for shorter contexts
            logger.info(f"Enabling standard MoE with {config.num_experts} experts")
            model.moe_layer = ExpertGating(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                k=config.top_k_experts,
                capacity_factor=config.moe_capacity_factor,
                dropout=config.expert_dropout
            )
            
            # Apply MoE to model components if needed
            if hasattr(model, 'transformer'):
                if hasattr(model.transformer, 'layers') and len(model.transformer.layers) > 0:
                    logger.info(f"Enabling Mixture of Experts with {config.num_experts} experts in transformer layers")
                    # Replace or augment FFN layers with MoE in transformer
                    for layer_idx, layer in enumerate(model.transformer.layers):
                        if hasattr(layer, 'feed_forward'):
                            # Create a layer-specific expert gating
                            layer.expert_gating = ExpertGating(
                                input_size=config.hidden_size,
                                num_experts=config.num_experts // 2,  # Use fewer experts per layer
                                capacity_factor=config.moe_capacity_factor,
                                top_k=config.top_k_experts,
                                dropout=config.expert_dropout
                            )
                            
                            # Hook the expert gating into the feed forward path
                            original_feed_forward = layer.feed_forward.forward
                            
                            def feed_forward_with_moe(self, x):
                                # First apply expert gating
                                moe_output = self.expert_gating(x)
                                # Then apply original feed forward as a residual
                                ff_output = original_feed_forward(self, x)
                                # Combine the outputs
                                return moe_output + ff_output
                            
                            # Apply the new forward function
                            layer.feed_forward.forward = types.MethodType(
                                feed_forward_with_moe, layer.feed_forward
                            )
                    logger.info(f"Successfully enabled Mixture of Experts with {config.num_experts} experts")
                else:
                    logger.warning("No transformer layers found, cannot apply Mixture of Experts to transformer")
    
    # Inside setup_advanced_model
    if config.use_moe and config.max_seq_length >= 32768:
        # Validate GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if gpu_mem < 24 and config.max_seq_length >= 65536:
            logger.warning(f"Insufficient GPU memory ({gpu_mem:.1f}GB) for 64K context. Reducing to 32K.")
            config.max_seq_length = 32768
    
    # Register causal forward hook if enabled
    if hasattr(model, 'causal_engine'):
        def causal_forward_hook(module, input, output):
            if hasattr(output, 'last_hidden_state'):
                batch_size, seq_len, hidden_size = output.last_hidden_state.shape
                causal_outputs = model.causal_engine(output.last_hidden_state)
                output.causal_outputs = causal_outputs
                output.causal_loss = causal_outputs.get('loss', torch.zeros(1, device=output.last_hidden_state.device))
            return output

        model.register_forward_hook(causal_forward_hook)
        def causal_forward_hook(module, input, output):
            if hasattr(output, 'last_hidden_state'):
                batch_size, seq_len, hidden_size = output.last_hidden_state.shape
                causal_outputs = model.causal_engine(output.last_hidden_state)
                output.causal_outputs = causal_outputs
                output.causal_loss = causal_outputs.get('loss', torch.zeros(1, device=output.last_hidden_state.device))
            return output

        model.register_forward_hook(causal_forward_hook)
        def causal_forward_hook(module, input, output):
            if hasattr(output, 'last_hidden_state'):
                batch_size, seq_len, hidden_size = output.last_hidden_state.shape
                causal_outputs = model.causal_engine(output.last_hidden_state)
                output.causal_outputs = causal_outputs
                output.causal_loss = causal_outputs.get('loss', torch.zeros(1, device=output.last_hidden_state.device))
            return output

        model.register_forward_hook(causal_forward_hook)

    # Register causal forward hook if enabled
    if hasattr(model, 'causal_engine'):
        def causal_forward_hook(module, input, output):
            if hasattr(output, 'last_hidden_state'):
                batch_size, seq_len, hidden_size = output.last_hidden_state.shape
                causal_outputs = model.causal_engine(output.last_hidden_state)
                output.causal_outputs = causal_outputs
                output.causal_loss = causal_outputs.get('loss', torch.zeros(1, device=output.last_hidden_state.device))
            return output

        model.register_forward_hook(causal_forward_hook)

    # Set up memory components if configured
    if config.use_memory_augmentation:
        # Enable Memory Bank
        model.enable_memory_bank(
            memory_size=config.memory_size,
            use_episodic=config.use_episodic_memory,
            use_working=config.use_working_memory,
            use_long_term=config.use_long_term_memory
        )
        
        # Set up Cache Manager for efficient inference
        model.enable_cache_manager(cache_size=1000)
        
        # Set up memory compaction if needed
        if config.use_memory_compaction:
            if not hasattr(model, 'memory_bank'):
                logger.warning("Memory compaction enabled but memory bank is not. Enabling memory bank...")
                model.enable_memory_bank(memory_size=config.memory_size)
            
            model.memory_bank.enable_compaction(
                threshold=config.memory_compaction_threshold,
                strategy=config.compaction_strategy,
                interval=config.compaction_interval
            )
            
        logger.info(f"Enabled memory augmentation with size {config.memory_size}")
    
    # Set up attention mechanisms if configured
    if config.use_efficient_attention:
        # Enable enhanced attention
        model.enable_enhanced_attention()
        
        # Set up specific attention mechanisms based on configuration
        model.enable_attention_mechanisms(
            use_flash=config.attention_implementation == "flash",
            use_sliding_window=hasattr(config, 'sliding_window_size'),
            use_grouped_query=hasattr(config, 'num_query_groups')
        )
        
        logger.info(f"Enabled efficient attention with implementation: {config.attention_implementation}")
    
    # Initialize tree reasoning if configured
    if config.use_tree_reasoning and not hasattr(model, 'tree_reasoning'):
        model = model.enable_reasoning('tree')
        logger.info("Enabled tree reasoning")
    
    # Initialize recursive reasoning if configured
    if config.use_recursive_reasoning and not hasattr(model, 'recursive_reasoning'):
        model = model.enable_reasoning('recursive')
        logger.info("Enabled recursive reasoning")
    
    # Initialize neural symbolic reasoning if configured
    if config.use_neural_symbolic and not hasattr(model, 'neural_symbolic_reasoning'):
        model = model.enable_reasoning('neural_symbolic')
        logger.info("Enabled neural symbolic reasoning")
        
    # Initialize knowledge reasoning if configured
    if config.use_knowledge_reasoning and not hasattr(model, 'knowledge_reasoning'):
        model = model.enable_reasoning('knowledge')
        logger.info("Enabled knowledge reasoning")
    
    # If using all reasoning types, enable adaptive reasoning
    if (config.use_tree_reasoning and config.use_recursive_reasoning and 
        config.use_neural_symbolic and config.use_knowledge_reasoning and 
        config.use_mcts):
        model = model.enable_reasoning('adaptive')
        logger.info("Enabled adaptive reasoning with all reasoning types")
    
    # Add the following code to the setup_advanced_model function after the TreeLSTM initialization section:

    # Add Tree LSTM for hierarchical reasoning if configured
    if config.use_tree_lstm:
        # Create Tree LSTM configuration
        tree_lstm_config = TreeLSTMConfig(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            cell_type="child_sum",  # Use child-sum variant by default
            n_ary_factor=2,         # Binary trees by default
            max_children=10,
            max_depth=config.tree_lstm_max_depth,
            dropout=config.dropout,
            layer_norm=True,
            residual_connections=True,
            recursive_mode=True if config.use_recursive_reasoning else False,
            share_weights=False
        )
        
        # Create the TreeLSTM module
        model.tree_lstm = TreeLSTM(tree_lstm_config)
        
        # Create the integration module for cleaner interface
        model.tree_lstm_integration = TreeLSTMIntegration(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            cell_type="child_sum",
            max_depth=config.tree_lstm_max_depth,
            dropout=config.dropout,
            predict_tree_structure=True  # Automatically infer tree structures
        )
        
        logger.info(f"Initialized enhanced TreeLSTM with max depth {config.tree_lstm_max_depth}")
        
        # Integrate with reasoning manager if available
        if hasattr(model, 'reasoning_manager'):
            # Add TreeLSTM as a reasoning component
            model.reasoning_manager.tree_lstm = model.tree_lstm_integration
            
            # Register a method to use TreeLSTM for reasoning
            def tree_lstm_reasoning(self, hidden_states, attention_mask=None, **kwargs):
                """Apply TreeLSTM reasoning to hidden states"""
                return self.tree_lstm(hidden_states, attention_mask=attention_mask)
                
            if not hasattr(model.reasoning_manager, 'apply_tree_lstm_reasoning'):
                model.reasoning_manager.apply_tree_lstm_reasoning = types.MethodType(
                    tree_lstm_reasoning, model.reasoning_manager
                )
    
    return model
def apply_computational_optimizations(model: ValkyrieLLM, config: AdvancedTrainingConfig) -> ValkyrieLLM:
    """Apply computational optimizations to the model."""
    # Apply Hyena operator if enabled
    if config.use_hyena:
        # Import the replacement function
        from model.hyena import replace_attention_with_hyena
        # Apply the replacement using the config and specified layer indices
        model = replace_attention_with_hyena(
            model=model,
            config=config, # Pass the full config object
            layer_indices=config.hyena_layer_indices
        )
        # Logger message is now handled inside replace_attention_with_hyena

    # FlashAttention v2 integration
    if config.use_flash_attention_v2:
        try:
            # Try importing flash attention if available
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            logger.info("Using FlashAttention-2 for improved memory efficiency")
            
            # Replace standard attention with flash attention v2 where possible
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
                for layer_idx, layer in enumerate(model.transformer.layers):
                    if hasattr(layer, 'self_attention'):
                        # Update the attention forward function to use flash_attn
                        original_attention_forward = layer.self_attention.forward
                        
                        def flash_attention_wrapper(self, q, k, v, mask=None):
                            # Ensure shapes are compatible with flash_attn
                            # flash_attn expects [batch, seqlen, n_heads, head_dim]
                            batch_size = q.size(0)
                            q_len = q.size(1)
                            k_len = k.size(1)
                            
                            # Reshape for flash_attn
                            q = q.view(batch_size, q_len, self.num_heads, -1)
                            k = k.view(batch_size, k_len, self.num_heads, -1)
                            v = v.view(batch_size, k_len, self.num_heads, -1)
                            
                            # Apply flash attention
                            if mask is None:
                                # Use the standard flash_attn function
                                output = flash_attn_func(
                                    q, k, v, 
                                    dropout_p=config.flash_attention_v2_config.get("dropout_p", 0.0),
                                    softmax_scale=config.flash_attention_v2_config.get("softmax_scale", None),
                                    causal=config.flash_attention_v2_config.get("causal", True)
                                )
                            else:
                                # Use the variable length version for masked attention
                                # First convert the mask to cu_seqlens format
                                from flash_attn.bert_padding import pad_input, unpad_input
                                indices = torch.nonzero(mask.sum(dim=-1))
                                max_len = mask.sum(dim=-1).max().item()
                                cu_seqlens, indices, q_unpad, k_unpad, v_unpad = unpad_input(q, mask)
                                
                                output_unpad = flash_attn_varlen_func(
                                    q_unpad, k_unpad, v_unpad,
                                    cu_seqlens, cu_seqlens,
                                    max_len, max_len,
                                    dropout_p=config.flash_attention_v2_config.get("dropout_p", 0.0),
                                    softmax_scale=config.flash_attention_v2_config.get("softmax_scale", None),
                                    causal=config.flash_attention_v2_config.get("causal", True)
                                )
                                # Pad the output back
                                output = pad_input(output_unpad, indices, batch_size, q_len)
                            
                            # Reshape output back to original format [batch, seqlen, hidden_dim]
                            output = output.contiguous().view(batch_size, q_len, -1)
                            return output
                        
                        # Replace the attention forward method
                        layer.self_attention.forward = types.MethodType(flash_attention_wrapper, layer.self_attention)
                        logger.info(f"Replaced attention in layer {layer_idx} with FlashAttention-2")
        except ImportError:
            logger.warning("FlashAttention-2 not available, falling back to standard attention")
    
    # Memory compaction strategies
    if config.use_memory_compaction:
        from model.memory.compaction import MemoryCompactor
        model.memory_compaction = MemoryCompactor(
            hidden_size=config.hidden_size,
            compaction_strategy=config.compaction_strategy,
            threshold=config.memory_compaction_threshold,
            interval=config.compaction_interval
        )
        
        # Hook into the forward method to apply compaction periodically
        original_forward = model.forward
        
        def forward_with_compaction(self, *args, **kwargs):
            # Get the result from original forward
            result = original_forward(self, *args, **kwargs)
            
            # Apply compaction if needed
            if hasattr(self, 'memory_compaction') and hasattr(self, 'compaction_step'):
                self.compaction_step += 1
                if self.compaction_step % self.memory_compaction.interval == 0:
                    result = self.memory_compaction(result)
            
            return result
        
        # Patch the forward method
        model.compaction_step = 0
        model.forward = types.MethodType(forward_with_compaction, model)
        logger.info(f"Enabled memory compaction with strategy: {config.compaction_strategy}")
    
    # Apply gradient checkpointing based on strategy
    if config.use_gradient_checkpointing:
        if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'layers'):
            logger.warning("Cannot apply gradient checkpointing - model structure not compatible")
        else:
            layers = model.transformer.layers
            
            if config.checkpointing_strategy == "uniform":
                # Checkpoint every n layers
                for i in range(0, len(layers), config.checkpoint_every_n_layers):
                    layers[i].gradient_checkpointing = True
                logger.info(f"Applied uniform gradient checkpointing every {config.checkpoint_every_n_layers} layers")
                
            elif config.checkpointing_strategy == "end_of_layer":
                # Checkpoint only the end of each layer
                for layer in layers:
                    if hasattr(layer, 'feed_forward'):
                        layer.feed_forward.gradient_checkpointing = True
                logger.info("Applied end-of-layer gradient checkpointing")
                
            elif config.checkpointing_strategy == "selective":
                # Checkpoint specific layers
                for i in config.checkpointing_layers:
                    if i < len(layers):
                        layers[i].gradient_checkpointing = True
                logger.info(f"Applied selective gradient checkpointing to layers: {config.checkpointing_layers}")
                
            else:
                logger.warning(f"Unknown checkpointing strategy: {config.checkpointing_strategy}")
    
    # Apply tensor parallelism if enabled
    if config.use_tensor_parallelism:
        try:
            from model.parallelism import apply_tensor_parallelism
            model = apply_tensor_parallelism(
                model, 
                config.tensor_parallel_size,
                config.tensor_parallel_communication
            )
            logger.info(f"Applied tensor parallelism with {config.tensor_parallel_size} devices")
        except ImportError:
            logger.warning("Tensor parallelism module not available, skipping")
    
    # Apply sequence parallelism if enabled
    if config.use_sequence_parallelism:
        try:
            from model.parallelism import apply_sequence_parallelism
            model = apply_sequence_parallelism(
                model,
                config.sequence_parallel_size,
                config.sequence_parallel_chunk_size,
                config.sequence_parallel_overlap
            )
            logger.info(f"Applied sequence parallelism with {config.sequence_parallel_size} devices")
        except ImportError:
            logger.warning("Sequence parallelism module not available, skipping")
    
    # Apply expert parallelism if enabled
    if config.use_expert_parallelism and config.use_moe:
        try:
            from model.parallelism import apply_expert_parallelism
            model = apply_expert_parallelism(
                model,
                config.expert_parallel_size,
                config.expert_parallel_communication
            )
            logger.info(f"Applied expert parallelism with {config.expert_parallel_size} devices")
        except ImportError:
            logger.warning("Expert parallelism module not available, skipping")
    
    # Apply dynamic architecture modifications
    if config.use_progressive_layer_dropping:
        # Setup for progressive layer dropping during training
        model.layer_drop_rate = 0.0  # Start with no dropping
        model.max_layer_drop_rate = config.layer_dropping_rate
        model.use_progressive_layer_dropping = True
        
        # Hook into forward to implement dynamic layer dropping
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            original_layer_forward = model.transformer.layers[0].forward
            
            def forward_with_dropping(self, x, *args, **kwargs):
                # Check if we should drop this layer
                if hasattr(model, 'use_progressive_layer_dropping') and \
                   hasattr(model, 'layer_drop_rate') and \
                   torch.rand(1).item() < model.layer_drop_rate:
                    # Skip this layer
                    return x
                # Otherwise process normally
                return original_layer_forward(self, x, *args, **kwargs)
            
            # Apply to all layers
            for layer in model.transformer.layers:
                layer.forward = types.MethodType(forward_with_dropping, layer)
                
            logger.info(f"Enabled progressive layer dropping with max rate: {config.layer_dropping_rate}")
    
    if config.use_dynamic_width_adjustment:
        # Setup for dynamic width adjustment
        model.width_adjustment_interval = config.width_adjustment_interval
        model.current_width_multiplier = 1.0
        model.width_adjustment_step = 0
        
        # Function to adjust width based on task complexity
        def adjust_model_width(self, step):
            if step % self.width_adjustment_interval == 0:
                # Get validation metrics if available
                if hasattr(self, 'recent_val_metrics') and 'loss' in self.recent_val_metrics:
                    # If loss is improving, reduce width to save compute
                    # If loss is plateauing, increase width for more capacity
                    loss_delta = self.recent_val_metrics.get('loss_delta', 0)
                    
                    if loss_delta < -0.05:  # Loss improving significantly
                        self.current_width_multiplier = max(0.8, self.current_width_multiplier * 0.95)
                    elif loss_delta > -0.01:  # Loss plateauing
                        self.current_width_multiplier = min(1.2, self.current_width_multiplier * 1.05)
                    
                    # Apply the width adjustment
                    self._apply_width_adjustment()
        
        # Function to actually modify the network width
        def _apply_width_adjustment(self):
            if not hasattr(self, 'transformer') or not hasattr(self.transformer, 'layers'):
                return
                
            # Adjust hidden dimension for each layer's feed forward network
            for layer in self.transformer.layers:
                if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'intermediate'):
                    intermediate = layer.feed_forward.intermediate
                    output = layer.feed_forward.output
                    
                    orig_dim = intermediate.weight.size(0)
                    new_dim = int(orig_dim * self.current_width_multiplier)
                    
                    # Dynamically resize the layer
                    # This is a simplified implementation - in practice this would require
                    # more careful handling of weights and may not be possible during training
                    # without specialized infrastructure
                    logger.info(f"Adjusted layer width from {orig_dim} to {new_dim}")
        
        # Attach methods to model
        model.adjust_model_width = types.MethodType(adjust_model_width, model)
        model._apply_width_adjustment = types.MethodType(_apply_width_adjustment, model)
        logger.info(f"Enabled dynamic width adjustment with interval: {config.width_adjustment_interval}")
    
    # Implement expert capacity adaptation for MoE
    if config.use_expert_capacity_adaptation and config.use_moe:
        if hasattr(model, 'moe_layer'):
            # Store original capacity factor
            model.moe_layer.original_capacity_factor = model.moe_layer.capacity_factor
            model.moe_layer.capacity_adjustment_interval = config.expert_capacity_adjustment_interval
            model.moe_layer.adjustment_step = 0
            
            # Create method to update capacity based on expert utilization
            def adjust_expert_capacity(self):
                # Get expert utilization statistics
                if hasattr(self, 'expert_utilization'):
                    util_stats = self.expert_utilization
                    
                    # Calculate standard deviation of utilization
                    mean_util = torch.mean(util_stats)
                    std_util = torch.std(util_stats)
                    
                    # If utilization is very uneven, increase capacity factor
                    if std_util / mean_util > 0.5:  # High variation
                        self.capacity_factor = min(2.0, self.capacity_factor * 1.05)
                    # If utilization is even, decrease capacity factor
                    elif std_util / mean_util < 0.2:  # Low variation
                        self.capacity_factor = max(1.0, self.capacity_factor * 0.98)
                        
                # Reset utilization stats
                self.expert_utilization = torch.zeros(self.num_experts, device=self.expert_gates.weight.device)
                self.adjustment_step += 1
            
            # Attach the method
            model.moe_layer.adjust_expert_capacity = types.MethodType(adjust_expert_capacity, model.moe_layer)
            
            # Hook into the forward to track utilization and adjust periodically
            original_moe_forward = model.moe_layer.forward
            
            def moe_forward_with_adaptation(self, x):
                # Call original forward
                result = original_moe_forward(self, x)
                
                # Track expert utilization
                if not hasattr(self, 'expert_utilization'):
                    self.expert_utilization = torch.zeros(self.num_experts, device=self.expert_gates.weight.device)
                
                # Update utilization based on dispatch weights
                if hasattr(self, 'dispatch_weights'):
                    self.expert_utilization += torch.sum(self.dispatch_weights, dim=0)
                
                # Adjust capacity periodically
                if self.adjustment_step % self.capacity_adjustment_interval == 0:
                    self.adjust_expert_capacity()
                
                return result
            
            # Apply the patched forward
            model.moe_layer.forward = types.MethodType(moe_forward_with_adaptation, model.moe_layer)
            logger.info(f"Enabled expert capacity adaptation with interval: {config.expert_capacity_adjustment_interval}")
    
    # Apply advanced optimization techniques
    if config.use_gradient_centralization:
        # Hook into optimizer step to apply gradient centralization
        if hasattr(model, 'optimizer'):
            original_step = model.optimizer.step
            
            def step_with_gradient_centralization(self, *args, **kwargs):
                # Apply gradient centralization before step
                for param_group in self.param_groups:
                    for p in param_group['params']:
                        if p.dim() > 1 and p.grad is not None:  # Only apply to parameters with dim > 1
                            p.grad.add_(-p.grad.mean(dim=tuple(range(1, p.dim())), keepdim=True))
                
                # Call original step
                return original_step(*args, **kwargs)
            
            # Replace optimizer step method
            model.optimizer.step = types.MethodType(step_with_gradient_centralization, model.optimizer)
            logger.info("Enabled gradient centralization")
    
    if config.use_layerwise_adaptive_scaling:
        # Apply different learning rates to different layers
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers') and hasattr(model, 'optimizer'):
            # Create parameter groups with different learning rates
            param_groups = []
            
            # Add embedding parameters with base learning rate
            if hasattr(model, 'embeddings'):
                param_groups.append({
                    'params': model.embeddings.parameters(),
                    'lr': config.learning_rate,
                    'name': 'embeddings'
                })
            
            # Add transformer layers with scaled learning rates
            for i, layer in enumerate(model.transformer.layers):
                scaling = config.layerwise_scaling_factors.get(f"layer_{i}", 1.0)
                param_groups.append({
                    'params': layer.parameters(),
                    'lr': config.learning_rate * scaling,
                    'name': f'layer_{i}'
                })
            
            # Add final layer norm and output layers
            remaining_params = []
            for name, param in model.named_parameters():
                if not any(p is param for group in param_groups for p in group['params']):
                    remaining_params.append(param)
            
            if remaining_params:
                param_groups.append({
                    'params': remaining_params,
                    'lr': config.learning_rate,
                    'name': 'other'
                })
            
            # Create a new optimizer with these parameter groups
            if isinstance(model.optimizer, torch.optim.Adam):
                model.optimizer = torch.optim.Adam(
                    param_groups,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_epsilon,
                    weight_decay=config.weight_decay
                )
            elif isinstance(model.optimizer, SophiaG):
                model.optimizer = SophiaG(
                    param_groups,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_epsilon,
                    weight_decay=config.weight_decay
                )
                
            logger.info(f"Applied layerwise adaptive scaling with {len(param_groups)} parameter groups")
    
    # Apply alternative optimizers if configured
    if hasattr(model, 'optimizer'):
        if config.use_lion_optimizer:
            try:
                from lion_pytorch import Lion
                # Create Lion optimizer
                model.optimizer = Lion(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay
                )
                logger.info("Enabled Lion optimizer")
            except ImportError:
                logger.warning("Lion optimizer not available, skipping")
        
        elif config.use_adan_optimizer:
            try:
                from adan_pytorch import Adan
                # Create Adan optimizer
                model.optimizer = Adan(
                    model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    betas=(config.adam_beta1, config.adam_beta2, 0.999),
                    eps=config.adam_epsilon
                )
                logger.info("Enabled Adan optimizer")
            except ImportError:
                logger.warning("Adan optimizer not available, skipping")
    
    # Apply training stability enhancements
    if config.gradient_clipping_strategy != "global":
        # Setup for layerwise gradient clipping
        if config.gradient_clipping_strategy == "layerwise":
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
                original_clip_grad_norm = torch.nn.utils.clip_grad_norm_
                
                def layerwise_clip_grad_norm(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
                    # If parameters is a list of layers, clip each layer individually
                    if isinstance(parameters, list) and all(isinstance(p, nn.Module) for p in parameters):
                        for i, layer in enumerate(parameters):
                            layer_max_norm = config.layerwise_clip_thresholds.get(f"layer_{i}", max_norm)
                            torch.nn.utils.clip_grad_norm_(
                                layer.parameters(),
                                layer_max_norm,
                                norm_type=norm_type,
                                error_if_nonfinite=error_if_nonfinite
                            )
                        return None
                    else:
                        # Fall back to standard clipping
                        return original_clip_grad_norm(parameters, max_norm, norm_type, error_if_nonfinite)
                
                # Patch the clip_grad_norm function
                torch.nn.utils.clip_grad_norm_ = layerwise_clip_grad_norm
                logger.info("Enabled layerwise gradient clipping")
        
        elif config.gradient_clipping_strategy == "parameter_group":
            # Setup for parameter group-based clipping
            if hasattr(model, 'optimizer'):
                original_step = model.optimizer.step
                
                def step_with_group_clipping(self, *args, **kwargs):
                    # Clip gradients by parameter group
                    for param_group in self.param_groups:
                        group_name = param_group.get('name', 'default')
                        max_norm = config.layerwise_clip_thresholds.get(group_name, config.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(
                            param_group['params'], 
                            max_norm
                        )
                    
                    # Call original step
                    return original_step(*args, **kwargs)
                
                # Replace optimizer step method
                model.optimizer.step = types.MethodType(step_with_group_clipping, model.optimizer)
                logger.info("Enabled parameter group gradient clipping")
    
    # Apply automatic mixed precision scaling
    if config.use_automatic_mixed_precision_scaling and config.use_mixed_precision:
        # Implement dynamic loss scaling
        if config.loss_scaling_strategy == "dynamic":
            if hasattr(model, 'scaler'):
                # Update scaler configuration for better stability
                model.scaler.set_growth_factor(2.0)
                model.scaler.set_backoff_factor(0.5)
                model.scaler.set_growth_interval(100)
                logger.info("Configured dynamic loss scaling for mixed precision")
        
        elif config.loss_scaling_strategy == "logit":
            # Special scaling for logits to improve numerical stability
            if hasattr(model, 'lm_head') or hasattr(model, 'output_layer'):
                def logit_scaling_hook(module, input, output):
                    # Scale logits to a more numerically stable range
                    scale_factor = 1.0
                    if output.isnan().any() or output.isinf().any():
                        scale_factor = 0.1  # Reduce scale if issues detected
                    return output * scale_factor
                
                # Register forward hook
                if hasattr(model, 'lm_head'):
                    model.lm_head.register_forward_hook(logit_scaling_hook)
                elif hasattr(model, 'output_layer'):
                    model.output_layer.register_forward_hook(logit_scaling_hook)
                
                logger.info("Enabled logit scaling for improved stability")
    
    # Implement NaN/Inf detection and recovery
    if config.nan_inf_detection:
        # Setup NaN detection during training
        original_forward = model.forward
        
        def forward_with_nan_detection(self, *args, **kwargs):
            # Run forward pass
            outputs = original_forward(self, *args, **kwargs)
            
            # Check for NaN/Inf
            if self.training and hasattr(self, 'nan_inf_check_step'):
                self.nan_inf_check_step += 1
                
                if self.nan_inf_check_step % config.nan_inf_check_interval == 0:
                    has_issue = False
                    
                    # Check output tensors
                    if isinstance(outputs, torch.Tensor):
                        has_issue = outputs.isnan().any() or outputs.isinf().any()
                    elif isinstance(outputs, (list, tuple)):
                        for out in outputs:
                            if isinstance(out, torch.Tensor):
                                has_issue = has_issue or out.isnan().any() or out.isinf().any()
                    
                    # If issues found, take action based on recovery strategy
                    if has_issue:
                        logger.warning("NaN/Inf detected in model outputs")
                        
                        if config.recovery_strategy == "rollback":
                            # Signal to training loop that we should roll back
                            self.should_rollback = True
                        elif config.recovery_strategy == "skip":
                            # Signal to training loop that we should skip this batch
                            self.should_skip_batch = True
                        elif config.recovery_strategy == "reduce_lr":
                            # Signal to optimizer to reduce learning rate
                            self.should_reduce_lr = True
            
            return outputs
        
        # Patch the forward method
        model.nan_inf_check_step = 0
        model.should_rollback = False
        model.should_skip_batch = False
        model.should_reduce_lr = False
        model.forward = types.MethodType(forward_with_nan_detection, model)
        logger.info(f"Enabled NaN/Inf detection with {config.recovery_strategy} recovery strategy")
    
    # Enable memory profiling
    if config.enable_memory_profiling:
        try:
            from model.profiling import MemoryProfiler
            model.memory_profiler = MemoryProfiler(interval=config.memory_profile_interval)
            model.memory_profiler.start()
            logger.info(f"Enabled memory profiling with interval: {config.memory_profile_interval}")
        except ImportError:
            logger.warning("Memory profiling module not available, skipping")
    
    # Enable communication profiling
    if config.enable_communication_profiling and config.use_distributed:
        try:
            from model.profiling import CommunicationProfiler
            model.comm_profiler = CommunicationProfiler(interval=config.communication_profile_interval)
            model.comm_profiler.start()
            logger.info(f"Enabled communication profiling with interval: {config.communication_profile_interval}")
        except ImportError:
            logger.warning("Communication profiling module not available, skipping")
    
    # Implement differential checkpointing
    if config.use_differential_checkpointing:
        try:
            from model.checkpointing import DifferentialCheckpointer
            model.diff_checkpointer = DifferentialCheckpointer(
                model=model,
                interval=config.differential_checkpoint_interval,
                compression=config.checkpoint_compression
            )
            logger.info(f"Enabled differential checkpointing with {config.checkpoint_compression} compression")
        except ImportError:
            logger.warning("Differential checkpointing module not available, skipping")
    
    # Apply specialized initialization for RWKV layers
    if config.use_rwkv:
        from model.rwkv.initialization import initialize_rwkv_layers
        initialize_rwkv_layers(model, config.rwkv_initialization)
        logger.info(f"Applied specialized {config.rwkv_initialization} initialization to RWKV layers")
    
    # Apply CoLT5 conditional computation if enabled
    if config.use_colt5:
        from model.colt5 import apply_colt5_conditional_computation
        apply_colt5_conditional_computation(
            model,
            config.colt5_routing_strategy,
            config.colt5_min_capacity,
            config.colt5_max_capacity
        )
        logger.info(f"Enabled CoLT5 conditional computation with {config.colt5_routing_strategy} routing")
    
    # Apply expert-specific initialization
    if config.use_moe:
        from model.moe.initialization import initialize_experts
        initialize_experts(model, config.expert_initialization)
        logger.info(f"Applied specialized {config.expert_initialization} initialization to MoE experts")
    
    return model
def setup_generation(model: ValkyrieLLM, config: AdvancedTrainingConfig) -> Any:
    """Set up text generation and beam search components."""
    logger.info(f"Setting up generation components with beam size: {config.beam_size}")
    
    if config.use_logical_beam_search:
        # Create a logical beam search generator with reasoning capabilities
        logger.info("Using logical beam search with consistency threshold: "
                   f"{config.consistency_threshold}")
        
        beam_config = type('BeamConfig', (), {
            'beam_size': config.beam_size,
            'max_seq_length': config.max_seq_length,
            'length_penalty': config.length_penalty,
            'consistency_threshold': config.consistency_threshold,
            'logical_reward_weight': config.logical_reward_weight
        })
        
        generator = LogicalBeamSearch(model, beam_config)
        
        # Register generator with model for direct access
        if hasattr(model, 'register_generator'):
            model.register_generator(generator)
        else:
            model.generator = generator
        
        return generator
    else:
        # Create standard beam search generator
        logger.info(f"Using standard beam search with beam size: {config.beam_size}")
        generator = BeamSearchGenerator(
            model=model,
            tokenizer=None,  # Will be set later if needed
            beam_size=config.beam_size
        )
        
        # Register generator with model
        if hasattr(model, 'register_generator'):
            model.register_generator(generator)
        else:
            model.generator = generator
            
        return generator
def train(
    model: ValkyrieLLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    training_config: AdvancedTrainingConfig,
    device: torch.device,
    output_dir: str,
    experiment_name: str = "llm_training",
    resume_from_checkpoint: Optional[str] = None,
    use_mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    num_train_epochs: int = 3,
    warmup_steps: int = 1000,
    save_steps: int = 1000,
    eval_steps: int = 500,
    use_curriculum: bool = False,
    use_distributed: bool = False,
     local_rank: int = -1
 ) -> Dict[str, Any]:
    """
    Train the language model with comprehensive features.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        training_config: Training configuration
        device: Device to train on
        output_dir: Directory to save checkpoints
        experiment_name: Name of the experiment
        resume_from_checkpoint: Path to checkpoint to resume from
        use_mixed_precision: Whether to use mixed precision training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        num_train_epochs: Number of training epochs
        warmup_steps: Number of warmup steps for learning rate
        save_steps: Number of steps between checkpoints
        eval_steps: Number of steps between evaluations
        use_curriculum: Whether to use curriculum learning
        use_distributed: Whether to use distributed training
        local_rank: Local rank for distributed training
    Returns:
        Dictionary containing training metrics and results
    """
    # Initialize wandb for experiment tracking
    wandb.init(project=experiment_name, name=f"epoch_{start_epoch}")
    # Set up distributed training if enabled
    if use_distributed:
        setup_distributed(local_rank)
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
            model = model.to(device)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    # Initialize training engine with causal reasoning support
    training_engine = TrainingEngine(
        model=model,
        training_config=training_config,
        device=device,
        use_mixed_precision=use_mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm
    )
    # Apply FSDP/Pipeline Parallelism if using distributed training
    if use_distributed:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.pipeline.sync import Pipe
        
        # Wrap model with FSDP using proper auto wrap policy
        model = FSDP(
            model,
            auto_wrap_policy=default_auto_wrap_policy,
            mixed_precision=torch.float16 if use_mixed_precision else None,
            device_id=torch.cuda.current_device(),
            use_orig_params=True  # Required for Sophia optimizer compatibility
        )
    
        # Setup pipeline parallelism if model has more than 1 layer
        if hasattr(model, 'transformer') and len(model.transformer.layers) > 1:
            # Split model into pipeline stages
            chunks = len(model.transformer.layers) // 2  # Split into 2 stages
            model = Pipe(
                model,
                chunks=chunks,
                checkpoint=training_config.use_activation_checkpointing
            ).to(device)
    
    # Set up Sophia optimizer
    optimizer = SophiaG(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        weight_decay=training_config.weight_decay,
        eps=training_config.adam_epsilon
    )
    
    # Apply FSDP/Pipeline Parallelism if using distributed training
    if use_distributed:
        # Wrap model with FSDP
        model = FSDP(
            model,
            auto_wrap_policy=default_auto_wrap_policy,
            mixed_precision=torch.float16 if use_mixed_precision else None,
            device_id=torch.cuda.current_device()
        )
        
        # Setup pipeline parallelism if model has more than 1 layer
        if hasattr(model, 'transformer') and len(model.transformer.layers) > 1:
            # Split model into pipeline stages
            chunks = len(model.transformer.layers) // 2  # Split into 2 stages
            model = Pipe(
                model,
                chunks=chunks,
                checkpoint='except_last' if config.use_activation_checkpointing else 'never'
            )
    
    # Enable Dynamic Sparse Training
    if hasattr(config, 'use_dynamic_sparse') and config.use_dynamic_sparse:
        from torch.nn.utils import prune
        # Gradually increase model sparsity during training
        def apply_sparsity(epoch):
            sparsity = min(0.8, 0.1 + (epoch / num_train_epochs) * 0.7)  # Ramp from 10% to 80%
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                    prune.remove(module, 'weight')  # Make pruning permanent
            return sparsity
        
        training_engine.add_epoch_callback(apply_sparsity)
        logger.info(f"Enabled dynamic sparse training (will ramp from 10% to 80% sparsity)")
    
    training_engine.set_optimizer(optimizer)
    # Setup Speculative Decoding for inference
    if hasattr(config, 'use_speculative_decoding') and config.use_speculative_decoding:
        from model.generation.speculative import SpeculativeDecoder
        model.speculative_decoder = SpeculativeDecoder(
            draft_model=copy.deepcopy(model.base_model),  # Smaller draft model
            target_model=model,
            max_speculative_tokens=5,
            acceptance_threshold=0.5
        )
        logger.info("Enabled speculative decoding with draft model")
    
    # Set up custom learning rate scheduler
    custom_lr_scheduler = get_lr_scheduler(
        optimizer=training_engine.optimizer,
        scheduler_type="cosine_with_warmup",
        num_warmup_steps=warmup_steps,
        num_training_steps=len(train_dataloader) * num_train_epochs,
        lr=training_config.learning_rate
    )
    
    training_engine.setup_lr_scheduler(
        num_epochs=num_train_epochs,
        steps_per_epoch=len(train_dataloader),
        warmup_steps=warmup_steps,
        custom_scheduler=custom_lr_scheduler
    )
    
    # Set up curriculum learning if enabled
    if use_curriculum:
        curriculum_scheduler = CurriculumScheduler(
            training_config=training_config,
            total_steps=len(train_dataloader) * num_train_epochs
        )
        training_engine.set_curriculum_scheduler(curriculum_scheduler)
    
    # Set up RLHF if enabled
    if training_config.use_rlhf:
        rlhf_integration = AdvancedRLHFIntegration(
            model=model,
            tokenizer=training_config.tokenizer,
            config=training_config,
            device=device
        )
        training_engine.set_rlhf_integration(rlhf_integration)
    
    # Apply Constitutional AI constraints if enabled
    if training_config.use_constitutional_ai and hasattr(model, 'constitutional_ai'):
        logger.info("Applying Constitutional AI constraints during training")
        training_engine.set_constitutional_ai(model.constitutional_ai)
    
    # Validate setup
    validation_result = validate_model_setup(training_engine)
    if not validation_result.success:
        for error in validation_result.errors:
            logger.error(f"Validation error: {error}")
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Validation warning: {warning}")
                
        if validation_result.critical_errors:
            raise TrainingError("Critical validation errors found: " + 
                              ", ".join(validation_result.critical_errors))
    else:
        logger.info("Model setup validation passed")
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Validation warning: {warning}")
    
    # Load checkpoint if resuming
    start_epoch = 0
    if resume_from_checkpoint:
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        training_engine.load_checkpoint(checkpoint)
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
    
    # Training metrics
    metrics = {
        "train_losses": [],
        "val_losses": [],
        "learning_rates": [],
        "best_val_loss": float("inf"),
        "best_epoch": -1
    }
    
    # Training loop
    for epoch in range(start_epoch, num_train_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_train_epochs}")
        
        # Initialize knowledge bank if using RAG
        if hasattr(model, 'rag') and hasattr(model, 'knowledge_bank'):
            logger.info("Initializing knowledge bank for RAG")
            # Sample from training data to build initial knowledge bank
            knowledge_entries = []
            for batch_idx, batch in enumerate(train_dataloader):
                if batch_idx >= 100:  # Sample first 100 batches
                    break
                with torch.no_grad():
                    # Get embeddings for knowledge bank
                    embeddings = model.base_model.get_input_embeddings()(batch['input_ids'])
                    knowledge_entries.extend([
                        KnowledgeEntry(
                            content=model.tokenizer.decode(ids, skip_special_tokens=True),
                            embedding=emb.mean(dim=0)  # Average token embeddings
                        )
                        for ids, emb in zip(batch['input_ids'], embeddings)
                    ])
            
            # Initialize knowledge bank
            model.knowledge_bank = KnowledgeBankManager(
                embedding_dim=model.config.rag_retriever_dim,
                max_entries=model.config.rag_max_knowledge_items * 10  # Allow room for growth
            )
            model.knowledge_bank.add_entries(knowledge_entries)
            logger.info(f"Initialized knowledge bank with {len(knowledge_entries)} entries")
        
        # Train for one epoch
        train_loss = training_engine.train_epoch(
            train_dataloader=train_dataloader,
            epoch=epoch
        )
        metrics["train_losses"].append(train_loss)
        
        # Evaluate if needed
        if (epoch + 1) % eval_steps == 0:
            val_loss, val_metrics = validate_model(
                model=model,
                val_dataloader=val_dataloader,
                device=device,
                use_mixed_precision=use_mixed_precision
            )
            metrics["val_losses"].append(val_loss)
            
            # Log metrics including causal stats if available
            causal_stats = ""
            if hasattr(model, 'causal_engine'):
                causal_loss = val_metrics.get('causal_loss', 0)
                causal_acc = val_metrics.get('causal_accuracy', 0)
                metrics["causal_losses"].append(causal_loss)
                metrics["causal_accuracy"].append(causal_acc)
                causal_stats = f", Causal Loss = {causal_loss:.4f}, Causal Acc = {causal_acc:.4f}"
            
            logger.info(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}{causal_stats}")
            
            # Save best model
                        # Save best model
            if val_loss < metrics["best_val_loss"]:
                metrics["best_val_loss"] = val_loss
                metrics["best_epoch"] = epoch + 1
                
                if not use_distributed or local_rank == 0:
                    checkpoint_path = os.path.join(output_dir, f"{experiment_name}_best.pt")
                    training_engine.save_checkpoint(checkpoint_path, epoch, val_metrics)
                    logger.info(f"Saved best model checkpoint to {checkpoint_path}")
                    
                    # Save safetensors version
                    safetensors_path = checkpoint_path.replace('.pt', '.safetensors')
                    # Note: Getting state_dict might need adjustment for DDP/FSDP wrappers (e.g., model.module.state_dict())
                    # Handle potential DDP/FSDP wrappers when getting state_dict
                    state_dict_model = model.module if hasattr(model, 'module') else model
                    state_dict = state_dict_model.state_dict()
                    save_file(state_dict, safetensors_path)
                    logger.info(f"Saved best model state dict to {safetensors_path}")
        # Log validation loss to wandb
        wandb.log({"val_loss": val_loss}, step=epoch + 1)
        # Save checkpoint if needed
        if (epoch + 1) % save_steps == 0 and (not use_distributed or local_rank == 0):
            checkpoint_path = os.path.join(output_dir, f"{experiment_name}_epoch_{epoch + 1}.pt")
            training_engine.save_checkpoint(checkpoint_path, epoch)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save safetensors version
            safetensors_path = checkpoint_path.replace('.pt', '.safetensors')
            # Handle potential DDP/FSDP wrappers when getting state_dict
            state_dict_model = model.module if hasattr(model, 'module') else model
            state_dict = state_dict_model.state_dict()
            save_file(state_dict, safetensors_path)
            logger.info(f"Saved state dict to {safetensors_path}")
    # Save final model
    if not use_distributed or local_rank == 0:
        final_path = os.path.join(output_dir, f"{experiment_name}_final.pt")
        training_engine.save_checkpoint(final_path, num_train_epochs - 1)
        logger.info(f"Saved final model to {final_path}")
    # Cleanup distributed setup if used
    if use_distributed:
        cleanup_distributed()
    # Finish wandb run
    wandb.finish()
    return metrics
def main():
    """Main training function."""
    # Parse arguments
    config = parse_args()
    
    # Set up logging
    os.makedirs(config.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.output_dir, f"{config.experiment_name}.log")),
            logging.StreamHandler()
        ]
    )
    
    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("No GPU available, using CPU")
    
    # Set up distributed training if needed
    if config.use_distributed:
        setup_distributed(config.local_rank)
        device = torch.device(f"cuda:{config.local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    logger.info("Loading datasets")
    train_dataset, val_dataset = load_dataset_from_config(config)
    
    # Create data loaders
    logger.info("Creating data loaders")
    train_sampler = None
    
    # For streaming datasets, we can't use distributed or curriculum samplers
    is_streaming = config.use_streaming or isinstance(train_dataset, IterableDataset)
    
    if config.use_distributed and not is_streaming:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    # Create curriculum sampler if enabled and not streaming
    if config.use_curriculum and not is_streaming:
        logger.info("Using curriculum learning")
        train_sampler = CurriculumSampler(
            train_dataset, 
            starting_difficulty=0.5, 
            difficulty_step=0.05, 
            distributed_sampler=train_sampler if config.use_distributed else None
        )
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None and not is_streaming,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_dataloader = create_dataloader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Set up base transformer model
    logger.info("Setting up transformer model")
    base_model = setup_transformer_model(
        model_type=config.model_type,
        config=config
    )
    
    # Set up advanced model with reasoning components
    logger.info("Setting up model with advanced reasoning capabilities")
    model = setup_advanced_model(config)
    
    # Initialize model weights properly
    logger.info("Initializing model weights")
    initialize_model(model, config)
    
    # Set up causal reasoning
    if config.use_causal_inference:
        from model.reasoning.causal_inference import CausalInferenceEngine, CausalInferenceConfig
        causal_config = CausalInferenceConfig(
            hidden_size=config.hidden_size,
            use_do_calculus=True,
            use_counterfactual_reasoning=False,
            use_structural_causal_models=True
        )
        model.causal_engine = CausalInferenceEngine(causal_config)
        logger.info("Initialized causal inference engine")

    # Set up adaptive reasoning
    if config.use_adaptive_reasoning:
        logger.info("Initializing adaptive reasoning with:")
        logger.info(f"  - MCTS: {config.use_mcts}")
        logger.info(f"  - Recursive reasoning: {config.use_recursive_reasoning}")
        logger.info(f"  - Neural symbolic reasoning: {config.use_neural_symbolic}")
        
        reasoning_manager = model.reasoning_manager
        
        # Initialize specialized reasoners if not already present
        if not hasattr(reasoning_manager, 'confidence_predictor'):
            # Add confidence predictor for adaptive reasoning
            confidence_predictor = ConfidencePredictor(
                hidden_size=config.hidden_size,
                threshold=config.strategy_selection_threshold
            )
            reasoning_manager.confidence_predictor = confidence_predictor
        
        # Add MCTS reasoner if enabled and not already present
        if config.use_mcts and not any(isinstance(r, AdaptiveMCTSReasoner) for r in getattr(reasoning_manager, 'reasoners', [])):
            adaptive_mcts = AdaptiveMCTSReasoner(
                hidden_size=config.hidden_size,
                max_simulations=config.mcts_simulations,
                use_adaptive_simulation=config.use_adaptive_simulation_count
            )
            reasoning_manager.register_mcts_reasoner(adaptive_mcts)
        
        # Add recursive reasoner if enabled and not already present
        if config.use_recursive_reasoning and not any(isinstance(r, AdaptiveRecursiveReasoner) for r in getattr(reasoning_manager, 'reasoners', [])):
            adaptive_recursive = AdaptiveRecursiveReasoner(
                hidden_size=config.hidden_size,
                max_depth=config.recursive_depth,
                early_stopping=config.early_stopping
            )
            reasoning_manager.register_recursive_reasoner(adaptive_recursive)
        
        # Add neural symbolic reasoner if enabled and not already present
        if config.use_neural_symbolic and not any(isinstance(r, NeuralSymbolicReasoner) for r in getattr(reasoning_manager, 'reasoners', [])):
            neural_symbolic = NeuralSymbolicReasoner(
                hidden_size=config.hidden_size,
                use_verification=config.use_symbolic_verification
            )
            reasoning_manager.register_symbolic_reasoner(neural_symbolic)
    
    # Apply causal reasoning optimizations if enabled
    if hasattr(model, 'causal_engine'):
        def causal_forward_hook(module, input, output):
            if hasattr(output, 'last_hidden_state'):
                batch_size, seq_len, hidden_size = output.last_hidden_state.shape
                causal_outputs = model.causal_engine(output.last_hidden_state)
                output.causal_outputs = causal_outputs
                output.causal_loss = causal_outputs.get('loss', torch.zeros(1, device=output.last_hidden_state.device))
            return output

        model.register_forward_hook(causal_forward_hook)

    # Apply computational optimizations
    model = apply_computational_optimizations(model, config)
    
    # Set up text generation (beam search)
    generator = setup_generation(model, config)
    
    # Move model to device
    model = model.to(device)
    
    # Create learning rate scheduler
    lr_scheduler = get_lr_scheduler(
        optimizer=None,  # Will be set by the training engine
        scheduler_type="cosine",
        num_warmup_steps=config.warmup_steps,
        num_training_steps=len(train_dataloader) * config.num_train_epochs,
        lr=config.learning_rate
    )
    
    # Train the model
    logger.info("Starting training")
    train_stats = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        training_config=config,
        device=device,
        output_dir=config.output_dir,
        experiment_name=config.experiment_name,
        resume_from_checkpoint=config.resume_from_checkpoint,
        use_mixed_precision=config.use_mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=config.warmup_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        use_curriculum=config.use_curriculum,
        use_distributed=config.use_distributed,
        local_rank=config.local_rank
    )
    
    # Clean up distributed training
    if config.use_distributed:
        cleanup_distributed()
    
    logger.info("Training completed")
    logger.info(f"Final training stats: {train_stats}")
    
    # Final validation
    logger.info("Running final validation")
    eval_metrics = validate_model(model, val_dataloader, device, compute_perplexity=True, compute_metrics=True)
    logger.info(f"Final evaluation metrics: {eval_metrics}")
    
    # Save final model
    if not config.use_distributed or config.local_rank == 0:
        final_model_path = os.path.join(config.output_dir, f"{config.experiment_name}_final")
        logger.info(f"Saving final model to {final_model_path}")
        model.save_pretrained(final_model_path)
        
        # Save safetensors version
        safetensors_path = os.path.join(final_model_path, "model.safetensors")
        # Handle potential DDP/FSDP wrappers when getting state_dict
        state_dict_model = model.module if hasattr(model, 'module') else model
        state_dict = state_dict_model.state_dict()
        save_file(state_dict, safetensors_path)
        logger.info(f"Saved final safetensors model to {safetensors_path}")
if __name__ == "__main__":
    main()