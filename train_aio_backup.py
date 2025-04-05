import os
import sys
import torch
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import asdict, dataclass, field
import time
import re
from contextlib import nullcontext
import datetime
import random
import numpy as np
import torch.nn as nn
import math
import glob
from torch.utils.data import Dataset, ConcatDataset
from types import SimpleNamespace
import argparse
import traceback
import copy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import core model components
from model.transformer import EfficientTransformer, TransformerModel
# Create a mock BaseModel since it doesn't exist in model.core_model
class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = kwargs.get('config', None)
        logger.warning("Using mock BaseModel class as it could not be imported")
        
    def forward(self, *args, **kwargs):
        return None

# Create a mock NanoGPT since it doesn't exist in model.gpt
class NanoGPT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = kwargs.get('config', None)
        logger.warning("Using mock NanoGPT class as it could not be imported")
        
    def forward(self, *args, **kwargs):
        return None

try:
    from model.gpt import NanoGPT
except ImportError:
    # We already defined a mock NanoGPT class above
    pass
from model.logical_nanogpt import LogicalGPT, TreeLSTM, MemoryAugmentedNetwork

# Create a mock NanoGPTBase since it doesn't exist in model.nanogpt
class NanoGPTBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = kwargs.get('config', None)
        logger.warning("Using mock NanoGPTBase class as it could not be imported")
        
    def forward(self, *args, **kwargs):
        return None

# Import already handled above
    pass
# Import already handled above

# Import advanced reasoning modules
from model.reasoning import ChainOfThoughtReasoner
from model.recursive_reasoning import RecursiveReasoningModule, RecursiveReasoningConfig
from model.tree_reasoning import TreeReasoningModule, TreeOfThoughtConfig as TreeReasoningConfig
from model.tree_reasoning_mcts import MCTSEnhancedTreeReasoningModule, MCTSConfig
try:
    from model.advanced_reasoning import AdvancedReasoningModule, MultiStepReasoner
except ImportError:
    # Create mock classes
    class AdvancedReasoningModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            logger.warning("Using mock AdvancedReasoningModule")
        def forward(self, *args, **kwargs):
            return None
    class MultiStepReasoner(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            logger.warning("Using mock MultiStepReasoner")
        def forward(self, *args, **kwargs):
            return None

from model.neural_symbolic import NeuralSymbolicIntegration as NeuralSymbolicProcessor, NeuralSymbolicConfig as SymbolicReasoningLayer
from model.knowledge_reasoning import KnowledgeReasoningModule as KnowledgeEnhancedReasoner, KnowledgeReasoningConfig as KnowledgeGraph
try:
    from model.adaptive_reasoning import AdaptiveReasoningController as AdaptiveReasoner, AdaptiveReasoningConfig as ReasoningStrategySelector
except ImportError:
    # Create mock classes
    class AdaptiveReasoner(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            logger.warning("Using mock AdaptiveReasoner")
        def forward(self, *args, **kwargs):
            return None
    class ReasoningStrategySelector(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            logger.warning("Using mock ReasoningStrategySelector")
        def forward(self, *args, **kwargs):
            return None
from model.math_reasoning import (
    SymbolicMathTransformer, 
    MathReasoningConfig, 
    FormalVerificationModule,
    TheoremProver
)
from model.verifiable_computation import VerifiableComputationModule, ProofGenerator
from model.logic import LogicModule, FOLProcessor
from model.sat import SATSolver

# Import memory components
from model.memory import (
    EnhancedMemory, 
    EpisodicMemory, 
    WorkingMemory,
    LongTermMemory,
    MemoryRouter
)
from model.memory_bank import MemoryBank
from model.cache import CacheManager, KVCache

# Import attention mechanisms
from model.attention import (
    AttentionLayer,
    MultiHeadAttention,
    EfficientAttention, 
    LinearAttention,
    FlashAttention,
    RotaryAttention,
    LocalAttention
)
from model.attention_mechanisms import (
    AttentionRouter,
    SparseAttention, 
    LongRangeAttention,
    HierarchicalAttention
)

# Import transformer and efficiency components
from model.computational_efficiency import (
    ComputationalOptimizer,
    ActivationCheckpointer,
    EfficientForwardModule,
    MixedPrecisionManager
)
from model.efficient_transformer import EfficientTransformerEnhanced
from model.efficient_layers import EfficientLayerNorm, EfficientMLP
from model.feed_forward import EnhancedFeedForward
from model.layers import EnhancedTransformerLayer, AdaptiveLayer
from model.lora import LoRALayer
from model.adapters import AdapterModule
from model.embedding import EnhancedEmbedding
from model.positional import (
    PositionalEncoding,
    RelativePositionalEncoding,
    RotaryPositionalEncoding
)
from model.rotary_embeddings import RotaryEmbedding

# Import numerical precision components
from model.numerical_precision import (
    NumericalPrecisionConfig,
    NumericalPrecisionModule,
    NumericallyStableOperations
)
from model.math_precision_integration import MathPrecisionManager

# Import distillation and training components
from model.knowledge_distillation import KnowledgeDistillationManager
from model.api_distillation import APIBasedDistillation
from model.distillation import DistillationHelper
from model.optimization import Optimizer
from model.meta_learning import MetaLearner

# Import reinforcement learning components
from model.reinforcement.ppo import PPOTrainer
from model.reinforcement.dpo import DPOTrainer
try:
    from model.reinforcement.actor_critic import ActorCritic
except ImportError:
    # Create mock ActorCritic class
    logger.warning("Could not import ActorCritic, using mock class")
    class ActorCritic:
        def __init__(self, *args, **kwargs):
            pass
        def forward(self, *args, **kwargs):
            return None
from model.constitutional_ai import ConstitutionalAITrainer

# Import MoE components
from model.moe import (
    MixtureOfExperts,
    ExpertGating,
    HierarchicalMoE
)
from model.expert_layer import ExpertLayer

# Import generation and uncertainty components
try:
    from model.generation.beam_search import BeamSearchGenerator
except ImportError:
    # Create mock BeamSearchGenerator class
    logger.warning("Could not import BeamSearchGenerator, using mock class")
    class BeamSearchGenerator:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, *args, **kwargs):
            return None

try:
    from model.generation.sampling import SamplingStrategies
except ImportError:
    # Create mock SamplingStrategies class
    logger.warning("Could not import SamplingStrategies, using mock class")
    class SamplingStrategies:
        def __init__(self, *args, **kwargs):
            pass
        def sample(self, *args, **kwargs):
            return None

from model.uncertainty.calibration import UncertaintyCalibration
try:
    from model.nlp.classification import TextClassifier
except ImportError:
    # Create mock TextClassifier class
    logger.warning("Could not import TextClassifier, using mock class")
    class TextClassifier:
        def __init__(self, *args, **kwargs):
            pass
        def classify(self, *args, **kwargs):
            return None

try:
    from model.nlp.tokenization import EnhancedTokenizer
except ImportError:
    # Create mock EnhancedTokenizer class
    logger.warning("Could not import EnhancedTokenizer, using mock class")
    class EnhancedTokenizer:
        def __init__(self, *args, **kwargs):
            pass
        def tokenize(self, *args, **kwargs):
            return None
        def decode(self, *args, **kwargs):
            return None

# Import utility components from model
from model.utils import ModelUtils
from model.architecture_components import ModelRegistry, ModelFactory
from model.integration import ModelIntegrator
from model.inference import InferenceOptimizer

# Import data components
from data.dataloader import DataLoaderFactory, EfficientDataLoader, collate_fn
from data.dataset import OptimizedDataset, DataProcessor, DomainSpecificDataManager
from data.run_inference import ReasoningEvaluator

# Import optimization utilities
from utils.optimization import (
    OptimizationConfig,
    ModelOptimizer,
    MemoryOptimizer,
    GradientOptimizer,
    find_optimal_checkpoint_config
)
from utils.io_utils import save_json, load_json

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Optional imports with fallbacks for training modules
try:
    import wandb
except ImportError:
    logger.warning("wandb not found. Wandb logging will be disabled.")
    # Create a mock wandb module for compatibility
    class MockWandb:
        def init(self, **kwargs): 
            return None
        def log(self, metrics): 
            return None
        def finish(self): 
            return None
    wandb = MockWandb()

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
except ImportError:
    logger.warning("torch.distributed not available. Distributed training will be disabled.")
    # Create mock distributed modules
    class MockDist:
        def init_process_group(self, *args, **kwargs): pass
        def get_rank(self): return 0
        def get_world_size(self): return 1
    dist = MockDist()
    
    class MockDDP:
        def __init__(self, model, **kwargs): 
            self.module = model
            
    DDP = MockDDP

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    logger.warning("tensorboard not found. Tensorboard logging will be disabled.")
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

try:
    from safetensors.torch import save_file as save_safetensors
    def save_model(model, path):
        state_dict = model.state_dict()
        save_safetensors(state_dict, path)
except ImportError:
    logger.warning("safetensors not found. Model saving will use PyTorch format only.")
    def save_model(model, path): 
        torch.save(model.state_dict(), path)

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file if present
except ImportError:
    logger.warning("dotenv not installed - environment variables will not be loaded from .env")

# Fallback for tqdm if not available
try:
    from tqdm import tqdm
except ImportError:
    logger.warning("tqdm not found. Using simple progress tracking.")
    def tqdm(iterable, **kwargs):
        total = len(iterable) if hasattr(iterable, "__len__") else None
        prefix = kwargs.get('desc', '')
        
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{prefix} {i+1}/{total} ({(i+1)/total*100:.1f}%)", end="")
            else:
                print(f"\r{prefix} {i+1}", end="")
            yield item
        print()

# Configuration classes
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

# Define the main model class, utilizing components from the codebase
class ValkyrieLLM(nn.Module):
    """
    Valkyrie Language Model with advanced reasoning capabilities.
    This model integrates multiple reasoning approaches, memory systems,
    and efficiency optimizations.
    """
    
    def __init__(
        self, 
        vocab_size=30000, 
        hidden_size=768, 
        num_layers=12, 
        num_heads=12, 
        max_seq_length=1024, 
        dropout=0.1, 
        use_position_embeddings=False,
        config=None
    ):
        super().__init__()
        
        # Use provided config or create new one
        if config is None:
            self.config = AdvancedModelConfig(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_heads,
                vocab_size=vocab_size,
                max_seq_len=max_seq_length,
                dropout=dropout,
                use_rotary_embeddings=not use_position_embeddings
            )
        else:
            self.config = config
        
        # Set up efficient transformer from the codebase
        if getattr(self.config, 'attention_mechanism', 'efficient') == 'efficient':
            self.transformer = EfficientTransformerEnhanced(
                config=self.config, 
                vocab_size=vocab_size,
                use_cache=getattr(self.config, 'use_cache', True)
            )
        else:
            self.transformer = TransformerModel(
                config=self.config,
                vocab_size=vocab_size
            )
        
        # Set up math reasoning module
        if getattr(self.config, 'use_neural_symbolic', True):
            math_config = MathReasoningConfig(
                hidden_size=hidden_size,
                intermediate_size=hidden_size * 4,
                num_attention_heads=num_heads,
                numerical_precision=NumericalPrecisionConfig(
                    precision_mode=getattr(self.config, 'numerical_precision_mode', 'auto'),
                    use_numerical_precision=getattr(self.config, 'use_numerical_precision', True)
                )
            )
            self.math_module = SymbolicMathTransformer(config=math_config)
            
            # Set up formal verification if enabled
            if getattr(self.config, 'use_formal_verification', True):
                self.formal_verification = FormalVerificationModule(
                    hidden_size=hidden_size
                )
                
            # Set up theorem prover if enabled
            if getattr(self.config, 'use_theorem_proving', True):
                self.theorem_prover = TheoremProver(
                    hidden_size=hidden_size
                )
        
        # Set up tree reasoning if enabled
        if getattr(self.config, 'use_tree_reasoning', True):
            tree_config = TreeReasoningConfig(
                hidden_size=hidden_size,
                max_depth=getattr(self.config, 'reasoning_depth', 4),
                num_samples=10
            )
            self.tree_reasoning = TreeReasoningModule(
                config=tree_config,
                hidden_size=hidden_size
            )
            
            # Set up MCTS enhanced tree reasoning if enabled
            if getattr(self.config, 'use_mcts', True):
                mcts_config = MCTSConfig(
                    num_simulations=getattr(self.config, 'mcts_simulations', 100),
                    c_puct=1.0
                )
                self.mcts_reasoning = MCTSEnhancedTreeReasoningModule(
                    config=mcts_config,
                    hidden_size=hidden_size
                )
        
        # Set up recursive reasoning if enabled
        if getattr(self.config, 'use_recursive_reasoning', True):
            recursive_config = RecursiveReasoningConfig(
                max_depth=getattr(self.config, 'recursive_depth', 3),
                hidden_size=hidden_size
            )
            self.recursive_reasoner = RecursiveReasoningModule(
                config=recursive_config,
                hidden_size=hidden_size
            )
        
        # Set up knowledge reasoning if enabled
        if getattr(self.config, 'use_knowledge_reasoning', True):
            knowledge_config = KnowledgeGraph(
                hidden_size=hidden_size,
                graph_size=getattr(self.config, 'knowledge_graph_size', 1000)
            )
            self.knowledge_reasoner = KnowledgeEnhancedReasoner(
                config=knowledge_config,
                hidden_size=hidden_size
            )
        
        # Set up neural symbolic integration if enabled
        if getattr(self.config, 'use_neural_symbolic', True):
            symbolic_config = SymbolicReasoningLayer(
                hidden_size=hidden_size
            )
            self.neural_symbolic = NeuralSymbolicProcessor(
                config=symbolic_config,
                hidden_size=hidden_size
            )
        
        # Set up chain of thought reasoning
        self.chain_of_thought = ChainOfThoughtReasoner()
        
        # Set up adaptive reasoning if enabled
        if getattr(self.config, 'use_adaptive_reasoning', True):
            self.adaptive_reasoner = AdaptiveReasoner(
                hidden_size=hidden_size,
                reasoners={
                    'tree': getattr(self, 'tree_reasoning', None),
                    'recursive': getattr(self, 'recursive_reasoner', None),
                    'knowledge': getattr(self, 'knowledge_reasoner', None),
                    'symbolic': getattr(self, 'neural_symbolic', None),
                    'chain_of_thought': self.chain_of_thought
                }
            )
        
        # Set up memory systems if enabled
        if getattr(self.config, 'use_memory_augmentation', True):
            self.memory = EnhancedMemory(
                hidden_size=hidden_size,
                memory_size=getattr(self.config, 'memory_size', 1024)
            )
            
            # Set up episodic memory if enabled
            if getattr(self.config, 'use_episodic_memory', True):
                self.episodic_memory = EpisodicMemory(
                    hidden_size=hidden_size,
                    memory_size=getattr(self.config, 'episodic_memory_size', 1024)
                )
                
            # Set up working memory if enabled
            if getattr(self.config, 'use_working_memory', True):
                self.working_memory = WorkingMemory(
                    hidden_size=hidden_size,
                    memory_size=getattr(self.config, 'working_memory_size', 512)
                )
                
            # Set up long-term memory if enabled
            if getattr(self.config, 'use_long_term_memory', True):
                self.long_term_memory = LongTermMemory(
                    hidden_size=hidden_size,
                    memory_size=getattr(self.config, 'long_term_memory_size', 4096)
                )
                
            # Set up memory router if enabled
            if getattr(self.config, 'use_memory_router', True):
                self.memory_router = MemoryRouter(
                    hidden_size=hidden_size,
                    memories={
                        'episodic': getattr(self, 'episodic_memory', None),
                        'working': getattr(self, 'working_memory', None),
                        'long_term': getattr(self, 'long_term_memory', None)
                    }
                )
                
            # Set up memory bank for additional storage
            self.memory_bank = MemoryBank(
                hidden_size=hidden_size,
                capacity=getattr(self.config, 'memory_bank_size', 2048)
            )
        
        # Set up numerical precision module if enabled
        if getattr(self.config, 'use_numerical_precision', True):
            numerical_config = NumericalPrecisionConfig(
                precision_mode=getattr(self.config, 'numerical_precision_mode', 'auto'),
                use_fp8_matmul=getattr(self.config, 'use_fp8_matmul', False)
            )
            self.numerical_module = NumericalPrecisionModule(
                config=numerical_config,
                hidden_size=hidden_size
            )
            
            # Set up math precision manager
            self.math_precision = MathPrecisionManager(
                hidden_size=hidden_size,
                numerical_config=numerical_config
            )
            
            # Set up numerically stable operations
            self.numerical_ops = NumericallyStableOperations(
                config=numerical_config
            )
        
        # Set up MoE if enabled
        if getattr(self.config, 'use_moe', False):
            self.moe = MixtureOfExperts(
                hidden_size=hidden_size,
                num_experts=getattr(self.config, 'num_experts', 8),
                capacity_factor=getattr(self.config, 'moe_capacity_factor', 1.25),
                dropout=getattr(self.config, 'expert_dropout', 0.1),
                expert_module=ExpertLayer(
                    hidden_size=hidden_size,
                    intermediate_size=hidden_size * 4
                )
            )
            
            # Set up expert gating mechanism
            self.expert_gating = ExpertGating(
                hidden_size=hidden_size,
                num_experts=getattr(self.config, 'num_experts', 8)
            )
            
            # Set up hierarchical MoE if enabled
            if getattr(self.config, 'use_hierarchical_moe', False):
                self.hierarchical_moe = HierarchicalMoE(
                    hidden_size=hidden_size,
                    num_experts=getattr(self.config, 'num_experts', 8),
                    expert_layer=self.moe
                )
        
        # Set up LoRA if enabled
        if getattr(self.config, 'use_lora', False):
            self.lora = LoRALayer(
                hidden_size=hidden_size,
                rank=getattr(self.config, 'lora_rank', 8)
            )
            
        # Set up adapters if enabled
        if getattr(self.config, 'use_adapters', False):
            self.adapters = AdapterModule(
                hidden_size=hidden_size,
                adapter_size=getattr(self.config, 'adapter_size', 64)
            )
            
        # Set up computational efficiency
        self.computational_optimizer = ComputationalOptimizer(
            activation_checkpointing=getattr(self.config, 'use_gradient_checkpointing', True),
            mixed_precision=getattr(self.config, 'use_mixed_precision', True)
        )
        
        # Set up cache manager
        self.cache_manager = CacheManager(
            max_length=max_seq_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # Set up verifiable computation
        self.verifiable_computation = VerifiableComputationModule(
            hidden_size=hidden_size
        )
        
        # Set up proof generator
        self.proof_generator = ProofGenerator(
            hidden_size=hidden_size
        )
        
        # Set up logic module
        self.logic_module = LogicModule(
            hidden_size=hidden_size
        )
        
        # Set up FOL processor
        self.fol_processor = FOLProcessor(
            hidden_size=hidden_size
        )
        
        # Set up SAT solver
        self.sat_solver = SATSolver(
            hidden_size=hidden_size
        )
        
        # Set up inference optimizer
        self.inference_optimizer = InferenceOptimizer(
            kv_cache=self.cache_manager,
            use_flash_attention=getattr(self.config, 'use_flash_attention', True)
        )
        
        # Set up generation components
        self.beam_search = BeamSearchGenerator(
            model=self,
            max_length=max_seq_length
        )
        
        self.sampling_strategies = SamplingStrategies()
        
        # Set up training components
        self.knowledge_distillation = KnowledgeDistillationManager(
            hidden_size=hidden_size
        )
        
        # Set up API-based distillation
        self.api_distillation = APIBasedDistillation(
            hidden_size=hidden_size
        )
        
        # Set up distillation helper
        self.distillation_helper = DistillationHelper(
            hidden_size=hidden_size
        )
        
        # Set up meta-learner
        self.meta_learner = MetaLearner(
            hidden_size=hidden_size
        )
        
        # Set up uncertainty calibration
        self.uncertainty_calibration = UncertaintyCalibration(
            hidden_size=hidden_size
        )
        
        # Set up text classifier
        self.text_classifier = TextClassifier(
            hidden_size=hidden_size,
            num_classes=getattr(self.config, 'num_classes', 2)
        )
        
        # Set up enhanced tokenizer
        self.enhanced_tokenizer = EnhancedTokenizer(
            vocab_size=vocab_size
        )
        
        # Integration module to tie everything together
        self.integrator = ModelIntegrator(
            modules={
                'transformer': self.transformer,
                'tree_reasoning': getattr(self, 'tree_reasoning', None),
                'recursive_reasoning': getattr(self, 'recursive_reasoner', None),
                'knowledge_reasoning': getattr(self, 'knowledge_reasoner', None),
                'memory': getattr(self, 'memory', None),
                'numerical': getattr(self, 'numerical_module', None),
                'moe': getattr(self, 'moe', None),
                'uncertainty': getattr(self, 'uncertainty_calibration', None),
                'classifier': getattr(self, 'text_classifier', None),
                'tokenizer': getattr(self, 'enhanced_tokenizer', None),
                'logic': getattr(self, 'logic_module', None),
                'sat': getattr(self, 'sat_solver', None),
                'fol': getattr(self, 'fol_processor', None)
            }
        )
        
        # Register with model registry
        ModelRegistry.register_model(self, model_type='valkyrie')
        
        logger.info(f"ValkyrieLLM initialized with {sum(p.numel() for p in self.parameters())} parameters")
        
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """Forward pass integrating all advanced components"""
        
        # Extract helpful kwargs
        use_reasoning = kwargs.pop('use_reasoning', False)
        reasoning_type = kwargs.pop('reasoning_type', 'auto')
        use_memory = kwargs.pop('use_memory', True)
        use_moe = kwargs.pop('use_moe', getattr(self.config, 'use_moe', False))
        use_uncertainty = kwargs.pop('use_uncertainty', False)
        use_classification = kwargs.pop('use_classification', False)
        use_logic = kwargs.pop('use_logic', False)
        use_enhanced_tokenizer = kwargs.pop('use_enhanced_tokenizer', False)
        use_meta_learning = kwargs.pop('use_meta_learning', False)
        
        # Use enhanced tokenizer if requested
        if use_enhanced_tokenizer and hasattr(self, 'enhanced_tokenizer') and input_ids is not None:
            # Store original input_ids for reference
            original_input_ids = input_ids
            
            # Decode and re-encode with enhanced tokenizer
            if hasattr(self.enhanced_tokenizer, 'decode') and hasattr(self.enhanced_tokenizer, 'tokenize'):
                batch_texts = []
                for ids in input_ids:
                    text = self.enhanced_tokenizer.decode(ids.tolist())
                    batch_texts.append(text)
                
                # Re-tokenize with enhanced features
                enhanced_encoding = self.enhanced_tokenizer.tokenize(batch_texts)
                input_ids = enhanced_encoding['input_ids']
                
                # Update attention mask if provided in the enhanced encoding
                if 'attention_mask' in enhanced_encoding and attention_mask is None:
                    attention_mask = enhanced_encoding['attention_mask']
        
        # Use cache manager if available
        if hasattr(self, 'cache_manager') and getattr(self.config, 'use_cache', True):
            kwargs['past_key_values'] = self.cache_manager.get_cache(
                batch_size=input_ids.shape[0] if input_ids is not None else 1
            )
        
        # Apply numerical precision if enabled
        if hasattr(self, 'numerical_module') and getattr(self.config, 'use_numerical_precision', True):
            with self.numerical_module.precision_context():
                # Main transformer pass
                outputs = self.transformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
        else:
            # Main transformer pass without numerical precision
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # Extract hidden states for additional processing
        if isinstance(outputs, dict):
            hidden_states = outputs.get('hidden_states', None)
            logits = outputs.get('logits', None)
        else:
            # Handle tuple returns
            logits = outputs[0]
            hidden_states = outputs[1] if len(outputs) > 1 else None
        
        # Apply MoE if enabled and requested
        if use_moe and hasattr(self, 'moe'):
            # Apply expert gating if available
            if hasattr(self, 'expert_gating'):
                # Get expert weights
                expert_weights = self.expert_gating(hidden_states)
                # Apply MoE with expert weights
                enhanced_hidden_states = self.moe(hidden_states, expert_weights=expert_weights)
            else:
                enhanced_hidden_states = self.moe(hidden_states)
                
            # Update logits with MoE output
            logits = self.transformer.output_projection(enhanced_hidden_states)
            if isinstance(outputs, dict):
                outputs['hidden_states'] = enhanced_hidden_states
                outputs['logits'] = logits
        
        # Apply memory augmentation if enabled and requested
        if use_memory and hasattr(self, 'memory'):
            # Update memory with current hidden states
            self.memory.update(hidden_states, attention_mask)
            
            # Augment hidden states with memory
            memory_enhanced = self.memory.augment(hidden_states)
            
            # Use memory bank if available
            if hasattr(self, 'memory_bank'):
                memory_bank_enhanced = self.memory_bank.retrieve(hidden_states)
                # Combine with memory enhanced states
                memory_enhanced = memory_enhanced + 0.5 * memory_bank_enhanced
                # Update memory bank
                self.memory_bank.store(hidden_states)
            
            # Use memory router if available
            if hasattr(self, 'memory_router'):
                routed_memory = self.memory_router.route(hidden_states, attention_mask)
                # Combine with memory enhanced states
                memory_enhanced = 0.7 * memory_enhanced + 0.3 * routed_memory
            
            # Update outputs
            if isinstance(outputs, dict):
                outputs['memory_enhanced_states'] = memory_enhanced
                
            # Update hidden states for downstream components
            hidden_states = memory_enhanced
        
        # Apply reasoning if requested
        if use_reasoning:
            if reasoning_type == 'auto' and hasattr(self, 'adaptive_reasoner'):
                # Use adaptive reasoning to select the best reasoning type
                reasoning_outputs = self.adaptive_reasoner.reason(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            elif reasoning_type == 'tree' and hasattr(self, 'tree_reasoning'):
                reasoning_outputs = self.tree_reasoning(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            elif reasoning_type == 'mcts' and hasattr(self, 'mcts_reasoning'):
                reasoning_outputs = self.mcts_reasoning(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            elif reasoning_type == 'recursive' and hasattr(self, 'recursive_reasoner'):
                reasoning_outputs = self.recursive_reasoner.reason(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            elif reasoning_type == 'mathematical' and hasattr(self, 'math_module'):
                reasoning_outputs = self.math_module(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            elif reasoning_type == 'symbolic' and hasattr(self, 'neural_symbolic'):
                reasoning_outputs = self.neural_symbolic(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            elif reasoning_type == 'knowledge' and hasattr(self, 'knowledge_reasoner'):
                reasoning_outputs = self.knowledge_reasoner(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            elif reasoning_type == 'chain_of_thought':
                reasoning_outputs = self.chain_of_thought.reason(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            else:
                # Fallback to chain of thought
                reasoning_outputs = self.chain_of_thought.reason(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
            # Add reasoning outputs to the results
            if isinstance(outputs, dict):
                outputs['reasoning_outputs'] = reasoning_outputs
                
            # Update hidden states for downstream components if reasoning produced new states
            if hasattr(reasoning_outputs, 'hidden_states'):
                hidden_states = reasoning_outputs.hidden_states
        
        # Apply logic processing if requested
        if use_logic and hasattr(self, 'logic_module'):
            logic_outputs = self.logic_module(hidden_states)
            
            # Apply FOL processing if available
            if hasattr(self, 'fol_processor'):
                fol_outputs = self.fol_processor(logic_outputs)
                
                # Apply SAT solver if available
                if hasattr(self, 'sat_solver'):
                    sat_outputs = self.sat_solver(fol_outputs)
                    
                    # Add to outputs
                    if isinstance(outputs, dict):
                        outputs['logic_outputs'] = {
                            'logic': logic_outputs,
                            'fol': fol_outputs,
                            'sat': sat_outputs
                        }
                else:
                    # Add to outputs without SAT
                    if isinstance(outputs, dict):
                        outputs['logic_outputs'] = {
                            'logic': logic_outputs,
                            'fol': fol_outputs
                        }
            else:
                # Add to outputs with just logic
                if isinstance(outputs, dict):
                    outputs['logic_outputs'] = {
                        'logic': logic_outputs
                    }
        
        # Apply uncertainty calibration if requested
        if use_uncertainty and hasattr(self, 'uncertainty_calibration'):
            uncertainty_outputs = self.uncertainty_calibration(logits, hidden_states)
            
            # Add to outputs
            if isinstance(outputs, dict):
                outputs['uncertainty'] = uncertainty_outputs
                
            # Apply calibrated logits
            if hasattr(uncertainty_outputs, 'calibrated_logits'):
                logits = uncertainty_outputs.calibrated_logits
                if isinstance(outputs, dict):
                    outputs['logits'] = logits
        
        # Apply text classification if requested
        if use_classification and hasattr(self, 'text_classifier'):
            classification_outputs = self.text_classifier.classify(hidden_states)
            
            # Add to outputs
            if isinstance(outputs, dict):
                outputs['classification'] = classification_outputs
        
        # Apply meta-learning if requested
        if use_meta_learning and hasattr(self, 'meta_learner'):
            meta_outputs = self.meta_learner(hidden_states, logits)
            
            # Add to outputs
            if isinstance(outputs, dict):
                outputs['meta_learning'] = meta_outputs
                
            # Update logits if meta-learning produced new logits
            if hasattr(meta_outputs, 'logits'):
                logits = meta_outputs.logits
                if isinstance(outputs, dict):
                    outputs['logits'] = logits
        
        # Apply LoRA if enabled
        if hasattr(self, 'lora') and getattr(self.config, 'use_lora', False):
            if isinstance(outputs, dict) and 'hidden_states' in outputs:
                outputs['hidden_states'] = self.lora(outputs['hidden_states'])
                # Update logits
                outputs['logits'] = self.transformer.output_projection(outputs['hidden_states'])
        
        # Apply adapters if enabled
        if hasattr(self, 'adapters') and getattr(self.config, 'use_adapters', False):
            if isinstance(outputs, dict) and 'hidden_states' in outputs:
                outputs['hidden_states'] = self.adapters(outputs['hidden_states'])
                # Update logits
                outputs['logits'] = self.transformer.output_projection(outputs['hidden_states'])
        
        return outputs
    
    def generate(self, input_ids, attention_mask=None, max_length=100, **kwargs):
        """
        Generate text using the model with various advanced generation strategies.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum length to generate
            **kwargs: Additional arguments
                - generation_method: Method to use for generation ('greedy', 'beam', 'sampling', 'nucleus', 'top_k', 'top_p')
                - use_reasoning: Whether to use reasoning during generation
                - reasoning_type: Type of reasoning to use ('auto', 'tree', 'recursive', 'mathematical', 'chain_of_thought', etc.)
                - use_memory: Whether to use memory during generation
                - use_uncertainty: Whether to use uncertainty calibration
                - use_logic: Whether to use logic processing
                - num_beams: Number of beams for beam search
                - temperature: Temperature for sampling
                - top_k: Top-k value for sampling
                - top_p: Top-p value for nucleus sampling
                - repetition_penalty: Penalty for repetition
                - no_repeat_ngram_size: Size of n-grams to avoid repeating
                
        Returns:
            torch.Tensor: Generated token IDs
        """
        # Extract generation method
        generation_method = kwargs.pop('generation_method', 'greedy')
        
        # Extract reasoning settings
        use_reasoning = kwargs.pop('use_reasoning', False)
        reasoning_type = kwargs.pop('reasoning_type', 'auto')
        
        # Extract other advanced features
        use_memory = kwargs.pop('use_memory', True)
        use_uncertainty = kwargs.pop('use_uncertainty', False)
        use_logic = kwargs.pop('use_logic', False)
        use_enhanced_tokenizer = kwargs.pop('use_enhanced_tokenizer', False)
        use_meta_learning = kwargs.pop('use_meta_learning', False)
        
        # Set up generation parameters
        kwargs['use_reasoning'] = use_reasoning
        kwargs['reasoning_type'] = reasoning_type
        kwargs['use_memory'] = use_memory
        kwargs['use_uncertainty'] = use_uncertainty
        kwargs['use_logic'] = use_logic
        kwargs['use_enhanced_tokenizer'] = use_enhanced_tokenizer
        kwargs['use_meta_learning'] = use_meta_learning
        
        # Choose generation method
        if generation_method == 'beam' and hasattr(self, 'beam_search'):
            # Extract beam search specific parameters
            num_beams = kwargs.pop('num_beams', 5)
            length_penalty = kwargs.pop('length_penalty', 1.0)
            early_stopping = kwargs.pop('early_stopping', False)
            
            return self.beam_search.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                **kwargs
            )
        elif generation_method in ['sampling', 'nucleus', 'top_k', 'top_p'] and hasattr(self, 'sampling_strategies'):
            # Extract sampling specific parameters
            temperature = kwargs.pop('temperature', 1.0)
            top_k = kwargs.pop('top_k', 50)
            top_p = kwargs.pop('top_p', 0.9)
            repetition_penalty = kwargs.pop('repetition_penalty', 1.0)
            no_repeat_ngram_size = kwargs.pop('no_repeat_ngram_size', 0)
            
            return self.sampling_strategies.generate(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                method=generation_method,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                **kwargs
            )
        elif generation_method == 'tree_search' and hasattr(self, 'tree_reasoning'):
            # Use tree reasoning for generation
            tree_config = TreeReasoningConfig(
                hidden_size=self.config.hidden_size,
                max_depth=kwargs.pop('tree_depth', 5),
                num_samples=kwargs.pop('num_samples', 5)
            )
            
            # Initialize with input
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_reasoning=True,
                    reasoning_type='tree',
                    **kwargs
                )
                
                # Extract reasoning outputs
                if isinstance(outputs, dict) and 'reasoning_outputs' in outputs:
                    reasoning_outputs = outputs['reasoning_outputs']
                    # Extract generated sequence if available
                    if hasattr(reasoning_outputs, 'generated_ids'):
                        return reasoning_outputs.generated_ids
            
            # Fallback to greedy if tree reasoning didn't generate
            return self._greedy_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                **kwargs
            )
        elif generation_method == 'mcts' and hasattr(self, 'mcts_reasoning'):
            # Use MCTS for generation
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_reasoning=True,
                    reasoning_type='mcts',
                    **kwargs
                )
                
                # Extract reasoning outputs
                if isinstance(outputs, dict) and 'reasoning_outputs' in outputs:
                    reasoning_outputs = outputs['reasoning_outputs']
                    # Extract generated sequence if available
                    if hasattr(reasoning_outputs, 'generated_ids'):
                        return reasoning_outputs.generated_ids
            
            # Fallback to greedy if MCTS didn't generate
            return self._greedy_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                **kwargs
            )
        elif generation_method == 'adaptive' and hasattr(self, 'adaptive_reasoner'):
            # Use adaptive reasoning for generation
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_reasoning=True,
                    reasoning_type='auto',
                    **kwargs
                )
                
                # Extract reasoning outputs
                if isinstance(outputs, dict) and 'reasoning_outputs' in outputs:
                    reasoning_outputs = outputs['reasoning_outputs']
                    # Extract generated sequence if available
                    if hasattr(reasoning_outputs, 'generated_ids'):
                        return reasoning_outputs.generated_ids
            
            # Fallback to greedy if adaptive reasoning didn't generate
            return self._greedy_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                **kwargs
            )
        else:
            # Default greedy generation
            return self._greedy_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                **kwargs
            )
    
    def _greedy_generate(self, input_ids, attention_mask=None, max_length=100, **kwargs):
        """
        Simple greedy generation with advanced features.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum length to generate
            **kwargs: Additional arguments from generate method
            
        Returns:
            torch.Tensor: Generated token IDs
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize cache manager
        if hasattr(self, 'cache_manager'):
            self.cache_manager.reset(batch_size)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Initialize sequence
        current_ids = input_ids.clone()
        current_mask = attention_mask.clone()
        
        # Extract advanced features
        use_reasoning = kwargs.get('use_reasoning', False)
        reasoning_type = kwargs.get('reasoning_type', 'auto')
        use_memory = kwargs.get('use_memory', True)
        use_uncertainty = kwargs.get('use_uncertainty', False)
        use_logic = kwargs.get('use_logic', False)
        
        # Set up generation parameters
        repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        no_repeat_ngram_size = kwargs.get('no_repeat_ngram_size', 0)
        bad_words_ids = kwargs.get('bad_words_ids', None)
        min_length = kwargs.get('min_length', 0)
        
        # Set up memory tracking if using memory
        if use_memory and hasattr(self, 'memory'):
            # Initialize memory with input
            with torch.no_grad():
                initial_outputs = self.forward(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    use_memory=True,
                    use_cache=True
                )
        
        # Generate tokens
        for i in range(max_length):
            with torch.no_grad():
                # Forward pass with all requested features
                outputs = self.forward(
                    input_ids=current_ids,
                    attention_mask=current_mask,
                    use_cache=True,
                    use_reasoning=use_reasoning,
                    reasoning_type=reasoning_type,
                    use_memory=use_memory,
                    use_uncertainty=use_uncertainty,
                    use_logic=use_logic,
                    **kwargs
                )
                
                # Extract logits
                if isinstance(outputs, dict):
                    next_token_logits = outputs['logits'][:, -1, :]
                else:
                    next_token_logits = outputs[0][:, -1, :]
                
                # Apply min length constraint
                if i < min_length:
                    # Set eos token prob to -inf
                    if hasattr(self, 'config') and hasattr(self.config, 'eos_token_id'):
                        next_token_logits[:, self.config.eos_token_id] = -float('inf')
                
                # Apply repetition penalty
                if repetition_penalty > 1.0:
                    for batch_idx in range(batch_size):
                        for prev_token in current_ids[batch_idx]:
                            if prev_token.item() < next_token_logits.shape[-1]:
                                next_token_logits[batch_idx, prev_token.item()] /= repetition_penalty
                
                # Apply no repeat ngram size
                if no_repeat_ngram_size > 0:
                    # For each batch
                    for batch_idx in range(batch_size):
                        # Get the current sequence
                        gen_tokens = current_ids[batch_idx].tolist()
                        # Check if we have enough tokens for ngram check
                        if len(gen_tokens) >= no_repeat_ngram_size:
                            # Get the last n-1 tokens
                            ngram_prefix = gen_tokens[-(no_repeat_ngram_size-1):]
                            # Find all ngrams in the sequence
                            for i in range(len(gen_tokens) - no_repeat_ngram_size + 1):
                                ngram = gen_tokens[i:i+no_repeat_ngram_size]
                                # If the prefix matches the start of an existing ngram
                                if ngram[:-1] == ngram_prefix:
                                    # Prevent the last token of the ngram from being generated
                                    next_token_logits[batch_idx, ngram[-1]] = -float('inf')
                
                # Apply bad words ids
                if bad_words_ids is not None:
                    for bad_word_ids in bad_words_ids:
                        # Check if the last tokens match the bad word prefix
                        bad_word_length = len(bad_word_ids)
                        if bad_word_length > 0 and current_ids.shape[1] >= bad_word_length - 1:
                            # Check if the last tokens match the bad word prefix
                            if (current_ids[:, -(bad_word_length-1):] == torch.tensor(bad_word_ids[:-1], device=device)).all(dim=1).any():
                                # Prevent the last token of the bad word from being generated
                                next_token_logits[:, bad_word_ids[-1]] = -float('inf')
                
                # Get next token
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                current_mask = torch.cat([current_mask, torch.ones_like(next_token)], dim=1)
                
                # Check for EOS token
                if hasattr(self, 'config') and hasattr(self.config, 'eos_token_id'):
                    if (next_token == self.config.eos_token_id).all():
                        break
        
        return current_ids
        
    @classmethod
    def with_moe(cls, **kwargs):
        """Create a Mixture of Experts variant of the model"""
        # Set MoE flag to True
        kwargs['use_moe'] = True
        
        # Define config if not provided
        if 'config' not in kwargs:
            config = AdvancedModelConfig(**kwargs)
            config.use_moe = True
            kwargs['config'] = config
        else:
            # Update existing config
            kwargs['config'].use_moe = True
        
        return cls(**kwargs)
    
    @classmethod
    def with_recursive_reasoning(cls, **kwargs):
        """Create a variant with enhanced recursive reasoning capabilities"""
        # Set recursive reasoning flag to True
        kwargs['use_recursive_reasoning'] = True
        
        # Define config if not provided
        if 'config' not in kwargs:
            config = AdvancedModelConfig(**kwargs)
            config.use_recursive_reasoning = True
            config.recursive_depth = kwargs.get('recursive_depth', 3)
            kwargs['config'] = config
        else:
            # Update existing config
            kwargs['config'].use_recursive_reasoning = True
            kwargs['config'].recursive_depth = kwargs.get('recursive_depth', 3)
        
        return cls(**kwargs)
    
    def enable_reasoning(self, reasoning_type='adaptive'):
        """
        Enable specific reasoning capabilities in the model.
        
        Args:
            reasoning_type (str): Type of reasoning to enable ('mathematical', 'logical', 
                                 'tree', 'mcts', 'recursive', 'knowledge', 'symbolic', 
                                 'adaptive', 'chain_of_thought', or 'all').
            
        Returns:
            None
        """
        logger.info(f"Enabling {reasoning_type} reasoning capabilities")
        
        if reasoning_type == 'all':
            # Enable all reasoning types
            self.use_math_reasoning = True
            self.use_logical_reasoning = True
            self.use_tree_reasoning = True
            self.use_mcts_reasoning = True
            self.use_recursive_reasoning = True
            self.use_knowledge_reasoning = True
            self.use_symbolic_reasoning = True
            self.use_adaptive_reasoning = True
            self.use_chain_of_thought = True
            logger.info("All reasoning capabilities enabled")
            
        elif reasoning_type == 'mathematical':
            self.use_math_reasoning = True
            logger.info("Mathematical reasoning enabled")
            
        elif reasoning_type == 'logical':
            self.use_logical_reasoning = True
            logger.info("Logical reasoning enabled")
            
        elif reasoning_type == 'tree':
            self.use_tree_reasoning = True
            logger.info("Tree reasoning enabled")
            
        elif reasoning_type == 'mcts':
            self.use_mcts_reasoning = True
            logger.info("MCTS reasoning enabled")
            
        elif reasoning_type == 'recursive':
            self.use_recursive_reasoning = True
            logger.info("Recursive reasoning enabled")
            
        elif reasoning_type == 'knowledge':
            self.use_knowledge_reasoning = True
            logger.info("Knowledge reasoning enabled")
            
        elif reasoning_type == 'symbolic':
            self.use_symbolic_reasoning = True
            logger.info("Symbolic reasoning enabled")
            
        elif reasoning_type == 'adaptive':
            self.use_adaptive_reasoning = True
            logger.info("Adaptive reasoning enabled")
            
        elif reasoning_type == 'chain_of_thought':
            self.use_chain_of_thought = True
            logger.info("Chain of thought reasoning enabled")
            
        else:
            logger.warning(f"Unknown reasoning type: {reasoning_type}, defaulting to adaptive")
            self.use_adaptive_reasoning = True
            
        return
    
    def setup_rlhf(self, reward_model=None, reference_model=None, rlhf_type='ppo'):
        """
        Set up Reinforcement Learning from Human Feedback (RLHF) components.
        
        Args:
            reward_model (nn.Module, optional): Model used to compute rewards for RLHF.
            reference_model (nn.Module, optional): Reference model used for KL divergence in RLHF.
            rlhf_type (str): Type of RLHF to use ('ppo', 'dpo', or 'constitutional').
            
        Returns:
            None
        """
        logger.info(f"Setting up RLHF components with type: {rlhf_type}")
        
        # Store reference to models
        self.reward_model = reward_model
        self.reference_model = reference_model
        
        # Set up RLHF components based on type
        if rlhf_type.lower() == 'ppo':
            logger.info("Initializing PPO trainer")
            self.ppo_trainer = PPOTrainer(
                model=self,
                reward_model=reward_model,
                reference_model=reference_model,
                tokenizer=self.tokenizer,
                kl_coef=0.1,  # KL penalty coefficient
                gamma=0.99,   # Discount factor
                lam=0.95,     # GAE lambda parameter
                cliprange=0.2 # PPO clip range
            )
            self.rlhf_trainer = self.ppo_trainer
            
        elif rlhf_type.lower() == 'dpo':
            logger.info("Initializing DPO trainer")
            self.dpo_trainer = DPOTrainer(
                model=self,
                reference_model=reference_model,
                tokenizer=self.tokenizer,
                beta=0.1,     # Temperature parameter for DPO
                max_length=1024,
                max_prompt_length=512
            )
            self.rlhf_trainer = self.dpo_trainer
            
        elif rlhf_type.lower() == 'constitutional':
            logger.info("Initializing Constitutional AI trainer")
            self.constitutional_trainer = ConstitutionalAITrainer(
                model=self,
                tokenizer=self.tokenizer,
                constitution_file="config/constitution.json",
                max_length=1024
            )
            self.rlhf_trainer = self.constitutional_trainer
            
        else:
            logger.warning(f"Unknown RLHF type: {rlhf_type}, defaulting to PPO")
            self.ppo_trainer = PPOTrainer(
                model=self,
                reward_model=reward_model,
                reference_model=reference_model,
                tokenizer=self.tokenizer,
                kl_coef=0.1,
                gamma=0.99,
                lam=0.95,
                cliprange=0.2
            )
            self.rlhf_trainer = self.ppo_trainer
        
        # Set up Actor-Critic components if using PPO
        if rlhf_type.lower() == 'ppo':
            logger.info("Initializing Actor-Critic components")
            self.actor_critic = ActorCritic(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size
            )
        
        logger.info("RLHF components initialized successfully")
        return
    
    def train_with_rlhf(self, prompts, chosen_responses, rejected_responses=None, reward_model=None, num_iterations=None):
        """
        Train the model using Reinforcement Learning from Human Feedback (RLHF).
        
        Args:
            prompts (List[str]): List of prompts for RLHF training.
            chosen_responses (List[str]): List of chosen (preferred) responses for each prompt.
            rejected_responses (List[str], optional): List of rejected responses for each prompt.
                Required for DPO training, optional for PPO.
            reward_model (nn.Module, optional): Model to compute rewards. If not provided,
                will use the reward_model set during setup_rlhf.
            num_iterations (int, optional): Number of RLHF training iterations.
                
        Returns:
            dict: Dictionary containing training metrics.
        """
        logger.info("Starting RLHF training")
        
        # Ensure RLHF components are set up
        if not hasattr(self, 'rlhf_trainer'):
            logger.warning("RLHF trainer not set up. Please call setup_rlhf first.")
            return {'error': 'RLHF trainer not set up'}
        
        # Use provided reward model or fall back to the one set during setup
        if reward_model is not None:
            self.reward_model = reward_model
        
        # Set default number of iterations if not provided
        if num_iterations is None:
            num_iterations = 1
        
        # Determine which RLHF algorithm to use based on the trainer type
        if isinstance(self.rlhf_trainer, PPOTrainer):
            logger.info(f"Training with PPO for {num_iterations} iterations")
            return self._train_with_ppo(prompts, chosen_responses, num_iterations)
            
        elif isinstance(self.rlhf_trainer, DPOTrainer):
            logger.info(f"Training with DPO for {num_iterations} iterations")
            if rejected_responses is None:
                logger.error("Rejected responses are required for DPO training")
                return {'error': 'Rejected responses are required for DPO training'}
            return self._train_with_dpo(prompts, chosen_responses, rejected_responses, num_iterations)
            
        elif isinstance(self.rlhf_trainer, ConstitutionalAITrainer):
            logger.info(f"Training with Constitutional AI for {num_iterations} iterations")
            return self._train_with_constitutional_ai(prompts, chosen_responses, num_iterations)
            
        else:
            logger.error(f"Unknown RLHF trainer type: {type(self.rlhf_trainer)}")
            return {'error': f'Unknown RLHF trainer type: {type(self.rlhf_trainer)}'}
    
    def _train_with_ppo(self, prompts, chosen_responses, num_iterations):
        """
        Train the model using Proximal Policy Optimization (PPO).
        
        Args:
            prompts (List[str]): List of prompts for PPO training.
            chosen_responses (List[str]): List of chosen responses for each prompt.
            num_iterations (int): Number of PPO training iterations.
                
        Returns:
            dict: Dictionary containing PPO training metrics.
        """
        metrics = {}
        
        # Ensure we have a reward model
        if self.reward_model is None:
            logger.error("Reward model is required for PPO training")
            return {'error': 'Reward model is required for PPO training'}
        
        # Prepare data for PPO training
        batch_size = min(len(prompts), 16)  # Use smaller batch size for PPO
        
        for iteration in range(num_iterations):
            logger.info(f"PPO iteration {iteration+1}/{num_iterations}")
            
            # Generate responses using current policy
            query_tensors = [self.tokenizer.encode(prompt, return_tensors="pt").to(self.device) for prompt in prompts[:batch_size]]
            response_tensors = []
            
            for query in query_tensors:
                with torch.no_grad():
                    response = self.generate(
                        input_ids=query,
                        max_length=query.shape[1] + 100,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.9,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                response_tensors.append(response.sequences[0])
            
            # Compute rewards using reward model
            rewards = []
            for query, response in zip(query_tensors, response_tensors):
                with torch.no_grad():
                    reward = self.reward_model(
                        input_ids=query,
                        decoder_input_ids=response
                    ).reward
                rewards.append(reward.item())
            
            # Run PPO step
            ppo_stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Update metrics
            metrics[f"iteration_{iteration+1}"] = {
                "mean_reward": np.mean(rewards),
                "kl_divergence": ppo_stats.get("kl_divergence", 0),
                "policy_loss": ppo_stats.get("policy_loss", 0),
                "value_loss": ppo_stats.get("value_loss", 0),
                "entropy": ppo_stats.get("entropy", 0)
            }
            
            logger.info(f"PPO stats: {metrics[f'iteration_{iteration+1}']}")
        
        return metrics
    
    def _train_with_dpo(self, prompts, chosen_responses, rejected_responses, num_iterations):
        """
        Train the model using Direct Preference Optimization (DPO).
        
        Args:
            prompts (List[str]): List of prompts for DPO training.
            chosen_responses (List[str]): List of chosen (preferred) responses for each prompt.
            rejected_responses (List[str]): List of rejected responses for each prompt.
            num_iterations (int): Number of DPO training iterations.
                
        Returns:
            dict: Dictionary containing DPO training metrics.
        """
        metrics = {}
        
        # Ensure we have a reference model
        if self.reference_model is None:
            logger.warning("Reference model not provided for DPO training. Using a copy of the current model.")
            self.reference_model = copy.deepcopy(self)
            self.reference_model.eval()
        
        # Prepare data for DPO training
        batch_size = min(len(prompts), 32)  # Can use larger batch size for DPO
        
        for iteration in range(num_iterations):
            logger.info(f"DPO iteration {iteration+1}/{num_iterations}")
            
            # Encode inputs
            prompt_tensors = [self.tokenizer.encode(prompt, return_tensors="pt").to(self.device) for prompt in prompts[:batch_size]]
            chosen_tensors = [self.tokenizer.encode(response, return_tensors="pt").to(self.device) for response in chosen_responses[:batch_size]]
            rejected_tensors = [self.tokenizer.encode(response, return_tensors="pt").to(self.device) for response in rejected_responses[:batch_size]]
            
            # Run DPO step
            dpo_stats = self.dpo_trainer.step(prompt_tensors, chosen_tensors, rejected_tensors)
            
            # Update metrics
            metrics[f"iteration_{iteration+1}"] = {
                "dpo_loss": dpo_stats.get("loss", 0),
                "policy_chosen_logps": dpo_stats.get("policy_chosen_logps", 0),
                "policy_rejected_logps": dpo_stats.get("policy_rejected_logps", 0),
                "reference_chosen_logps": dpo_stats.get("reference_chosen_logps", 0),
                "reference_rejected_logps": dpo_stats.get("reference_rejected_logps", 0)
            }
            
            logger.info(f"DPO stats: {metrics[f'iteration_{iteration+1}']}")
        
        return metrics
    
    def _train_with_constitutional_ai(self, prompts, chosen_responses, num_iterations):
        """
        Train the model using Constitutional AI.
        
        Args:
            prompts (List[str]): List of prompts for Constitutional AI training.
            chosen_responses (List[str]): List of chosen responses for each prompt.
            num_iterations (int): Number of Constitutional AI training iterations.
                
        Returns:
            dict: Dictionary containing Constitutional AI training metrics.
        """
        metrics = {}
        
        # Prepare data for Constitutional AI training
        batch_size = min(len(prompts), 32)
        
        for iteration in range(num_iterations):
            logger.info(f"Constitutional AI iteration {iteration+1}/{num_iterations}")
            
            # Encode inputs
            prompt_tensors = [self.tokenizer.encode(prompt, return_tensors="pt").to(self.device) for prompt in prompts[:batch_size]]
            response_tensors = [self.tokenizer.encode(response, return_tensors="pt").to(self.device) for response in chosen_responses[:batch_size]]
            
            # Run Constitutional AI step
            cai_stats = self.constitutional_trainer.step(prompt_tensors, response_tensors)
            
            # Update metrics
            metrics[f"iteration_{iteration+1}"] = {
                "cai_loss": cai_stats.get("loss", 0),
                "constitution_violations": cai_stats.get("constitution_violations", 0),
                "revised_responses": cai_stats.get("revised_responses", 0)
            }
            
            logger.info(f"Constitutional AI stats: {metrics[f'iteration_{iteration+1}']}")
        
        return metrics
    
    def train_with_rlhf(self, prompts, chosen_responses, rejected_responses=None, reward_model=None, num_iterations=None):
        """
        Train the model using RLHF.
        
        Args:
            prompts: List of prompts
            chosen_responses: List of chosen responses (for DPO)
            rejected_responses: List of rejected responses (for DPO)
            reward_model: Model to use for reward computation (for PPO)
            num_iterations: Number of RLHF iterations
            
        Returns:
            dict: Training metrics
        """
        logger.info("Training with RLHF")
        
        # Set up RLHF if not already set up
        if not hasattr(self, 'ppo_trainer'):
            self.setup_rlhf(reward_model=reward_model)
        
        # Use provided num_iterations or default from config
        if num_iterations is None:
            num_iterations = self.rlhf_config.num_iterations
        
        # Choose algorithm based on config
        if self.rlhf_config.algorithm == "ppo":
            logger.info("Using PPO algorithm for RLHF")
            
            # Train with PPO
            metrics = self.ppo_trainer.train(
                prompts=prompts,
                num_rollouts=self.rlhf_config.num_rollouts,
                num_iterations=num_iterations
            )
            
        elif self.rlhf_config.algorithm == "dpo":
            logger.info("Using DPO algorithm for RLHF")
            
            # Check if we have chosen and rejected responses
            if rejected_responses is None:
                raise ValueError("DPO requires both chosen and rejected responses")
            
            # Train with DPO
            metrics = self.dpo_trainer.train(
                prompts=prompts,
                chosen_responses=chosen_responses,
                rejected_responses=rejected_responses,
                num_iterations=num_iterations
            )
            
        elif self.rlhf_config.algorithm == "constitutional":
            logger.info("Using Constitutional AI for RLHF")
            
            # Train with Constitutional AI
            metrics = self.constitutional_trainer.train(
                prompts=prompts,
                num_iterations=num_iterations
            )
            
        else:
            raise ValueError(f"Unknown RLHF algorithm: {self.rlhf_config.algorithm}")
        
        return metrics

# Minimal ChainOfThoughtReasoner for compatibility
class ChainOfThoughtReasoner:
    """Minimal ChainOfThoughtReasoner implementation"""
    def __init__(self, model=None, tokenizer=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        
    def reason(self, prompt, **kwargs):
        return "Mock reasoning output"

# Mock classes for data processing modules
class DataLoaderFactory:
    """Factory for creating data loaders"""
    def __init__(self, batch_size=8, num_workers=2, pin_memory=True, drop_last=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
    def create_dataloader(self, dataset, shuffle=True, use_efficient_loader=False):
        from torch.utils.data import DataLoader
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )
        
    def create_distributed_dataloader(self, dataset, local_rank, use_efficient_loader=False):
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, rank=local_rank)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )

class EfficientDataLoader:
    """Efficient data loader for Valkyrie training"""
    def __init__(self, dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._dataloader = DataLoaderFactory(batch_size, num_workers, pin_memory, drop_last).create_dataloader(dataset, shuffle)
        
    def __iter__(self):
        return iter(self._dataloader)
        
    def __len__(self):
        return len(self._dataloader)

class OptimizedDataset(Dataset):
    """Optimized dataset for Valkyrie training"""
    def __init__(self, dataset, tokenizer=None, max_seq_length=1024, preprocessing_workers=4):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item
        
    def __len__(self):
        return len(self.dataset)

class DataProcessor:
    """Data processor for loading and processing datasets"""
    def __init__(self, data_dir, cache_dir=None, tokenizer=None, max_seq_length=1024):
        self.data_dir = data_dir
        self.cache_dir = cache_dir or os.path.join(data_dir, 'cache')
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
    def get_dataset(self, split='train', preprocessing_config=None):
        # Simple dataset creation for testing
        preprocessing_config = preprocessing_config or {}
        from torch.utils.data import TensorDataset
        # Create a small random dataset
        input_ids = torch.randint(0, 100, (10, 20))
        attention_mask = torch.ones_like(input_ids)
        return TensorDataset(input_ids, attention_mask)
        
    def load_reasoning_dataset(self, dataset_name, split='validation', max_samples=None, format_for_reasoning=True):
        # Create a simple dataset for testing
        samples = []
        for i in range(max_samples or 10):
            samples.append({
                "input": f"Question {i}: What is 2+2?",
                "target": "4"
            })
        return samples

class DomainSpecificDataManager:
    """Manager for domain-specific data"""
    def __init__(self, base_data_dir, domains, domain_weights=None, tokenizer=None, max_seq_length=1024, cache_dir=None):
        self.base_data_dir = base_data_dir
        self.domains = domains
        self.domain_weights = domain_weights or {}
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir or os.path.join(base_data_dir, 'cache', 'domains')
        
    def get_domain_dataset(self, domain, split='train'):
        # Create a simple dataset for testing
        from torch.utils.data import TensorDataset
        input_ids = torch.randint(0, 100, (5, 20))
        attention_mask = torch.ones_like(input_ids)
        return TensorDataset(input_ids, attention_mask)

class DomainDataBridge:
    """Bridge for handling domain-specific data"""
    def __init__(self, domain_manager, batch_size=8, num_workers=2, pin_memory=True, drop_last=False, mixing_strategy='proportional'):
        self.domain_manager = domain_manager
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.mixing_strategy = mixing_strategy
        
    def get_domain_dataloaders(self, use_distributed=False, local_rank=-1, use_efficient_loader=True):
        dataloaders = {}
        for domain in self.domain_manager.domains:
            dataset = self.domain_manager.get_domain_dataset(domain)
            factory = DataLoaderFactory(self.batch_size, self.num_workers, self.pin_memory, self.drop_last)
            if use_distributed and local_rank >= 0:
                dataloaders[domain] = factory.create_distributed_dataloader(dataset, local_rank, use_efficient_loader)
            else:
                dataloaders[domain] = factory.create_dataloader(dataset, True, use_efficient_loader)
        return dataloaders

class TokenizerManager:
    """Manager for tokenization"""
    def __init__(self, tokenizer_path=None, vocab_size=32000):
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size
        
    def get_tokenizer(self):
        # Define a minimal tokenizer for testing
        class MinimalTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
                
            def __len__(self):
                return self.vocab_size
                
            def encode(self, text, **kwargs):
                return [1, 2, 3, 4, 5]
                
            def decode(self, ids, **kwargs):
                return "decoded text"
                
            def batch_encode_plus(self, texts, **kwargs):
                batch_size = len(texts)
                return {
                    'input_ids': torch.ones(batch_size, 10, dtype=torch.long),
                    'attention_mask': torch.ones(batch_size, 10)
                }
        
        return MinimalTokenizer(self.vocab_size)

# Mock ReasoningEvaluator
class ReasoningEvaluator:
    """Evaluator for reasoning capabilities"""
    def __init__(self, model, tokenizer, device=None, max_length=512, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self.max_length = max_length
        self.batch_size = batch_size
        
    def evaluate_all(self, datasets, data_dir, split='validation', max_samples=100, comprehensive=True):
        # Return mock metrics
        metrics = {}
        for dataset in datasets:
            metrics[f"{dataset}_accuracy"] = random.random()
            metrics[f"{dataset}_f1"] = random.random()
        return metrics

# Define the EnhancedTrainingConfig class
class EnhancedTrainingConfig:
    """
    Enhanced training configuration for Valkyrie LLM.
    Includes all parameters needed for advanced training features.
    """
    def __init__(self, **kwargs):
        # Basic training parameters
        self.batch_size = kwargs.get('batch_size', 8)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.epochs = kwargs.get('epochs', 10)
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        self.warmup_steps = kwargs.get('warmup_steps', 0)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        
        # Data parameters
        self.data_dir = kwargs.get('data_dir', 'data')
        self.max_seq_length = kwargs.get('max_seq_length', 1024)
        self.min_seq_length = kwargs.get('min_seq_length', 16)
        
        # Mixed precision and efficiency
        self.use_mixed_precision = kwargs.get('use_mixed_precision', False)
        self.mixed_precision_dtype = kwargs.get('mixed_precision_dtype', 'float16')
        self.use_activation_checkpointing = kwargs.get('use_activation_checkpointing', False)
        self.checkpoint_every_n_layers = kwargs.get('checkpoint_every_n_layers', 0)
        
        # Distributed training
        self.use_distributed_training = kwargs.get('use_distributed_training', False)
        
        # Computational efficiency options
        self.use_computational_efficiency = kwargs.get('use_computational_efficiency', False)
        self.use_efficient_attention = kwargs.get('use_efficient_attention', False)
        self.use_efficient_loader = kwargs.get('use_efficient_loader', True)
        
        # Memory module options
        self.use_enhanced_memory = kwargs.get('use_enhanced_memory', False)
        self.memory_config = kwargs.get('memory_config', None)
        
        # Mixture of Experts options
        self.use_moe = kwargs.get('use_moe', False)
        self.num_experts = kwargs.get('num_experts', 4)
        self.expert_capacity = kwargs.get('expert_capacity', 0)
        self.moe_layers = kwargs.get('moe_layers', [])
        
        # Reasoning module options
        self.use_recursive_reasoning = kwargs.get('use_recursive_reasoning', False)
        self.use_tree_reasoning = kwargs.get('use_tree_reasoning', False)
        self.use_neural_symbolic = kwargs.get('use_neural_symbolic', False)
        self.use_knowledge_integration = kwargs.get('use_knowledge_integration', False)
        
        # Domain-specific training options
        self.use_domain_specific_data = kwargs.get('use_domain_specific_data', False)
        self.domains = kwargs.get('domains', [])
        self.domain_weights = kwargs.get('domain_weights', {})
        self.mixing_strategy = kwargs.get('mixing_strategy', 'proportional')
        
        # Knowledge distillation options
        self.use_knowledge_distillation = kwargs.get('use_knowledge_distillation', False)
        self.teacher_model_path = kwargs.get('teacher_model_path', None)
        self.distillation_alpha = kwargs.get('distillation_alpha', 0.5)
        self.distillation_temperature = kwargs.get('distillation_temperature', 2.0)
        
        # Logging and tracking options
        self.track_with_wandb = kwargs.get('track_with_wandb', False)
        self.wandb_project = kwargs.get('wandb_project', 'valkyrie')
        self.log_every_n_steps = kwargs.get('log_every_n_steps', 10)
        self.log_dir = kwargs.get('log_dir', 'logs')
        self.log_level = kwargs.get('log_level', 'INFO')
        
        # Evaluation options
        self.evaluate_reasoning = kwargs.get('evaluate_reasoning', True)
        self.reasoning_datasets = kwargs.get('reasoning_datasets', 'logical,mathematical,symbolic,recursive')
        self.reasoning_max_samples = kwargs.get('reasoning_max_samples', 100)
        self.reasoning_max_length = kwargs.get('reasoning_max_length', 512)
        self.reasoning_batch_size = kwargs.get('reasoning_batch_size', 8)
        self.reasoning_data_dir = kwargs.get('reasoning_data_dir', 'data/reasoning')
        
        # Validation options
        self.exit_on_validation_failure = kwargs.get('exit_on_validation_failure', False)
    
    def __str__(self):
        """String representation of the config"""
        return f"EnhancedTrainingConfig(batch_size={self.batch_size}, lr={self.learning_rate}, epochs={self.epochs})"
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# Define the AdaptiveReasoningConfig class
class AdaptiveReasoningConfig:
    """
    Configuration for adaptive reasoning in Valkyrie LLM.
    Controls how the model selects and applies different reasoning strategies.
    """
    def __init__(self, **kwargs):
        # Adaptive reasoning strategy selection
        self.use_strategy_selection = kwargs.get('use_strategy_selection', True)
        self.strategy_selection_method = kwargs.get('strategy_selection_method', 'confidence')  # 'confidence', 'meta', 'hybrid'
        
        # Reasoning module activation thresholds
        self.recursive_threshold = kwargs.get('recursive_threshold', 0.7)
        self.tree_threshold = kwargs.get('tree_threshold', 0.6)
        self.symbolic_threshold = kwargs.get('symbolic_threshold', 0.8)
        self.verifiable_threshold = kwargs.get('verifiable_threshold', 0.9)
        
        # Dynamic depth control
        self.use_dynamic_depth = kwargs.get('use_dynamic_depth', True)
        self.max_reasoning_depth = kwargs.get('max_reasoning_depth', 5)
        self.depth_increase_factor = kwargs.get('depth_increase_factor', 1.5)
        
        # Confidence estimation
        self.confidence_method = kwargs.get('confidence_method', 'entropy')  # 'entropy', 'variance', 'ensemble'
        self.min_confidence_threshold = kwargs.get('min_confidence_threshold', 0.4)
        
        # Learning rates for adaptive components
        self.meta_controller_lr = kwargs.get('meta_controller_lr', 1e-4)
        self.strategy_selector_lr = kwargs.get('strategy_selector_lr', 2e-4)
        
        # Adaptive mixing
        self.use_adaptive_mixing = kwargs.get('use_adaptive_mixing', True)
        self.mixing_temperature = kwargs.get('mixing_temperature', 0.5)
        
        # Task-specific adaptation
        self.task_specific_strategies = kwargs.get('task_specific_strategies', {})
        
    def __str__(self):
        """String representation of the config"""
        return f"AdaptiveReasoningConfig(strategy={self.strategy_selection_method}, max_depth={self.max_reasoning_depth})"
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# Model imports
from model.logical_nanogpt import LogicalGPT
from model.core_model import ValkyrieLLM
from model.reasoning import ChainOfThoughtReasoner
from model.knowledge_distillation import KnowledgeDistillationModule, KnowledgeDistillationConfig
from model.computational_efficiency import ComputationalEfficiencyOptimizer
from model.adaptive_reasoning import AdaptiveReasoningController

# Training imports
from training.trainer import MemoryEfficientTrainer
from training.advanced_trainer import AdvancedTrainer
from training.distillation import DistillationTrainer

# Data imports
from data.preprocessor import LogicalDataPreprocessor, DataLoaderFactory
from data.domain_specific_data import DomainDataManager, DomainDataConfig, DomainSpecificDataset
from data.domain_data_bridge import load_domain_data_for_training

# Utilities
from utils.enhanced_memory_manager import EnhancedMemoryManager
from utils.memory_profiler import MemoryProfiler
from utils.training_efficiency import TrainingEfficiencyManager

# Try to import validators, or define them locally if import fails
try:
    from validators import ModelValidator, ConfigValidator, ValidationResult
except ImportError:
    # Define ValidationResult locally if import fails
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class ValidationResult:
        is_valid: bool
        errors: List[str]
        warnings: List[str]
    
    # Simple validator classes if imports fail
    class ModelValidator:
        """
        Comprehensive validator for the Valkyrie model that checks all components and configurations
        to ensure they're properly initialized and compatible with each other.
        """
        
        @staticmethod
        def validate_model(model):
            """
            Validate the Valkyrie model to ensure all components are properly initialized.
            
            Returns:
                ValidationResult: Object containing validation results, warnings, and errors.
            """
            logger.info("Validating Valkyrie model and all its components")
            
            result = ValidationResult()
            
            try:
                # Check if model is an instance of ValkyrieLLM or wrapped in DDP
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_to_validate = model.module
                    if not isinstance(model_to_validate, ValkyrieLLM):
                        result.add_error("Model is wrapped in DDP but not an instance of ValkyrieLLM")
                        return result
                else:
                    if not isinstance(model, ValkyrieLLM):
                        result.add_error("Model is not an instance of ValkyrieLLM")
                        return result
                    model_to_validate = model
                
                # Validate core transformer components
                ModelValidator._validate_transformer_components(model_to_validate, result)
                
                # Validate model parameter counts
                ModelValidator._validate_parameter_counts(model_to_validate, result)
                
                # Validate advanced reasoning modules if enabled
                if hasattr(model_to_validate, 'has_recursive_reasoning') and model_to_validate.has_recursive_reasoning:
                    ModelValidator._validate_recursive_reasoning(model_to_validate, result)
                
                if hasattr(model_to_validate, 'has_tree_reasoning') and model_to_validate.has_tree_reasoning:
                    ModelValidator._validate_tree_reasoning(model_to_validate, result)
                
                if hasattr(model_to_validate, 'has_neural_symbolic') and model_to_validate.has_neural_symbolic:
                    ModelValidator._validate_neural_symbolic(model_to_validate, result)
                
                # Validate Mixture of Experts if enabled
                if hasattr(model_to_validate, 'has_moe') and model_to_validate.has_moe:
                    ModelValidator._validate_moe(model_to_validate, result)
                
                # Validate enhanced memory if enabled
                if hasattr(model_to_validate, 'has_enhanced_memory') and model_to_validate.has_enhanced_memory:
                    ModelValidator._validate_enhanced_memory(model_to_validate, result)
                
                # Validate knowledge incorporation if enabled
                if hasattr(model_to_validate, 'has_knowledge_incorporation') and model_to_validate.has_knowledge_incorporation:
                    ModelValidator._validate_knowledge_incorporation(model_to_validate, result)
                
                # Validate computational efficiency features
                ModelValidator._validate_computational_efficiency(model_to_validate, result)
                
                # Run forward pass with dummy input for final validation
                ModelValidator._validate_forward_pass(model_to_validate, result)
                
                logger.info(f"Model validation completed with {len(result.errors)} errors and {len(result.warnings)} warnings")
                
            except Exception as e:
                result.add_error(f"Exception during model validation: {str(e)}")
                logger.error(f"Exception during model validation: {str(e)}", exc_info=True)
            
            return result
        
        @staticmethod
        def _validate_transformer_components(model, result):
            """Validate the core transformer components of the model."""
            try:
                # Check if all required components exist
                if not hasattr(model, 'transformer') and not hasattr(model, 'model'):
                    result.add_error("Model missing transformer component")
                    return
                
                transformer = getattr(model, 'transformer', None) or getattr(model, 'model', None)
                
                # Check embedding layer
                if not hasattr(transformer, 'wte') and not hasattr(transformer, 'word_embeddings'):
                    result.add_error("Transformer missing word embedding layer")
                
                # Check position embeddings if used
                if model.config.use_position_embeddings and not (hasattr(transformer, 'wpe') or hasattr(transformer, 'position_embeddings')):
                    result.add_error("Model configured to use position embeddings but they're missing")
                
                # Check number of layers
                layers_attr_name = None
                for attr_name in ['h', 'layers', 'encoder', 'blocks']:
                    if hasattr(transformer, attr_name):
                        layers_attr_name = attr_name
                        break
                
                if layers_attr_name is None:
                    result.add_error("Cannot find transformer layers")
                    return
                
                layers = getattr(transformer, layers_attr_name)
                if not isinstance(layers, (list, nn.ModuleList)):
                    result.add_error(f"Transformer {layers_attr_name} is not a ModuleList or list")
                    return
                
                if len(layers) != model.config.num_layers:
                    result.add_warning(f"Number of layers ({len(layers)}) doesn't match config ({model.config.num_layers})")
                
                # Check LM head
                if not hasattr(model, 'lm_head') and not hasattr(model, 'head'):
                    result.add_error("Model missing language model head")
                
                logger.info("Core transformer components validated")
                
            except Exception as e:
                result.add_error(f"Error validating transformer components: {str(e)}")
        
        @staticmethod
        def _validate_parameter_counts(model, result):
            """Validate that parameter counts are reasonable and as expected."""
            try:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Check if parameter count is non-zero
                if total_params == 0:
                    result.add_error("Model has zero parameters")
                    return
                
                # Check if at least some parameters are trainable
                if trainable_params == 0:
                    result.add_error("Model has zero trainable parameters")
                    return
                
                # Check if parameter count is within reasonable bounds
                if total_params < 1_000_000:  # 1M
                    result.add_warning(f"Model has only {total_params:,} parameters. This seems very small for a Valkyrie LLM.")
                
                # Check if all parameters are trainable (expected for from-scratch training)
                if trainable_params < total_params:
                    trainable_ratio = trainable_params / total_params
                    if trainable_ratio < 0.99:  # Allow small differences due to special embeddings etc.
                        result.add_warning(f"Only {trainable_ratio:.2%} of parameters are trainable. Expected close to 100% for from-scratch training.")
                
                logger.info(f"Parameter count validation: {total_params:,} total, {trainable_params:,} trainable")
                
            except Exception as e:
                result.add_error(f"Error validating parameter counts: {str(e)}")
        
        @staticmethod
        def _validate_recursive_reasoning(model, result):
            """Validate the recursive reasoning module."""
            try:
                if not hasattr(model, 'recursive_reasoning_module'):
                    result.add_error("Model has recursive reasoning enabled but no module found")
                    return
                
                module = model.recursive_reasoning_module
                
                # Check reasoning depth
                if not hasattr(module, 'max_depth') or module.max_depth <= 0:
                    result.add_error("Invalid recursive reasoning depth")
                
                # Check reasoning type
                if not hasattr(module, 'reasoning_type'):
                    result.add_warning("Recursive reasoning module missing reasoning_type attribute")
                
                logger.info("Recursive reasoning module validated")
                
            except Exception as e:
                result.add_error(f"Error validating recursive reasoning: {str(e)}")
        
        @staticmethod
        def _validate_tree_reasoning(model, result):
            """Validate the tree reasoning module."""
            try:
                if not hasattr(model, 'tree_reasoning_module'):
                    result.add_error("Model has tree reasoning enabled but no module found")
                    return
                
                module = model.tree_reasoning_module
                
                # Check tree depth
                if not hasattr(module, 'max_depth') or module.max_depth <= 0:
                    result.add_error("Invalid tree reasoning depth")
                
                # Check tree width
                if not hasattr(module, 'tree_width') or module.tree_width <= 0:
                    result.add_error("Invalid tree reasoning width")
                
                logger.info("Tree reasoning module validated")
                
            except Exception as e:
                result.add_error(f"Error validating tree reasoning: {str(e)}")
        
        @staticmethod
        def _validate_neural_symbolic(model, result):
            """Validate the neural-symbolic module."""
            try:
                if not hasattr(model, 'neural_symbolic_module'):
                    result.add_error("Model has neural-symbolic reasoning enabled but no module found")
                    return
                
                module = model.neural_symbolic_module
                
                # Check integration method
                if not hasattr(module, 'integration_method'):
                    result.add_warning("Neural-symbolic module missing integration_method attribute")
                
                # Check symbolic rules if defined
                if hasattr(module, 'has_symbolic_rules') and module.has_symbolic_rules:
                    if not hasattr(module, 'symbolic_rules') or not module.symbolic_rules:
                        result.add_error("Neural-symbolic module missing symbolic rules")
                
                logger.info("Neural-symbolic module validated")
                
            except Exception as e:
                result.add_error(f"Error validating neural-symbolic module: {str(e)}")
        
        @staticmethod
        def _validate_moe(model, result):
            """Validate the Mixture of Experts module."""
            try:
                # Check if MoE layers exist
                moe_layers_found = False
                for name, module in model.named_modules():
                    if 'MoELayer' in module.__class__.__name__:
                        moe_layers_found = True
                        
                        # Check experts
                        if not hasattr(module, 'experts') or not isinstance(module.experts, nn.ModuleList):
                            result.add_error(f"MoE layer {name} missing experts ModuleList")
                        elif len(module.experts) == 0:
                            result.add_error(f"MoE layer {name} has zero experts")
                        
                        # Check router
                        if not hasattr(module, 'router'):
                            result.add_error(f"MoE layer {name} missing router")
                
                if not moe_layers_found:
                    result.add_error("Model has MoE enabled but no MoE layers found")
                
                logger.info("Mixture of Experts module validated")
                
            except Exception as e:
                result.add_error(f"Error validating MoE: {str(e)}")
        
        @staticmethod
        def _validate_enhanced_memory(model, result):
            """Validate the enhanced memory module."""
            try:
                if not hasattr(model, 'memory_module'):
                    result.add_error("Model has enhanced memory enabled but no module found")
                    return
                
                module = model.memory_module
                
                # Check memory size
                if not hasattr(module, 'memory_size') or module.memory_size <= 0:
                    result.add_error("Invalid memory size")
                
                # Check memory type
                if not hasattr(module, 'memory_type'):
                    result.add_warning("Memory module missing memory_type attribute")
                
                logger.info("Enhanced memory module validated")
                
            except Exception as e:
                result.add_error(f"Error validating enhanced memory: {str(e)}")
        
        @staticmethod
        def _validate_knowledge_incorporation(model, result):
            """Validate the knowledge incorporation module."""
            try:
                if not hasattr(model, 'knowledge_module'):
                    result.add_error("Model has knowledge incorporation enabled but no module found")
                    return
                
                module = model.knowledge_module
                
                # Check knowledge source
                if not hasattr(module, 'knowledge_source'):
                    result.add_warning("Knowledge module missing knowledge_source attribute")
                
                logger.info("Knowledge incorporation module validated")
                
            except Exception as e:
                result.add_error(f"Error validating knowledge incorporation: {str(e)}")
        
        @staticmethod
        def _validate_computational_efficiency(model, result):
            """Validate computational efficiency features."""
            try:
                # Check activation checkpointing
                has_checkpointing = False
                for name, module in model.named_modules():
                    if hasattr(module, '_checkpoint_activations') and module._checkpoint_activations:
                        has_checkpointing = True
                        break
                
                if hasattr(model, 'is_using_activation_checkpointing') and model.is_using_activation_checkpointing and not has_checkpointing:
                    result.add_warning("Model claims to use activation checkpointing but no modules have it enabled")
                
                # Check efficient attention
                if hasattr(model, 'is_using_efficient_attention') and model.is_using_efficient_attention:
                    efficient_attention_found = False
                    for name, module in model.named_modules():
                        if 'EfficientAttention' in module.__class__.__name__ or 'FlashAttention' in module.__class__.__name__:
                            efficient_attention_found = True
                            break
                    
                    if not efficient_attention_found:
                        result.add_warning("Model claims to use efficient attention but no efficient attention modules found")
                
                logger.info("Computational efficiency features validated")
                
            except Exception as e:
                result.add_error(f"Error validating computational efficiency features: {str(e)}")
        
        @staticmethod
        def _validate_forward_pass(model, result):
            """Validate that the model can perform a forward pass with dummy input."""
            try:
                # Save original device
                original_device = next(model.parameters()).device
                
                # Create dummy input
                batch_size = 2
                seq_length = 16
                
                # If model has a vocab_size attribute, use it, otherwise default to 32000
                vocab_size = getattr(model.config, 'vocab_size', 32000)
                
                inputs = {
                    'input_ids': torch.randint(0, vocab_size, (batch_size, seq_length), device=original_device),
                    'attention_mask': torch.ones(batch_size, seq_length, device=original_device)
                }
                
                # Set model to eval mode temporarily
                training_mode = model.training
                model.eval()
                
                # Perform forward pass
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Restore training mode
                model.train(training_mode)
                
                # Check outputs
                if outputs is None:
                    result.add_error("Model forward pass returned None")
                    return
                
                # Check if logits tensor has correct shape
                if not hasattr(outputs, 'logits') and not (isinstance(outputs, tuple) and len(outputs) > 0):
                    result.add_error("Model outputs missing logits")
                    return
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                expected_shape = (batch_size, seq_length, vocab_size)
                
                if logits.shape != expected_shape:
                    result.add_error(f"Output logits have incorrect shape: got {logits.shape}, expected {expected_shape}")
                
                # Check for NaN values
                if torch.isnan(logits).any():
                    result.add_error("Output logits contain NaN values")
                
                logger.info("Forward pass validation successful")
                
            except Exception as e:
                result.add_error(f"Error during forward pass validation: {str(e)}")


class ValidationResult:
    """Store validation results including errors and warnings."""
    
    def __init__(self, is_valid=None, errors=None, warnings=None):
        self.errors = errors or []
        self.warnings = warnings or []
        self._is_valid = is_valid
        
    @property
    def is_valid(self):
        if self._is_valid is not None:
            return self._is_valid
        return len(self.errors) == 0
    
    def add_error(self, message):
        self.errors.append(message)
        
    def add_warning(self, message):
        self.warnings.append(message)

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log', mode='a'),
    ]
)

# Create logs directory and add file handler
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True, parents=True)
file_handler = logging.FileHandler(os.path.join('logs', f'training_{int(time.time())}.log'))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)

def setup_logging(args):
    """
    Set up logging with advanced configuration options.
    
    Args:
        args: Command line arguments with logging settings
    """
    # Set up log level
    log_level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    log_level = log_level_map.get(args.log_level.lower() if hasattr(args, 'log_level') else "info", logging.INFO)
    
    # Set up log file handler if specified
    handlers = [logging.StreamHandler()]
    
    if hasattr(args, 'log_file') and args.log_file:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        # Create file handler
        file_handler = logging.FileHandler(args.log_file, mode='a')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=handlers
    )
    
    # Set up library-specific loggers with appropriate levels
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("bitsandbytes").setLevel(logging.WARNING)
    
    # Set up advanced logging if requested
    if hasattr(args, 'log_memory_usage') and args.log_memory_usage:
        # Log memory usage periodically
        memory_logger = logging.getLogger("memory_usage")
        memory_logger.setLevel(log_level)
        
        def log_memory_usage():
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    memory_logger.info(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        # Log initial memory usage
        log_memory_usage()
    
    # Set up learning rate logging
    if hasattr(args, 'log_learning_rate') and args.log_learning_rate:
        lr_logger = logging.getLogger("learning_rate")
        lr_logger.setLevel(log_level)
    
    # Set up gradient norm logging
    if hasattr(args, 'log_gradient_norm') and args.log_gradient_norm:
        grad_logger = logging.getLogger("gradient_norm")
        grad_logger.setLevel(log_level)
    
    # Set up parameter logging
    if hasattr(args, 'log_parameters') and args.log_parameters:
        param_logger = logging.getLogger("parameters")
        param_logger.setLevel(log_level)
    
    # Set up Weights & Biases if requested
    if hasattr(args, 'use_wandb') and args.use_wandb and 'wandb' in sys.modules and not isinstance(wandb, type):
        try:
            wandb.init(
                project=args.wandb_project if hasattr(args, 'wandb_project') else "valkyrie",
                entity=args.wandb_entity if hasattr(args, 'wandb_entity') else None,
                name=args.experiment_name if hasattr(args, 'experiment_name') else f"run_{time.strftime('%Y%m%d_%H%M%S')}",
                config=vars(args)
            )
            logger.info(f"Weights & Biases initialized with project: {args.wandb_project if hasattr(args, 'wandb_project') else 'valkyrie'}")
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {str(e)}")
    
    # Set up TensorBoard if requested
    if hasattr(args, 'use_tensorboard') and args.use_tensorboard and 'SummaryWriter' in globals():
        try:
            tensorboard_dir = args.tensorboard_dir if hasattr(args, 'tensorboard_dir') else os.path.join(args.output_dir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tensorboard_dir)
            logger.info(f"TensorBoard initialized with log directory: {tensorboard_dir}")
            
            # Add hyperparameters to TensorBoard
            writer.add_text("hyperparameters", str(vars(args)))
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {str(e)}")
    
    logger.info("Logging is set up")
    
    # Log important information
    logger.info(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA not available")
    
    logger.info(f"Command line arguments: {args}")
    
    # Return any created resources
    return {
        "writer": writer if hasattr(args, 'use_tensorboard') and args.use_tensorboard and 'writer' in locals() else None
    }

def parse_args():
    """
    Parse command line arguments for the training script.
    Includes all parameters needed for advanced components.
    """
    parser = argparse.ArgumentParser(description="Train Valkyrie LLM with all advanced components")
    
    # Basic training parameters
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model checkpoints and logs")
    parser.add_argument("--experiment_name", type=str, default="valkyrie", help="Name of the experiment")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to pretrained tokenizer")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--distributed", action="store_true", help="Whether to use distributed training")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON file with training configuration")
    parser.add_argument("--model_config", type=str, default=None, help="Path to a JSON file with model configuration")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="valkyrie", choices=["valkyrie", "valkyrie-moe", "valkyrie-recursive", "logical-gpt", "nanogpt", "efficient-transformer"], help="Type of model to train")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size of the model")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=3072, help="Size of the intermediate layer")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="Dropout probability for hidden layers")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1, help="Dropout probability for attention probabilities")
    parser.add_argument("--max_position_embeddings", type=int, default=2048, help="Maximum sequence length for position embeddings")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for training")
    parser.add_argument("--use_rotary_embeddings", action="store_true", help="Whether to use rotary position embeddings")
    parser.add_argument("--rotary_emb_base", type=int, default=10000, help="Base for rotary position embeddings")
    parser.add_argument("--tie_weights", action="store_true", help="Whether to tie weights of embedding and output layer")
    
    # Advanced model components
    parser.add_argument("--use_moe", action="store_true", help="Whether to use Mixture of Experts")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts for MoE")
    parser.add_argument("--moe_capacity_factor", type=float, default=1.25, help="Capacity factor for MoE")
    parser.add_argument("--use_tree_reasoning", action="store_true", help="Whether to use tree reasoning")
    parser.add_argument("--reasoning_depth", type=int, default=4, help="Depth for tree reasoning")
    parser.add_argument("--use_neural_symbolic", action="store_true", help="Whether to use neural symbolic reasoning")
    parser.add_argument("--use_formal_verification", action="store_true", help="Whether to use formal verification")
    parser.add_argument("--use_mcts", action="store_true", help="Whether to use Monte Carlo Tree Search")
    parser.add_argument("--mcts_simulations", type=int, default=100, help="Number of simulations for MCTS")
    parser.add_argument("--use_recursive_reasoning", action="store_true", help="Whether to use recursive reasoning")
    parser.add_argument("--recursive_depth", type=int, default=3, help="Depth for recursive reasoning")
    parser.add_argument("--use_knowledge_reasoning", action="store_true", help="Whether to use knowledge reasoning")
    parser.add_argument("--knowledge_graph_size", type=int, default=1000, help="Size of knowledge graph")
    parser.add_argument("--use_enhanced_attention", action="store_true", help="Whether to use enhanced attention")
    parser.add_argument("--attention_mechanism", type=str, default="efficient", choices=["standard", "flash", "efficient", "linear"], help="Type of attention mechanism")
    parser.add_argument("--use_hierarchical_attention", action="store_true", help="Whether to use hierarchical attention")
    parser.add_argument("--use_sparse_attention", action="store_true", help="Whether to use sparse attention")
    parser.add_argument("--use_local_attention", action="store_true", help="Whether to use local attention")
    parser.add_argument("--local_window_size", type=int, default=128, help="Window size for local attention")
    parser.add_argument("--use_memory_augmentation", action="store_true", help="Whether to use memory augmentation")
    parser.add_argument("--memory_size", type=int, default=1024, help="Size of memory")
    parser.add_argument("--use_episodic_memory", action="store_true", help="Whether to use episodic memory")
    parser.add_argument("--use_working_memory", action="store_true", help="Whether to use working memory")
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank for LoRA")
    parser.add_argument("--use_adapters", action="store_true", help="Whether to use adapters")
    parser.add_argument("--adapter_size", type=int, default=64, help="Size of adapters")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps. Overrides num_train_epochs if set to a positive value")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for gradient clipping")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Linear warmup over warmup_steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Linear warmup over warmup_ratio fraction of training steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help="The scheduler type to use")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every X updates steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size per GPU/TPU core/CPU for training")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Batch size per GPU/TPU core/CPU for evaluation")
    
    # Data parameters
    parser.add_argument("--train_file", type=str, default=None, help="Path to training data file")
    parser.add_argument("--validation_file", type=str, default=None, help="Path to validation data file")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing data files")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4, help="Number of workers for preprocessing")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--streaming", action="store_true", help="Whether to use dataset streaming mode")
    parser.add_argument("--num_train_samples", type=int, default=None, help="Number of training examples to use")
    parser.add_argument("--num_eval_samples", type=int, default=None, help="Number of evaluation examples to use")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Number of evaluation examples to use for evaluation tasks")
    
    # Efficiency parameters
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Whether to use gradient checkpointing")
    parser.add_argument("--use_activation_checkpointing", action="store_true", help="Whether to use activation checkpointing")
    parser.add_argument("--checkpoint_every_n_layers", type=int, default=2, help="Checkpoint every N layers")
    parser.add_argument("--use_efficient_attention", action="store_true", help="Whether to use efficient attention")
    parser.add_argument("--use_flash_attention", action="store_true", help="Whether to use Flash Attention")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether to use 8-bit Adam optimizer")
    parser.add_argument("--use_mixed_precision", action="store_true", help="Whether to use mixed precision training")
    parser.add_argument("--mixed_precision_dtype", type=str, default=None, choices=["float16", "bfloat16"], help="Mixed precision data type")
    parser.add_argument("--optimize_memory", action="store_true", help="Whether to optimize memory usage")
    parser.add_argument("--max_memory", type=int, default=None, help="Maximum memory in MB to use")
    parser.add_argument("--quantize_model", action="store_true", help="Whether to quantize the model")
    parser.add_argument("--quantization_bits", type=int, default=8, help="Number of bits for quantization")
    parser.add_argument("--use_computational_efficiency", action="store_true", help="Whether to use computational efficiency optimizations")
    parser.add_argument("--use_fused_operations", action="store_true", help="Whether to use fused operations")
    parser.add_argument("--use_fused_adam", action="store_true", help="Whether to use fused Adam optimizer")
    parser.add_argument("--cpu_offload", action="store_true", help="Whether to offload to CPU")
    parser.add_argument("--model_parallelism", action="store_true", help="Whether to use model parallelism")
    
    # Advanced training features
    parser.add_argument("--use_distillation", action="store_true", help="Whether to use knowledge distillation")
    parser.add_argument("--teacher_model_path", type=str, default=None, help="Path to teacher model for distillation")
    parser.add_argument("--distillation_alpha", type=float, default=0.5, help="Alpha for knowledge distillation")
    parser.add_argument("--distillation_temperature", type=float, default=2.0, help="Temperature for knowledge distillation")
    parser.add_argument("--perform_knowledge_distillation", action="store_true", help="Whether to perform knowledge distillation during training")
    parser.add_argument("--distillation_epochs", type=int, default=2, help="Number of epochs for knowledge distillation")
    
    # Reinforcement learning parameters
    parser.add_argument("--use_reinforcement_learning", action="store_true", help="Whether to use reinforcement learning")
    parser.add_argument("--rl_algorithm", type=str, default="ppo", choices=["ppo", "dpo", "constitutional"], help="Reinforcement learning algorithm to use")
    parser.add_argument("--perform_rl_training", action="store_true", help="Whether to perform reinforcement learning after training")
    parser.add_argument("--rl_epochs", type=int, default=2, help="Number of epochs for reinforcement learning")
    parser.add_argument("--rl_learning_rate", type=float, default=1e-5, help="Learning rate for reinforcement learning")
    parser.add_argument("--rl_batch_size", type=int, default=8, help="Batch size for reinforcement learning")
    parser.add_argument("--rl_mini_batch_size", type=int, default=2, help="Mini batch size for PPO")
    parser.add_argument("--rl_gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps for reinforcement learning")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of epochs for PPO")
    parser.add_argument("--kl_penalty", type=float, default=0.1, help="KL penalty for PPO")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="Beta for DPO")
    parser.add_argument("--constitutional_principles_file", type=str, default=None, help="Path to constitutional principles for Constitutional AI")
    
    # Numerical precision parameters
    parser.add_argument("--use_numerical_precision", action="store_true", help="Whether to use numerical precision")
    parser.add_argument("--numerical_precision_mode", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16", "mixed"], help="Numerical precision mode")
    parser.add_argument("--use_fp8_matmul", action="store_true", help="Whether to use FP8 for matrix multiplication")
    parser.add_argument("--use_stable_embedding", action="store_true", help="Whether to use stable embedding")
    parser.add_argument("--quantization_aware", action="store_true", help="Whether to use quantization-aware training")
    parser.add_argument("--quantization_precision", type=int, default=8, help="Precision for quantization")
    
    # Memory mechanisms parameters
    parser.add_argument("--use_memory_mechanisms", action="store_true", help="Whether to use memory mechanisms")
    parser.add_argument("--use_memory_router", action="store_true", help="Whether to use memory router")
    parser.add_argument("--use_long_term_memory", action="store_true", help="Whether to use long-term memory")
    
    # Domain-specific parameters
    parser.add_argument("--use_domain_adaptation", action="store_true", help="Whether to use domain adaptation")
    parser.add_argument("--domains", type=str, default=None, help="Comma-separated list of domains")
    parser.add_argument("--domain_weights", type=str, default=None, help="Comma-separated list of domain weights")
    
    # Evaluation parameters
    parser.add_argument("--evaluate_reasoning", action="store_true", help="Whether to evaluate reasoning capabilities")
    parser.add_argument("--reasoning_datasets", type=str, default=None, help="Comma-separated list of reasoning datasets to evaluate on")
    parser.add_argument("--reasoning_data_dir", type=str, default=None, help="Directory containing reasoning datasets")
    
    # Logging and monitoring
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="valkyrie", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity name")
    parser.add_argument("--log_level", type=str, default="info", choices=["debug", "info", "warning", "error", "critical"], help="Logging level")
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file")
    parser.add_argument("--use_tensorboard", action="store_true", help="Whether to use Tensorboard for logging")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="Directory for Tensorboard logs")
    
    # Tokenizer parameters
    parser.add_argument("--use_enhanced_tokenizer", action="store_true", help="Whether to use enhanced tokenizer")
    parser.add_argument("--tokenizer_vocab_size", type=int, default=32000, help="Vocabulary size for training a new tokenizer")
    
    # Device parameters
    parser.add_argument("--device", type=str, default=None, help="Device to use for training")
    parser.add_argument("--use_cpu", action="store_true", help="Whether to use CPU for training")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Post-processing of arguments
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"
    
    # Set output directories
    if args.tensorboard_dir is None:
        args.tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
    
    # Set up reasoning data directory
    if args.reasoning_data_dir is None:
        args.reasoning_data_dir = args.data_dir
    
    # Parse domain weights if provided
    if args.domain_weights is not None:
        try:
            domain_weights = args.domain_weights.split(",")
            args.domain_weights = [float(w) for w in domain_weights]
        except (ValueError, TypeError):
            logger.warning("Could not parse domain weights. Using equal weights.")
            args.domain_weights = None
    
    # Parse domains if provided
    if args.domains is not None:
        args.domains = args.domains.split(",")
    
    # Parse reasoning datasets if provided
    if args.reasoning_datasets is not None:
        args.reasoning_datasets = args.reasoning_datasets.split(",")
    
    return args

def setup_training_config(args):
    """
    Set up training configuration with all advanced features.
    
    Args:
        args: Command line arguments
    
    Returns:
        Training configuration object
    """
    logger.info("Setting up training configuration")
    
    try:
        if hasattr(args, 'config') and args.config:
            # Load training config from file
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
                
            # Use EnhancedTrainingConfig for Valkyrie's advanced features
            config = EnhancedTrainingConfig(**config_dict)
        else:
            # Use default config
            config = EnhancedTrainingConfig()
        
        # Override with command-line arguments if provided
        # Basic training parameters
        if hasattr(args, 'learning_rate'):
            config.learning_rate = args.learning_rate
        if hasattr(args, 'weight_decay'):
            config.weight_decay = args.weight_decay
        if hasattr(args, 'num_train_epochs'):
            config.num_train_epochs = args.num_train_epochs
        if hasattr(args, 'max_steps') and args.max_steps > 0:
            config.max_steps = args.max_steps
        if hasattr(args, 'gradient_accumulation_steps'):
            config.gradient_accumulation_steps = args.gradient_accumulation_steps
        if hasattr(args, 'max_grad_norm'):
            config.max_grad_norm = args.max_grad_norm
            
        # Advanced training parameters
        if hasattr(args, 'warmup_steps'):
            config.warmup_steps = args.warmup_steps
        if hasattr(args, 'warmup_ratio'):
            config.warmup_ratio = args.warmup_ratio
        if hasattr(args, 'lr_scheduler_type'):
            config.lr_scheduler_type = args.lr_scheduler_type
            
        # Batch sizes
        if hasattr(args, 'train_batch_size'):
            config.train_batch_size = args.train_batch_size
        if hasattr(args, 'eval_batch_size'):
            config.eval_batch_size = args.eval_batch_size
            
        # Efficiency parameters
        if hasattr(args, 'use_mixed_precision') and args.use_mixed_precision:
            config.use_mixed_precision = True
            if hasattr(args, 'mixed_precision_dtype') and args.mixed_precision_dtype:
                config.mixed_precision_dtype = args.mixed_precision_dtype
                
        if hasattr(args, 'use_gradient_checkpointing') and args.use_gradient_checkpointing:
            config.use_gradient_checkpointing = True
            
        if hasattr(args, 'use_8bit_adam') and args.use_8bit_adam:
            config.use_8bit_adam = True
            
        if hasattr(args, 'optimize_memory') and args.optimize_memory:
            config.optimize_memory = True
            
        if hasattr(args, 'use_computational_efficiency') and args.use_computational_efficiency:
            config.use_computational_efficiency = True
            
        if hasattr(args, 'use_activation_checkpointing') and args.use_activation_checkpointing:
            config.use_activation_checkpointing = True
            if hasattr(args, 'checkpoint_every_n_layers') and args.checkpoint_every_n_layers is not None:
                config.checkpoint_every_n_layers = args.checkpoint_every_n_layers
                
        if hasattr(args, 'use_efficient_attention') and args.use_efficient_attention:
            config.use_efficient_attention = True
            if hasattr(args, 'attention_implementation') and args.attention_implementation is not None:
                config.attention_implementation = args.attention_implementation
                
        if hasattr(args, 'use_early_exit') and args.use_early_exit:
            config.use_early_exit = True
            if hasattr(args, 'exit_threshold') and args.exit_threshold is not None:
                config.exit_threshold = args.exit_threshold
                
        if hasattr(args, 'use_kv_caching') and args.use_kv_caching:
            config.use_kv_caching = True
            
        # Advanced training features
        if hasattr(args, 'use_distillation') and args.use_distillation:
            config.use_distillation = True
            if hasattr(args, 'distillation_alpha'):
                config.distillation_alpha = args.distillation_alpha
            if hasattr(args, 'distillation_temperature'):
                config.distillation_temperature = args.distillation_temperature
                
        if hasattr(args, 'use_domain_specific_data') and args.use_domain_specific_data:
            config.use_domain_specific_data = True
            if hasattr(args, 'domain_weights') and args.domain_weights:
                config.domain_weights = args.domain_weights
                
        # Adaptive reasoning
        if hasattr(args, 'use_adaptive_reasoning') and args.use_adaptive_reasoning:
            config.use_adaptive_reasoning = True
            # Create adaptive reasoning config if not already present
            if not hasattr(config, 'adaptive_reasoning_config'):
                config.adaptive_reasoning_config = AdaptiveReasoningConfig()
        
        # Logging and tracking options
        if hasattr(args, 'use_wandb'):
            config.track_with_wandb = args.use_wandb
        if hasattr(args, 'wandb_project'):
            config.wandb_project = args.wandb_project
        if hasattr(args, 'wandb_entity'):
            config.wandb_entity = args.wandb_entity
            
        if hasattr(args, 'logging_steps'):
            config.log_every_n_steps = args.logging_steps
        if hasattr(args, 'eval_steps'):
            config.eval_every_n_steps = args.eval_steps
        if hasattr(args, 'save_steps'):
            config.save_every_n_steps = args.save_steps
            
        # Set output and log directories
        if hasattr(args, 'output_dir'):
            config.output_dir = args.output_dir
        if hasattr(args, 'log_dir'):
            config.log_dir = args.log_dir
        else:
            config.log_dir = os.path.join(config.output_dir, "logs")
        
        # Create necessary directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Log the full configuration
        logger.info("Training configuration set up successfully")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Full training configuration: {config}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error setting up training config: {str(e)}")
        raise e

def setup_model_config(args):
    """
    Set up model configuration with all advanced features.
    
    Args:
        args: Command line arguments
    
    Returns:
        Model configuration object
    """
    logger.info("Setting up model configuration")
    
    try:
        if hasattr(args, 'model_config') and args.model_config:
            # Load model config from file
            with open(args.model_config, 'r') as f:
                config_dict = json.load(f)
                
            # Use AdvancedModelConfig for Valkyrie's advanced features
            config = AdvancedModelConfig(**config_dict)
        else:
            # Use default config
            config = AdvancedModelConfig()
        
        # Override with command-line arguments if provided
        # Basic model parameters
        if hasattr(args, 'hidden_size'):
            config.hidden_size = args.hidden_size
        if hasattr(args, 'num_hidden_layers'):
            config.num_layers = args.num_hidden_layers
        if hasattr(args, 'num_attention_heads'):
            config.num_attention_heads = args.num_attention_heads
        if hasattr(args, 'intermediate_size'):
            config.intermediate_size = args.intermediate_size
        if hasattr(args, 'hidden_dropout_prob'):
            config.dropout = args.hidden_dropout_prob
        if hasattr(args, 'attention_probs_dropout_prob'):
            config.attention_dropout = args.attention_probs_dropout_prob
        if hasattr(args, 'max_position_embeddings'):
            config.max_seq_len = args.max_position_embeddings
        if hasattr(args, 'use_rotary_embeddings'):
            config.use_rotary_embeddings = args.use_rotary_embeddings
        if hasattr(args, 'rotary_emb_base'):
            config.rotary_emb_base = args.rotary_emb_base
        if hasattr(args, 'tie_weights'):
            config.tie_weights = args.tie_weights
            
        # Advanced model parameters
        if hasattr(args, 'use_moe') and args.use_moe:
            config.use_moe = True
            if hasattr(args, 'num_experts'):
                config.num_experts = args.num_experts
            if hasattr(args, 'moe_capacity_factor'):
                config.moe_capacity_factor = args.moe_capacity_factor
                
        if hasattr(args, 'use_tree_reasoning') and args.use_tree_reasoning:
            config.use_tree_reasoning = True
            if hasattr(args, 'reasoning_depth'):
                config.reasoning_depth = args.reasoning_depth
                
        if hasattr(args, 'use_neural_symbolic') and args.use_neural_symbolic:
            config.use_neural_symbolic = True
            
        if hasattr(args, 'use_formal_verification') and args.use_formal_verification:
            config.use_formal_verification = True
            
        if hasattr(args, 'use_mcts') and args.use_mcts:
            config.use_mcts = True
            if hasattr(args, 'mcts_simulations'):
                config.mcts_simulations = args.mcts_simulations
                
        if hasattr(args, 'use_recursive_reasoning') and args.use_recursive_reasoning:
            config.use_recursive_reasoning = True
            if hasattr(args, 'recursive_depth'):
                config.recursive_depth = args.recursive_depth
                
        if hasattr(args, 'use_knowledge_reasoning') and args.use_knowledge_reasoning:
            config.use_knowledge_reasoning = True
            if hasattr(args, 'knowledge_graph_size'):
                config.knowledge_graph_size = args.knowledge_graph_size
                
        if hasattr(args, 'use_enhanced_attention') and args.use_enhanced_attention:
            config.use_enhanced_attention = True
            if hasattr(args, 'attention_mechanism'):
                config.attention_mechanism = args.attention_mechanism
                
        if hasattr(args, 'use_hierarchical_attention') and args.use_hierarchical_attention:
            config.use_hierarchical_attention = True
            
        if hasattr(args, 'use_sparse_attention') and args.use_sparse_attention:
            config.use_sparse_attention = True
            
        if hasattr(args, 'use_local_attention') and args.use_local_attention:
            config.use_local_attention = True
            if hasattr(args, 'local_window_size'):
                config.local_window_size = args.local_window_size
                
        if hasattr(args, 'use_memory_augmentation') and args.use_memory_augmentation:
            config.use_memory_augmentation = True
            if hasattr(args, 'memory_size'):
                config.memory_size = args.memory_size
                
        if hasattr(args, 'use_episodic_memory') and args.use_episodic_memory:
            config.use_episodic_memory = True
            
        if hasattr(args, 'use_working_memory') and args.use_working_memory:
            config.use_working_memory = True
            
        if hasattr(args, 'use_lora') and args.use_lora:
            config.use_lora = True
            if hasattr(args, 'lora_rank'):
                config.lora_rank = args.lora_rank
                
        if hasattr(args, 'use_adapters') and args.use_adapters:
            config.use_adapters = True
            if hasattr(args, 'adapter_size'):
                config.adapter_size = args.adapter_size
                
        if hasattr(args, 'use_numerical_precision') and args.use_numerical_precision:
            config.use_numerical_precision = True
            if hasattr(args, 'numerical_precision_mode'):
                config.numerical_precision_mode = args.numerical_precision_mode
            if hasattr(args, 'use_fp8_matmul'):
                config.use_fp8_matmul = args.use_fp8_matmul
            if hasattr(args, 'use_stable_embedding'):
                config.use_stable_embedding = args.use_stable_embedding
                
        if hasattr(args, 'use_computational_efficiency') and args.use_computational_efficiency:
            config.use_computational_efficiency = True
            
        if hasattr(args, 'use_activation_checkpointing') and args.use_activation_checkpointing:
            config.use_activation_checkpointing = True
            if hasattr(args, 'checkpoint_every_n_layers'):
                config.checkpoint_every_n_layers = args.checkpoint_every_n_layers
                
        if hasattr(args, 'use_efficient_attention') and args.use_efficient_attention:
            config.use_efficient_attention = True
            if hasattr(args, 'attention_implementation'):
                config.attention_implementation = args.attention_implementation
                
        if hasattr(args, 'use_early_exit') and args.use_early_exit:
            config.use_early_exit = True
            if hasattr(args, 'exit_threshold'):
                config.exit_threshold = args.exit_threshold
                
        if hasattr(args, 'use_kv_caching') and args.use_kv_caching:
            config.use_cache = True
        
        # Log the full configuration
        logger.info("Model configuration set up successfully")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Full model configuration: {config}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error setting up model config: {str(e)}")
        raise e

def setup_tokenizer(args):
    """
    Set up tokenizer for the Valkyrie model using the TokenizerManager.
    
    Args:
        args: Command line arguments
        
    Returns:
        tokenizer: The tokenizer for Valkyrie
    """
    logger.info("Setting up tokenizer using TokenizerManager")
    
    try:
        # Create tokenizer manager
        tokenizer_manager = TokenizerManager(
            tokenizer_path=args.tokenizer_path,
            tokenizer_type=args.tokenizer_type,
            vocab_size=args.vocab_size or 32000,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
        # Get or create tokenizer
        tokenizer = tokenizer_manager.get_tokenizer(
            create_if_not_exists=True,
            min_frequency=args.tokenizer_min_frequency or 2
        )
        
        logger.info(f"Tokenizer setup complete with vocabulary size {tokenizer.vocab_size}")
        return tokenizer
        
    except Exception as e:
        logger.error(f"Error setting up tokenizer: {str(e)}", exc_info=True)
        raise e


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        logger.info(f"Using random seed: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    # Additional seed setting for better reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed} for full reproducibility")


class TrainingEngine:
    """
    Advanced training engine for Valkyrie LLM that integrates all training components.
    
    This engine handles standard training, knowledge distillation, domain-specific training,
    reinforcement learning, and other advanced training techniques.
    
    Args:
        model: The Valkyrie model to train
        teacher_model: Optional teacher model for knowledge distillation
        optimizer: Optimizer for training (can be None and set up later)
        lr_scheduler: Learning rate scheduler (can be None and set up later)
        training_config: Training configuration
        tokenizer: Tokenizer for the model
    """
    
    def __init__(self, model, teacher_model=None, optimizer=None, lr_scheduler=None, training_config=None, tokenizer=None):
        """
        Initialize the training engine with all components.
        
        Args:
            model: The Valkyrie model to train
            teacher_model: Optional teacher model for knowledge distillation
            optimizer: Optimizer for training (can be None and set up later)
            lr_scheduler: Learning rate scheduler (can be None and set up later)
            training_config: Training configuration
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.training_config = training_config
        self.tokenizer = tokenizer
        
        # Set up mixed precision training if enabled
        self.use_mixed_precision = hasattr(training_config, 'use_mixed_precision') and training_config.use_mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        # Set up knowledge distillation if enabled
        self.use_knowledge_distillation = self.teacher_model is not None
        
        # Set up API distillation if enabled
        self.use_api_distillation = hasattr(training_config, 'use_api_distillation') and training_config.use_api_distillation
        if self.use_api_distillation and hasattr(model, 'api_distillation'):
            self.api_distillation = model.api_distillation
        else:
            self.api_distillation = None
        
        # Set up meta-learning if enabled
        self.use_meta_learning = hasattr(training_config, 'use_meta_learning') and training_config.use_meta_learning
        if self.use_meta_learning and hasattr(model, 'meta_learner'):
            self.meta_learner = model.meta_learner
        else:
            self.meta_learner = None
        
        # Set up domain-specific training if enabled
        self.use_domain_specific = hasattr(training_config, 'use_domain_specific_data') and training_config.use_domain_specific_data
        
        # Set up reinforcement learning if enabled
        self.use_rlhf = hasattr(training_config, 'use_rlhf') and training_config.use_rlhf
        if self.use_rlhf:
            # Set up RLHF components if not already set up
            if not hasattr(model, 'ppo_trainer'):
                self.model.setup_rlhf()
            
            # Get RLHF trainers from model
            self.ppo_trainer = model.ppo_trainer
            self.dpo_trainer = model.dpo_trainer
            self.constitutional_trainer = model.constitutional_trainer
            
            # Set RLHF algorithm
            self.rlhf_algorithm = getattr(training_config, 'rlhf_algorithm', 'ppo')
        
        # Set up distributed training if enabled
        self.use_distributed = hasattr(training_config, 'use_distributed_training') and training_config.use_distributed_training
        self.local_rank = -1
        
        # Set up gradient accumulation
        self.gradient_accumulation_steps = getattr(training_config, 'gradient_accumulation_steps', 1)
        
        # Set up gradient clipping
        self.max_grad_norm = getattr(training_config, 'max_grad_norm', 1.0)
        
        # Set up device
        self.device = next(model.parameters()).device
        
        # Set up model quantization if enabled
        self.use_quantization = hasattr(training_config, 'use_quantization') and training_config.use_quantization
        if self.use_quantization:
            from model.quantization import QuantizationManager
            self.quantization_manager = QuantizationManager(
                quantization_method=getattr(training_config, 'quantization_method', 'dynamic'),
                quantization_bits=getattr(training_config, 'quantization_bits', 8),
                quantization_scheme=getattr(training_config, 'quantization_scheme', 'symmetric')
            )
        
        # Set up model compilation if enabled (PyTorch 2.0+)
        self.use_compilation = hasattr(training_config, 'compile_model') and training_config.compile_model
        if self.use_compilation and hasattr(torch, 'compile'):
            self.model = torch.compile(
                self.model, 
                backend=getattr(training_config, 'dynamo_backend', 'inductor')
            )
            logger.info(f"Model compiled with backend: {getattr(training_config, 'dynamo_backend', 'inductor')}")
        
        # Log initialization
        logger.info(f"Training engine initialized with device: {self.device}")
        logger.info(f"Mixed precision: {self.use_mixed_precision}")
        logger.info(f"Knowledge distillation: {self.use_knowledge_distillation}")
        logger.info(f"API distillation: {self.use_api_distillation}")
        logger.info(f"Meta-learning: {self.use_meta_learning}")
        logger.info(f"Domain-specific training: {self.use_domain_specific}")
        logger.info(f"RLHF: {self.use_rlhf}")
        logger.info(f"Distributed training: {self.use_distributed}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Max gradient norm: {self.max_grad_norm}")
        logger.info(f"Quantization: {self.use_quantization}")
        logger.info(f"Model compilation: {self.use_compilation}")
    
    def setup_distributed(self, local_rank):
        """
        Set up distributed training.
        
        Args:
            local_rank: Local rank of the process
        """
        self.local_rank = local_rank
        logger.info(f"Setting up distributed training with local rank: {local_rank}")
    
    def train_epoch(self, train_dataloader, domain_dataloaders=None, epoch=0):
        """
        Train for one epoch.
        
        Args:
            train_dataloader: Main training data loader
            domain_dataloaders: Optional domain-specific data loaders
            epoch: Current epoch number
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()
            
        total_loss = 0.0
        total_steps = 0
        
        # Set up progress bar
        if not self.use_distributed or self.local_rank == 0:
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        
        # Determine if we're using domain-specific training for this epoch
        use_domains_this_epoch = self.use_domain_specific and domain_dataloaders is not None
        
        # If using domain-specific training, set up domain iterators
        domain_iterators = {}
        if use_domains_this_epoch:
            for domain, dataloader in domain_dataloaders.items():
                domain_iterators[domain] = iter(dataloader)
            
            # Get domain weights
            if hasattr(self.training_config, 'domain_weights'):
                domain_weights = self.training_config.domain_weights
            else:
                # Equal weights if not specified
                domain_weights = {domain: 1.0 / len(domain_dataloaders) for domain in domain_dataloaders}
                
            logger.info(f"Using domain-specific training with weights: {domain_weights}")
        
        # Main training loop
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Determine if this step uses gradient accumulation
            is_accumulation_step = (step + 1) % self.gradient_accumulation_steps != 0
            
            # Forward and backward pass for main batch
            loss = self._training_step(batch, is_accumulation_step)
            
            # Update metrics
            if not is_accumulation_step:
                total_loss += loss
                total_steps += 1
                
                # Update progress bar
                if not self.use_distributed or self.local_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": f"{loss:.4f}"})
                
                # Domain-specific training steps
                if use_domains_this_epoch:
                    self._domain_training_steps(domain_iterators, domain_weights, domain_dataloaders, is_accumulation_step)
            
            # Log every n steps
            if not is_accumulation_step and (step // self.gradient_accumulation_steps) % self.training_config.log_every_n_steps == 0:
                logger.info(f"Epoch {epoch+1}, Step {step//self.gradient_accumulation_steps}: loss = {loss:.4f}")
        
        # Close progress bar
        if not self.use_distributed or self.local_rank == 0:
            progress_bar.close()
        
        # Calculate average loss
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        
        return avg_loss
    
    def _training_step(self, batch, is_accumulation_step):
        """
        Perform a single training step with all advanced features.
        
        Args:
            batch: Batch of data
            is_accumulation_step: Whether this is an accumulation step
            
        Returns:
            float: Loss value
        """
        # Set up context managers
        autocast_context = torch.cuda.amp.autocast() if self.use_mixed_precision else nullcontext()
        no_sync_context = self.model.no_sync() if self.use_distributed and is_accumulation_step else nullcontext()
        
        # Forward pass
        with autocast_context:
            with no_sync_context:
                # Apply meta-learning if enabled
                if self.use_meta_learning and self.meta_learner is not None:
                    # Meta-learning forward pass
                    outputs = self.meta_learner.meta_forward(self.model, **batch)
                else:
                    # Standard forward pass
                    outputs = self.model(**batch)
                
                # Extract loss
                if isinstance(outputs, dict):
                    loss = outputs.get('loss', None)
                    logits = outputs.get('logits', None)
                else:
                    loss = outputs[0]
                    logits = outputs[1] if len(outputs) > 1 else None
                
                # Knowledge distillation if enabled
                if self.use_knowledge_distillation and self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(**batch)
                    
                    # Calculate distillation loss
                    if isinstance(teacher_outputs, dict):
                        teacher_logits = teacher_outputs.get('logits', None)
                    else:
                        teacher_logits = teacher_outputs[0]
                    
                    distillation_loss = self._calculate_distillation_loss(
                        logits, 
                        teacher_logits, 
                        batch.get('attention_mask', None)
                    )
                    
                    # Combine losses
                    alpha = getattr(self.training_config, 'distillation_alpha', 0.5)
                    loss = alpha * loss + (1 - alpha) * distillation_loss
                
                # API distillation if enabled
                if self.use_api_distillation and self.api_distillation is not None:
                    api_loss = self.api_distillation(
                        model_outputs=outputs,
                        batch=batch
                    )
                    
                    # Combine losses
                    api_alpha = getattr(self.training_config, 'api_distillation_alpha', 0.3)
                    loss = (1 - api_alpha) * loss + api_alpha * api_loss
                
                # Scale loss for gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights if not an accumulation step
        if not is_accumulation_step:
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)
                
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update weights
            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
        
        return loss.item()
    
    def _domain_training_steps(self, domain_iterators, domain_weights, domain_dataloaders, is_accumulation_step):
        """
        Perform domain-specific training steps.
        
        Args:
            domain_iterators: Dictionary of domain iterators
            domain_weights: Dictionary of domain weights
            domain_dataloaders: Dictionary of domain-specific data loaders
            is_accumulation_step: Whether this is an accumulation step
        """
        for domain, iterator in domain_iterators.items():
            try:
                # Get batch from domain
                domain_batch = next(iterator)
            except StopIteration:
                # Reinitialize iterator if exhausted
                domain_iterators[domain] = iter(domain_dataloaders[domain])
                domain_batch = next(domain_iterators[domain])
            
            # Move batch to device
            domain_batch = {k: v.to(self.device) for k, v in domain_batch.items()}
            
            # Set up context managers
            autocast_context = torch.cuda.amp.autocast() if self.use_mixed_precision else nullcontext()
            no_sync_context = self.model.no_sync() if self.use_distributed and is_accumulation_step else nullcontext()
            
            # Forward pass
            with autocast_context:
                with no_sync_context:
                    # Get model outputs
                    outputs = self.model(**domain_batch)
                    loss = outputs.loss
                    
                    # Apply domain weight
                    loss = loss * domain_weights[domain]
                    
                    # Scale loss for gradient accumulation
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
    
    def _calculate_distillation_loss(self, student_logits, teacher_logits, attention_mask):
        """
        Calculate knowledge distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            attention_mask: Attention mask
            
        Returns:
            torch.Tensor: Distillation loss
        """
        # Get temperature
        temperature = getattr(self.training_config, 'distillation_temperature', 2.0)
        
        # Apply temperature
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature
        
        # Calculate KL divergence loss
        loss_fn = torch.nn.KLDivLoss(reduction='none')
        loss = loss_fn(
            torch.nn.functional.log_softmax(student_logits, dim=-1),
            torch.nn.functional.softmax(teacher_logits, dim=-1)
        )
        
        # Apply attention mask
        mask = attention_mask.unsqueeze(-1).expand_as(loss)
        loss = (loss * mask).sum() / mask.sum()
        
        # Scale by temperature squared
        loss = loss * (temperature ** 2)
        
        return loss
    
    def validate(self, val_dataloader, comprehensive=False):
        """
        Validate the model.
        
        Args:
            val_dataloader: Validation data loader
            comprehensive: Whether to perform comprehensive validation
            
        Returns:
            tuple: (validation loss, perplexity)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        # Set up progress bar
        if not self.use_distributed or self.local_rank == 0:
            progress_bar = tqdm(total=len(val_dataloader), desc="Validation")
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Update metrics
                total_loss += loss.item() * batch['attention_mask'].sum().item()
                total_tokens += batch['attention_mask'].sum().item()
                
                # Update progress bar
                if not self.use_distributed or self.local_rank == 0:
                    progress_bar.update(1)
        
        # Close progress bar
        if not self.use_distributed or self.local_rank == 0:
            progress_bar.close()
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        # Perform comprehensive validation if requested
        if comprehensive:
            logger.info("Performing comprehensive validation")
            
            # Additional validation metrics could be added here
            # For example, evaluating on specific tasks or datasets
            
        return avg_loss, perplexity
        
    def setup_optimizer(self, learning_rate, weight_decay):
        """
        Set up the optimizer for training with advanced features.
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        logger.info(f"Setting up AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")
        
        # Group parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Use fused Adam if available and enabled
        use_fused_adam = getattr(self.training_config, 'use_fused_adam', False)
        
        if use_fused_adam and 'fused' in dir(torch.optim.AdamW):
            logger.info("Using fused AdamW optimizer")
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=True
            )
        else:
            # Standard AdamW
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.95),
                eps=1e-8
            )
        
        # Use optimizer from model.optimization if available
        if hasattr(self.training_config, 'use_advanced_optimizer') and self.training_config.use_advanced_optimizer:
            from model.optimization import Optimizer as AdvancedOptimizer
            self.optimizer = AdvancedOptimizer.create_optimizer(
                model=self.model,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                optimizer_type=getattr(self.training_config, 'optimizer_type', 'adamw')
            )
            logger.info(f"Using advanced optimizer: {self.optimizer.__class__.__name__}")
        
        return self.optimizer
    
    def setup_lr_scheduler(self, num_epochs, steps_per_epoch):
        """
        Set up the learning rate scheduler.
        
        Args:
            num_epochs: Number of training epochs
            steps_per_epoch: Number of steps per epoch
        """
        logger.info("Setting up learning rate scheduler")
        
        # Get warmup steps
        warmup_steps = getattr(self.training_config, 'warmup_steps', 0)
        if warmup_steps == 0:
            # Default to 10% of total steps if not specified
            warmup_steps = int(0.1 * num_epochs * steps_per_epoch)
        
        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: min(1.0, step / warmup_steps) if step < warmup_steps else 
                         max(0.1, 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / 
                                                       (num_epochs * steps_per_epoch - warmup_steps))))
        )
        
        logger.info(f"Learning rate scheduler configured with {warmup_steps} warmup steps")
        
    def train(self, train_dataloader, val_dataloader, domain_dataloaders=None, epochs=1, output_dir="output", experiment_name="Valkyrie"):
        """
        Train the Valkyrie model for the specified number of epochs.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            domain_dataloaders: Optional domain-specific data loaders
            epochs: Number of training epochs
            output_dir: Directory to save model checkpoints
            experiment_name: Name of the experiment
            
        Returns:
            dict: Dictionary of training metrics
        """
        logger.info(f"Starting Valkyrie model training for {epochs} epochs")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save initial model and configuration
        initial_model_path = output_path / f"{experiment_name}_initial.pt"
        logger.info(f"Saving initial model to {initial_model_path}")
        torch.save(self.model.state_dict(), initial_model_path)
        
        # Initialize tracking
        if getattr(self.training_config, 'track_with_wandb', False) and (not self.use_distributed or self.local_rank == 0):
            logger.info("Initializing wandb tracking")
            
            # Prepare config dictionary
            config_dict = {}
            if hasattr(self.training_config, '__dataclass_fields__'):
                config_dict.update(asdict(self.training_config))
            elif self.training_config is not None:
                config_dict.update(vars(self.training_config))
                
            # Add model parameters info
            config_dict["model_parameters"] = sum(p.numel() for p in self.model.parameters())
            config_dict["trainable_parameters"] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Initialize wandb
            wandb.init(
                project=getattr(self.training_config, 'wandb_project', 'valkyrie'),
                name=f"{experiment_name}_{int(time.time())}",
                config=config_dict
            )
        
        # Initialize metrics
        best_val_loss = float('inf')
        training_metrics = {
            'train_losses': [],
            'val_losses': [],
            'perplexities': [],
            'learning_rates': []
        }
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch(
                train_dataloader=train_dataloader,
                domain_dataloaders=domain_dataloaders,
                epoch=epoch
            )
            
            # Log training metrics
            logger.info(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}")
            training_metrics['train_losses'].append(train_loss)
            
            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            training_metrics['learning_rates'].append(current_lr)
            logger.info(f"Current learning rate: {current_lr:.6f}")
    
    # Validate model
            val_loss, perplexity = self.validate(val_dataloader)
            logger.info(f"Epoch {epoch+1}: Validation loss = {val_loss:.4f}, Perplexity = {perplexity:.2f}")
            training_metrics['val_losses'].append(val_loss)
            training_metrics['perplexities'].append(perplexity)
            
            # Log to wandb
            if getattr(self.training_config, 'track_with_wandb', False) and (not self.use_distributed or self.local_rank == 0):
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "perplexity": perplexity,
                    "learning_rate": current_lr
                })
            
            # Save checkpoint
            if not self.use_distributed or self.local_rank == 0:
                checkpoint_path = output_path / f"{experiment_name}_epoch_{epoch+1}.pt"
                logger.info(f"Saving model checkpoint to {checkpoint_path}")
                torch.save(self.model.state_dict(), checkpoint_path)
                
                # Save config with the model
                config_path = output_path / f"{experiment_name}_config.json"
                with open(config_path, 'w') as f:
                    if hasattr(self.model, 'config') and hasattr(self.model.config, 'to_dict'):
                        json.dump(self.model.config.to_dict(), f, indent=2)
                    elif hasattr(self.model, 'config'):
                        json.dump(vars(self.model.config), f, indent=2)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = output_path / f"{experiment_name}_best.pt"
                    logger.info(f"New best model! Saving to {best_model_path}")
                    torch.save(self.model.state_dict(), best_model_path)
                    
                    # Save in safetensors format as well (if available)
                    try:
                        best_model_safetensors_path = output_path / f"{experiment_name}_best.safetensors"
                        logger.info(f"Saving best model in safetensors format to {best_model_safetensors_path}")
                        save_model(self.model, best_model_safetensors_path)
                    except Exception as e:
                        logger.warning(f"Failed to save model in safetensors format: {str(e)}")
        
            # Save final model
            if not self.use_distributed or self.local_rank == 0:
                final_model_path = output_path / f"{experiment_name}_final.pt"
                logger.info(f"Saving final model to {final_model_path}")
                torch.save(self.model.state_dict(), final_model_path)
                
                # Save in safetensors format as well (if available)
                try:
                    final_model_safetensors_path = output_path / f"{experiment_name}_final.safetensors"
                    logger.info(f"Saving final model in safetensors format to {final_model_safetensors_path}")
                    save_model(self.model, final_model_safetensors_path)
                except Exception as e:
                    logger.warning(f"Failed to save model in safetensors format: {str(e)}")
        
            # Clean up wandb
            if getattr(self.training_config, 'track_with_wandb', False) and (not self.use_distributed or self.local_rank == 0):
                wandb.finish()
        
        logger.info("Training completed successfully")
        return training_metrics
    
    def train_with_rlhf(self, train_dataloader, val_dataloader, rlhf_dataloader, epochs=1, output_dir="output", experiment_name="Valkyrie_RLHF"):
        """
        Train the model using Reinforcement Learning from Human Feedback (RLHF).
        
        Args:
            train_dataloader: Main training data loader
            val_dataloader: Validation data loader
            rlhf_dataloader: Data loader for RLHF (prompts, chosen responses, rejected responses)
            epochs: Number of epochs to train
            output_dir: Directory to save outputs
            experiment_name: Name of the experiment
            
        Returns:
            dict: Dictionary of training metrics
        """
        logger.info(f"Starting RLHF training for {epochs} epochs")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize metrics
        rlhf_metrics = {
            'ppo_rewards': [],
            'kl_divergences': [],
            'policy_losses': [],
            'value_losses': [],
            'dpo_losses': [],
            'constitutional_losses': []
        }
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"Starting RLHF epoch {epoch+1}/{epochs}")
            
            # Extract prompts, chosen responses, and rejected responses from dataloader
            prompts = []
            chosen_responses = []
            rejected_responses = []
            
            for batch in rlhf_dataloader:
                if isinstance(batch, dict):
                    prompts.extend(batch.get('prompts', []))
                    chosen_responses.extend(batch.get('chosen', []))
                    rejected_responses.extend(batch.get('rejected', []))
                else:
                    # Assume batch is a tuple of (prompts, chosen, rejected)
                    batch_prompts, batch_chosen, batch_rejected = batch
                    prompts.extend(batch_prompts)
                    chosen_responses.extend(batch_chosen)
                    rejected_responses.extend(batch_rejected)
            
            # Train with RLHF
            if self.rlhf_algorithm == "ppo":
                # Train with PPO
                metrics = self.model.train_with_rlhf(
                    prompts=prompts,
                    chosen_responses=chosen_responses,
                    num_iterations=getattr(self.training_config, 'rlhf_iterations_per_epoch', 4)
                )
                
                # Update metrics
                rlhf_metrics['ppo_rewards'].append(metrics.get('mean_reward', 0.0))
                rlhf_metrics['kl_divergences'].append(metrics.get('mean_kl', 0.0))
                rlhf_metrics['policy_losses'].append(metrics.get('policy_loss', 0.0))
                rlhf_metrics['value_losses'].append(metrics.get('value_loss', 0.0))
                
            elif self.rlhf_algorithm == "dpo":
                # Train with DPO
                metrics = self.model.train_with_rlhf(
                    prompts=prompts,
                    chosen_responses=chosen_responses,
                    rejected_responses=rejected_responses,
                    num_iterations=getattr(self.training_config, 'rlhf_iterations_per_epoch', 4)
                )
                
                # Update metrics
                rlhf_metrics['dpo_losses'].append(metrics.get('dpo_loss', 0.0))
                
            elif self.rlhf_algorithm == "constitutional":
                # Train with Constitutional AI
                metrics = self.model.train_with_rlhf(
                    prompts=prompts,
                    chosen_responses=chosen_responses,
                    num_iterations=getattr(self.training_config, 'rlhf_iterations_per_epoch', 4)
                )
                
                # Update metrics
                rlhf_metrics['constitutional_losses'].append(metrics.get('constitutional_loss', 0.0))
            
            # Log metrics
            logger.info(f"RLHF epoch {epoch+1} metrics: {metrics}")
            
            # Validate model
            val_loss, perplexity = self.validate(val_dataloader)
            logger.info(f"Validation after RLHF epoch {epoch+1}: loss = {val_loss:.4f}, perplexity = {perplexity:.2f}")
            
            # Save checkpoint
            checkpoint_path = output_path / f"{experiment_name}_rlhf_epoch_{epoch+1}.pt"
            logger.info(f"Saving RLHF checkpoint to {checkpoint_path}")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'rlhf_metrics': rlhf_metrics,
                'epoch': epoch
            }, checkpoint_path)
            
            # Log to wandb if enabled
            if getattr(self.training_config, 'track_with_wandb', False):
                wandb.log({
                    'rlhf_epoch': epoch + 1,
                    'val_loss': val_loss,
                    'perplexity': perplexity,
                    **{f'rlhf/{k}': v[-1] for k, v in rlhf_metrics.items() if v}
                })
        
        # Save final model
        final_model_path = output_path / f"{experiment_name}_rlhf_final.pt"
        logger.info(f"Saving final RLHF model to {final_model_path}")
        torch.save(self.model.state_dict(), final_model_path)
        
        return rlhf_metrics
    
    def quantize_model(self, quantization_method='dynamic', quantization_bits=8):
        """
        Quantize the model for efficient inference.
        
        Args:
            quantization_method: Method to use for quantization ('dynamic', 'static', 'aware_training')
            quantization_bits: Number of bits to use for quantization (8, 4, etc.)
            
        Returns:
            The quantized model
        """
        logger.info(f"Quantizing model with method: {quantization_method}, bits: {quantization_bits}")
        
        if not self.use_quantization:
            logger.warning("Quantization was not enabled during initialization. Setting up quantization manager.")
            from model.quantization import QuantizationManager
            self.quantization_manager = QuantizationManager(
                quantization_method=quantization_method,
                quantization_bits=quantization_bits
            )
            self.use_quantization = True
        
        # Quantize the model
        quantized_model = self.quantization_manager.quantize(self.model)
        
        # Update model reference
        self.model = quantized_model
        
        return quantized_model

def setup_train_dataloader(args, tokenizer, training_config):
    """
    Set up the training data loader using existing data processing modules.
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer to use for tokenizing data
        training_config: Training configuration
        
    Returns:
        DataLoader: Training data loader
    """
    logger.info("Setting up training data loader with optimized modules")
    
    try:
        # Create data processor
        data_processor = DataProcessor(
            data_dir=training_config.data_dir,
            cache_dir=os.path.join(training_config.data_dir, 'cache'),
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length or 1024
        )
        
        # Process training data
        train_dataset = data_processor.get_dataset(
            split='train',
            preprocessing_config={
                'shuffle': True,
                'filter_by_length': True,
                'min_length': args.min_seq_length or 16,
                'use_cache': getattr(args, 'use_cache', True)
            }
        )
        
        # If optimized datasets are enabled, wrap with OptimizedDataset
        if getattr(args, 'use_optimized_dataset', True):
            train_dataset = OptimizedDataset(
                dataset=train_dataset,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length or 1024,
                preprocessing_workers=args.preprocessing_workers or 4
            )
        
        logger.info(f"Training dataset has {len(train_dataset):,} samples")
        
        # Store dataset size for reference
        args.train_dataset_size = len(train_dataset)
        
        # Create dataloader using factory
        dataloader_factory = DataLoaderFactory(
            batch_size=training_config.batch_size,
            num_workers=args.num_workers or 2,
            pin_memory=True,
            drop_last=args.drop_last
        )
        
        # Set up distributed training if enabled
        if training_config.use_distributed_training and args.local_rank != -1:
            train_dataloader = dataloader_factory.create_distributed_dataloader(
                dataset=train_dataset,
                local_rank=args.local_rank,
                use_efficient_loader=getattr(args, 'use_efficient_loader', True)
            )
        else:
            train_dataloader = dataloader_factory.create_dataloader(
                dataset=train_dataset,
                shuffle=True,
                use_efficient_loader=getattr(args, 'use_efficient_loader', True)
            )
        
        logger.info(f"Created training dataloader with {len(train_dataloader)} batches")
        return train_dataloader
        
    except Exception as e:
        logger.error(f"Error setting up training dataloader: {str(e)}", exc_info=True)
        raise e


def setup_val_dataloader(args, tokenizer, training_config):
    """
    Set up the validation data loader using existing data processing modules.
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer to use for tokenizing data
        training_config: Training configuration
        
    Returns:
        DataLoader: Validation data loader
    """
    logger.info("Setting up validation data loader with optimized modules")
    
    try:
        # Create data processor
        data_processor = DataProcessor(
            data_dir=training_config.data_dir,
            cache_dir=os.path.join(training_config.data_dir, 'cache'),
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length or 1024
        )
        
        # Process validation data
        val_dataset = data_processor.get_dataset(
            split='validation',
            preprocessing_config={
                'shuffle': False,
                'filter_by_length': True,
                'min_length': args.min_seq_length or 16,
                'use_cache': getattr(args, 'use_cache', True)
            }
        )
        
        # If optimized datasets are enabled, wrap with OptimizedDataset
        if getattr(args, 'use_optimized_dataset', True):
            val_dataset = OptimizedDataset(
                dataset=val_dataset,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length or 1024,
                preprocessing_workers=args.preprocessing_workers or 4
            )
        
        logger.info(f"Validation dataset has {len(val_dataset):,} samples")
        
        # Store dataset size for reference
        args.val_dataset_size = len(val_dataset)
        
        # Create dataloader using factory
        dataloader_factory = DataLoaderFactory(
            batch_size=training_config.batch_size,
            num_workers=args.num_workers or 2,
            pin_memory=True,
            drop_last=False
        )
        
        # Create validation dataloader (no need for distributed for validation)
        val_dataloader = dataloader_factory.create_dataloader(
            dataset=val_dataset,
            shuffle=False,
            use_efficient_loader=getattr(args, 'use_efficient_loader', True)
        )
        
        logger.info(f"Created validation dataloader with {len(val_dataloader)} batches")
        return val_dataloader
        
    except Exception as e:
        logger.error(f"Error setting up validation dataloader: {str(e)}", exc_info=True)
        raise e



def setup_domain_dataloaders(args, tokenizer, training_config):
    """
    Set up domain-specific data loaders using existing domain data modules.
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer to use for tokenizing data
        training_config: Training configuration
        
    Returns:
        dict: Dictionary of domain-specific data loaders
    """
    logger.info("Setting up domain-specific data loaders with specialized modules")
    
    if not hasattr(training_config, 'domains') or not training_config.domains:
        logger.error("No domains specified in training config")
        return None
    
    try:
        # Create domain data manager
        domain_manager = DomainSpecificDataManager(
            base_data_dir=training_config.data_dir,
            domains=training_config.domains,
            domain_weights=getattr(training_config, 'domain_weights', None),
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length or 1024,
            cache_dir=os.path.join(training_config.data_dir, 'cache', 'domains')
        )
        
        # Create domain data bridge for efficient domain data handling
        domain_bridge = DomainDataBridge(
            domain_manager=domain_manager,
            batch_size=training_config.batch_size,
            num_workers=args.num_workers or 2,
            pin_memory=True,
            drop_last=args.drop_last,
            mixing_strategy=getattr(training_config, 'mixing_strategy', 'proportional')
        )
        
        # Get domain dataloaders
        domain_dataloaders = domain_bridge.get_domain_dataloaders(
            use_distributed=training_config.use_distributed_training,
            local_rank=args.local_rank if training_config.use_distributed_training else -1,
            use_efficient_loader=getattr(args, 'use_efficient_loader', True)
        )
        
        if not domain_dataloaders:
            logger.warning("No domain dataloaders created")
            return None
            
        logger.info(f"Created {len(domain_dataloaders)} domain dataloaders")
        return domain_dataloaders
        
    except Exception as e:
        logger.error(f"Error setting up domain dataloaders: {str(e)}", exc_info=True)
        raise e


def load_reasoning_dataset(dataset_name, data_dir, split='validation', max_samples=None):
    """
    Load a reasoning evaluation dataset using the DataProcessor.
    
    Args:
        dataset_name: Name of the reasoning dataset to load
        data_dir: Directory containing the datasets
        split: Split to load (e.g., 'train', 'validation', 'test')
        max_samples: Maximum number of samples to load (or None for all)
        
    Returns:
        list: List of samples in the format [{"input": "...", "target": "..."}, ...]
    """
    logger.info(f"Loading {dataset_name} reasoning dataset with DataProcessor")
    
    # Create data processor for reasoning data
    data_processor = DataProcessor(
        data_dir=data_dir,
        cache_dir=os.path.join(data_dir, 'cache', 'reasoning'),
        max_seq_length=512  # Default for reasoning tasks
    )
    
    # Load dataset with specific format for reasoning tasks
    dataset = data_processor.load_reasoning_dataset(
        dataset_name=dataset_name,
        split=split,
        max_samples=max_samples,
        format_for_reasoning=True
    )
    
    logger.info(f"Loaded {dataset_name} reasoning dataset with {len(dataset)} samples")
    return dataset

def evaluate_reasoning_capabilities(model, tokenizer, args):
    """
    Evaluate the reasoning capabilities of the model using various reasoning datasets.
    This function tests all the advanced reasoning components of the model.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer to use
        args: Command line arguments
        
    Returns:
        dict: Dictionary of evaluation results
    """
    logger.info("Evaluating model reasoning capabilities")
    
    try:
        # Set model to evaluation mode
        training_mode = model.training
        model.eval()
        
        # Create evaluator
        device = next(model.parameters()).device
        evaluator = ReasoningEvaluator(model, tokenizer, device=device)
        
        # Determine which datasets to evaluate on
        eval_datasets = []
        
        # Mathematical reasoning
        if hasattr(model, 'math_module') or hasattr(model, 'neural_symbolic'):
            eval_datasets.append("mathematical")
            eval_datasets.append("gsm8k")
            eval_datasets.append("math")
            
        # Logical reasoning
        if hasattr(model, 'logic_module') or hasattr(model, 'neural_symbolic'):
            eval_datasets.append("logical")
            eval_datasets.append("logical_deduction")
            
        # Symbolic reasoning
        if hasattr(model, 'neural_symbolic'):
            eval_datasets.append("symbolic")
            
        # Tree reasoning
        if hasattr(model, 'tree_reasoning'):
            eval_datasets.append("tree_reasoning")
            
        # Recursive reasoning
        if hasattr(model, 'recursive_reasoner'):
            eval_datasets.append("recursive")
            
        # Knowledge reasoning
        if hasattr(model, 'knowledge_reasoner'):
            eval_datasets.append("knowledge")
            
        # Chain of thought
        if hasattr(model, 'chain_of_thought'):
            eval_datasets.append("chain_of_thought")
            
        # MCTS reasoning
        if hasattr(model, 'mcts_reasoning'):
            eval_datasets.append("mcts")
            
        # Adaptive reasoning
        if hasattr(model, 'adaptive_reasoner'):
            eval_datasets.append("adaptive")
            
        # Common sense reasoning
        eval_datasets.append("commonsense")
        
        # Consistency checking
        eval_datasets.append("consistency")
        
        # Counterfactual reasoning
        eval_datasets.append("counterfactual")
        
        # Deduplicate datasets
        eval_datasets = list(set(eval_datasets))
        
        logger.info(f"Evaluating on datasets: {', '.join(eval_datasets)}")
        
        # Initialize results dictionary
        results = {}
        
        # Evaluate on each dataset
        for dataset_name in eval_datasets:
            logger.info(f"Evaluating on {dataset_name} dataset")
            
            # Determine max samples
            max_samples = args.max_eval_samples if hasattr(args, 'max_eval_samples') else 100
            
            # Evaluate on dataset
            dataset_results = evaluator.evaluate_dataset(
                dataset_name=dataset_name,
                data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                split="validation",
                max_samples=max_samples
            )
            
            # Add to results
            results[dataset_name] = dataset_results
            
            # Log results
            logger.info(f"Results for {dataset_name}:")
            for metric, value in dataset_results.items():
                logger.info(f"  {metric}: {value:.4f}")
                
                # Log to wandb if enabled
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({f"reasoning/{dataset_name}/{metric}": value})
        
        # Perform specialized evaluations for specific components
        
        # Evaluate mathematical reasoning if available
        if "mathematical" in eval_datasets and hasattr(model, 'math_module'):
            logger.info("Evaluating specialized mathematical reasoning")
            
            # Use the math module directly
            math_eval = model.math_module.evaluate(
                data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                max_samples=max_samples
            )
            
            # Add to results
            results["specialized_math"] = math_eval
            
            # Log results
            logger.info("Specialized mathematical reasoning results:")
            for metric, value in math_eval.items():
                logger.info(f"  {metric}: {value:.4f}")
                
                # Log to wandb if enabled
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({f"reasoning/specialized_math/{metric}": value})
        
        # Evaluate tree reasoning if available
        if "tree_reasoning" in eval_datasets and hasattr(model, 'tree_reasoning'):
            logger.info("Evaluating specialized tree reasoning")
            
            # Use the tree reasoning module directly
            tree_eval = model.tree_reasoning.evaluate(
                data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                max_samples=max_samples
            )
            
            # Add to results
            results["specialized_tree"] = tree_eval
            
            # Log results
            logger.info("Specialized tree reasoning results:")
            for metric, value in tree_eval.items():
                logger.info(f"  {metric}: {value:.4f}")
                
                # Log to wandb if enabled
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({f"reasoning/specialized_tree/{metric}": value})
        
        # Evaluate MCTS reasoning if available
        if "mcts" in eval_datasets and hasattr(model, 'mcts_reasoning'):
            logger.info("Evaluating specialized MCTS reasoning")
            
            # Use the MCTS reasoning module directly
            mcts_eval = model.mcts_reasoning.evaluate(
                data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                max_samples=max_samples
            )
            
            # Add to results
            results["specialized_mcts"] = mcts_eval
            
            # Log results
            logger.info("Specialized MCTS reasoning results:")
            for metric, value in mcts_eval.items():
                logger.info(f"  {metric}: {value:.4f}")
                
                # Log to wandb if enabled
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({f"reasoning/specialized_mcts/{metric}": value})
        
        # Evaluate recursive reasoning if available
        if "recursive" in eval_datasets and hasattr(model, 'recursive_reasoner'):
            logger.info("Evaluating specialized recursive reasoning")
            
            # Use the recursive reasoning module directly
            recursive_eval = model.recursive_reasoner.evaluate(
                data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                max_samples=max_samples
            )
            
            # Add to results
            results["specialized_recursive"] = recursive_eval
            
            # Log results
            logger.info("Specialized recursive reasoning results:")
            for metric, value in recursive_eval.items():
                logger.info(f"  {metric}: {value:.4f}")
                
                # Log to wandb if enabled
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({f"reasoning/specialized_recursive/{metric}": value})
        
        # Evaluate knowledge reasoning if available
        if "knowledge" in eval_datasets and hasattr(model, 'knowledge_reasoner'):
            logger.info("Evaluating specialized knowledge reasoning")
            
            # Use the knowledge reasoning module directly
            knowledge_eval = model.knowledge_reasoner.evaluate(
                data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                max_samples=max_samples
            )
            
            # Add to results
            results["specialized_knowledge"] = knowledge_eval
            
            # Log results
            logger.info("Specialized knowledge reasoning results:")
            for metric, value in knowledge_eval.items():
                logger.info(f"  {metric}: {value:.4f}")
                
                # Log to wandb if enabled
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({f"reasoning/specialized_knowledge/{metric}": value})
        
        # Evaluate neural symbolic reasoning if available
        if "symbolic" in eval_datasets and hasattr(model, 'neural_symbolic'):
            logger.info("Evaluating specialized neural symbolic reasoning")
            
            # Use the neural symbolic module directly
            symbolic_eval = model.neural_symbolic.evaluate(
                data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                max_samples=max_samples
            )
            
            # Add to results
            results["specialized_symbolic"] = symbolic_eval
            
            # Log results
            logger.info("Specialized neural symbolic reasoning results:")
            for metric, value in symbolic_eval.items():
                logger.info(f"  {metric}: {value:.4f}")
                
                # Log to wandb if enabled
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({f"reasoning/specialized_symbolic/{metric}": value})
        
        # Evaluate logical reasoning if available
        if "logical" in eval_datasets and hasattr(model, 'logic_module'):
            logger.info("Evaluating specialized logical reasoning")
            
            # Use the logic module directly
            logic_eval = model.logic_module.evaluate(
                data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                max_samples=max_samples
            )
            
            # Add to results
            results["specialized_logic"] = logic_eval
            
            # Log results
            logger.info("Specialized logical reasoning results:")
            for metric, value in logic_eval.items():
                logger.info(f"  {metric}: {value:.4f}")
                
                # Log to wandb if enabled
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({f"reasoning/specialized_logic/{metric}": value})
        
        # Evaluate adaptive reasoning if available
        if "adaptive" in eval_datasets and hasattr(model, 'adaptive_reasoner'):
            logger.info("Evaluating specialized adaptive reasoning")
            
            # Use the adaptive reasoning module directly
            adaptive_eval = model.adaptive_reasoner.evaluate(
                data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                max_samples=max_samples
            )
            
            # Add to results
            results["specialized_adaptive"] = adaptive_eval
            
            # Log results
            logger.info("Specialized adaptive reasoning results:")
            for metric, value in adaptive_eval.items():
                logger.info(f"  {metric}: {value:.4f}")
                
                # Log to wandb if enabled
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({f"reasoning/specialized_adaptive/{metric}": value})
        
        # Evaluate verifiable computation if available
        if hasattr(model, 'verifiable_computation'):
            logger.info("Evaluating verifiable computation")
            
            # Use the verifiable computation module directly
            verifiable_eval = model.verifiable_computation.evaluate(
                data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                max_samples=max_samples
            )
            
            # Add to results
            results["verifiable_computation"] = verifiable_eval
            
            # Log results
            logger.info("Verifiable computation results:")
            for metric, value in verifiable_eval.items():
                logger.info(f"  {metric}: {value:.4f}")
                
                # Log to wandb if enabled
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({f"reasoning/verifiable_computation/{metric}": value})
        
        # Evaluate consistency checking if available
        if "consistency" in eval_datasets:
            logger.info("Evaluating consistency checking")
            
            # Use verifiable computation if available, otherwise use standard evaluator
            if hasattr(model, 'verifiable_computation'):
                consistency_eval = model.verifiable_computation.evaluate_consistency(
                    data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                    max_samples=max_samples
                )
            else:
                # Use standard evaluator
                consistency_eval = evaluator.evaluate_dataset(
                    dataset_name="consistency_checking",
                    data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                    split="validation",
                    max_samples=max_samples
                )
                
            results["consistency_checking"] = consistency_eval
            
            # Log detailed consistency results
            logger.info("  Consistency checking evaluation:")
            for k, v in consistency_eval.items():
                logger.info(f"    {k}: {v:.4f}")
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({f"reasoning/consistency/{k}": v})
        
        # Perform counterfactual reasoning evaluation if available
        if "counterfactual" in eval_datasets:
            logger.info("Evaluating counterfactual reasoning...")
            
            counterfactual_eval = evaluator.evaluate_dataset(
                dataset_name="counterfactual",
                data_dir=args.data_dir if hasattr(args, 'data_dir') else "data",
                split="validation",
                max_samples=max_samples
            )
            results["counterfactual"] = counterfactual_eval
            
            # Log detailed counterfactual results
            logger.info("  Counterfactual reasoning evaluation:")
            for k, v in counterfactual_eval.items():
                logger.info(f"    {k}: {v:.4f}")
                if hasattr(args, 'use_wandb') and args.use_wandb:
                    wandb.log({f"reasoning/counterfactual/{k}": v})
        
        # Calculate average metrics across all reasoning tasks
        average_results = {}
        for result_type, metrics in results.items():
            for metric_name, value in metrics.items():
                if metric_name not in average_results:
                    average_results[metric_name] = []
                if isinstance(value, (int, float)):
                    average_results[metric_name].append(value)
        
        # Compute averages
        averages = {k: sum(v) / len(v) if v else 0.0 for k, v in average_results.items()}
        results["average"] = averages
        
        # Log average results
        logger.info("Average reasoning performance across all tasks:")
        for k, v in averages.items():
            logger.info(f"  {k}: {v:.4f}")
            if hasattr(args, 'use_wandb') and args.use_wandb:
                wandb.log({f"reasoning/average/{k}": v})
        
        return results
    
    except Exception as e:
        logger.error(f"Error during reasoning evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}
    
    finally:
        # Restore original training mode
        model.train(training_mode)

def setup_model(args, model_config, tokenizer, training_config):
    """
    Create and set up the model with all advanced components from the codebase.
    This function sets up the model with all available reasoning, memory, and efficiency enhancements.
    
    Args:
        args: Command line arguments
        model_config: Model configuration
        tokenizer: Tokenizer to use
        training_config: Training configuration
        
    Returns:
        The fully configured model
    """
    logger.info("Setting up Valkyrie model with advanced reasoning capabilities")
    
    try:
        # Configure model type based on configuration
        if hasattr(model_config, 'model_type'):
            model_type = model_config.model_type
        elif hasattr(args, 'model_type'):
            model_type = args.model_type
        else:
            model_type = 'valkyrie'
            
        logger.info(f"Using model type: {model_type}")
        
        # Determine vocab size from tokenizer
        vocab_size = len(tokenizer)
        logger.info(f"Setting vocab size to {vocab_size} from tokenizer")
        
        # Determine device
        device = args.device if hasattr(args, 'device') else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Create advanced model configuration
        advanced_config = AdvancedModelConfig(
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            num_attention_heads=model_config.num_attention_heads,
            vocab_size=vocab_size,
            max_seq_len=model_config.max_seq_len,
            dropout=model_config.dropout,
            
            # Reasoning configuration
            use_tree_reasoning=getattr(model_config, 'use_tree_reasoning', True),
            reasoning_depth=getattr(model_config, 'reasoning_depth', 4),
            use_neural_symbolic=getattr(model_config, 'use_neural_symbolic', True),
            use_formal_verification=getattr(model_config, 'use_formal_verification', True),
            use_mcts=getattr(model_config, 'use_mcts', True),
            mcts_simulations=getattr(model_config, 'mcts_simulations', 100),
            use_recursive_reasoning=getattr(model_config, 'use_recursive_reasoning', True),
            recursive_depth=getattr(model_config, 'recursive_depth', 3),
            use_knowledge_reasoning=getattr(model_config, 'use_knowledge_reasoning', True),
            knowledge_graph_size=getattr(model_config, 'knowledge_graph_size', 1000),
            
            # Attention configuration
            use_enhanced_attention=getattr(model_config, 'use_enhanced_attention', True),
            attention_mechanism=getattr(model_config, 'attention_mechanism', 'efficient'),
            use_hierarchical_attention=getattr(model_config, 'use_hierarchical_attention', True),
            use_sparse_attention=getattr(model_config, 'use_sparse_attention', False),
            use_local_attention=getattr(model_config, 'use_local_attention', False),
            
            # Memory configuration
            use_memory_augmentation=getattr(model_config, 'use_memory_augmentation', True),
            memory_size=getattr(model_config, 'memory_size', 1024),
            use_episodic_memory=getattr(model_config, 'use_episodic_memory', True),
            use_working_memory=getattr(model_config, 'use_working_memory', True),
            use_long_term_memory=getattr(model_config, 'use_long_term_memory', True),
            use_memory_router=getattr(model_config, 'use_memory_router', True),
            
            # Numerical precision
            use_numerical_precision=getattr(model_config, 'use_numerical_precision', True),
            numerical_precision_mode=getattr(model_config, 'numerical_precision_mode', 'auto'),
            use_fp8_matmul=getattr(model_config, 'use_fp8_matmul', False),
            
            # MoE configuration
            use_moe=getattr(model_config, 'use_moe', False),
            num_experts=getattr(model_config, 'num_experts', 8),
            moe_capacity_factor=getattr(model_config, 'moe_capacity_factor', 1.25),
            top_k_experts=getattr(model_config, 'top_k_experts', 2),
            expert_dropout=getattr(model_config, 'expert_dropout', 0.1),
            
            # LoRA and Adapters
            use_lora=getattr(model_config, 'use_lora', False),
            lora_rank=getattr(model_config, 'lora_rank', 8),
            use_adapters=getattr(model_config, 'use_adapters', False),
            adapter_size=getattr(model_config, 'adapter_size', 64),
            
            # Transformer configuration
            use_rotary_embeddings=getattr(model_config, 'use_rotary_embeddings', True),
            rotary_emb_base=getattr(model_config, 'rotary_emb_base', 10000),
            use_cache=getattr(model_config, 'use_cache', True),
            
            # Additional components
            use_adaptive_reasoning=getattr(model_config, 'use_adaptive_reasoning', True),
            use_uncertainty_calibration=getattr(model_config, 'use_uncertainty_calibration', True),
            use_text_classification=getattr(model_config, 'use_text_classification', True),
            use_enhanced_tokenizer=getattr(model_config, 'use_enhanced_tokenizer', True),
            use_meta_learning=getattr(model_config, 'use_meta_learning', True),
            use_logic_processing=getattr(model_config, 'use_logic_processing', True),
            use_sat_solver=getattr(model_config, 'use_sat_solver', True),
            use_fol_processor=getattr(model_config, 'use_fol_processor', True)
        )
        
        # Create the model with all components
        model = ValkyrieLLM(
            vocab_size=vocab_size,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_attention_heads,
            max_seq_length=model_config.max_seq_len,
            dropout=model_config.dropout,
            config=advanced_config
        )
        
        # Move model to device
        model = model.to(device)
        logger.info(f"Model moved to device: {device}")
        
        # Set up distributed training if enabled
        if hasattr(args, 'local_rank') and args.local_rank != -1:
            logger.info(f"Setting up distributed training with local rank: {args.local_rank}")
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
            )
        
        # Create training engine
        engine = TrainingEngine(model, training_config=training_config, tokenizer=tokenizer)
        
        # Check if we should load from a checkpoint
        if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
            logger.info(f"Loading model from checkpoint: {args.resume_from_checkpoint}")
            
            # Check if we're loading a safetensors file
            if args.resume_from_checkpoint.endswith('.safetensors'):
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(args.resume_from_checkpoint)
                    logger.info(f"Loaded state dict from safetensors file")
                    
                    # Load state dict
                    if hasattr(model, 'module'):
                        model.module.load_state_dict(state_dict)
                    else:
                        model.load_state_dict(state_dict)
                except ImportError:
                    logger.warning("safetensors not available, falling back to torch loading")
                    
                    # Load with torch
                    checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
                    if 'model_state_dict' in checkpoint:
                        if hasattr(model, 'module'):
                            model.module.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        if hasattr(model, 'module'):
                            model.module.load_state_dict(checkpoint)
                        else:
                            model.load_state_dict(checkpoint)
            else:
                # Load with torch
                checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
                if 'model_state_dict' in checkpoint:
                    if hasattr(model, 'module'):
                        model.module.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    if hasattr(model, 'module'):
                        model.module.load_state_dict(checkpoint)
                    else:
                        model.load_state_dict(checkpoint)
                        
                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint and engine.optimizer is not None:
                    engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                # Load scheduler state if available
                if 'scheduler_state_dict' in checkpoint and engine.scheduler is not None:
                    engine.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Set up optimizer
        engine.setup_optimizer(args.learning_rate, args.weight_decay)
        logger.info("Setting up optimizer")
            
        # Set up learning rate scheduler with advanced features
        logger.info("Setting up learning rate scheduler with advanced features")
        steps_per_epoch = args.steps_per_epoch if hasattr(args, 'steps_per_epoch') else None
        engine.setup_lr_scheduler(args.num_train_epochs, args.gradient_accumulation_steps, steps_per_epoch=steps_per_epoch)
        
        # Validate model components
        if hasattr(args, 'validate_model') and args.validate_model:
            logger.info("Validating model components")
            from model.core_model import ModelValidator
            validation_result = ModelValidator.validate_model(model)
            
            if validation_result.is_valid:
                logger.info("Model validation successful")
            else:
                logger.warning("Model validation found issues:")
                for error in validation_result.errors:
                    logger.warning(f"  Error: {error}")
                for warning in validation_result.warnings:
                    logger.warning(f"  Warning: {warning}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error setting up model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main():
    """
    Main function for training the Valkyrie LLM model.
    Sets up all components and runs the training pipeline with
    all advanced features enabled.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed if hasattr(args, 'seed') else 42)
    
    # Set up logging
    setup_logging(args)
    logger.info("Starting Valkyrie LLM training process")
    
    # Set up model configuration
    model_config = setup_model_config(args)
    logger.info(f"Model configuration: {model_config}")
    
    # Set up training configuration
    training_config = setup_training_config(args)
    logger.info(f"Training configuration: {training_config}")
    
    # Set up tokenizer
    tokenizer = setup_tokenizer(args)
    logger.info(f"Tokenizer initialized with vocabulary size: {len(tokenizer)}")
    
    # Set up model with all advanced components
    model = setup_model(args, model_config, tokenizer, training_config)
    logger.info("Model set up successfully")
    
    # Set up training dataloader
    train_dataloader = setup_train_dataloader(args, tokenizer, training_config)
    logger.info("Training dataloader set up successfully")
    
    # Set up validation dataloader
    val_dataloader = setup_val_dataloader(args, tokenizer, training_config)
    logger.info("Validation dataloader set up successfully")
    
    # Set up domain dataloaders if enabled
    domain_dataloaders = None
    if hasattr(args, 'use_domain_adaptation') and args.use_domain_adaptation:
        domain_dataloaders = setup_domain_dataloaders(args, tokenizer, training_config)
        logger.info("Domain-specific dataloaders set up successfully")
    
    # Create training engine
    engine = TrainingEngine(
        model=model,
        training_config=training_config,
        tokenizer=tokenizer
    )
    
    # Set up optimizer
    engine.setup_optimizer(
        learning_rate=args.learning_rate if hasattr(args, 'learning_rate') else 1e-4,
        weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.01
    )
    logger.info("Optimizer set up successfully")
    
    # Set up learning rate scheduler
    engine.setup_lr_scheduler(
        num_epochs=args.num_train_epochs if hasattr(args, 'num_train_epochs') else 3,
        steps_per_epoch=len(train_dataloader) // args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else len(train_dataloader)
    )
    logger.info("Learning rate scheduler set up successfully")
    
    # Enable specific reasoning capabilities if requested
    if hasattr(args, 'enable_reasoning') and args.enable_reasoning:
        reasoning_type = args.reasoning_type if hasattr(args, 'reasoning_type') else 'adaptive'
        model.enable_reasoning(reasoning_type=reasoning_type)
        logger.info(f"Enabled {reasoning_type} reasoning capabilities")
    
    # Set up RLHF if enabled
    if hasattr(args, 'use_rlhf') and args.use_rlhf:
        # Set up reward model if provided
        reward_model = None
        if hasattr(args, 'reward_model_path') and args.reward_model_path:
            logger.info(f"Loading reward model from {args.reward_model_path}")
            # Load reward model with same architecture but different weights
            reward_model = setup_model(args, model_config, tokenizer, training_config)
            reward_model.load_state_dict(torch.load(args.reward_model_path))
        
        # Set up reference model if provided
        reference_model = None
        if hasattr(args, 'reference_model_path') and args.reference_model_path:
            logger.info(f"Loading reference model from {args.reference_model_path}")
            # Load reference model with same architecture but different weights
            reference_model = setup_model(args, model_config, tokenizer, training_config)
            reference_model.load_state_dict(torch.load(args.reference_model_path))
        
        # Set up RLHF components
        model.setup_rlhf(
            reward_model=reward_model,
            reference_model=reference_model
        )
        logger.info("RLHF components set up successfully")
        
        # Set up RLHF dataloader if provided
        rlhf_dataloader = None
        if hasattr(args, 'rlhf_data_path') and args.rlhf_data_path:
            logger.info(f"Setting up RLHF dataloader from {args.rlhf_data_path}")
            # Load RLHF dataset
            from data.rlhf_dataset import RLHFDataset
            rlhf_dataset = RLHFDataset(
                data_path=args.rlhf_data_path,
                tokenizer=tokenizer,
                max_length=args.max_seq_length if hasattr(args, 'max_seq_length') else 1024
            )
            
            # Create dataloader
            rlhf_dataloader = torch.utils.data.DataLoader(
                rlhf_dataset,
                batch_size=args.rlhf_batch_size if hasattr(args, 'rlhf_batch_size') else 4,
                shuffle=True,
                num_workers=args.num_workers if hasattr(args, 'num_workers') else 2
            )
            logger.info("RLHF dataloader set up successfully")
    
    # Determine training mode
    training_mode = "standard"
    if hasattr(args, 'training_mode'):
        training_mode = args.training_mode
    
    # Run training based on mode
    if training_mode == "standard":
        # Train the model with standard training
        training_results = engine.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            domain_dataloaders=domain_dataloaders,
            epochs=args.num_train_epochs if hasattr(args, 'num_train_epochs') else 3,
            output_dir=args.output_dir if hasattr(args, 'output_dir') else "output",
            experiment_name=args.experiment_name if hasattr(args, 'experiment_name') else "valkyrie"
        )
        logger.info("Standard training completed successfully")
    
    elif training_mode == "rlhf" and 'rlhf_dataloader' in locals() and rlhf_dataloader is not None:
        # Train the model with RLHF
        rlhf_results = engine.train_with_rlhf(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            rlhf_dataloader=rlhf_dataloader,
            epochs=args.rlhf_epochs if hasattr(args, 'rlhf_epochs') else 1,
            output_dir=args.output_dir if hasattr(args, 'output_dir') else "output",
            experiment_name=args.experiment_name if hasattr(args, 'experiment_name') else "valkyrie_rlhf"
        )
        logger.info("RLHF training completed successfully")
    
    elif training_mode == "quantize":
        # Quantize the model for efficient inference
        quantized_model = engine.quantize_model(
            quantization_method=args.quantization_method if hasattr(args, 'quantization_method') else 'dynamic',
            quantization_bits=args.quantization_bits if hasattr(args, 'quantization_bits') else 8
        )
        logger.info("Model quantization completed successfully")
        
        # Save quantized model
        quantized_model_path = os.path.join(
            args.output_dir if hasattr(args, 'output_dir') else "output",
            f"{args.experiment_name if hasattr(args, 'experiment_name') else 'valkyrie'}_quantized.pt"
        )
        torch.save(quantized_model.state_dict(), quantized_model_path)
        logger.info(f"Quantized model saved to {quantized_model_path}")
    
    else:
        logger.warning(f"Unknown training mode: {training_mode}, defaulting to standard training")
        # Train the model with standard training
        training_results = engine.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            domain_dataloaders=domain_dataloaders,
            epochs=args.num_train_epochs if hasattr(args, 'num_train_epochs') else 3,
            output_dir=args.output_dir if hasattr(args, 'output_dir') else "output",
            experiment_name=args.experiment_name if hasattr(args, 'experiment_name') else "valkyrie"
        )
    
    # Evaluate reasoning capabilities if enabled
    if hasattr(args, 'evaluate_reasoning') and args.evaluate_reasoning:
        reasoning_results = evaluate_reasoning_capabilities(model, tokenizer, args)
        logger.info(f"Reasoning evaluation results: {reasoning_results}")
    
    # Save final model
    final_model_path = os.path.join(
        args.output_dir if hasattr(args, 'output_dir') else "output",
        f"{args.experiment_name if hasattr(args, 'experiment_name') else 'valkyrie'}_final.pt"
    )
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    return 0

if __name__ == "__main__":
    main()
