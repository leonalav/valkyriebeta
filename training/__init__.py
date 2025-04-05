"""
Training module with comprehensive training components for the Valkyrie model
"""

# Core components
from .components import (
    MemoryConfig, 
    TrainingEfficiencyConfig, 
    ModelConfig, 
    AdvancedModelConfig
)

# Data loading
from .data_loaders import (
    setup_train_dataloader,
    setup_val_dataloader, 
    setup_domain_dataloaders,
    setup_rlhf_dataloader,
    setup_rwkv_dataloader,
    RWKVChunkedDataset
)

# Model setup
from .model_setup import (
    setup_model,
    setup_optimizer,
    setup_scheduler,
    initialize_distributed,
    setup_tokenizer,
    setup_rwkv_model,
    setup_hybrid_model
)

# Training engine
from .training_engine import (
    TrainingEngine,
    save_model,
    load_model
)

# Evaluation functions
from .evaluation import (
    evaluate_model,
    compute_metrics,
    evaluate_reasoning_capabilities,
    evaluate_sequence_modeling
)

# RWKV specific components
from .layers.rwkv_layer import (
    RWKVTimeFirst,
    RWKVChannelMixer,
    EnhancedRWKVBlock,
    StructuralRepresentationLayer,
    SymbolicIntegrationModule,
    HierarchicalTimeAttention
)

from .layers.hybrid_model import (
    LayerRouter,
    TransformerBlock,
    HybridRWKVTransformerModel
)

# Enhanced MoE components
from .moe_router import (
    HierarchicalRouter,
    EnhancedMoE,
    MoEIntegrationLayer,
    SelfRefiningMoE
)

# Adaptive reasoning components
from .adaptive_reasoning import (
    ConfidencePredictor,
    AdaptiveMCTSReasoner,
    AdaptiveRecursiveReasoner,
    NeuralSymbolicReasoner,
    ReasoningManager
)

# Utilities
from .curriculum import (
    CurriculumScheduler,
    DifficultyEstimator
)

from .validation import (
    validate_model, 
    validate_config
)

# For backward compatibility
from .components import RWKVIntegrator, HybridModelConfigurator

__all__ = [
    # Core configs
    'MemoryConfig', 'TrainingEfficiencyConfig', 'ModelConfig', 'AdvancedModelConfig',
    
    # Data loaders
    'setup_train_dataloader', 'setup_val_dataloader', 'setup_domain_dataloaders', 
    'setup_rlhf_dataloader', 'setup_rwkv_dataloader', 'RWKVChunkedDataset',
    
    # Model setup
    'setup_model', 'setup_optimizer', 'setup_scheduler', 'initialize_distributed',
    'setup_tokenizer', 'setup_rwkv_model', 'setup_hybrid_model',
    
    # Training engine
    'TrainingEngine', 'save_model', 'load_model',
    
    # Evaluation
    'evaluate_model', 'compute_metrics', 'evaluate_reasoning_capabilities',
    'evaluate_sequence_modeling',
    
    # RWKV components
    'RWKVTimeFirst', 'RWKVChannelMixer', 'EnhancedRWKVBlock', 
    'StructuralRepresentationLayer', 'SymbolicIntegrationModule', 'HierarchicalTimeAttention',
    
    # Hybrid model components
    'LayerRouter', 'TransformerBlock', 'HybridRWKVTransformerModel',
    
    # MoE components
    'HierarchicalRouter', 'EnhancedMoE', 'MoEIntegrationLayer', 'SelfRefiningMoE',
    
    # Adaptive reasoning
    'ConfidencePredictor', 'AdaptiveMCTSReasoner', 'AdaptiveRecursiveReasoner',
    'NeuralSymbolicReasoner', 'ReasoningManager',
    
    # Utilities
    'CurriculumScheduler', 'DifficultyEstimator', 'validate_model', 'validate_config',
    
    # Backward compatibility
    'RWKVIntegrator', 'HybridModelConfigurator'
] 