from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ArchitectureConfig:
    # Core architecture
    hidden_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 8192  # 4x hidden_size
    
    # Memory efficiency
    use_flash_attention: bool = True
    use_memory_efficient_linear: bool = True
    use_gradient_checkpointing: bool = True
    use_activation_checkpointing: bool = True
    
    # Attention mechanisms
    use_alibi: bool = False  # Position embedding type
    use_rotary: bool = True  # RoPE embeddings
    attention_dropout: float = 0.1
    max_position_embeddings: int = 32768
    
    # Reasoning components
    use_tree_reasoning: bool = True
    max_reasoning_depth: int = 4
    num_reasoning_layers: int = 2
    reasoning_ffn_size: int = 1024
    
    # Expert routing
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity: int = 32
    routing_algorithm: str = "top_2"
    
    # Memory bank
    use_enhanced_memory: bool = True
    memory_size: int = 1024
    num_memory_heads: int = 4
    use_memory_compression: bool = True
    memory_compression_factor: int = 4
    use_hierarchical_memory: bool = True
    num_memory_hierarchies: int = 3
    memory_update_rate: float = 0.1
    use_gated_memory: bool = True
    use_learnable_memory: bool = True

    # Knowledge integration
    use_knowledge_integration: bool = True
    knowledge_bank_size: int = 1024
    num_knowledge_heads: int = 4
    knowledge_dropout: float = 0.1
    
    # Tree of Thought reasoning
    use_tree_of_thought: bool = True
    max_tree_depth: int = 3
    branching_factor: int = 4
    pruning_threshold: float = 0.3
    use_value_function: bool = True
    use_beam_search: bool = True
    beam_size: int = 4
    use_monte_carlo: bool = True
    monte_carlo_samples: int = 8
    use_self_consistency: bool = True
    consistency_threshold: float = 0.7
    
    # Recurrent reasoning
    use_recurrent_reasoning: bool = True
    recurrent_hidden_size: int = 2048
    num_recurrent_layers: int = 1
    
    # Uncertainty estimation
    use_uncertainty_estimation: bool = True
    uncertainty_samples: int = 5
    
    # Adaptive computation
    use_adaptive_computation: bool = True
    max_computation_steps: int = 10
    early_stopping_threshold: float = 0.95
    
    # Model name for loading pretrained components
    model_name: str = "meta-llama/Llama-2-7b-hf"  # Default model to use for components
    
    # Neural-Symbolic Integration
    use_neural_symbolic: bool = True
    symbol_vocabulary_size: int = 128
    num_symbolic_layers: int = 2
    use_symbolic_reasoning: bool = True
    use_neural_guided_search: bool = True
    max_symbolic_steps: int = 5
    use_symbolic_verification: bool = True
    use_symbolic_abstraction: bool = True
    abstraction_levels: int = 3
    use_symbolic_composition: bool = True
    composition_depth: int = 2
    
    # Recursive Reasoning
    use_recursive_reasoning_transformer: bool = True
    max_recursion_depth: int = 5
    use_recursive_attention: bool = True
    use_recursive_memory: bool = True
    use_recursive_gating: bool = True
    use_recursive_routing: bool = True
    num_reasoning_experts: int = 4
    use_recursive_verification: bool = True
    use_recursive_composition: bool = True
    
    # Multi-Hop Knowledge Reasoning
    use_knowledge_reasoning: bool = True
    max_hops: int = 3
    use_adaptive_hops: bool = True
    knowledge_source: str = "conceptnet"  # "conceptnet", "wordnet", "wikidata", "custom"
    knowledge_embedding_dim: int = 768
    knowledge_graph_path: str = "data/knowledge_graphs/"
    use_knowledge_retrieval: bool = True
    max_knowledge_items: int = 50
    use_knowledge_fusion: bool = True
    fusion_layers: int = 2
    use_multi_hop_attention: bool = True
    use_knowledge_routing: bool = True
    num_knowledge_experts: int = 4
    
    # Verifiable Computation
    use_verifiable_computation: bool = True
    num_computation_units: int = 3
    verification_threshold: float = 0.7
    use_cross_verification: bool = True
    use_self_verification: bool = True
    use_verification_routing: bool = True
    num_verification_experts: int = 3
    use_verification_feedback: bool = True
    feedback_iterations: int = 2
    use_verification_memory: bool = True
    verification_memory_size: int = 64
