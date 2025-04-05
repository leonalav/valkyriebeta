"""
Configuration for 3B parameter model with 16k context length
"""

from training.components import AdvancedModelConfig, TrainingEfficiencyConfig, MemoryConfig

# Model configuration for 3B parameters
model_config = AdvancedModelConfig(
    # Core architecture
    hidden_size=2560,          # Increased from 768
    num_layers=32,             # Increased from 12
    num_attention_heads=32,    # Increased from 12
    vocab_size=32000,          # Standard GPT-2 vocabulary size
    max_seq_len=16384,         # Increased to 16k
    dropout=0.1,
    
    # Advanced features
    use_moe=True,              # Enable Mixture of Experts for efficiency
    num_experts=16,            # Increased number of experts
    moe_capacity_factor=1.25,
    top_k_experts=4,
    expert_dropout=0.1,
    
    # Reasoning capabilities
    use_tree_reasoning=True,
    reasoning_depth=4,
    use_neural_symbolic=True,
    use_formal_verification=True,
    use_mcts=True,
    mcts_simulations=100,
    use_recursive_reasoning=True,
    recursive_depth=3,
    use_knowledge_reasoning=True,
    knowledge_graph_size=1000,
    
    # Attention configuration
    use_enhanced_attention=True,
    attention_mechanism="efficient",
    use_hierarchical_attention=True,
    use_sparse_attention=True,  # Enable sparse attention for efficiency
    sparse_attention_pattern="fixed",
    use_local_attention=True,
    local_window_size=2048,    # Increased window size
    
    # Memory configuration
    use_memory_augmentation=True,
    memory_size=4096,          # Increased memory size
    use_episodic_memory=True,
    use_working_memory=True,
    
    # Numerical precision
    use_numerical_precision=True,
    numerical_precision_mode="auto",
    use_fp8_matmul=True,       # Enable FP8 for efficiency
    use_stable_embedding=True,
    math_precision_enabled=True
)

# Training efficiency configuration
training_config = TrainingEfficiencyConfig(
    use_mixed_precision=True,
    optimize_cuda_kernels=True,
    optimize_grouping=True,
    compile_model=True,        # Enable model compilation
    dynamo_backend="inductor", # Use PyTorch 2.0's inductor backend
    use_fused_adam=True,
    use_fused_layer_norm=True,
    
    # Advanced efficiency options
    activation_checkpointing=True,
    checkpoint_every_n_layers=2,
    use_sharded_ddp=True,      # Enable sharded DDP for memory efficiency
    use_fsdp=True,             # Enable Fully Sharded Data Parallel
    use_offload=True,          # Enable CPU offloading
    use_cpu_offload=True,      # Enable CPU offloading for optimizer states
    gradient_accumulation_steps=4  # Increased for memory efficiency
)

# Memory configuration
memory_config = MemoryConfig(
    use_gradient_checkpointing=True,
    use_flash_attention=True,
    activation_checkpointing=True,
    optimize_memory_use=True,
    mem_efficient_linear=True,
    cpu_offload=True,
    low_cpu_mem_usage=True,
    max_memory_MB=24000,       # Set maximum GPU memory usage
    
    # Advanced memory mechanisms
    use_episodic_memory=True,
    episodic_memory_size=2048,  # Increased memory sizes
    use_working_memory=True,
    working_memory_size=1024,
    use_long_term_memory=True,
    long_term_memory_size=8192,
    use_memory_router=True,
    memory_update_frequency=10
)

# Training hyperparameters
training_params = {
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "num_train_epochs": 3,
    "batch_size": 4,           # Reduced batch size for memory efficiency
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "lr_scheduler": "cosine",
    "optimizer": "adamw",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8
}

# Model architecture parameters
architecture_params = {
    "ffn_hidden_size": 10240,  # Increased from 3072
    "layer_norm_epsilon": 1e-5,
    "attention_dropout": 0.1,
    "use_bias": True,
    "activation_function": "gelu_new",
    
    # Advanced architecture options
    "use_rmsnorm": True,
    "use_parallel_attention": True,
    "use_swiglu": True,
    "use_flash_attention": True,
    "use_efficient_qkv": True,
    "use_fused_operations": True,
    "use_rope_scaling": True,
    "rope_scaling_factor": 2.0  # Increased for longer context
} 