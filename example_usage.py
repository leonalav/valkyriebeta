import torch
import torch.nn as nn
from config.architecture_config import ArchitectureConfig
from config.training_config import EnhancedTrainingConfig, DistillationConfig, DomainSpecificConfig, ComputationalEfficiencyConfig, AdaptiveReasoningConfig
from model.core_model import EnhancedLanguageModel
from model.knowledge_distillation import KnowledgeDistillationModule, KnowledgeDistillationConfig
from model.computational_efficiency import ComputationalEfficiencyOptimizer
from model.adaptive_reasoning import AdaptiveReasoningController
from data.domain_specific_data import DomainDataManager, DomainDataConfig
from tokenizer.tokenizer import Tokenizer
import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

logger = logging.getLogger(__name__)

def basic_usage_example():
    """Basic usage example with core functionality"""
    print("Initializing Enhanced Language Model with Advanced Reasoning Capabilities")
    
    # Create configuration with all reasoning components enabled
    config = ArchitectureConfig(
        # Core architecture (using smaller values for example)
        hidden_size=256,
        num_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        
        # Memory efficiency
        use_flash_attention=True,
        use_memory_efficient_linear=True,
        
        # Expert routing
        use_moe=True,
        num_experts=4,
        num_experts_per_token=2,
        
        # Memory bank
        use_enhanced_memory=True,
        memory_size=128,
        use_hierarchical_memory=True,
        
        # Tree reasoning
        use_tree_reasoning=True,
        num_branches=3,
        max_tree_depth=3,
        
        # Neural-Symbolic Integration
        use_neural_symbolic=True,
        num_symbols=128,
        num_rules=32,
        
        # Recursive Reasoning
        use_recursive_reasoning=True,
        max_recursion_depth=3,
        
        # Knowledge Reasoning
        use_knowledge_reasoning=True,
        knowledge_size=1024,
        
        # Verifiable Computation
        use_verifiable_computation=True,
        num_verifiers=3,
        
        # Vocabulary size
        vocab_size=10000
    )
    
    # Create model
    model = EnhancedLanguageModel(config)
    
    # Generate input
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    labels = torch.randint(0, config.vocab_size, (1, 10))
    
    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    
    # Print outputs
    print("Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Print loss
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    print("Basic usage example completed")

def knowledge_distillation_example():
    """Example using knowledge distillation"""
    print("\nInitializing model with knowledge distillation")
    
    # Create configuration with knowledge distillation enabled
    config = ArchitectureConfig(
        # Core architecture (smaller for example)
        hidden_size=256,
        num_layers=4,
        num_attention_heads=8,
        
        # Enable knowledge distillation
        use_knowledge_distillation=True,
        
        # Vocabulary size
        vocab_size=10000
    )
    
    # Create model
    student_model = EnhancedLanguageModel(config)
    
    # Create a simple teacher model (for demonstration)
    teacher_config = ArchitectureConfig(
        hidden_size=512,  # Larger teacher model
        num_layers=8,
        num_attention_heads=16,
        vocab_size=10000
    )
    teacher_model = EnhancedLanguageModel(teacher_config)
    
    # Create distillation configuration
    distillation_config = KnowledgeDistillationConfig(
        use_teacher_model=True,
        temperature=2.0,
        alpha=0.5,
        domain_adaptation_layers=1
    )
    
    # Create distillation module
    distillation_module = KnowledgeDistillationModule(distillation_config, config.hidden_size)
    
    # Generate input
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    labels = torch.randint(0, config.vocab_size, (1, 10))
    
    # Get teacher outputs
    with torch.no_grad():
        teacher_outputs = teacher_model(input_ids=input_ids, labels=labels)
    
    # Forward pass with student model and teacher outputs
    student_outputs = student_model(
        input_ids=input_ids, 
        labels=labels,
        teacher_model_outputs=teacher_outputs
    )
    
    # Print outputs
    print("Student model outputs with distillation:")
    for key, value in student_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: shape={tuple(v.shape)}")
                else:
                    print(f"    {k}: {type(v)}")
        else:
            print(f"  {key}: {type(value)}")
    
    print("Knowledge distillation example completed")

def domain_specific_data_example():
    """Example using domain-specific data"""
    print("\nInitializing domain-specific data handling")
    
    # Create domain data config
    domain_config = DomainDataConfig(
        domains=["general", "math", "science"],
        domain_weights={
            "general": 1.0,
            "math": 1.5,
            "science": 1.2
        },
        use_curriculum=True,
        mixing_strategy="proportional"
    )
    
    # For this example, we'll create some dummy domain directories and files
    data_dir = Path("example_data")
    data_dir.mkdir(exist_ok=True)
    
    for domain in domain_config.domains:
        domain_dir = data_dir / domain
        domain_dir.mkdir(exist_ok=True)
        
        # Create dummy data file
        with open(domain_dir / "data.json", "w") as f:
            f.write(f'{{"examples": [{{"text": "This is a {domain} example"}}]}}\n')
        
        # Create dummy vocab file
        with open(domain_dir / "vocab.json", "w") as f:
            f.write(f'["{domain}_token_1", "{domain}_token_2"]\n')
    
    # Update domain paths
    domain_config.domain_data_paths = {
        domain: str(data_dir / domain / "data.json") for domain in domain_config.domains
    }
    domain_config.domain_vocab_files = {
        domain: str(data_dir / domain / "vocab.json") for domain in domain_config.domains
    }
    
    # Create a simple tokenizer for demonstration
    class SimpleTokenizer:
        def __init__(self):
            self.vocab_size = 1000
            
        def __call__(self, text, max_length=None, padding=None, truncation=None, return_tensors=None):
            # Simple tokenizer implementation
            batch_size = 1 if isinstance(text, str) else len(text)
            seq_len = max_length or 10
            
            # Create dummy tensors
            input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
            attention_mask = torch.ones_like(input_ids)
            
            # Return as PyTorch tensors if requested
            if return_tensors == "pt":
                return SimpleEncodings(input_ids, attention_mask)
            
            # Otherwise return as dictionary
            return {
                "input_ids": input_ids.tolist(),
                "attention_mask": attention_mask.tolist()
            }
            
        def add_tokens(self, tokens):
            # Simulate adding tokens to vocabulary
            return len(tokens)
    
    class SimpleEncodings:
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    
    # Create domain data manager
    domain_manager = DomainDataManager(
        config=domain_config,
        tokenizer=tokenizer,
        max_length=20,
        seed=42
    )
    
    # Create mixed dataloader
    train_loader = domain_manager.create_mixed_dataloader(
        batch_size=4,
        shuffle=True,
        num_workers=0  # Use 0 for example
    )
    
    # Print information about the dataloader
    print(f"Mixed domain dataloader created with {len(train_loader)} batches")
    
    # Get a batch from the dataloader
    batch = next(iter(train_loader))
    
    print("Batch contents:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)}")
        else:
            print(f"  {key}: {value}")
    
    # Cleanup example data
    import shutil
    shutil.rmtree(data_dir)
    
    print("Domain-specific data example completed")

def computational_efficiency_example():
    """Example using computational efficiency optimizations"""
    print("\nInitializing model with computational efficiency optimizations")
    
    # Create configuration with default reasoning components
    config = ArchitectureConfig(
        # Core architecture (smaller for example)
        hidden_size=256,
        num_layers=4,
        num_attention_heads=8,
        
        # Enable computational efficiency
        use_computational_efficiency=True,
        
        # Vocabulary size
        vocab_size=10000
    )
    
    # Create model
    model = EnhancedLanguageModel(config)
    
    # Create efficiency configuration
    efficiency_config = ComputationalEfficiencyConfig(
        use_activation_checkpointing=True,
        checkpoint_every_n_layers=2,
        use_efficient_attention=True,
        attention_implementation="memory_efficient",  # Use memory_efficient as flash requires CUDA
        use_early_exit=True,
        exit_threshold=0.9,
        use_kv_caching=True,
        use_mixed_precision=False  # Disable for example
    )
    
    # Create efficiency optimizer
    optimizer = ComputationalEfficiencyOptimizer(efficiency_config)
    
    # Optimize model
    optimized_model = optimizer.optimize_model(model)
    
    # Create KV cache for generation
    kv_cache = optimizer.create_kv_cache()
    
    # Create early exit controller
    early_exit_controller = optimizer.create_early_exit_controller()
    
    # Generate input for inference
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    
    # Prepare model for generation
    optimized_model.prepare_for_generation()
    
    # Forward pass with KV caching (simulating generation)
    with torch.no_grad():
        # First forward pass
        outputs = optimized_model(
            input_ids=input_ids,
            use_cache=True
        )
        
        # Get logits and past_key_values
        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        
        # Simulate generating next token
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        
        # Second forward pass with past_key_values
        next_outputs = optimized_model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True
        )
    
    print("Generated optimized outputs:")
    for key, value in next_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)}")
        elif key == "past_key_values":
            print(f"  {key}: {type(value)} with {len(value)} layers")
        else:
            print(f"  {key}: {type(value)}")
    
    print("Computational efficiency example completed")

def adaptive_reasoning_example():
    """Example using adaptive reasoning"""
    print("\nInitializing model with adaptive reasoning")
    
    # Create configuration with adaptive reasoning enabled
    config = ArchitectureConfig(
        # Core architecture (smaller for example)
        hidden_size=256,
        num_layers=4,
        num_attention_heads=8,
        
        # Enable adaptive reasoning
        use_adaptive_reasoning=True,
        
        # Enable various reasoning components
        use_moe=True,
        use_neural_symbolic=True,
        use_recursive_reasoning=True,
        use_tree_reasoning=True,
        
        # Vocabulary size
        vocab_size=10000
    )
    
    # Create model
    model = EnhancedLanguageModel(config)
    
    # Generate inputs of varying complexity
    simple_input = torch.randint(0, config.vocab_size, (1, 5))  # Short input
    complex_input = torch.randint(0, config.vocab_size, (1, 20))  # Longer input
    
    # Forward pass with simple input
    print("Processing simple input:")
    simple_outputs = model(input_ids=simple_input)
    
    # Forward pass with complex input
    print("Processing complex input:")
    complex_outputs = model(input_ids=complex_input)
    
    print("Adaptive reasoning example completed")

def combined_example():
    """Example combining all features"""
    print("\nInitializing model with all enhanced features")
    
    # Create model configuration
    model_config = ArchitectureConfig(
        # Core architecture (smaller for example)
        hidden_size=256,
        num_layers=4,
        num_attention_heads=8,
        
        # Enable enhanced features
        use_knowledge_distillation=True,
        use_computational_efficiency=True,
        use_adaptive_reasoning=True,
        
        # Reasoning components
        use_moe=True,
        use_neural_symbolic=True,
        use_recursive_reasoning=True,
        use_tree_reasoning=True,
        
        # Vocabulary size
        vocab_size=10000
    )
    
    # Create training configuration
    training_config = EnhancedTrainingConfig(
        # Basic training parameters
        batch_size=8,
        learning_rate=5e-5,
        num_epochs=3,
        
        # Knowledge distillation
        use_knowledge_distillation=True,
        distillation_alpha=0.5,
        progressive_distillation=True,
        
        # Domain-specific data
        use_domain_specific_data=True,
        domains=["general", "math", "science"],
        domain_weights={
            "general": 1.0,
            "math": 1.5,
            "science": 1.2
        },
        
        # Computational efficiency
        use_computational_efficiency=True,
        use_activation_checkpointing=True,
        use_efficient_attention=True,
        use_early_exit=True,
        
        # Adaptive reasoning
        use_adaptive_reasoning=True
    )
    
    # Create model
    model = EnhancedLanguageModel(model_config)
    
    # Print model configuration summary
    print("Model Configuration:")
    for key, value in vars(model_config).items():
        if not key.startswith("_"):
            print(f"  {key}: {value}")
    
    # Print training configuration summary
    print("\nTraining Configuration:")
    for key, value in vars(training_config).items():
        if not key.startswith("_") and not isinstance(value, dict) and not isinstance(value, list):
            print(f"  {key}: {value}")
    
    # Generate input
    input_ids = torch.randint(0, model_config.vocab_size, (1, 10))
    labels = torch.randint(0, model_config.vocab_size, (1, 10))
    
    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    
    # Print outputs
    print("\nModel outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)}")
        else:
            print(f"  {key}: {type(value)}")
    
    print("Combined example completed")

def example_mcts_reasoning():
    """Example of using Tree-of-Thought reasoning with MCTS"""
    # Similar to the example in README
    from model import TransformerModel, ModelConfig, create_transformer_model
    from model import LanguageModelMCTSIntegration, MCTSConfig
    from transformers import AutoTokenizer
    
    # Create a model with Tree-of-Thought reasoning
    config = ModelConfig(
        vocab_size=50257,  # GPT-2 vocabulary size
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=1024,
        use_tree_reasoning=True,
        use_flash_attention=True
    )
    model = create_transformer_model(config)
    
    # Load a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure MCTS
    mcts_config = MCTSConfig(
        max_iterations=50,
        exploration_weight=1.5,
        rollout_depth=3,
        top_k_candidates=5
    )
    
    # Create integration
    integration = LanguageModelMCTSIntegration(
        language_model=model,
        tokenizer=tokenizer,
        mcts_config=mcts_config
    )
    
    # Solve a problem
    problem = "Solve the following problem step by step: If a train travels at 60 mph, how long will it take to travel 150 miles?"
    solution, trace = integration.solve_problem(problem)
    
    print("Solution:")
    print(solution)
    print("\nReasoning steps:")
    for i, step in enumerate(trace["steps"]):
        print(f"{i+1}. {step}")

def example_upgrade_to_1_3b_with_32k_context():
    """
    Example of creating or upgrading to a 1.3B parameter model with 32K+ context length.
    
    This example demonstrates:
    1. Creating a new 1.3B parameter model from scratch
    2. Upgrading an existing model to 1.3B parameters with extended context
    3. Loading pretrained weights and extending the context length
    
    The example requires access to a pretrained model or checkpoints.
    """
    import torch
    from model import (
        ModelConfig, 
        create_transformer_model, 
        TransformerModel,
        upgrade_to_1_3b_with_extended_context
    )
    from transformers import AutoTokenizer
    
    print("Example 1: Creating a new 1.3B parameter model with 32K context from scratch")
    
    # Create a 1.3B parameter configuration with 32K context length
    config_1_3b = ModelConfig.create_1_3b_config(context_length=32768)
    
    # Create a new model from this configuration
    model_1_3b = create_transformer_model(config_1_3b)
    
    # Display model statistics
    total_params = sum(p.numel() for p in model_1_3b.parameters() if p.requires_grad)
    print(f"Model created with {total_params / 1e9:.2f} billion parameters")
    print(f"Hidden size: {config_1_3b.hidden_size}")
    print(f"Layers: {config_1_3b.num_hidden_layers}")
    print(f"Attention heads: {config_1_3b.num_attention_heads}")
    print(f"Context length: {config_1_3b.max_position_embeddings}")
    
    print("\nExample 2: Upgrading an existing model to 1.3B with extended context")
    
    # For demonstration, create a smaller model to upgrade
    # In a real scenario, you would load a pretrained model
    config_small = ModelConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048
    )
    small_model = create_transformer_model(config_small)
    
    # Upgrade the model to 1.3B with 32K context
    upgraded_model = upgrade_to_1_3b_with_extended_context(
        model=small_model,
        context_length=32768,
        save_path=None  # Set to a path to save the model
    )
    
    # Display upgraded model statistics
    total_params_upgraded = sum(p.numel() for p in upgraded_model.parameters() if p.requires_grad)
    print(f"Model upgraded to {total_params_upgraded / 1e9:.2f} billion parameters")
    print(f"New context length: {upgraded_model.config.max_position_embeddings}")
    
    print("\nExample 3: Using the model with long context")
    
    # Load a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create a long input sequence for testing (repeated text to simulate long context)
    base_text = "This is a test of the long context capabilities. "
    long_input = base_text * 1000  # Repeat to create long context
    
    # Tokenize with truncation disabled
    input_ids = tokenizer.encode(
        long_input, 
        return_tensors='pt',
        truncation=False,
        max_length=32768
    )
    
    # Print context statistics
    print(f"Input length: {input_ids.shape[1]} tokens")
    print(f"Maximum context length: {upgraded_model.config.max_position_embeddings}")
    
    # For demo purposes, only process a small segment to avoid long computation
    # In a real scenario, you would process the full context
    with torch.no_grad():
        if input_ids.shape[1] > 1000:
            print("For demo, truncating long input to 1000 tokens...")
            input_ids = input_ids[:, :1000]
        
        # Process the input (forward pass only)
        outputs = upgraded_model(
            input_ids=input_ids,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        
    print("Successfully processed long context input")
    
    # Generate a short continuation to verify the model works
    print("\nGenerating a short continuation from the context:")
    generate_length = 50
    
    outputs, _ = upgraded_model.generate(
        input_ids=input_ids[:, -512:],  # Use the last 512 tokens for generation
        max_length=input_ids.shape[1] + generate_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        use_tree_reasoning=True,
    )
    
    # Decode and print generated text
    generated_text = tokenizer.decode(outputs[0][-generate_length:], skip_special_tokens=True)
    print(f"Generated continuation: {generated_text}")
    
    print("\nModel upgrade and long context handling successful!")

def main():
    # Run examples
    try:
        basic_usage_example()
        knowledge_distillation_example()
        domain_specific_data_example()
        computational_efficiency_example()
        adaptive_reasoning_example()
        combined_example()
        example_mcts_reasoning()
        example_upgrade_to_1_3b_with_32k_context()
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
