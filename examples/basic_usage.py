#!/usr/bin/env python3
"""
Example usage of the tree reasoning transformer model.

This script demonstrates how to create, load, and use the transformer model for 
text generation, with and without tree reasoning.
"""

import torch
import logging
import os
import sys
import argparse
from typing import Optional, List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import (
    TransformerModel, 
    ModelConfig, 
    create_transformer_model,
    LanguageModelMCTSIntegration,
    MCTSConfig
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_tokenizer():
    """Set up a tokenizer for the model."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except ImportError:
        logger.error("Please install the transformers library: pip install transformers")
        raise

def create_model(config_path: Optional[str] = None) -> TransformerModel:
    """
    Create a transformer model with the specified configuration.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        TransformerModel instance
    """
    if config_path and os.path.exists(config_path):
        # Load configuration from file
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
    else:
        # Use default configuration for a small model
        config = ModelConfig(
            vocab_size=50257,  # GPT-2 vocabulary size
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=1024,
            use_tree_reasoning=True,
            use_rotary_embeddings=True,
            use_flash_attention=True
        )
    
    # Create model
    model = create_transformer_model(config)
    return model

def generate_text(
    model: TransformerModel,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    use_tree_reasoning: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> str:
    """
    Generate text using the transformer model.
    
    Args:
        model: TransformerModel instance
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Input prompt for generation
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        use_tree_reasoning: Whether to use tree reasoning
        device: Device to use for inference
        
    Returns:
        Generated text
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Calculate attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Generate with model
    with torch.no_grad():
        output_ids, _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            use_tree_reasoning=use_tree_reasoning
        )
    
    # Decode output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def generate_with_mcts(
    model: TransformerModel,
    tokenizer,
    prompt: str,
    max_iterations: int = 50,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Any]:
    """
    Generate a solution using explicit MCTS integration.
    
    Args:
        model: TransformerModel instance
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Problem to solve
        max_iterations: Maximum MCTS iterations
        device: Device to use for inference
        
    Returns:
        Dictionary with solution and reasoning trace
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Configure MCTS
    mcts_config = MCTSConfig(
        max_iterations=max_iterations,
        exploration_weight=1.5,
        rollout_depth=2,
        top_k_candidates=3,
        num_simulations=2,
        enable_visualization=True
    )
    
    # Create MCTS integration
    integration = LanguageModelMCTSIntegration(
        language_model=model,
        tokenizer=tokenizer,
        mcts_config=mcts_config,
        temperature=0.8
    )
    
    # Solve the problem
    solution_text, trace = integration.solve_problem(prompt, max_iterations)
    
    # Return results
    return {
        "solution": solution_text,
        "steps": trace["steps"],
        "value": trace["value"],
        "iterations": trace["iterations"],
        "tree_visualization": trace["tree_visualization"]
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Transformer model with tree reasoning")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt for generation")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--use-mcts", action="store_true", help="Use explicit MCTS integration")
    parser.add_argument("--use-tree-reasoning", action="store_true", help="Use tree reasoning module")
    parser.add_argument("--config", type=str, help="Path to model configuration file")
    parser.add_argument("--model-path", type=str, help="Path to pretrained model")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                      help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Set up tokenizer
    tokenizer = setup_tokenizer()
    
    # Create or load model
    if args.model_path:
        try:
            logger.info(f"Loading model from {args.model_path}")
            model = TransformerModel.from_pretrained(args.model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return
    else:
        logger.info("Creating new model")
        model = create_model(args.config)
    
    # Print model information
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params:,} parameters")
    
    # Generate text
    if args.use_mcts:
        logger.info("Generating with explicit MCTS integration")
        result = generate_with_mcts(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=args.device
        )
        
        # Print results
        print("\n" + "="*50)
        print("PROMPT:")
        print(args.prompt)
        print("\nSOLUTION:")
        print(result["solution"])
        print("\nREASONING STEPS:")
        for i, step in enumerate(result["steps"]):
            print(f"{i+1}. {step}")
        print(f"\nFinal value: {result['value']:.4f}")
        print(f"MCTS Iterations: {result['iterations']}")
        if result.get("tree_visualization"):
            print("\nTREE VISUALIZATION:")
            print(result["tree_visualization"])
    else:
        logger.info(f"Generating text with{' tree reasoning' if args.use_tree_reasoning else 'out tree reasoning'}")
        output_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            use_tree_reasoning=args.use_tree_reasoning,
            device=args.device
        )
        
        # Print results
        print("\n" + "="*50)
        print("PROMPT:")
        print(args.prompt)
        print("\nGENERATED TEXT:")
        print(output_text)

if __name__ == "__main__":
    main() 