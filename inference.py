#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for ValkyrieLLM
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Import model modules
from model.valkyrie_llm import ValkyrieLLM
from data.nanogpt_data import TokenizerManager

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run inference with ValkyrieLLM")
    
    # Basic arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to model file")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for generation")
    parser.add_argument("--prompt_file", type=str, default=None, help="File containing text prompt")
    parser.add_argument("--output_file", type=str, default=None, help="File to save generated text")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated sequence")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--use_reasoning", action="store_true", help="Use reasoning capabilities")
    parser.add_argument("--reasoning_type", type=str, default="adaptive", help="Reasoning type (tree, recursive, neural_symbolic, knowledge, mcts, adaptive)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check that either prompt or prompt_file is provided
    if args.prompt is None and args.prompt_file is None:
        parser.error("Either --prompt or --prompt_file must be provided")
        
    return args

def load_model(model_path):
    """
    Load model from file
    
    Args:
        model_path: Path to model file
        
    Returns:
        model: Loaded model
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from {model_path}")
    
    # Load model
    if os.path.isfile(model_path):
        # Load state dict
        state_dict = torch.load(model_path, map_location="cpu")
        
        # Check if it's a checkpoint or state dict
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            # It's a checkpoint
            state_dict = state_dict["model_state_dict"]
            
        # Create model and load state dict
        model = ValkyrieLLM()
        model.load_state_dict(state_dict)
        
        return model
    else:
        raise ValueError(f"Model file not found: {model_path}")

def get_prompt(args):
    """
    Get prompt from arguments
    
    Args:
        args: Command-line arguments
        
    Returns:
        prompt: Text prompt
    """
    if args.prompt is not None:
        return args.prompt
    elif args.prompt_file is not None:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        raise ValueError("Either prompt or prompt_file must be provided")

def generate_text(model, prompt, args):
    """
    Generate text using the model
    
    Args:
        model: Model to use
        prompt: Text prompt
        args: Command-line arguments
        
    Returns:
        generated_text: Generated text
    """
    logger = logging.getLogger(__name__)
    
    # Create tokenizer
    tokenizer_manager = TokenizerManager(vocab_size=model.vocab_size)
    tokenizer = tokenizer_manager.get_tokenizer()
    
    # Enable reasoning if requested
    if args.use_reasoning:
        logger.info(f"Enabling {args.reasoning_type} reasoning")
        model.enable_reasoning(reasoning_type=args.reasoning_type)
        
    # Tokenize prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    
    # Move model and inputs to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_ids = input_ids.to(device)
    
    # Set model to eval mode
    model.eval()
    
    # Generate text
    logger.info("Generating text...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=args.max_length,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        
    # Decode generated text
    generated_text = tokenizer.decode(output_ids[0].tolist())
    
    return generated_text

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Load model
    model = load_model(args.model_path)
    
    # Get prompt
    prompt = get_prompt(args)
    
    # Generate text
    generated_text = generate_text(model, prompt, args)
    
    # Print generated text
    print("\nGenerated text:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)
    
    # Save generated text if requested
    if args.output_file is not None:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(generated_text)
        logger.info(f"Generated text saved to {args.output_file}")

if __name__ == "__main__":
    main() 