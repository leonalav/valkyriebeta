#!/usr/bin/env python
# Script to train a model using the HuggingFaceFW/fineweb dataset
import os
import sys
import logging
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Train model using HuggingFaceFW/fineweb dataset")
    
    # Basic parameters
    parser.add_argument("--output_dir", type=str, default="output/fineweb", 
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--experiment_name", type=str, default="fineweb_training",
                        help="Name of the experiment")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to pretrained tokenizer")
    
    # Model configuration
    parser.add_argument("--model_type", type=str, default="gpt2",
                        help="Base model architecture to use")
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--num_attention_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--max_seq_length", type=int, default=1024, 
                        help="Maximum sequence length")
    
    # Dataset configuration
    parser.add_argument("--dataset_subset", type=str, default="CC-MAIN-2024-10",
                        help="FineWeb dataset subset to use")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--use_mixed_precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--use_activation_checkpointing", action="store_true",
                        help="Use activation checkpointing to save memory")
    parser.add_argument("--save_steps", type=int, default=10000,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=5000,
                        help="Evaluate every X steps")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare command line arguments for maintrain.py
    cmd_args = [
        # Data source (HuggingFace dataset)
        f"--huggingface_dataset=HuggingFaceFW/fineweb",
        f"--huggingface_subset={args.dataset_subset}",
        "--use_streaming",  # Use streaming mode for large dataset
        f"--data_split=train",
        
        # Model configuration
        f"--model_type={args.model_type}",
        f"--hidden_size={args.hidden_size}",
        f"--num_layers={args.num_layers}",
        f"--num_attention_heads={args.num_attention_heads}",
        f"--max_seq_length={args.max_seq_length}",
        
        # Training parameters
        f"--learning_rate={args.learning_rate}",
        f"--num_train_epochs={args.num_train_epochs}",
        f"--batch_size={args.batch_size}",
        f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
        f"--save_steps={args.save_steps}",
        f"--eval_steps={args.eval_steps}",
        
        # Output configuration
        f"--output_dir={args.output_dir}",
        f"--experiment_name={args.experiment_name}",
        f"--tokenizer_path={args.tokenizer_path}",
    ]
    
    # Add boolean flags if specified
    if args.use_mixed_precision:
        cmd_args.append("--use_mixed_precision")
        
    if args.use_activation_checkpointing:
        cmd_args.append("--use_activation_checkpointing")
    
    # Add performance optimizations for fineweb dataset
    cmd_args.extend([
        "--use_kv_caching",  # Enable KV caching
        "--attention_implementation=flash",  # Use flash attention if available
        "--use_efficient_attention",  # Enable efficient attention
    ])
    
    # Convert arguments to string
    cmd_str = " ".join(cmd_args)
    
    # Print the command being executed
    print(f"Running: python -m training.maintrain {cmd_str}")
    
    # Import and run maintrain directly with the arguments
    # Add current directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    # Set sys.argv for argparse in maintrain.py
    sys.argv = ['training.maintrain'] + cmd_args
    
    # Import and run maintrain
    from training.maintrain import main as train_main
    train_main()

if __name__ == "__main__":
    main() 