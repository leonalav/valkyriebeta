import os
import sys
import logging
import random
import numpy as np
import torch
import argparse
from typing import Dict, Any, Optional

def setup_logging(args):
    """
    Set up structured logging configuration for the training process.
    Uses centralized logging configuration from config/logging_config.py.
    
    Args:
        args: Command line arguments
    """
    from config.logging_config import configure_structured_logging
    
    log_level = getattr(args, 'log_level', 'INFO').upper()
    log_file = getattr(args, 'log_file', None)
    
    # Configure structured logging with centralized configuration
    logger = configure_structured_logging(
        name=__name__,
        level=log_level,
        log_file=log_file,
        enable_json=True,
        enable_console=True,
        enable_file=log_file is not None,
        enable_metrics=True
    )
    
    # Add additional context to logs
    logger = logger.bind(
        experiment_name=getattr(args, 'experiment_name', 'default'),
        model_type=getattr(args, 'model_type', 'unknown')
    )
    
    logger.info("Structured logging initialized")
    
    return logger

def parse_args():
    """
    Parse command line arguments for the training script.
    
    Returns:
        args: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Train Valkyrie LLM with advanced features")
    
    # Basic training parameters
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save model checkpoints and logs")
    parser.add_argument("--experiment_name", type=str, default="valkyrie", help="Name of the experiment")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Data parameters
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="valkyrie", help="Model type")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Advanced features
    parser.add_argument("--enable_reasoning", action="store_true", help="Enable reasoning capabilities")
    parser.add_argument("--reasoning_type", type=str, default="adaptive", help="Type of reasoning to use")
    parser.add_argument("--use_moe", action="store_true", help="Use Mixture of Experts")
    parser.add_argument("--use_domain_adaptation", action="store_true", help="Use domain adaptation")
    parser.add_argument("--domain_data_paths", type=str, nargs="+", help="Paths to domain-specific data")
    parser.add_argument("--domain_weights", type=float, nargs="+", help="Weights for domain-specific data")
    
    # RLHF parameters
    parser.add_argument("--use_rlhf", action="store_true", help="Use RLHF")
    parser.add_argument("--rlhf_data_path", type=str, help="Path to RLHF data")
    parser.add_argument("--rlhf_batch_size", type=int, default=4, help="Batch size for RLHF")
    parser.add_argument("--rlhf_epochs", type=int, default=1, help="Number of RLHF epochs")
    parser.add_argument("--reward_model_path", type=str, help="Path to reward model")
    parser.add_argument("--reference_model_path", type=str, help="Path to reference model")
    
    # Training mode
    parser.add_argument("--training_mode", type=str, default="standard", choices=["standard", "rlhf", "quantize"], help="Training mode")
    
    # Evaluation
    parser.add_argument("--evaluate_reasoning", action="store_true", help="Evaluate reasoning capabilities")
    
    # Logging
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--log_file", type=str, help="Path to log file")
    
    return parser.parse_args()

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    if hasattr(torch, 'set_deterministic'):
        torch.set_deterministic(True)
    
    # Set deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger = logging.getLogger(__name__)
    logger.info(f"Random seed set to {seed} for full reproducibility")

def setup_model_config(args):
    """
    Set up model configuration from command line arguments.
    
    Args:
        args: Command line arguments
    
    Returns:
        model_config: Model configuration
    """
    from config import AdvancedModelConfig
    
    # Create model configuration
    model_config = AdvancedModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_length,
        dropout=args.dropout,
        
        # Set advanced features based on arguments
        use_moe=args.use_moe if hasattr(args, 'use_moe') else False,
        use_tree_reasoning=args.enable_reasoning if hasattr(args, 'enable_reasoning') else False,
        use_neural_symbolic=args.enable_reasoning if hasattr(args, 'enable_reasoning') else False,
        use_recursive_reasoning=args.enable_reasoning if hasattr(args, 'enable_reasoning') else False,
        use_knowledge_reasoning=args.enable_reasoning if hasattr(args, 'enable_reasoning') else False,
        use_mcts=args.enable_reasoning if hasattr(args, 'enable_reasoning') else False,
    )
    
    return model_config

def setup_training_config(args):
    """
    Set up training configuration from command line arguments.
    
    Args:
        args: Command line arguments
    
    Returns:
        training_config: Training configuration
    """
    from config import TrainingEfficiencyConfig
    
    # Create training configuration
    training_config = TrainingEfficiencyConfig(
        use_mixed_precision=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') else 1,
    )
    
    return training_config

def setup_tokenizer(args):
    """
    Set up tokenizer for the model.
    
    Args:
        args: Command line arguments
    
    Returns:
        tokenizer: Tokenizer for the model
    """
    from data.tokenizer import EnhancedTokenizer
    
    # Create tokenizer
    tokenizer = EnhancedTokenizer(vocab_size=args.vocab_size)
    
    return tokenizer 