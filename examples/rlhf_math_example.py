#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use the integrated RLHF and mathematical reasoning capabilities.

This script shows how to:
1. Load a pretrained model
2. Configure RLHF with mathematical reasoning integration
3. Prepare datasets for training
4. Train the model with RLHF and mathematical reasoning
5. Evaluate the model on mathematical reasoning tasks
"""

import os
import sys
import argparse
import torch
import json
import logging
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Add parent directory to path to import from model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.transformer import TransformerModel
from model.config import ModelConfig
from model.reinforcement.config import RLConfig, RLHFConfig
from model.reinforcement.rlhf_math_integration import (
    RLHFMathConfig, 
    RLHFMathIntegration,
    MathReasoningDataset
)
from model.math_reasoning import MathReasoningConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model with RLHF and mathematical reasoning")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large"], 
                        help="Model size if not loading from path")
    
    # Data parameters
    parser.add_argument("--math_data_path", type=str, required=True, help="Path to mathematical reasoning dataset")
    parser.add_argument("--preference_data_path", type=str, required=True, help="Path to preference dataset")
    parser.add_argument("--eval_data_path", type=str, default=None, help="Path to evaluation dataset")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--rl_algorithm", type=str, default="dpo", choices=["ppo", "dpo", "expert_iteration"],
                        help="RL algorithm to use")
    
    # RLHF parameters
    parser.add_argument("--use_math_reward", action="store_true", help="Use math-specific reward model")
    parser.add_argument("--use_curriculum", action="store_true", help="Use curriculum learning")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save model every X steps")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    
    return parser.parse_args()

def create_model_config(args) -> ModelConfig:
    """Create model configuration based on arguments."""
    if args.model_size == "small":
        return ModelConfig(
            vocab_size=50257,  # GPT-2 vocabulary size
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=1024,
            use_tree_reasoning=True,
            use_flash_attention=True
        )
    elif args.model_size == "base":
        return ModelConfig(
            vocab_size=50257,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            max_position_embeddings=1024,
            use_tree_reasoning=True,
            use_flash_attention=True
        )
    elif args.model_size == "large":
        return ModelConfig(
            vocab_size=50257,
            hidden_size=1536,
            num_hidden_layers=36,
            num_attention_heads=24,
            intermediate_size=6144,
            max_position_embeddings=1024,
            use_tree_reasoning=True,
            use_flash_attention=True
        )
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")

def create_rlhf_math_config(args) -> RLHFMathConfig:
    """Create RLHF with math reasoning configuration based on arguments."""
    # Create RL config
    rl_config = RLConfig(
        learning_rate=args.learning_rate,
        use_ppo=args.rl_algorithm == "ppo",
        use_dpo=args.rl_algorithm == "dpo",
        use_expert_iteration=args.rl_algorithm == "expert_iteration",
        dpo_beta=0.1,  # Temperature parameter for DPO
        ppo_epochs=4,
        ppo_batch_size=args.batch_size,
        max_grad_norm=1.0,
        weight_decay=0.01,
        warmup_steps=100
    )
    
    # Create RLHF config
    rlhf_config = RLHFConfig(
        reward_model_hidden_size=1024,
        reward_model_num_layers=2,
        reward_model_lr=args.learning_rate,
        rl_algorithm=args.rl_algorithm,
        rl_epochs=args.num_epochs,
        rl_batch_size=args.batch_size
    )
    
    # Create math reasoning config
    math_config = MathReasoningConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=16,
        use_symbolic_processor=True,
        use_theorem_prover=True,
        use_verification=True
    )
    
    # Create integrated config
    return RLHFMathConfig(
        rl_config=rl_config,
        rlhf_config=rlhf_config,
        math_config=math_config,
        use_math_reward_bonus=args.use_math_reward,
        use_curriculum=args.use_curriculum,
        use_symbolic_verification=True,
        use_numerical_verification=True
    )

def load_math_dataset(data_path: str, tokenizer: Any, batch_size: int) -> DataLoader:
    """Load mathematical reasoning dataset."""
    logger.info(f"Loading math dataset from {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    dataset = MathReasoningDataset(
        problems=data,
        tokenizer=tokenizer,
        max_length=512
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    logger.info(f"Loaded {len(dataset)} math problems")
    return dataloader

def load_preference_dataset(data_path: str, tokenizer: Any, batch_size: int) -> DataLoader:
    """Load preference dataset for RLHF."""
    logger.info(f"Loading preference dataset from {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Split data into chosen and rejected
    chosen_data = []
    rejected_data = []
    
    for item in data:
        chosen_data.append({
            "prompt": item["prompt"],
            "response": item["chosen"]
        })
        rejected_data.append({
            "prompt": item["prompt"],
            "response": item["rejected"]
        })
    
    # Import here to avoid circular imports
    from model.reinforcement.dpo import PreferenceDataset
    
    dataset = PreferenceDataset(
        chosen_data=chosen_data,
        rejected_data=rejected_data,
        tokenizer=tokenizer,
        max_length=512
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    logger.info(f"Loaded {len(dataset)} preference pairs")
    return dataloader

def train_and_evaluate(args):
    """Train and evaluate the model."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or create model
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        model = TransformerModel.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        logger.info(f"Creating new {args.model_size} model")
        config = create_model_config(args)
        model = TransformerModel(config)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    math_dataloader = load_math_dataset(args.math_data_path, tokenizer, args.batch_size)
    preference_dataloader = load_preference_dataset(args.preference_data_path, tokenizer, args.batch_size)
    
    # Load evaluation dataset if provided
    eval_dataloader = None
    if args.eval_data_path:
        eval_dataloader = load_math_dataset(args.eval_data_path, tokenizer, args.batch_size)
    
    # Create RLHF with math reasoning configuration
    rlhf_math_config = create_rlhf_math_config(args)
    
    # Create RLHF with math reasoning integration
    integration = RLHFMathIntegration(
        model=model,
        tokenizer=tokenizer,
        config=rlhf_math_config,
        device=args.device
    )
    
    # Define callback function for saving checkpoints
    def save_checkpoint_callback(epoch, model, metrics):
        if (epoch + 1) % (args.save_steps // args.batch_size) == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            # Save metrics
            with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Train the model
    logger.info("Starting training")
    metrics = integration.train(
        train_dataloader=preference_dataloader,
        num_epochs=args.num_epochs,
        eval_dataloader=None,
        math_eval_dataloader=math_dataloader,
        callback=save_checkpoint_callback
    )
    
    # Save final model
    final_model_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Save final metrics
    with open(os.path.join(final_model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Training completed. Final model saved to {final_model_dir}")
    
    # Evaluate final model if evaluation dataset provided
    if eval_dataloader:
        logger.info("Evaluating final model")
        eval_metrics = integration.evaluate_math(eval_dataloader)
        
        logger.info(f"Evaluation metrics: {eval_metrics}")
        
        # Save evaluation metrics
        with open(os.path.join(final_model_dir, "eval_metrics.json"), "w") as f:
            json.dump(eval_metrics, f, indent=2)

def main():
    """Main function."""
    args = parse_args()
    train_and_evaluate(args)

if __name__ == "__main__":
    main() 