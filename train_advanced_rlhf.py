import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from dataclasses import asdict
import time
from tqdm import tqdm
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

# Import configuration classes
from model.reinforcement.config import RLConfig, RLHFConfig
from model.reinforcement.rlhf_math_integration import RLHFMathConfig
from model.reinforcement.advanced_rlhf import AdvancedRLHFConfig, AdvancedRLHFIntegration
from model.nlp.natural_language_understanding import NLUConfig
from model.nlp.semantic_parser import SemanticParserConfig
from model.reasoning.logical_reasoner import LogicalReasoningConfig
from model.reasoning.causal_inference import CausalInferenceConfig
from model.constitutional_ai import ConstitutionalAIConfig

# Model imports
from model.logical_nanogpt import LogicalGPT
from model.core_model import EnhancedLanguageModel

# Utilities
from utils.enhanced_memory_manager import EnhancedMemoryManager
from utils.training_efficiency import TrainingEfficiencyManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('advanced_rlhf_training.log', mode='a'),
    ]
)

# Create logs directory and add file handler
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True, parents=True)
file_handler = logging.FileHandler(os.path.join('logs', f'advanced_rlhf_{int(time.time())}.log'))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced RLHF training for NanoGPT with comprehensive reasoning capabilities")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large", "xl"], 
                        help="Model size if not loading from path")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs")
    
    # Data parameters
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--preference_data_path", type=str, required=True, help="Path to preference data")
    parser.add_argument("--math_data_path", type=str, default=None, help="Path to math reasoning data")
    parser.add_argument("--logical_data_path", type=str, default=None, help="Path to logical reasoning data")
    parser.add_argument("--causal_data_path", type=str, default=None, help="Path to causal reasoning data")
    parser.add_argument("--eval_data_path", type=str, default=None, help="Path to evaluation data")
    
    # RLHF parameters
    parser.add_argument("--rl_algorithm", type=str, default="dpo", choices=["ppo", "dpo", "expert_iteration"],
                        help="RL algorithm to use")
    parser.add_argument("--use_recursive_rlhf", action="store_true", help="Use recursive RLHF")
    parser.add_argument("--recursive_depth", type=int, default=2, help="Depth for recursive RLHF")
    parser.add_argument("--use_multi_agent_debate", action="store_true", help="Use multi-agent debate")
    parser.add_argument("--num_debate_agents", type=int, default=3, help="Number of agents for debate")
    parser.add_argument("--use_reward_ensemble", action="store_true", help="Use reward model ensemble")
    parser.add_argument("--num_reward_models", type=int, default=3, help="Number of reward models in ensemble")
    
    # Component enablement
    parser.add_argument("--use_math_reasoning", action="store_true", help="Enable mathematical reasoning")
    parser.add_argument("--use_logical_reasoning", action="store_true", help="Enable logical reasoning")
    parser.add_argument("--use_causal_inference", action="store_true", help="Enable causal inference")
    parser.add_argument("--use_nlu", action="store_true", help="Enable NLU components")
    parser.add_argument("--use_constitutional_ai", action="store_true", help="Enable constitutional AI")
    
    # Component weights
    parser.add_argument("--language_weight", type=float, default=0.25, help="Weight for language modeling")
    parser.add_argument("--math_weight", type=float, default=0.20, help="Weight for mathematical reasoning")
    parser.add_argument("--logical_weight", type=float, default=0.20, help="Weight for logical reasoning")
    parser.add_argument("--causal_weight", type=float, default=0.15, help="Weight for causal inference")
    parser.add_argument("--constitutional_weight", type=float, default=0.20, help="Weight for constitutional alignment")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save model every X steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate model every X steps")
    parser.add_argument("--log_steps", type=int, default=100, help="Log metrics every X steps")
    
    # Distributed training
    parser.add_argument("--use_distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(args):
    """Create or load model based on arguments."""
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        model = EnhancedLanguageModel.from_pretrained(args.model_path)
    else:
        logger.info(f"Creating new {args.model_size} model")
        model_configs = {
            "small": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
            "base": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
            "large": {"hidden_size": 1536, "num_hidden_layers": 32, "num_attention_heads": 24},
            "xl": {"hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 32},
        }
        config = model_configs[args.model_size]
        model = EnhancedLanguageModel(**config)
    
    return model


def create_tokenizer(args, model):
    """Create or load tokenizer based on model."""
    if hasattr(model, "tokenizer"):
        return model.tokenizer
    
    from transformers import AutoTokenizer
    
    if args.model_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def create_dataloaders(args, tokenizer):
    """Create dataloaders for training and evaluation."""
    # This is a placeholder - in a real implementation, you would:
    # 1. Load datasets from the provided paths
    # 2. Create appropriate dataset classes
    # 3. Create and return dataloaders
    
    # For now, return dummy dataloaders
    class DummyDataset(Dataset):
        def __init__(self, size=1000):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 50257, (128,)),
                "attention_mask": torch.ones(128),
                "labels": torch.randint(0, 50257, (128,)),
                "chosen_input_ids": torch.randint(0, 50257, (128,)),
                "chosen_attention_mask": torch.ones(128),
                "rejected_input_ids": torch.randint(0, 50257, (128,)),
                "rejected_attention_mask": torch.ones(128),
            }
    
    train_dataset = DummyDataset()
    preference_dataset = DummyDataset()
    eval_dataset = DummyDataset(size=100)
    
    # Create specialized datasets if paths provided
    math_dataset = DummyDataset(size=500) if args.math_data_path else None
    logical_dataset = DummyDataset(size=500) if args.logical_data_path else None
    causal_dataset = DummyDataset(size=500) if args.causal_data_path else None
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    preference_dataloader = DataLoader(preference_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size) if eval_dataset else None
    
    # Create specialized dataloaders
    math_dataloader = DataLoader(math_dataset, batch_size=args.batch_size) if math_dataset else None
    logical_dataloader = DataLoader(logical_dataset, batch_size=args.batch_size) if logical_dataset else None
    causal_dataloader = DataLoader(causal_dataset, batch_size=args.batch_size) if causal_dataset else None
    
    return {
        "train": train_dataloader,
        "preference": preference_dataloader,
        "eval": eval_dataloader,
        "math": math_dataloader,
        "logical": logical_dataloader,
        "causal": causal_dataloader
    }


def create_advanced_rlhf_config(args):
    """Create advanced RLHF configuration from arguments."""
    # Create base RL config
    rl_config = RLConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        
        # Set appropriate flags based on algorithm
        use_ppo=args.rl_algorithm == "ppo",
        use_dpo=args.rl_algorithm == "dpo",
        use_expert_iteration=args.rl_algorithm == "expert_iteration",
    )
    
    # Create RLHF config
    rlhf_config = RLHFConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rl_algorithm=args.rl_algorithm,
    )
    
    # Create math RLHF config if enabled
    rlhf_math_config = RLHFMathConfig(
        use_math_reward_bonus=args.use_math_reasoning,
    )
    
    # Create NLU config if enabled
    nlu_config = NLUConfig() if args.use_nlu else None
    
    # Create reasoning configs if enabled
    logical_reasoning_config = LogicalReasoningConfig() if args.use_logical_reasoning else None
    causal_inference_config = CausalInferenceConfig() if args.use_causal_inference else None
    
    # Create constitutional AI config if enabled
    constitutional_ai_config = ConstitutionalAIConfig() if args.use_constitutional_ai else None
    
    # Create component weights
    component_weights = {
        "language_modeling": args.language_weight,
        "mathematical_reasoning": args.math_weight,
        "logical_reasoning": args.logical_weight,
        "causal_inference": args.causal_weight,
        "constitutional_alignment": args.constitutional_weight
    }
    
    # Create advanced RLHF config
    advanced_config = AdvancedRLHFConfig(
        rl_config=rl_config,
        rlhf_config=rlhf_config,
        rlhf_math_config=rlhf_math_config,
        
        # Set component configs
        nlu_config=nlu_config or NLUConfig(),
        logical_reasoning_config=logical_reasoning_config or LogicalReasoningConfig(),
        causal_inference_config=causal_inference_config or CausalInferenceConfig(),
        constitutional_ai_config=constitutional_ai_config or ConstitutionalAIConfig(),
        
        # Set component enablement
        use_nlu=args.use_nlu,
        use_logical_reasoning=args.use_logical_reasoning,
        use_causal_inference=args.use_causal_inference,
        use_constitutional_ai=args.use_constitutional_ai,
        
        # Set advanced RLHF parameters
        use_recursive_rlhf=args.use_recursive_rlhf,
        recursive_depth=args.recursive_depth,
        use_multi_agent_debate=args.use_multi_agent_debate,
        num_debate_agents=args.num_debate_agents,
        use_reward_ensemble=args.use_reward_ensemble,
        num_reward_models=args.num_reward_models,
        
        # Set component weights
        component_weights=component_weights,
        
        # Set training parameters
        num_epochs=args.num_epochs,
        log_frequency=args.log_steps,
        
        # Set distributed training
        use_distributed=args.use_distributed
    )
    
    return advanced_config


def train(args, model, tokenizer, dataloaders):
    """Train the model with advanced RLHF."""
    logger.info("Starting advanced RLHF training")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save arguments
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create advanced RLHF config
    advanced_config = create_advanced_rlhf_config(args)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(advanced_config), f, indent=2)
    
    # Create device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create advanced RLHF integration
    rlhf_integration = AdvancedRLHFIntegration(
        model=model,
        tokenizer=tokenizer,
        config=advanced_config,
        device=device
    )
    
    # Define training callback
    def training_callback(epoch, model, metrics):
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": rlhf_integration.optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Log metrics
        logger.info(f"Epoch {epoch+1} metrics:")
        for component, component_metrics in metrics.items():
            if f"epoch_{epoch}" in component_metrics:
                logger.info(f"  {component}: {component_metrics[f'epoch_{epoch}']}")
    
    # Train with advanced RLHF
    metrics = rlhf_integration.train(
        train_dataloader=dataloaders["preference"],  # Use preference data for RLHF
        num_epochs=args.num_epochs,
        eval_dataloader=dataloaders["eval"],
        math_eval_dataloader=dataloaders["math"],
        logical_eval_dataloader=dataloaders["logical"],
        nlu_eval_dataloader=dataloaders["eval"],  # Reuse general eval for NLU
        callback=training_callback
    )
    
    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(advanced_config),
        "metrics": metrics
    }, final_model_path)
    
    logger.info(f"Saved final model to {final_model_path}")
    
    # Save final metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Advanced RLHF training completed")
    
    return model, metrics


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create or load model
    model = create_model(args)
    
    # Create or load tokenizer
    tokenizer = create_tokenizer(args, model)
    
    # Create dataloaders
    dataloaders = create_dataloaders(args, tokenizer)
    
    # Train with advanced RLHF
    model, metrics = train(args, model, tokenizer, dataloaders)
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main() 