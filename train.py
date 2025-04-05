#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main training script for Valkyrie LLM 3B model with 16k context length
"""

import os
import logging
import torch
from pathlib import Path

# Import configurations
from training.configs.model_3b import (
    model_config,
    training_config,
    memory_config,
    training_params,
    architecture_params
)

# Import core training components
from training.training_engine import TrainingEngine
from training.model_setup import setup_model
from training.data_loaders import (
    setup_train_dataloader,
    setup_val_dataloader,
    setup_domain_dataloaders,
    setup_rlhf_dataloader
)
from training.evaluation import evaluate_reasoning_capabilities

# Import utility functions
from utils.setup_utils import (
    parse_args,
    set_seed,
    setup_logging,
    setup_tokenizer
)

logger = logging.getLogger(__name__)

def main():
    """Main training function that orchestrates all components"""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args)
    logger.info("Starting Valkyrie LLM 3B training with 16k context length")
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up tokenizer
    tokenizer = setup_tokenizer(args)
    
    # Set up model with 3B configuration
    model = setup_model(
        args=args,
        model_config=model_config,
        tokenizer=tokenizer,
        training_config=training_config,
        architecture_params=architecture_params
    )
    
    # Set up data loaders with adjusted batch sizes
    train_dataloader = setup_train_dataloader(
        args=args,
        tokenizer=tokenizer,
        training_config=training_config,
        batch_size=training_params["batch_size"]
    )
    
    val_dataloader = setup_val_dataloader(
        args=args,
        tokenizer=tokenizer,
        training_config=training_config,
        batch_size=training_params["batch_size"] // 2
    )
    
    # Set up domain-specific data loaders if enabled
    domain_dataloaders = None
    if args.use_domain_adaptation:
        domain_dataloaders = setup_domain_dataloaders(
            args=args,
            tokenizer=tokenizer,
            training_config=training_config,
            batch_size=training_params["batch_size"] // 2
        )
    
    # Set up RLHF data loader if enabled
    rlhf_dataloader = None
    if args.use_rlhf:
        rlhf_dataloader = setup_rlhf_dataloader(
            args=args,
            tokenizer=tokenizer,
            training_config=training_config,
            batch_size=training_params["batch_size"] // 2
        )
    
    # Create training engine
    engine = TrainingEngine(
        model=model,
        training_config=training_config,
        tokenizer=tokenizer,
        memory_config=memory_config
    )
    
    # Set up optimizer and scheduler with 3B parameters
    engine.setup_optimizer(
        learning_rate=training_params["learning_rate"],
        weight_decay=training_params["weight_decay"],
        beta1=training_params["adam_beta1"],
        beta2=training_params["adam_beta2"],
        epsilon=training_params["adam_epsilon"]
    )
    
    engine.setup_lr_scheduler(
        num_epochs=training_params["num_train_epochs"],
        steps_per_epoch=len(train_dataloader),
        warmup_steps=training_params["warmup_steps"]
    )
    
    # Validate setup
    validation_result = engine.validate_setup()
    if validation_result.errors:
        for error in validation_result.errors:
            logger.error(f"Validation error: {error}")
        raise ValueError("Setup validation failed")
    
    # Optimize training setup
    engine.optimize_training_setup()
    
    # Training loop
    if args.use_rlhf:
        # Train with RLHF
        results = engine.train_with_rlhf(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            rlhf_dataloader=rlhf_dataloader,
            epochs=training_params["num_train_epochs"],
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            max_grad_norm=training_params["max_grad_norm"],
            gradient_accumulation_steps=training_params["gradient_accumulation_steps"]
        )
    else:
        # Standard training
        results = engine.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            domain_dataloaders=domain_dataloaders,
            epochs=training_params["num_train_epochs"],
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            max_grad_norm=training_params["max_grad_norm"],
            gradient_accumulation_steps=training_params["gradient_accumulation_steps"]
        )
    
    # Evaluate reasoning capabilities if enabled
    if args.evaluate_reasoning:
        reasoning_results = evaluate_reasoning_capabilities(
            model=model,
            tokenizer=tokenizer,
            args=args
        )
        logger.info(f"Reasoning evaluation results: {reasoning_results}")
    
    # Prepare model for inference
    model = engine.prepare_for_inference()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f"{args.experiment_name}_3b_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed. Final 3B model saved to {final_model_path}")
    
    return results

if __name__ == "__main__":
    main() 