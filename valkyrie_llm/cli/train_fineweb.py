#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for training Valkyrie LLM on FineWeb dataset.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import time

# Import necessary modules
from valkyrie_llm.training.configs.model_3b import (
    model_config,
    training_config,
    memory_config,
    training_params,
    architecture_params
)
from valkyrie_llm.training.training_engine import TrainingEngine
from valkyrie_llm.training.model_setup import setup_model
from valkyrie_llm.training.evaluation import evaluate_reasoning_capabilities
from valkyrie_llm.utils.setup_utils import (
    set_seed,
    setup_logging,
    setup_tokenizer
)
from valkyrie_llm.data.fineweb import (
    FineWebDataset,
    setup_fineweb_dataloader
)
from valkyrie_llm.utils.tpu_utils import setup_tpu_strategy

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Valkyrie LLM on FineWeb sample-10BT dataset")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb",
                        help="HuggingFace dataset name for FineWeb")
    parser.add_argument("--dataset_config", type=str, default="sample-10BT",
                        help="Dataset config name (e.g. 'sample-10BT')")
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Validation split ratio (default: 0.05)")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--use_pretrained", action="store_true",
                        help="Whether to initialize with pretrained model")
    parser.add_argument("--use_rwkv", action="store_true",
                        help="Whether to use RWKV architecture")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="output/fineweb",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--experiment_name", type=str, default="valkyrie_fineweb",
                        help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--learning_rate", type=float, default=6e-4,
                        help="Peak learning rate for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay for AdamW optimizer")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Linear warmup steps")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for gradient clipping")
    
    # Advanced training options
    parser.add_argument("--use_mixed_precision", action="store_true",
                        help="Whether to use mixed precision training (fp16)")
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="Whether to use Flash Attention for efficient computation")
    parser.add_argument("--use_gradient_checkpointing", action="store_true",
                        help="Whether to use gradient checkpointing to save memory")
    parser.add_argument("--use_distributed", action="store_true",
                        help="Whether to use distributed training")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    parser.add_argument("--use_tpu", action="store_true",
                        help="Whether to use TPU for training (Kaggle)")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to use (for testing)")
    
    # Evaluation and features
    parser.add_argument("--evaluate_reasoning", action="store_true",
                        help="Whether to evaluate reasoning capabilities")
    parser.add_argument("--use_rlhf", action="store_true",
                        help="Whether to use RLHF training")
    parser.add_argument("--use_domain_adaptation", action="store_true",
                        help="Whether to use domain adaptation")
    
    return parser.parse_args()

def main():
    """Main function for FineWeb training CLI"""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, f"{args.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Starting Valkyrie LLM training on FineWeb sample-10BT dataset")
    logger.info(f"Arguments: {args}")
    
    # Set up TPU if requested
    if args.use_tpu:
        strategy, is_tpu_available = setup_tpu_strategy()
        logger.info(f"TPU strategy: {strategy}, TPU available: {is_tpu_available}")
        if not is_tpu_available:
            logger.warning("TPU requested but not available, falling back to CPU/GPU")
            args.use_tpu = False
    else:
        is_tpu_available = False
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up tokenizer
    tokenizer = setup_tokenizer(args)
    
    # Update training parameters from args
    training_params["learning_rate"] = args.learning_rate
    training_params["weight_decay"] = args.weight_decay
    training_params["warmup_steps"] = args.warmup_steps
    training_params["num_train_epochs"] = args.num_train_epochs
    training_params["batch_size"] = args.batch_size
    training_params["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    training_params["max_grad_norm"] = args.max_grad_norm
    
    # Update memory config from args
    memory_config.use_gradient_checkpointing = args.use_gradient_checkpointing
    memory_config.use_flash_attention = args.use_flash_attention
    
    # Update training config
    training_config.use_mixed_precision = args.use_mixed_precision
    training_config.use_distributed = args.use_distributed
    training_config.local_rank = args.local_rank
    
    # Set up model
    if args.use_tpu and is_tpu_available:
        with strategy.scope():
            model = setup_model(
                args=args,
                model_config=model_config,
                tokenizer=tokenizer,
                training_config=training_config,
                architecture_params=architecture_params
            )
    else:
        model = setup_model(
            args=args,
            model_config=model_config,
            tokenizer=tokenizer,
            training_config=training_config,
            architecture_params=architecture_params
        )
    
    # Set up FineWeb dataloader
    train_dataloader = setup_fineweb_dataloader(
        tokenizer=tokenizer,
        block_size=model_config.max_seq_len,
        batch_size=training_params["batch_size"],
        max_examples=args.max_examples,
        dataset_name=args.dataset_name,
        config_name=args.dataset_config
    )
    
    # Set up validation dataloader (using a portion of training data)
    train_size = int((1 - args.val_split) * len(train_dataloader.dataset))
    val_size = len(train_dataloader.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataloader.dataset, [train_size, val_size]
    )
    
    # Recreate train dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_params["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Create validation dataloader
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_params["batch_size"] // 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    # Create training engine
    engine = TrainingEngine(
        model=model,
        training_config=training_config,
        tokenizer=tokenizer,
        memory_config=memory_config
    )
    
    # Set up optimizer and scheduler
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
    start_time = time.time()
    logger.info("Starting training")
    
    # TPU-specific training
    if args.use_tpu and is_tpu_available:
        # Import PyTorch XLA libraries for TPU
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        
        # Create parallel loader for TPU
        train_loader = pl.ParallelLoader(train_dataloader, [xm.xla_device()]).per_device_loader(xm.xla_device())
        val_loader = pl.ParallelLoader(val_dataloader, [xm.xla_device()]).per_device_loader(xm.xla_device())
        
        # Initialize engine for TPU training
        engine.init_tpu_training()
        
        # Train with TPU
        results = engine.train_on_tpu(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
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
            epochs=training_params["num_train_epochs"],
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            max_grad_norm=training_params["max_grad_norm"],
            gradient_accumulation_steps=training_params["gradient_accumulation_steps"]
        )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
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
    final_model_path = os.path.join(args.output_dir, f"{args.experiment_name}_final.pt")
    if args.use_tpu and is_tpu_available:
        import torch_xla.core.xla_model as xm
        xm.save(model.state_dict(), final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    
    return results

if __name__ == "__main__":
    main() 