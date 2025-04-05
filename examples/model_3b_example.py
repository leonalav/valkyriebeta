#!/usr/bin/env python3
"""
3B Parameter Model Example

This script demonstrates how to instantiate and use the 3B parameter model.
"""

import os
import sys
import logging
import torch
import argparse
from pathlib import Path

# Add parent directory to path to be able to import model package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Example script for 3B parameter model")
    parser.add_argument("--save_model", action="store_true", help="Save the model to disk")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the model")
    parser.add_argument("--small_test", action="store_true", help="Use small model for testing")
    args = parser.parse_args()

    # Import model after parsing arguments
    from model.core_model import CoreModel
    
    # Check available GPU memory before creating the model
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
        free_memory = torch.cuda.memory_reserved(0) / (1024**3)  # in GB
        logger.info(f"GPU has {gpu_memory:.2f} GB total memory, {free_memory:.2f} GB reserved")
        device = torch.device("cuda")
    else:
        logger.info("No GPU available, using CPU")
        device = torch.device("cpu")
    
    # Create the model - either 3B or small test model
    if args.small_test:
        logger.info("Creating small model for testing (125M parameters)")
        model = CoreModel(
            vocab_size=50257,  # GPT-2 vocabulary size
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            max_seq_length=1024
        )
    else:
        logger.info("Creating 3B parameter model")
        model = CoreModel.from_3b_config()
    
    # Log model architecture
    logger.info(f"Model architecture:\n{model}")
    
    # Calculate and log parameter count
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {param_count:,} parameters ({param_count/1_000_000_000:.2f}B)")
    
    # Move to device if possible
    try:
        model = model.to(device)
        logger.info(f"Model moved to {device}")
    except RuntimeError as e:
        logger.error(f"Failed to move model to {device}: {e}")
        logger.info("Falling back to CPU")
        device = torch.device("cpu")
        model = model.to(device)
    
    # Example forward pass with random input
    with torch.no_grad():
        try:
            batch_size = 1
            seq_len = 32  # Small sequence length for testing
            
            logger.info(f"Testing forward pass with batch_size={batch_size}, seq_len={seq_len}")
            input_ids = torch.randint(0, 50000, (batch_size, seq_len), device=device)
            attention_mask = torch.ones((batch_size, seq_len), device=device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logger.info(f"Forward pass successful")
            logger.info(f"Output logits shape: {outputs['logits'].shape}")
            logger.info(f"Output hidden states shape: {outputs['hidden_states'].shape}")
        except RuntimeError as e:
            logger.error(f"Forward pass failed: {e}")
    
    # Save the model if requested
    if args.save_model:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "coremodel_3b.pt")
        
        try:
            # Save just the model parameters for efficiency
            logger.info(f"Saving model parameters to {model_path}")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved successfully")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    logger.info("Example completed")
    
if __name__ == "__main__":
    main() 