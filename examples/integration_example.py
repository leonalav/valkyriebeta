#!/usr/bin/env python3
"""
Integration example for testing the ValkyrieLLM model with GNN components.
"""
import torch
import logging
import sys
import os
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to be able to import model package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core models
from model.core_model import CoreModel

def main():
    """Main function for the integration example."""
    try:
        # Create a simple model instance
        logger.info("Creating CoreModel instance...")
        model = CoreModel(
            vocab_size=50000,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            max_seq_length=1024,
            dropout=0.1
        )
        
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create sample input data
        batch_size = 2
        seq_len = 32
        
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        
        # Forward pass through the model
        logger.info("Running forward pass...")
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Print output shapes
        logger.info(f"Output logits shape: {outputs['logits'].shape}")
        logger.info(f"Output hidden states shape: {outputs['hidden_states'].shape}")
        
        logger.info("Integration example completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in integration example: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    start_time = time.time()
    success = main()
    end_time = time.time()
    
    logger.info(f"Execution time: {end_time - start_time:.2f} seconds")
    sys.exit(0 if success else 1) 