"""
Example script demonstrating the mathematical precision enhancements.

This script shows how to:
1. Create a model with enhanced numerical precision
2. Use the formal verification capabilities
3. Apply precision-adaptive operations to mathematical computations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import sys
import os

# Add parent directory to path to import model modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import (
    GPTConfig,
    NumericalPrecisionConfig,
    FormalVerificationConfig,
    create_precision_enhanced_gpt,
    EnhancedMathematicalReasoning
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Mathematical precision enhancement example')
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of the model')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--use_double_precision', action='store_true', help='Use double precision for critical operations')
    parser.add_argument('--use_kahan_summation', action='store_true', help='Use Kahan summation for more accurate addition')
    parser.add_argument('--verify_reasoning', action='store_true', help='Demonstrate formal verification')
    return parser.parse_args()

def demonstrate_precision_improvements(model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Demonstrate precision improvements on mathematical operations"""
    logger.info("Demonstrating precision improvements on mathematical operations")
    
    # Create some test data
    batch_size = 2
    seq_len = 16
    hidden_size = model.config.n_embd
    vocab_size = model.config.vocab_size
    
    # Create input with some extreme values to test numerical stability
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Create attention mask
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    
    # Process through model
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
        hidden_states = output['hidden_states']
    
    # Check for NaN/Inf values in output
    has_nan = torch.isnan(hidden_states).any()
    has_inf = torch.isinf(hidden_states).any()
    
    logger.info(f"Output contains NaN: {has_nan}")
    logger.info(f"Output contains Inf: {has_inf}")
    
    # Demonstrate precision-adaptive operations
    logger.info("Demonstrating precision-adaptive operations")
    
    # Create a standalone enhancement module for testing
    enhancement = EnhancedMathematicalReasoning(
        hidden_size=hidden_size,
        precision_config=NumericalPrecisionConfig(
            use_double_precision=True,
            use_kahan_summation=True
        )
    ).to(device)
    
    # Create test data with extreme values
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Add some potentially problematic values
    x[0, 0] = 1e-10  # Very small values
    x[0, 1] = 1e10   # Very large values
    x[1, 0] = float('nan')  # NaN values (will be handled by verification)
    x[1, 1] = float('inf')  # Inf values (will be handled by verification)
    
    # Process through enhancement module
    enhanced_output = enhancement(x, attention_mask=attention_mask)
    
    # Check for NaN/Inf values in enhanced output
    has_nan = torch.isnan(enhanced_output).any()
    has_inf = torch.isinf(enhanced_output).any()
    
    logger.info(f"Enhanced output contains NaN: {has_nan}")
    logger.info(f"Enhanced output contains Inf: {has_inf}")
    
    return enhanced_output

def demonstrate_formal_verification(model, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Demonstrate formal verification of mathematical reasoning"""
    logger.info("Demonstrating formal verification of mathematical reasoning")
    
    # Create a standalone enhancement module with verification capabilities
    hidden_size = model.config.n_embd
    enhancement = EnhancedMathematicalReasoning(
        hidden_size=hidden_size,
        precision_config=NumericalPrecisionConfig(use_double_precision=True),
        verification_config=FormalVerificationConfig(
            hidden_size=hidden_size,
            quantify_uncertainty=True
        )
    ).to(device)
    
    # Create test data for verification
    batch_size = 2
    seq_len = 16
    num_premises = 2
    
    # Create premises, conclusion, and reasoning steps
    premises = torch.randn(batch_size, num_premises, seq_len, hidden_size, device=device)
    conclusion = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Create reasoning steps (a valid reasoning path)
    reasoning_steps = []
    current = premises[:, 0].clone()  # Start with first premise
    
    # Create 3 reasoning steps
    for i in range(3):
        # Move slightly toward conclusion
        alpha = (i + 1) / 4  # 0.25, 0.5, 0.75
        next_step = (1 - alpha) * current + alpha * conclusion
        reasoning_steps.append(next_step)
        current = next_step
    
    # Process through enhancement module with verification
    enhanced_output, verification_details = enhancement(
        premises[:, 0],  # Use first premise as input
        verify_reasoning=True,
        premises=premises,
        conclusion=conclusion,
        reasoning_steps=reasoning_steps
    )
    
    # Print verification results
    logger.info(f"Verification result: {verification_details['is_valid']}")
    logger.info(f"Verification confidence: {verification_details['confidence']:.4f}")
    logger.info(f"Verification details: {verification_details['verification_details']}")
    
    # Now try with invalid reasoning (random steps)
    invalid_steps = [torch.randn_like(premises[:, 0]) for _ in range(3)]
    
    # Process through enhancement module with verification
    _, invalid_verification = enhancement(
        premises[:, 0],
        verify_reasoning=True,
        premises=premises,
        conclusion=conclusion,
        reasoning_steps=invalid_steps
    )
    
    # Print verification results for invalid reasoning
    logger.info(f"Invalid verification result: {invalid_verification['is_valid']}")
    logger.info(f"Invalid verification confidence: {invalid_verification['confidence']:.4f}")
    logger.info(f"Invalid verification details: {invalid_verification['verification_details']}")

def main():
    try:
        args = parse_args()
        
        # Configure precision settings
        precision_config = NumericalPrecisionConfig(
            use_double_precision=args.use_double_precision,
            use_kahan_summation=args.use_kahan_summation
        )
        
        # Configure verification settings
        verification_config = FormalVerificationConfig(
            hidden_size=args.hidden_size,
            numerical_precision=precision_config,
            quantify_uncertainty=True
        )
        
        # Create model configuration
        model_config = GPTConfig(
            block_size=1024,  # Add block_size parameter
            vocab_size=50257,  # Add vocab_size parameter
            n_layer=args.num_layers,
            n_head=args.num_heads,
            n_embd=args.hidden_size
        )
        
        # Create precision-enhanced model
        logger.info("Creating precision-enhanced model")
        model = create_precision_enhanced_gpt(
            model_config,
            precision_config=precision_config,
            verification_config=verification_config
        )
        
        # Move model to appropriate device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        model = model.to(device)
        
        # Demonstrate precision improvements
        demonstrate_precision_improvements(model, device)
        
        # Demonstrate formal verification if requested
        if args.verify_reasoning:
            demonstrate_formal_verification(model, device)
        
        logger.info("Example completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 