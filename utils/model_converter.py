#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility script for converting models between different formats
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Import model modules
from model.valkyrie_llm import ValkyrieLLM

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
    parser = argparse.ArgumentParser(description="Convert ValkyrieLLM models between formats")
    
    # Basic arguments
    parser.add_argument("--input_model", type=str, required=True, help="Path to input model")
    parser.add_argument("--output_model", type=str, required=True, help="Path to output model")
    parser.add_argument("--input_format", type=str, default="pytorch", choices=["pytorch", "safetensors", "onnx"], help="Input model format")
    parser.add_argument("--output_format", type=str, default="pytorch", choices=["pytorch", "safetensors", "onnx"], help="Output model format")
    parser.add_argument("--quantize", action="store_true", help="Quantize the model")
    parser.add_argument("--quantization_method", type=str, default="dynamic", choices=["dynamic", "static", "aware"], help="Quantization method")
    parser.add_argument("--quantization_bits", type=int, default=8, choices=[8, 4], help="Number of bits for quantization")
    
    # Parse arguments
    args = parser.parse_args()
    
    return args

def load_model(model_path, model_format):
    """
    Load model from file
    
    Args:
        model_path: Path to model file
        model_format: Model format
        
    Returns:
        model: Loaded model
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model from {model_path} in {model_format} format")
    
    if model_format == "pytorch":
        # Load PyTorch model
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
    elif model_format == "safetensors":
        try:
            from safetensors.torch import load_file
            
            # Load safetensors model
            state_dict = load_file(model_path)
            
            # Create model and load state dict
            model = ValkyrieLLM()
            model.load_state_dict(state_dict)
            
            return model
        except ImportError:
            logger.error("safetensors package not found. Please install it with: pip install safetensors")
            sys.exit(1)
    elif model_format == "onnx":
        try:
            import onnx
            import onnxruntime
            
            # Load ONNX model
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            
            # Create PyTorch model
            model = ValkyrieLLM()
            
            # This is a placeholder - actual ONNX to PyTorch conversion would be more complex
            logger.warning("ONNX to PyTorch conversion is not fully implemented")
            
            return model
        except ImportError:
            logger.error("onnx or onnxruntime package not found. Please install them with: pip install onnx onnxruntime")
            sys.exit(1)
    else:
        raise ValueError(f"Unsupported model format: {model_format}")

def save_model(model, model_path, model_format, quantize=False, quantization_method="dynamic", quantization_bits=8):
    """
    Save model to file
    
    Args:
        model: Model to save
        model_path: Path to save model
        model_format: Model format
        quantize: Whether to quantize the model
        quantization_method: Quantization method
        quantization_bits: Number of bits for quantization
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Quantize model if requested
    if quantize:
        logger.info(f"Quantizing model using {quantization_method} method with {quantization_bits} bits")
        
        # Make sure model is in eval mode
        model.eval()
        
        if quantization_method == "dynamic":
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8 if quantization_bits == 8 else torch.qint4
            )
            model = quantized_model
        elif quantization_method == "static":
            logger.warning("Static quantization not fully implemented")
        elif quantization_method == "aware":
            logger.warning("Quantization-aware training not fully implemented")
    
    logger.info(f"Saving model to {model_path} in {model_format} format")
    
    if model_format == "pytorch":
        # Save PyTorch model
        torch.save(model.state_dict(), model_path)
    elif model_format == "safetensors":
        try:
            from safetensors.torch import save_file
            
            # Save safetensors model
            state_dict = model.state_dict()
            save_file(state_dict, model_path)
        except ImportError:
            logger.error("safetensors package not found. Please install it with: pip install safetensors")
            sys.exit(1)
    elif model_format == "onnx":
        try:
            import onnx
            import torch.onnx
            
            # Create dummy input
            dummy_input = torch.zeros(1, 16, dtype=torch.long)
            
            # Export model to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                model_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"}
                }
            )
        except ImportError:
            logger.error("onnx package not found. Please install it with: pip install onnx")
            sys.exit(1)
    else:
        raise ValueError(f"Unsupported model format: {model_format}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Load model
    model = load_model(args.input_model, args.input_format)
    
    # Save model
    save_model(
        model=model,
        model_path=args.output_model,
        model_format=args.output_format,
        quantize=args.quantize,
        quantization_method=args.quantization_method,
        quantization_bits=args.quantization_bits
    )
    
    logger.info("Model conversion complete!")

if __name__ == "__main__":
    main() 