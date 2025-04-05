#!/usr/bin/env python3
"""
Setup and Run Advanced RLHF Training

This script automates the process of setting up the environment and running
advanced RLHF training for enhanced reasoning capabilities.

Usage:
    python setup_and_run_rlhf.py --help
"""

import os
import sys
import subprocess
import argparse
import logging
import json
from pathlib import Path
import time
import shutil
import pkg_resources
from pkg_resources import DistributionNotFound

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup_rlhf.log', mode='a'),
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup and run advanced RLHF training"
    )
    
    # Setup options
    parser.add_argument(
        "--install_dependencies",
        action="store_true",
        help="Install required dependencies"
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Only check if environment is ready without running training"
    )
    parser.add_argument(
        "--verify_cuda",
        action="store_true",
        help="Verify CUDA availability"
    )
    
    # Model and data options
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model to enhance or HuggingFace model name"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./enhanced_model",
        help="Directory to save the enhanced model"
    )
    
    # Training options
    parser.add_argument(
        "--mode",
        type=str,
        default="guided",
        choices=["guided", "auto", "custom"],
        help="Enhancement mode: guided (interactive), auto (automatic), or custom (from config)"
    )
    parser.add_argument(
        "--components",
        type=str,
        nargs="+",
        choices=["math", "logical", "causal", "nlu", "constitutional", "all"],
        default=["all"],
        help="Components to enhance (default: all)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    
    # Advanced options
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to configuration file (for custom mode)"
    )
    parser.add_argument(
        "--use_synthetic_data",
        action="store_true",
        help="Generate synthetic data for enhancement"
    )
    
    # Misc options
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="GPU ID(s) to use (comma-separated)"
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable caching for dependency installation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0",
        "safetensors>=0.3.0",
        "tensorboard>=2.10.0"
    ]
    
    missing_packages = []
    outdated_packages = []
    
    for package_spec in required_packages:
        package_name = package_spec.split(">=")[0]
        required_version = package_spec.split(">=")[1] if ">=" in package_spec else None
        
        try:
            # Check if package is installed
            installed_version = pkg_resources.get_distribution(package_name).version
            
            # Check version if required
            if required_version and installed_version < required_version:
                outdated_packages.append((package_name, installed_version, required_version))
        except DistributionNotFound:
            missing_packages.append(package_name)
    
    return missing_packages, outdated_packages

def install_dependencies(missing_packages, outdated_packages, no_cache=False):
    """Install missing or outdated dependencies."""
    if not missing_packages and not outdated_packages:
        logger.info("All dependencies are already installed and up to date.")
        return True
    
    # Prepare pip command
    pip_cmd = [sys.executable, "-m", "pip", "install"]
    if no_cache:
        pip_cmd.append("--no-cache-dir")
    
    # Install missing packages
    if missing_packages:
        logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
        packages_cmd = pip_cmd + missing_packages
        try:
            subprocess.check_call(packages_cmd)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install missing packages: {str(e)}")
            return False
    
    # Upgrade outdated packages
    if outdated_packages:
        logger.info(f"Upgrading outdated packages: {', '.join([p[0] for p in outdated_packages])}")
        upgrade_packages = [f"{package}>={required}" for package, _, required in outdated_packages]
        upgrade_cmd = pip_cmd + ["--upgrade"] + upgrade_packages
        try:
            subprocess.check_call(upgrade_cmd)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to upgrade outdated packages: {str(e)}")
            return False
    
    logger.info("All dependencies installed successfully.")
    return True

def verify_cuda():
    """Verify CUDA availability and report GPU information."""
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            
            # Get memory info for each GPU
            for i in range(torch.cuda.device_count()):
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9  # Convert to GB
                mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                mem_allocated = torch.cuda.memory_allocated(i) / 1e9
                mem_free = mem_total - mem_reserved
                
                logger.info(f"GPU {i} - {torch.cuda.get_device_name(i)}")
                logger.info(f"  Total memory: {mem_total:.2f} GB")
                logger.info(f"  Reserved memory: {mem_reserved:.2f} GB")
                logger.info(f"  Allocated memory: {mem_allocated:.2f} GB")
                logger.info(f"  Free memory: {mem_free:.2f} GB")
            
            return True
        else:
            logger.warning("CUDA is not available. Training will be very slow on CPU.")
            return False
    except ImportError:
        logger.error("PyTorch is not installed. Cannot verify CUDA.")
        return False
    except Exception as e:
        logger.error(f"Error verifying CUDA: {str(e)}")
        return False

def check_model_path(model_path):
    """Check if model path exists or is a valid HuggingFace model."""
    if os.path.exists(model_path):
        logger.info(f"Model path exists: {model_path}")
        return True
    
    # If not a local path, assume it's a HuggingFace model
    logger.info(f"Model path not found locally. Assuming HuggingFace model: {model_path}")
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path)
        logger.info(f"HuggingFace model exists: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error checking HuggingFace model: {str(e)}")
        return False

def prepare_environment(args):
    """Prepare the environment for RLHF training."""
    # Check and install dependencies if requested
    if args.install_dependencies:
        logger.info("Checking dependencies...")
        missing_packages, outdated_packages = check_dependencies()
        
        if missing_packages or outdated_packages:
            logger.info("Installing required dependencies...")
            if not install_dependencies(missing_packages, outdated_packages, args.no_cache):
                logger.error("Failed to install dependencies. Exiting.")
                return False
    
    # Verify CUDA if requested
    if args.verify_cuda:
        logger.info("Verifying CUDA availability...")
        verify_cuda()
    
    # Check model path if provided
    if args.model_path:
        logger.info(f"Checking model path: {args.model_path}")
        if not check_model_path(args.model_path):
            logger.error(f"Model path is invalid: {args.model_path}")
            return False
    
    # Check data directory if provided
    if args.data_dir and not args.use_synthetic_data:
        if not os.path.exists(args.data_dir):
            logger.warning(f"Data directory does not exist: {args.data_dir}")
            logger.warning("Will use synthetic data instead.")
        else:
            logger.info(f"Data directory exists: {args.data_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Check configuration file if provided
    if args.mode == "custom" and args.config_path:
        if not os.path.exists(args.config_path):
            logger.error(f"Configuration file not found: {args.config_path}")
            return False
        
        try:
            with open(args.config_path, "r") as f:
                json.load(f)  # Just check if it's valid JSON
            logger.info(f"Configuration file is valid: {args.config_path}")
        except Exception as e:
            logger.error(f"Invalid configuration file: {str(e)}")
            return False
    
    logger.info("Environment preparation completed successfully.")
    return True

def run_rlhf_training(args):
    """Run the RLHF training script with provided arguments."""
    logger.info("Starting RLHF training...")
    
    # Set environment variables for GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # Build command for enhance_model_reasoning.py
    cmd = [sys.executable, "enhance_model_reasoning.py"]
    
    # Add arguments
    cmd.extend(["--mode", args.mode])
    cmd.extend(["--model_path", args.model_path])
    cmd.extend(["--output_dir", args.output_dir])
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--num_epochs", str(args.num_epochs)])
    cmd.extend(["--learning_rate", str(args.learning_rate)])
    
    # Add components
    if args.components:
        cmd.append("--components")
        cmd.extend(args.components)
    
    # Add optional arguments
    if args.data_dir:
        cmd.extend(["--data_dir", args.data_dir])
    
    if args.use_synthetic_data:
        cmd.append("--use_synthetic_data")
    
    if args.config_path:
        cmd.extend(["--config_path", args.config_path])
    
    if args.verbose:
        cmd.append("--verbose")
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        logger.info("RLHF training completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running RLHF training: {str(e)}")
        return False
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        return False

def main():
    """Main function to setup and run RLHF training."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=== Setup and Run Advanced RLHF Training ===")
    
    # Prepare environment
    logger.info("Preparing environment...")
    if not prepare_environment(args):
        logger.error("Environment preparation failed. Exiting.")
        return 1
    
    # Exit if check only
    if args.check_only:
        logger.info("Environment check completed. Exiting.")
        return 0
    
    # Check if model path is provided
    if not args.model_path:
        logger.error("Model path is required. Use --model_path to specify a model.")
        return 1
    
    # Run RLHF training
    success = run_rlhf_training(args)
    
    if success:
        logger.info("=== RLHF Training Completed Successfully ===")
        logger.info(f"Enhanced model saved to: {args.output_dir}")
        return 0
    else:
        logger.error("=== RLHF Training Failed ===")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 