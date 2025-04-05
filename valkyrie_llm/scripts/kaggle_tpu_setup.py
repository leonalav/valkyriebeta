#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for Kaggle TPU training.
This script checks TPU availability and installs necessary dependencies.
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def install_requirements():
    """Install TPU requirements"""
    logger.info("Installing TPU requirements...")
    
    try:
        # Install PyTorch/XLA for TPU
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch_xla[tpu]>=1.12", "-f", "https://storage.googleapis.com/libtpu-releases/index.html"])
        logger.info("PyTorch/XLA installed successfully")
        
        # Install other dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow>=2.8.0"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets>=2.10.0", "transformers>=4.26.0", "sentencepiece>=0.1.97"])
        logger.info("Additional dependencies installed successfully")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def check_tpu_availability():
    """Check if TPU is available"""
    logger.info("Checking TPU availability...")
    
    try:
        import tensorflow as tf
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        logger.info(f"TPU detected: {tpu.master()}")
        
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        logger.info("TPU initialized successfully")
        
        # Check PyTorch/XLA
        import torch_xla.core.xla_model as xm
        logger.info("PyTorch/XLA is available")
        
        return True
    except (ImportError, ValueError) as e:
        logger.error(f"TPU check failed: {e}")
        return False

def setup_kaggle_environment():
    """Set up Kaggle environment for TPU training"""
    logger.info("Setting up Kaggle environment for TPU training...")
    
    # Check if running on Kaggle
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        logger.warning("Not running on Kaggle, skipping setup")
        return False
    
    # Install requirements
    success = install_requirements()
    
    if not success:
        logger.error("Failed to install requirements")
        return False
    
    # Check TPU availability
    tpu_available = check_tpu_availability()
    
    if not tpu_available:
        logger.error("TPU is not available")
        return False
    
    logger.info("Kaggle TPU environment setup successfully")
    return True

def main():
    """Main function"""
    result = setup_kaggle_environment()
    
    if result:
        logger.info("Kaggle TPU setup completed successfully")
        return 0
    else:
        logger.error("Kaggle TPU setup failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 