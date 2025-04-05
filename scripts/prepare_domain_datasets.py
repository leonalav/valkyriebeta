#!/usr/bin/env python
"""
Domain Dataset Preparation Script

This script uses the domain data bridge to prepare datasets from existing data loaders
for use with the enhanced training pipeline.

Usage:
    python scripts/prepare_domain_datasets.py --config config/domain_bridge_config.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.domain_data_bridge import (
    prepare_all_domains,
    get_available_loaders,
    load_domain_data_for_training
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("domain_preparation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Prepare domain-specific datasets using existing loaders")
    parser.add_argument(
        "--config", 
        type=str,
        default="config/domain_bridge_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Directory to save processed data (overrides config)"
    )
    parser.add_argument(
        "--list_loaders", 
        action="store_true", 
        help="List available loaders and exit"
    )
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        help="Specific domains to process (e.g., math science coding)"
    )
    parser.add_argument(
        "--max_examples", 
        type=int, 
        help="Maximum examples per domain (overrides config)"
    )
    args = parser.parse_args()
    
    if args.list_loaders:
        print("Available loaders by domain:")
        for domain, loaders in get_available_loaders().items():
            print(f"  {domain}:")
            for loader in loaders:
                print(f"    - {loader}")
        return
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Override with command line arguments
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    if args.max_examples:
        config["max_examples_per_domain"] = args.max_examples
    
    # Filter loaders if specific domains are requested
    if args.domains:
        available_loaders = get_available_loaders()
        filtered_loaders = {}
        
        for domain in args.domains:
            if domain in available_loaders:
                for loader in available_loaders[domain]:
                    if loader in config["loaders"]:
                        filtered_loaders[loader] = config["loaders"][loader]
        
        if not filtered_loaders:
            logger.warning("No loaders found for the specified domains")
            return
        
        config["loaders"] = filtered_loaders
    
    # Prepare all domains
    output_dir = config.get("output_dir", "data/domain_specific")
    logger.info(f"Preparing domain-specific datasets in {output_dir}")
    
    stats = prepare_all_domains(
        config_file=args.config,
        output_dir=output_dir
    )
    
    # Log summary
    logger.info("Domain dataset preparation complete:")
    for domain, domain_stats in stats["domains"].items():
        logger.info(f"  {domain}: {domain_stats['total_examples']} examples")
    
    # Example of how to use the prepared datasets
    logger.info("\nTo use the prepared datasets for training:")
    logger.info("""
    # In your training script:
    from data.domain_data_bridge import load_domain_data_for_training
    from config.training_config import EnhancedTrainingConfig
    
    # Load domain data manager
    domain_manager = load_domain_data_for_training(
        data_dir="data/domain_specific",
        tokenizer=tokenizer,
        max_length=1024
    )
    
    # Create mixed dataloader for training
    train_loader = domain_manager.create_mixed_dataloader(
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # Create training configuration with domain-specific settings
    training_config = EnhancedTrainingConfig(
        use_domain_specific_data=True,
        domains=domain_manager.config.domains,
        domain_weights=domain_manager.config.domain_weights,
        batch_size=32,
        learning_rate=5e-5,
        num_epochs=10
    )
    
    # Start training with domain-specific data
    train(model, train_loader, training_config)
    """)

if __name__ == "__main__":
    main() 