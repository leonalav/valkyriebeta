import os
import sys
import logging
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent)
sys.path.append(root_dir)

from data.data_processor import UnifiedDataProcessor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Download and process datasets')
    parser.add_argument('--save-format', type=str, choices=['parquet', 'jsonl', 'both'], 
                      default='parquet', help='Format to save datasets in')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for data processing')
    parser.add_argument('--datasets', nargs='+', 
                      help='Specific datasets to process (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Custom output directory (default: datasets/processed)')
    parser.add_argument('--model-name', type=str, default='google/gemma-2-27b-it',
                      help='Model name/path for tokenizer (default: google/gemma-2-27b-it)') 
    parser.add_argument('--local-tokenizer', type=str,
                      help='Path to local tokenizer files (overrides model-name)')
    parser.add_argument('--tokenizer-type', type=str, 
                      choices=['gemma', 'custom'],
                      default='gemma',
                      help='Type of tokenizer to use')
    parser.add_argument('--custom-tokenizer-path', type=str,
                      help='Path to custom tokenizer files (required if tokenizer-type is custom)')
    parser.add_argument('--use-fast-tokenizer', action='store_true',
                      help='Use fast tokenizer implementation if available')
    return parser.parse_args()

def get_tokenizer_path(args):
    """Get appropriate tokenizer based on type"""
    if args.tokenizer_type == 'custom':
        if not args.custom_tokenizer_path:
            raise ValueError("custom-tokenizer-path required when tokenizer-type is custom")
        return args.custom_tokenizer_path
    
    # Base model mappings
    MODEL_MAPPINGS = {
        'gemma': 'google/gemma-2-27b-it'
    }
    return MODEL_MAPPINGS[args.tokenizer_type]

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Configuration
    DATA_ROOT = os.path.join(root_dir, "datasets")
    CACHE_DIR = os.path.join(DATA_ROOT, ".cache")
    OUTPUT_DIR = args.output_dir or os.path.join(DATA_ROOT, "processed")
    
    # Create directories
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Initialize tokenizer with proper model selection
        logger.info(f"Initializing tokenizer type: {args.tokenizer_type}")
        tokenizer_path = get_tokenizer_path(args)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=args.use_fast_tokenizer
            )
            logger.info(f"Successfully loaded tokenizer: {tokenizer.__class__.__name__}")
            logger.info(f"Vocabulary size: {len(tokenizer)}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            raise

        # Initialize data processor
        logger.info("Initializing data processor...")
        processor = UnifiedDataProcessor(
            tokenizer=tokenizer,
            cache_dir=CACHE_DIR,
            dataset_dir=DATA_ROOT
        )

        # Process datasets
        logger.info(f"Starting dataset processing (format: {args.save_format})...")
        results = processor.process_datasets(
            output_dir=OUTPUT_DIR,
            batch_size=args.batch_size,
            selected_datasets=args.datasets,
            save_format=args.save_format
        )

        # Print results
        logger.info("\nProcessing complete! Summary:")
        for name, info in results.items():
            logger.info(f"\nDataset: {name}")
            logger.info(f"- Number of examples: {info['num_examples']}")
            logger.info(f"- Output directory: {info['output_dir']}")
            if info['parquet_path']:
                logger.info(f"- Parquet file: {info['parquet_path']}")
            if info['jsonl_path']:
                logger.info(f"- JSONL file: {info['jsonl_path']}")

        # Print cache statistics
        stats = processor.get_cache_stats()
        logger.info(f"\nCache Statistics:")
        logger.info(f"- Cache size: {stats['cache_size_mb']:.2f} MB")
        logger.info(f"- Cached files: {stats['num_cached_files']}")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        # Cleanup
        if 'processor' in locals():
            processor.cleanup()

if __name__ == "__main__":
    main()