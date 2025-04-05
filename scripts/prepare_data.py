import os
import logging
from pathlib import Path
from typing import Optional, Dict
import torch
import random

from data.pipeline import IntegratedPipeline, PipelineConfig
from config.model_config import ModelConfig
from config.memory_config import MemoryConfig
from config.training_efficiency_config import TrainingEfficiencyConfig

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def prepare_datasets(
    data_root: str,
    output_root: str = "processed_data",
    config_override: Optional[Dict] = None,
    eval_split: float = 0.1,
    inference_split: float = 0.1
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    All-in-one function to prepare datasets for training
    
    Args:
        data_root: Root directory containing all data files
        output_root: Directory to save processed data
        config_override: Optional configuration overrides
        eval_split: Fraction of data to use for evaluation (default: 0.1)
        inference_split: Fraction of data to use for inference (default: 0.1)
    """
    logger = logging.getLogger(__name__)
    
    # Get all data files from single directory
    all_files = list(Path(data_root).glob("*.jsonl")) + list(Path(data_root).glob("*.parquet"))
    all_files = [str(f) for f in all_files]
    
    # Randomly split files into train/eval/inference
    random.shuffle(all_files)
    
    total_files = len(all_files)
    eval_size = int(total_files * eval_split)
    inference_size = int(total_files * inference_split)
    
    inference_files = all_files[:inference_size]
    eval_files = all_files[inference_size:inference_size + eval_size]
    train_files = all_files[inference_size + eval_size:]
    
    logger.info(f"Split {total_files} files into: {len(train_files)} train, "
                f"{len(eval_files)} eval, {len(inference_files)} inference")
    
    # Create pipeline configuration
    config_params = {
        'output_dir': output_root,
        'cache_dir': os.path.join(output_root, '.cache'),
    }
    if config_override:
        config_params.update(config_override)
    
    config = PipelineConfig(**config_params)
    
    # Initialize and run pipeline
    logger.info("Initializing data processing pipeline...")
    pipeline = IntegratedPipeline(config)
    
    try:
        # Process all data
        logger.info("Processing datasets...")
        result = pipeline.prepare_data(
            train_files=train_files,
            eval_files=eval_files,
            inference_files=inference_files
        )
        
        # Log statistics
        stats = pipeline.get_stats()
        logger.info(
            f"\nProcessing Complete!\n"
            f"Statistics:\n"
            f"- Cache Size: {stats['cache_size_mb']:.2f}MB\n"
            f"- Output Size: {stats['output_size_mb']:.2f}MB\n"
            f"- Processed Files: {stats['num_processed_files']}\n"
            f"\nProcessed data saved to: {output_root}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Prepare all datasets
        result = prepare_datasets(
            data_root="data",  # Your data root directory
            output_root="processed_data",  # Where to save processed data
            config_override={
                'max_seq_length': ModelConfig.max_seq_length,
                'train_batch_size': ModelConfig.batch_size,
                'eval_batch_size': ModelConfig.batch_size // 2
            }
        )
        
        logger.info("\nData preparation successful!")
        logger.info("You can now proceed with training using the processed data.")
        
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        raise 