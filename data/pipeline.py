import os
from typing import Dict, Optional, Union, List
import logging
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer
from dataclasses import dataclass
import torch

from .data_processor import UnifiedDataProcessor
from .preprocessor import LogicalExample
from config.model_config import ModelConfig  # Import our model config

@dataclass
class PipelineConfig:
    """Configuration for the data pipeline"""
    model_name: str = ModelConfig.model_name  # Use our model's name
    cache_dir: str = ".cache"
    output_dir: str = "processed_data"
    max_seq_length: int = ModelConfig.max_seq_length  # Use our model's sequence length
    train_batch_size: int = ModelConfig.batch_size  # Use our model's batch size
    eval_batch_size: int = ModelConfig.batch_size // 2  # Half the training batch size for eval
    enable_memory_mapping: bool = True
    enable_prefetch: bool = True
    enable_caching: bool = True

class IntegratedPipeline:
    """Integrated pipeline for data processing and loading"""
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        os.makedirs(self.config.cache_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize tokenizer - use model's tokenizer if available, otherwise create new
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        except Exception as e:
            self.logger.warning(f"Could not load pretrained tokenizer: {e}")
            self.logger.info("Creating new tokenizer with model's vocabulary size")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "gpt2",  # Use GPT2's tokenizer as base
                model_max_length=self.config.max_seq_length,
                vocab_size=ModelConfig.vocab_size,
                pad_token="<pad>",
                eos_token="</s>",
                bos_token="<s>",
                unk_token="<unk>"
            )
        
        # Initialize processor
        self.processor = UnifiedDataProcessor(
            tokenizer=self.tokenizer,
            cache_dir=self.config.cache_dir,
            max_seq_length=self.config.max_seq_length
        )
        
    def prepare_data(
        self,
        train_files: Optional[List[str]] = None,
        eval_files: Optional[List[str]] = None,
        inference_files: Optional[List[str]] = None
    ) -> Dict[str, Union[torch.utils.data.DataLoader, str]]:
        """Prepare data for training, evaluation, and/or inference"""
        result = {}
        
        # Process training data if provided
        if train_files:
            train_output = self.processor.process_data_files(
                data_files=train_files,
                output_dir=os.path.join(self.config.output_dir, "train"),
                batch_size=self.config.train_batch_size,
                is_inference=False
            )
            result['train_loader'] = train_output['train_loader']
            result['train_dir'] = train_output['preprocessed_dir']
            
        # Process evaluation data if provided
        if eval_files:
            eval_output = self.processor.process_data_files(
                data_files=eval_files,
                output_dir=os.path.join(self.config.output_dir, "eval"),
                batch_size=self.config.eval_batch_size,
                is_inference=False
            )
            result['eval_loader'] = eval_output['train_loader']
            result['eval_dir'] = eval_output['preprocessed_dir']
            
        # Process inference data if provided
        if inference_files:
            inference_output = self.processor.process_data_files(
                data_files=inference_files,
                output_dir=os.path.join(self.config.output_dir, "inference"),
                batch_size=self.config.eval_batch_size,
                is_inference=True
            )
            result['inference_loader'] = inference_output['data_loader']
            result['inference_dir'] = inference_output['preprocessed_dir']
            
        return result
        
    def preprocess_only(
        self,
        data_files: List[str],
        output_subdir: str = "preprocessed"
    ) -> List[LogicalExample]:
        """Preprocess data without creating loaders"""
        return self.processor.process_data_files(
            data_files=data_files,
            output_dir=os.path.join(self.config.output_dir, output_subdir),
            preprocess_only=True
        )
        
    def get_stats(self) -> Dict[str, float]:
        """Get pipeline statistics"""
        stats = self.processor.get_cache_stats()
        stats.update({
            'output_size_mb': self._get_dir_size(self.config.output_dir) / (1024 * 1024),
            'num_processed_files': len(list(Path(self.config.output_dir).rglob("*.json")))
        })
        return stats
        
    def cleanup(self):
        """Cleanup pipeline resources"""
        self.processor.cleanup()
        
    def _get_dir_size(self, dir_path: str) -> int:
        """Get total size of directory in bytes"""
        total = 0
        for dirpath, _, filenames in os.walk(dir_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total

# Example usage in training script:
def process_dataset(
    train_files: List[str],
    eval_files: Optional[List[str]] = None,
    inference_files: Optional[List[str]] = None,
    config_override: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Union[torch.utils.data.DataLoader, str]]:
    """Helper function for easy dataset processing"""
    
    # Create configuration with potential overrides
    config_params = {}
    if config_override:
        config_params.update(config_override)
    config_params.update(kwargs)
    
    config = PipelineConfig(**config_params)
    
    # Initialize and run pipeline
    pipeline = IntegratedPipeline(config)
    try:
        result = pipeline.prepare_data(
            train_files=train_files,
            eval_files=eval_files,
            inference_files=inference_files
        )
        
        # Log statistics
        stats = pipeline.get_stats()
        logging.info(
            f"Pipeline Statistics:\n"
            f"- Cache Size: {stats['cache_size_mb']:.2f}MB\n"
            f"- Output Size: {stats['output_size_mb']:.2f}MB\n"
            f"- Processed Files: {stats['num_processed_files']}"
        )
        
        return result
    finally:
        pipeline.cleanup() 