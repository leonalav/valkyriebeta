import os
from typing import List, Optional, Dict, Any, Union
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
import logging
import psutil
import torch
from pathlib import Path
from ..utils.tensor_cache import TensorCache

from .optimized_dataset import OptimizedDataset, DatasetConfig
from .run_inference import InferenceDataset

class DataLoaderFactory:
    """Creates optimized data loaders"""
    
    @staticmethod
    def create_loader(
        data_files: List[str],
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        max_seq_length: int = 2048,
        num_workers: Optional[int] = None,
        cache_dir: str = ".cache",
        enable_memory_mapping: bool = True,
        enable_prefetch: bool = True,
        enable_caching: bool = True,
        is_inference: bool = False
    ) -> DataLoader:
        """Create an optimized data loader for training or inference"""
        
        # Validate files
        valid_files = []
        for file in data_files:
            if not os.path.exists(file):
                logging.warning(f"File not found: {file}")
                continue
            valid_files.append(file)
            
        if not valid_files:
            raise ValueError("No valid data files provided")
            
        # Auto-configure settings
        config = DataLoaderFactory._get_optimal_config(
            valid_files,
            batch_size,
            max_seq_length,
            num_workers,
            enable_memory_mapping,
            enable_prefetch,
            enable_caching,
            is_inference
        )
        
        # Create appropriate dataset
        if is_inference:
            dataset = InferenceDataset(
                valid_files,
                tokenizer,
                config.max_seq_length
            )
        else:
            dataset = OptimizedDataset(
                data_files=valid_files,
                tokenizer=tokenizer,
                config=config
            )
        
        # Setup tensor cache if enabled
        tensor_cache = None
        if enable_caching:
            tensor_cache = TensorCache(max_size_gb=4.0)
        
        # Create and return data loader
        return DataLoader(
            dataset,
            batch_size=None if not is_inference else config.batch_size,  # Batch size handled by dataset during training
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor if not is_inference else 2,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            collate_fn=CollateFunction(
                pad_token_id=tokenizer.pad_token_id,
                tensor_cache=tensor_cache
            )
        )
        
    @staticmethod
    def _get_optimal_config(
        data_files: List[str],
        batch_size: int,
        max_seq_length: int,
        num_workers: Optional[int],
        enable_memory_mapping: bool,
        enable_prefetch: bool,
        enable_caching: bool,
        is_inference: bool
    ) -> DatasetConfig:
        """Get optimal configuration based on system resources and mode"""
        
        # Get system resources
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        num_cpus = psutil.cpu_count()
        
        # Calculate total dataset size
        total_size_gb = sum(
            os.path.getsize(f) / (1024**3)
            for f in data_files
        )
        
        # Adjust settings for inference
        if is_inference:
            # Use more conservative settings for inference
            num_workers = min(2, num_cpus - 1) if num_workers is None else num_workers
            memory_map_threshold = float('inf')  # Disable memory mapping for inference
            cache_size = int(total_memory * 0.1 * 1024)  # 10% of memory for cache
            buffer_size = 1000  # Smaller buffer for inference
        else:
            # Training settings (existing logic)
            if num_workers is None:
                if total_size_gb < 1:
                    num_workers = 1
                else:
                    num_workers = min(
                        num_cpus - 1,
                        int(total_memory / 4)  # 4GB per worker
                    )
                    num_workers = max(1, num_workers)
                    
            memory_map_threshold = min(
                1.0,  # Default 1GB
                total_memory * 0.1  # 10% of system memory
            ) if enable_memory_mapping else float('inf')
            
            cache_size = int(total_memory * 0.2 * 1024) if enable_caching else 0
            buffer_size = min(
                10000,  # Default
                int(total_memory * 1024 * 0.05)  # 5% of system memory
            )
            
        # Create configuration
        return DatasetConfig(
            cache_dir=os.path.abspath(".cache"),
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            num_buckets=8 if not is_inference else 1,  # No bucketing needed for inference
            prefetch_factor=2 if enable_prefetch else 0,
            num_workers=num_workers,
            tokenizer_batch_size=min(1000, batch_size * 10),
            cache_tokenization=enable_caching,
            enable_prefetch=enable_prefetch,
            memory_map_threshold_gb=memory_map_threshold,
            streaming_buffer_size=buffer_size,
            max_cached_samples=cache_size,
            augmentation_workers=max(1, num_workers // 2) if not is_inference else 0
        )
        
    @staticmethod
    def cleanup_cache(cache_dir: str = ".cache"):
        """Clean up cache directory"""
        cache_path = Path(cache_dir)
        if cache_path.exists():
            for file in cache_path.glob("**/*"):
                if file.is_file():
                    try:
                        file.unlink()
                    except Exception as e:
                        logging.warning(f"Failed to delete {file}: {e}")
            try:
                cache_path.rmdir()
            except Exception as e:
                logging.warning(f"Failed to delete cache directory: {e}")
                
    @staticmethod
    def get_cache_size(cache_dir: str = ".cache") -> float:
        """Get cache size in MB"""
        total_size = 0
        cache_path = Path(cache_dir)
        if cache_path.exists():
            for file in cache_path.glob("**/*"):
                if file.is_file():
                    total_size += file.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB

class CollateFunction:
    """Efficient collate function with caching"""
    
    def __init__(self, pad_token_id: int, tensor_cache: Optional[TensorCache] = None):
        self.pad_token_id = pad_token_id
        self.tensor_cache = tensor_cache
        
    def __call__(self, batch):
        # Try to get from cache first
        if self.tensor_cache is not None:
            cache_key = self._get_cache_key(batch)
            cached_batch = self.tensor_cache.get(cache_key)
            if cached_batch is not None:
                return cached_batch
                
        # Process batch
        max_len = max(len(x['input_ids']) for x in batch)
        
        # Prepare tensors
        batch_size = len(batch)
        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_token_id,
            dtype=torch.long
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long
        )
        
        # Fill tensors
        for i, item in enumerate(batch):
            input_ids[i, :len(item['input_ids'])] = torch.tensor(
                item['input_ids'],
                dtype=torch.long
            )
            attention_mask[i, :len(item['input_ids'])] = 1
            
        processed_batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Cache result
        if self.tensor_cache is not None:
            self.tensor_cache.add(cache_key, processed_batch)
            
        return processed_batch
        
    def _get_cache_key(self, batch) -> str:
        """Generate cache key for batch"""
        # Simple hash of batch contents
        return str(hash(str(batch)))