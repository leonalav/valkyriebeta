import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable, Dict, Any
import numpy as np
from utils.memory_manager import MemoryManager
import psutil
import os

class MemoryEfficientDataLoader(DataLoader):
    """Memory-efficient data loader with dynamic batching and prefetching"""
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        memory_manager: MemoryManager,
        shuffle: bool = True,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = None
    ):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, prefetch_factor=prefetch_factor, pin_memory=pin_memory, collate_fn=collate_fn)
        self.dataset = dataset
        self.initial_batch_size = batch_size
        self.memory_manager = memory_manager
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn
        
        # Initialize dataloader with dynamic batching
        self.current_batch_size = batch_size
        self._create_dataloader()
        
    def _create_dataloader(self):
        """Create a new dataloader with current batch size"""
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.current_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        self.iterator = iter(self.dataloader)
        
    def _adjust_batch_size(self):
        """Adjust batch size based on memory usage"""
        stats = self.memory_manager.get_memory_stats()
        new_batch_size = self.memory_manager.optimize_batch_size(
            stats.gpu_allocated / (torch.cuda.get_device_properties(0).total_memory / 1024**2)
        )
        
        if new_batch_size != self.current_batch_size:
            self.current_batch_size = new_batch_size
            self._create_dataloader()
            
    def _prefetch_to_memory(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prefetch batch to appropriate memory device"""
        if self.pin_memory and torch.cuda.is_available():
            # Check if we should use CPU offloading
            if self.memory_manager.should_offload_to_cpu(
                sum(x.numel() * x.element_size() for x in batch.values())
            ):
                return {k: v.pin_memory() for k, v in batch.items()}
            else:
                return {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        return batch
        
    def __iter__(self):
        return self
        
    def __next__(self) -> Dict[str, torch.Tensor]:
        try:
            # Adjust batch size based on memory usage
            self._adjust_batch_size()
            
            # Get next batch
            batch = next(self.iterator)
            
            # Clear memory before prefetching
            self.memory_manager.clear_memory()
            
            # Prefetch to appropriate memory
            batch = self._prefetch_to_memory(batch)
            
            return batch
            
        except StopIteration:
            self._create_dataloader()
            raise StopIteration
            
    def __len__(self):
        return len(self.dataloader)

class StreamingDataset(Dataset):
    """Memory-efficient dataset that streams data from disk"""
    def __init__(
        self,
        data_dir: str,
        memory_manager: MemoryManager,
        max_samples_in_memory: int = 1000,
        transform: Optional[Callable] = None
    ):
        self.data_dir = data_dir
        self.memory_manager = memory_manager
        self.max_samples_in_memory = max_samples_in_memory
        self.transform = transform
        
        # Create memory-mapped cache
        self.cache = {}
        self.cache_order = []
        
        # Index all files
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.num_samples = len(self.files)
        
    def _load_sample(self, idx: int) -> Dict[str, Any]:
        """Load a sample from disk or cache"""
        filename = self.files[idx]
        
        if filename in self.cache:
            return self.cache[filename]
            
        # Load sample from disk
        sample = torch.load(os.path.join(self.data_dir, filename))
        
        # Update cache
        if len(self.cache) >= self.max_samples_in_memory:
            # Remove oldest sample from cache
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
            
        self.cache[filename] = sample
        self.cache_order.append(filename)
        
        return sample
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._load_sample(idx)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
    def __len__(self):
        return self.num_samples