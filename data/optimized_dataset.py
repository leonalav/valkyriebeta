import os
import json
import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from torch.utils.data import Dataset, IterableDataset
import mmap
from pathlib import Path
from collections import defaultdict
import threading
from queue import Queue
import logging
from transformers import PreTrainedTokenizer
import h5py
from tqdm import tqdm
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
import psutil
from ..data.preprocessor import LogicalDataPreprocessor, LogicalExample
from ..data.tokenization import LogicalTokenizer

@dataclass
class DatasetConfig:
    """Configuration for optimized dataset loading"""
    cache_dir: str = ".cache"
    max_seq_length: int = 2048
    batch_size: int = 32
    num_buckets: int = 8
    prefetch_factor: int = 2
    num_workers: int = 4
    tokenizer_batch_size: int = 1000
    cache_tokenization: bool = True
    enable_prefetch: bool = True
    memory_map_threshold_gb: float = 1.0
    streaming_buffer_size: int = 10000
    max_cached_samples: int = 100000
    augmentation_workers: int = 2
    
class TokenizationCache:
    """Efficient caching system for tokenized data"""
    def __init__(self, cache_dir: str, tokenizer_name: str):
        self.cache_dir = Path(cache_dir) / "tokenization_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache filename based on tokenizer name
        tokenizer_hash = hashlib.md5(tokenizer_name.encode()).hexdigest()
        self.cache_file = self.cache_dir / f"cache_{tokenizer_hash}.h5"
        
        # Initialize cache file
        if not self.cache_file.exists():
            with h5py.File(self.cache_file, 'w') as f:
                f.create_group('tokens')
                f.create_group('masks')
                
    def get(self, text_hash: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get cached tokenization"""
        try:
            with h5py.File(self.cache_file, 'r') as f:
                if text_hash in f['tokens']:
                    return {
                        'input_ids': torch.tensor(f['tokens'][text_hash][:]),
                        'attention_mask': torch.tensor(f['masks'][text_hash][:])
                    }
        except Exception as e:
            logging.warning(f"Cache read error: {e}")
        return None
        
    def put(self, text_hash: str, tokens: Dict[str, torch.Tensor]):
        """Cache tokenization results"""
        try:
            with h5py.File(self.cache_file, 'a') as f:
                if text_hash not in f['tokens']:
                    f['tokens'].create_dataset(text_hash, data=tokens['input_ids'].numpy())
                    f['masks'].create_dataset(text_hash, data=tokens['attention_mask'].numpy())
        except Exception as e:
            logging.warning(f"Cache write error: {e}")

class SequenceBucketer:
    """Dynamic sequence length bucketing"""
    def __init__(self, num_buckets: int, max_seq_length: int):
        self.num_buckets = num_buckets
        self.max_seq_length = max_seq_length
        self.buckets = [[] for _ in range(num_buckets)]
        self.bucket_limits = np.linspace(0, max_seq_length, num_buckets + 1)
        
    def add_sample(self, sample: Dict[str, torch.Tensor], length: int):
        """Add sample to appropriate bucket"""
        bucket_idx = np.digitize(length, self.bucket_limits) - 1
        bucket_idx = min(bucket_idx, self.num_buckets - 1)
        self.buckets[bucket_idx].append(sample)
        
    def get_batch(self, batch_size: int) -> Optional[List[Dict[str, torch.Tensor]]]:
        """Get batch from most filled bucket"""
        max_samples = max(len(bucket) for bucket in self.buckets)
        if max_samples < batch_size:
            return None
            
        # Select bucket with most samples
        bucket_idx = max(range(self.num_buckets), 
                        key=lambda i: len(self.buckets[i]))
        
        # Get batch and remove from bucket
        batch = self.buckets[bucket_idx][:batch_size]
        self.buckets[bucket_idx] = self.buckets[bucket_idx][batch_size:]
        
        return batch

class DataPrefetcher:
    """Asynchronous data prefetching system"""
    def __init__(self, 
                 dataset: Dataset,
                 buffer_size: int = 1000,
                 num_workers: int = 4):
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        
        self.queue = Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.workers = []
        
        # Start prefetch workers
        for _ in range(num_workers):
            worker = threading.Thread(target=self._prefetch_worker)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
    def _prefetch_worker(self):
        """Worker thread for prefetching data"""
        while not self.stop_event.is_set():
            try:
                # Get next index to prefetch
                idx = self.dataset._get_next_index()
                if idx is None:
                    break
                    
                # Load and preprocess sample
                sample = self.dataset._load_and_process_sample(idx)
                
                # Add to queue
                self.queue.put(sample, timeout=1)
            except Exception as e:
                logging.warning(f"Prefetch error: {e}")
                continue
                
    def get_next(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get next prefetched sample"""
        try:
            return self.queue.get(timeout=1)
        except:
            return None
            
    def stop(self):
        """Stop prefetching"""
        self.stop_event.set()
        for worker in self.workers:
            worker.join()

class MemoryMappedDataset:
    """Memory-mapped dataset for efficient loading"""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_size = os.path.getsize(filepath)
        
        # Memory map the file
        self.file = open(filepath, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Index line positions
        self.line_positions = []
        pos = 0
        while pos < self.file_size:
            self.line_positions.append(pos)
            pos = self.mm.find(b'\n', pos) + 1
            if pos == 0:  # No more newlines
                break
                
    def __getitem__(self, idx: int) -> str:
        """Get line by index"""
        start = self.line_positions[idx]
        if idx + 1 < len(self.line_positions):
            end = self.line_positions[idx + 1] - 1
        else:
            end = self.file_size
            
        self.mm.seek(start)
        line = self.mm.read(end - start).decode('utf-8')
        return line
        
    def __len__(self) -> int:
        return len(self.line_positions)
        
    def close(self):
        """Close memory-mapped file"""
        self.mm.close()
        self.file.close()

class OptimizedDataset(IterableDataset):
    """Memory-efficient dataset with streaming and preprocessing optimizations"""
    def __init__(
        self,
        data_files: List[str],
        tokenizer: PreTrainedTokenizer,
        config: DatasetConfig
    ):
        super().__init__()
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize enhanced tokenizer and preprocessor
        self.logical_tokenizer = LogicalTokenizer(tokenizer)
        self.preprocessor = LogicalDataPreprocessor(self.logical_tokenizer, config)
        
        # Setup caching
        self.tokenization_cache = TokenizationCache(
            config.cache_dir,
            tokenizer.__class__.__name__
        )
        
        # Initialize other components
        self._setup_datasets()
        self.bucketer = SequenceBucketer(config.num_buckets, config.max_seq_length)
        
        if config.enable_prefetch:
            self.prefetcher = DataPrefetcher(
                self,
                config.streaming_buffer_size,
                config.num_workers
            )
        else:
            self.prefetcher = None
            
        # Setup augmentation thread pool
        self.augmentation_pool = ThreadPoolExecutor(
            max_workers=config.augmentation_workers
        )
        
        # Initialize state
        self.current_file_idx = 0
        self.current_position = 0
        self.lock = threading.Lock()
        
    def _setup_datasets(self):
        """Setup memory mapping or streaming based on file size"""
        for filepath in self.data_files:
            size_gb = os.path.getsize(filepath) / (1024**3)
            
            if size_gb >= self.config.memory_map_threshold_gb:
                # Use memory mapping for large files
                self.mm_datasets = {filepath: MemoryMappedDataset(filepath)}
            else:
                # Use streaming for smaller files
                self.streaming_datasets = [filepath]
                
    def _get_next_index(self) -> Optional[int]:
        """Get next sample index with thread safety"""
        with self.lock:
            if self.current_file_idx >= len(self.data_files):
                return None
                
            current_file = self.data_files[self.current_file_idx]
            
            if current_file in self.mm_datasets:
                dataset = self.mm_datasets[current_file]
                if self.current_position >= len(dataset):
                    self.current_file_idx += 1
                    self.current_position = 0
                    return self._get_next_index()
                    
                idx = self.current_position
                self.current_position += 1
                return idx
            else:
                # Handle streaming dataset
                try:
                    with open(current_file, 'r') as f:
                        f.seek(self.current_position)
                        line = f.readline()
                        if not line:
                            self.current_file_idx += 1
                            self.current_position = 0
                            return self._get_next_index()
                            
                        self.current_position = f.tell()
                        return self.current_position
                except Exception as e:
                    logging.error(f"Error reading file {current_file}: {e}")
                    self.current_file_idx += 1
                    self.current_position = 0
                    return self._get_next_index()
                    
    def _load_and_process_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and process a single sample with logical reasoning support"""
        # Get raw data
        data = self.mm_datasets[self.data_files[idx % len(self.data_files)]][idx]
        
        # Process based on file type
        if self.data_files[idx % len(self.data_files)].endswith('.parquet'):
            text = self._process_parquet_sample(data)
        else:
            text = self._process_jsonl_sample(data)
            
        # Check cache first
        text_hash = str(hash(text))
        cached_tokens = self.tokenization_cache.get(text_hash)
        if cached_tokens is not None:
            return cached_tokens
            
        # Preprocess with logical reasoning
        logical_example = self.preprocessor.preprocess_logical_example(
            text=text,
            logical_tree=data.get('logical_tree'),
            labels=data.get('labels')
        )
        
        # Tokenize with special tokens
        encoded = self.logical_tokenizer.tokenize(
            text=logical_example.text,
            max_length=self.config.max_seq_length,
            padding='max_length',
            truncation=True
        )
        
        # Add logical tree if available
        if logical_example.logical_tree:
            tree_text = self.logical_tokenizer.encode_logical_tree(
                logical_example.logical_tree
            )
            tree_tokens = self.logical_tokenizer.tokenize(
                text=tree_text,
                max_length=self.config.max_seq_length // 2,  # Use half length for tree
                padding='max_length',
                truncation=True
            )
            encoded.update(tree_tokens)
            
        # Cache the tokens
        self.tokenization_cache.put(text_hash, encoded)
        
        return encoded
        
    def _process_parquet_sample(self, data: Dict) -> str:
        """Process sample from parquet file"""
        if 'conversation' in data:
            return f"Instruction: {data['instruction']}\nConversation: {data['conversation']}\nOutput: {data['output']}"
        elif 'problem' in data:
            text = f"Problem: {data['problem']}\nSolution: {data['solution']}"
            if 'answer' in data:
                text += f"\nAnswer: {data['answer']}"
            return text
        return data.get('text', '')
        
    def _process_jsonl_sample(self, data: Dict) -> str:
        """Process sample from jsonl file"""
        if 'question' in data:
            text = f"Question: {data['question']}"
            if 'solution' in data:
                text += f"\nSolution: {data['solution']}"
            return text
        return data.get('text', '')
        
    def __iter__(self):
        """Iterator implementation"""
        while True:
            # Use prefetcher if enabled
            if self.prefetcher:
                sample = self.prefetcher.get_next()
                if sample is None:
                    break
            else:
                idx = self._get_next_index()
                if idx is None:
                    break
                sample = self._load_and_process_sample(idx)
                
            # Add to sequence bucketer
            seq_length = (sample['attention_mask'] == 1).sum().item()
            self.bucketer.add_sample(sample, seq_length)
            
            # Try to get a batch
            batch = self.bucketer.get_batch(self.config.batch_size)
            if batch:
                yield self._collate_batch(batch)
                
    def _collate_batch(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch with support for logical reasoning features"""
        collated = {}
        
        # Collate standard features
        for key in ['input_ids', 'attention_mask']:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
                
        # Collate logical features if present
        if 'tree_input_ids' in batch[0]:
            collated['tree_input_ids'] = torch.stack(
                [item['tree_input_ids'] for item in batch]
            )
            collated['tree_attention_mask'] = torch.stack(
                [item['tree_attention_mask'] for item in batch]
            )
            
        # Collate labels if present
        if 'labels' in batch[0]:
            collated['labels'] = torch.stack([item['labels'] for item in batch])
            
        return collated
        
    def close(self):
        """Cleanup resources"""
        # Stop prefetching
        if self.prefetcher:
            self.prefetcher.stop()
            
        # Close memory-mapped files
        for dataset in self.mm_datasets.values():
            dataset.close()
            
        # Shutdown augmentation pool
        self.augmentation_pool.shutdown() 