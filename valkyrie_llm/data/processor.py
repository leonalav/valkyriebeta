import torch
import numpy as np
import logging
import multiprocessing
import os
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple, Callable
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, IterableDataset
from collections import defaultdict
import random
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    
    # Processing parameters
    max_seq_length: int = 2048
    chunk_size: int = 1024
    stride: int = 512
    pad_to_multiple_of: Optional[int] = 8
    
    # Tokenization parameters
    truncation_side: str = "right"
    padding_side: str = "right"
    add_special_tokens: bool = True
    
    # Caching parameters
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    # Advanced processing
    filter_min_length: Optional[int] = 128
    filter_max_length: Optional[int] = None
    keep_incomplete_chunks: bool = False
    
    # Processing hooks
    pre_tokenization_hook: Optional[Callable] = None
    post_tokenization_hook: Optional[Callable] = None
    
    # Multiprocessing
    num_workers: int = 4
    
    def __post_init__(self):
        """Validate config and set defaults."""
        if self.cache_dir is None:
            self.cache_dir = os.path.join(os.getcwd(), "cache")
            
        # Create cache directory if it doesn't exist
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)


class StreamingProcessor:
    """
    Processes large datasets in a streaming fashion to minimize memory usage.
    
    This processor can handle datasets that are too large to fit in memory by
    processing them in chunks and yielding processed examples as an iterator.
    """
    
    def __init__(self, 
                 tokenizer: Any,
                 config: Optional[ProcessingConfig] = None):
        """
        Initialize the streaming processor.
        
        Args:
            tokenizer: Tokenizer to use for processing text
            config: Configuration for data processing
        """
        self.tokenizer = tokenizer
        self.config = config or ProcessingConfig()
        
        # Stats for monitoring
        self.stats = defaultdict(int)
    
    def process_file(self, 
                     file_path: str, 
                     output_format: str = "pt",
                     filter_fn: Optional[Callable] = None) -> Iterator[Dict[str, Any]]:
        """
        Process a file in a streaming fashion.
        
        Args:
            file_path: Path to the file to process
            output_format: Format of the output ("pt" for PyTorch tensors or "np" for numpy)
            filter_fn: Optional function to filter examples
            
        Yields:
            Processed examples
        """
        # Check if cached version exists
        cache_path = self._get_cache_path(file_path) if self.config.use_cache else None
        
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading processed data from cache: {cache_path}")
            yield from self._read_from_cache(cache_path)
            return
        
        # Process file based on extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".json" or ext == ".jsonl":
            yield from self._process_json_file(file_path, output_format, filter_fn, cache_path)
        elif ext == ".txt":
            yield from self._process_text_file(file_path, output_format, filter_fn, cache_path)
        else:
            logger.warning(f"Unsupported file extension: {ext}. Trying to process as text.")
            yield from self._process_text_file(file_path, output_format, filter_fn, cache_path)
    
    def process_dataset(self, 
                        dataset: Union[List[Dict[str, Any]], Any],
                        text_key: str = "text",
                        output_format: str = "pt",
                        filter_fn: Optional[Callable] = None) -> Iterator[Dict[str, Any]]:
        """
        Process a dataset in a streaming fashion.
        
        Args:
            dataset: Dataset to process (list of dictionaries or HuggingFace dataset)
            text_key: Key for the text field in the dataset
            output_format: Format of the output ("pt" for PyTorch tensors or "np" for numpy)
            filter_fn: Optional function to filter examples
            
        Yields:
            Processed examples
        """
        # Process each example
        for example in dataset:
            # Get text from example
            if isinstance(example, dict):
                text = example.get(text_key, "")
            else:
                text = getattr(example, text_key, "")
            
            if not text:
                self.stats["empty_examples"] += 1
                continue
            
            # Apply pre-tokenization hook if provided
            if self.config.pre_tokenization_hook:
                text = self.config.pre_tokenization_hook(text)
            
            # Tokenize text
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
                return_tensors=None,
                add_special_tokens=self.config.add_special_tokens
            )
            
            # Apply post-tokenization hook if provided
            if self.config.post_tokenization_hook:
                tokenized = self.config.post_tokenization_hook(tokenized)
            
            # Create chunks
            for chunk in self._create_chunks(tokenized, output_format):
                # Apply filter if provided
                if filter_fn and not filter_fn(chunk):
                    self.stats["filtered_chunks"] += 1
                    continue
                
                # Ensure all outputs are tensors if pt format
                if output_format == "pt":
                    chunk = {k: torch.tensor(v) for k, v in chunk.items()}
                
                # Add metadata from original example if available
                if isinstance(example, dict):
                    for key, value in example.items():
                        if key != text_key and key not in chunk:
                            chunk[f"meta_{key}"] = value
                
                yield chunk
                self.stats["processed_chunks"] += 1
    
    def _process_json_file(self, 
                          file_path: str, 
                          output_format: str,
                          filter_fn: Optional[Callable],
                          cache_path: Optional[str]) -> Iterator[Dict[str, Any]]:
        """Process a JSON or JSONL file."""
        cache_examples = [] if cache_path else None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Check if it's a JSONL file (one JSON object per line)
            first_line = f.readline().strip()
            f.seek(0)
            
            try:
                if first_line.startswith('['):
                    # Regular JSON file
                    data = json.load(f)
                    if isinstance(data, list):
                        for example in data:
                            for processed in self._process_example(example, output_format, filter_fn):
                                if cache_examples is not None:
                                    cache_examples.append(processed)
                                yield processed
                    else:
                        logger.warning(f"JSON file doesn't contain a list: {file_path}")
                else:
                    # JSONL file
                    for line in f:
                        if not line.strip():
                            continue
                        
                        example = json.loads(line)
                        for processed in self._process_example(example, output_format, filter_fn):
                            if cache_examples is not None:
                                cache_examples.append(processed)
                            yield processed
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON file: {file_path}")
        
        # Save to cache if enabled
        if cache_path and cache_examples:
            self._save_to_cache(cache_examples, cache_path)
    
    def _process_text_file(self, 
                          file_path: str, 
                          output_format: str,
                          filter_fn: Optional[Callable],
                          cache_path: Optional[str]) -> Iterator[Dict[str, Any]]:
        """Process a text file."""
        cache_examples = [] if cache_path else None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
            # Apply pre-tokenization hook if provided
            if self.config.pre_tokenization_hook:
                text = self.config.pre_tokenization_hook(text)
            
            # Tokenize text
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
                return_tensors=None,
                add_special_tokens=self.config.add_special_tokens
            )
            
            # Apply post-tokenization hook if provided
            if self.config.post_tokenization_hook:
                tokenized = self.config.post_tokenization_hook(tokenized)
            
            # Create chunks
            for chunk in self._create_chunks(tokenized, output_format):
                # Apply filter if provided
                if filter_fn and not filter_fn(chunk):
                    self.stats["filtered_chunks"] += 1
                    continue
                
                # Add metadata
                chunk["meta_file"] = os.path.basename(file_path)
                
                if cache_examples is not None:
                    cache_examples.append(chunk)
                
                yield chunk
                self.stats["processed_chunks"] += 1
        
        # Save to cache if enabled
        if cache_path and cache_examples:
            self._save_to_cache(cache_examples, cache_path)
    
    def _process_example(self, 
                        example: Dict[str, Any],
                        output_format: str,
                        filter_fn: Optional[Callable]) -> Iterator[Dict[str, Any]]:
        """Process a single example."""
        # Extract text from example (try common text field names)
        text = None
        for key in ["text", "content", "document", "passage", "context"]:
            if key in example:
                text = example[key]
                break
        
        if text is None:
            self.stats["examples_no_text"] += 1
            return
        
        # Apply pre-tokenization hook if provided
        if self.config.pre_tokenization_hook:
            text = self.config.pre_tokenization_hook(text)
        
        # Tokenize text
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=self.config.add_special_tokens
        )
        
        # Apply post-tokenization hook if provided
        if self.config.post_tokenization_hook:
            tokenized = self.config.post_tokenization_hook(tokenized)
        
        # Create chunks
        for chunk in self._create_chunks(tokenized, output_format):
            # Apply filter if provided
            if filter_fn and not filter_fn(chunk):
                self.stats["filtered_chunks"] += 1
                continue
            
            # Add metadata from original example
            for key, value in example.items():
                if key != "text" and key not in chunk:
                    chunk[f"meta_{key}"] = value
            
            yield chunk
            self.stats["processed_chunks"] += 1
    
    def _create_chunks(self, 
                      tokenized: Dict[str, List[int]], 
                      output_format: str) -> Iterator[Dict[str, Any]]:
        """Create chunks from tokenized text."""
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized.get("attention_mask", [1] * len(input_ids))
        
        # Skip if too short
        if self.config.filter_min_length and len(input_ids) < self.config.filter_min_length:
            self.stats["filtered_short"] += 1
            return
        
        # Skip if too long
        if self.config.filter_max_length and len(input_ids) > self.config.filter_max_length:
            self.stats["filtered_long"] += 1
            return
        
        # If short enough to fit in a single chunk, return as is
        if len(input_ids) <= self.config.chunk_size:
            # Pad to chunk size if needed
            if len(input_ids) < self.config.chunk_size and self.config.pad_to_multiple_of:
                padding_length = self.config.chunk_size - len(input_ids)
                if self.config.padding_side == "right":
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length
                else:
                    input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                    attention_mask = [0] * padding_length + attention_mask
            
            # Create chunk
            chunk = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "chunk_index": 0,
                "total_chunks": 1
            }
            
            # Convert to tensors if needed
            if output_format == "pt":
                chunk = {k: torch.tensor(v) for k, v in chunk.items()}
            
            yield chunk
            return
        
        # Create overlapping chunks
        chunk_index = 0
        
        for start_idx in range(0, len(input_ids), self.config.stride):
            # Get end index for this chunk
            end_idx = min(start_idx + self.config.chunk_size, len(input_ids))
            
            # Skip incomplete chunks if configured
            if end_idx - start_idx < self.config.chunk_size and not self.config.keep_incomplete_chunks:
                if start_idx > 0:  # Don't skip if it's the only chunk
                    continue
            
            # Extract chunk
            chunk_input_ids = input_ids[start_idx:end_idx]
            chunk_attention_mask = attention_mask[start_idx:end_idx]
            
            # Pad to chunk size if needed
            if len(chunk_input_ids) < self.config.chunk_size and self.config.pad_to_multiple_of:
                padding_length = self.config.chunk_size - len(chunk_input_ids)
                if self.config.padding_side == "right":
                    chunk_input_ids = chunk_input_ids + [self.tokenizer.pad_token_id] * padding_length
                    chunk_attention_mask = chunk_attention_mask + [0] * padding_length
                else:
                    chunk_input_ids = [self.tokenizer.pad_token_id] * padding_length + chunk_input_ids
                    chunk_attention_mask = [0] * padding_length + chunk_attention_mask
            
            # Create chunk
            chunk = {
                "input_ids": chunk_input_ids,
                "attention_mask": chunk_attention_mask,
                "chunk_index": chunk_index,
                "total_chunks": (len(input_ids) + self.config.stride - 1) // self.config.stride
            }
            
            # Convert to tensors if needed
            if output_format == "pt":
                chunk = {k: torch.tensor(v) for k, v in chunk.items()}
            
            yield chunk
            chunk_index += 1
            
            # Stop if we've processed the entire input
            if end_idx >= len(input_ids):
                break
    
    def _get_cache_path(self, file_path: str) -> str:
        """Get cache path for a file."""
        file_name = os.path.basename(file_path)
        return os.path.join(self.config.cache_dir, f"{file_name}.processed.pt")
    
    def _save_to_cache(self, examples: List[Dict[str, Any]], cache_path: str):
        """Save processed examples to cache."""
        logger.info(f"Saving {len(examples)} processed examples to cache: {cache_path}")
        torch.save(examples, cache_path)
    
    def _read_from_cache(self, cache_path: str) -> Iterator[Dict[str, Any]]:
        """Read processed examples from cache."""
        examples = torch.load(cache_path)
        for example in examples:
            yield example
            self.stats["cached_examples"] += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return dict(self.stats)


class StreamingDataset(IterableDataset):
    """
    Dataset that streams data from files or other datasets.
    
    This dataset is memory-efficient as it processes data on-the-fly
    and can handle datasets that are too large to fit in memory.
    """
    
    def __init__(self,
                 processor: StreamingProcessor,
                 data_source: Union[str, List[str], Any],
                 source_type: str = "file",
                 text_key: str = "text",
                 shuffle_buffer_size: int = 1000,
                 shuffle: bool = True,
                 seed: int = 42,
                 filter_fn: Optional[Callable] = None):
        """
        Initialize streaming dataset.
        
        Args:
            processor: Data processor
            data_source: Source of data (file path, list of file paths, or dataset)
            source_type: Type of data source ("file", "files", or "dataset")
            text_key: Key for text field in dataset
            shuffle_buffer_size: Size of shuffle buffer (0 to disable)
            shuffle: Whether to shuffle the data
            seed: Random seed
            filter_fn: Optional function to filter examples
        """
        self.processor = processor
        self.data_source = data_source
        self.source_type = source_type
        self.text_key = text_key
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle = shuffle
        self.seed = seed
        self.filter_fn = filter_fn
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate inputs."""
        if self.source_type == "file" and not isinstance(self.data_source, str):
            raise ValueError("For source_type='file', data_source must be a string (file path)")
        
        if self.source_type == "files" and not (isinstance(self.data_source, list) and all(isinstance(f, str) for f in self.data_source)):
            raise ValueError("For source_type='files', data_source must be a list of strings (file paths)")
            
        if self.shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be >= 0")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return iterator over the dataset."""
        # Set worker seed for reproducibility
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed
        
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id
            
        random.seed(worker_seed)
        
        # Create iterator based on source type
        if self.source_type == "file":
            iterator = self.processor.process_file(
                self.data_source, 
                output_format="pt", 
                filter_fn=self.filter_fn
            )
        elif self.source_type == "files":
            iterator = self._multi_file_iterator()
        else:  # dataset
            iterator = self.processor.process_dataset(
                self.data_source,
                text_key=self.text_key,
                output_format="pt",
                filter_fn=self.filter_fn
            )
        
        # Apply shuffling if enabled
        if self.shuffle and self.shuffle_buffer_size > 0:
            yield from self._shuffled(iterator)
        else:
            yield from iterator
    
    def _multi_file_iterator(self) -> Iterator[Dict[str, Any]]:
        """Iterator over multiple files."""
        # Shuffle file order if enabled
        files = list(self.data_source)
        if self.shuffle:
            random.shuffle(files)
        
        for file_path in files:
            yield from self.processor.process_file(
                file_path, 
                output_format="pt", 
                filter_fn=self.filter_fn
            )
    
    def _shuffled(self, iterator: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Apply shuffling to an iterator."""
        buffer = []
        
        for example in iterator:
            buffer.append(example)
            
            # If buffer is full, yield a random example
            if len(buffer) >= self.shuffle_buffer_size:
                idx = random.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)
        
        # Yield remaining examples
        random.shuffle(buffer)
        yield from buffer


class DataCollator:
    """
    Collates examples into batches for training.
    
    This collator handles padding, truncation, and other batch preparation
    tasks in an efficient way.
    """
    
    def __init__(self, 
                 tokenizer: Any,
                 pad_to_multiple_of: Optional[int] = 8,
                 label_pad_token_id: int = -100,
                 padding_side: str = "right",
                 max_length: Optional[int] = None,
                 return_tensors: str = "pt"):
        """
        Initialize data collator.
        
        Args:
            tokenizer: Tokenizer to use for padding
            pad_to_multiple_of: Pad to multiple of this value
            label_pad_token_id: Padding token ID for labels
            padding_side: Side to pad ("left" or "right")
            max_length: Maximum sequence length
            return_tensors: Return format ("pt" or "np")
        """
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.padding_side = padding_side
        self.max_length = max_length
        self.return_tensors = return_tensors
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate examples into a batch.
        
        Args:
            examples: List of examples to collate
            
        Returns:
            Collated batch
        """
        # Handle edge case with empty list
        if not examples or len(examples) == 0:
            return {}
        
        # Extract features present in all examples
        batch = {}
        first = examples[0]
        
        for key in first.keys():
            if key.startswith("meta_"):
                # Don't collate metadata
                continue
                
            if isinstance(first[key], torch.Tensor):
                # Default for tensors
                values = [example[key] for example in examples]
                batch[key] = self._collate_tensors(values, key)
            elif isinstance(first[key], (int, float, bool)):
                # Default for scalars
                batch[key] = torch.tensor([example[key] for example in examples])
            else:
                # Skip other types
                continue
        
        # Process metadata
        meta_keys = [k for k in first.keys() if k.startswith("meta_")]
        if meta_keys:
            for key in meta_keys:
                batch[key] = [example[key] if key in example else None for example in examples]
        
        return batch
    
    def _collate_tensors(self, 
                         tensors: List[torch.Tensor], 
                         key: str) -> torch.Tensor:
        """Collate tensors with appropriate padding."""
        # Check if all tensors have the same shape
        shapes = [t.shape for t in tensors]
        if all(len(s) == 1 for s in shapes) or all(s == shapes[0] for s in shapes):
            # All same shape, just stack
            return torch.stack(tensors)
        
        # Pad tensors
        if key == "labels":
            # Special handling for labels
            return self._pad_labels(tensors)
        else:
            # General case
            return self._pad_tensors(tensors, key)
    
    def _pad_tensors(self, 
                    tensors: List[torch.Tensor], 
                    key: str) -> torch.Tensor:
        """Pad tensors to the same length."""
        # Determine dimensions and max length
        if len(tensors[0].shape) == 1:
            # 1D tensors (e.g., input_ids)
            max_length = max(len(t) for t in tensors)
            
            # Adjust to pad_to_multiple_of if specified
            if self.pad_to_multiple_of:
                max_length = ((max_length + self.pad_to_multiple_of - 1) // 
                              self.pad_to_multiple_of) * self.pad_to_multiple_of
            
            # Don't exceed max_length if specified
            if self.max_length and max_length > self.max_length:
                max_length = self.max_length
            
            # Determine padding value
            pad_value = 0
            if key == "input_ids":
                pad_value = self.tokenizer.pad_token_id
            
            # Create padded tensor
            padded = torch.full((len(tensors), max_length), pad_value, dtype=tensors[0].dtype)
            
            # Fill with actual values
            for i, tensor in enumerate(tensors):
                length = min(len(tensor), max_length)
                if self.padding_side == "right":
                    padded[i, :length] = tensor[:length]
                else:
                    padded[i, -length:] = tensor[:length]
            
            return padded
        else:
            # For higher dimensional tensors, use simple stacking
            # (assumes all are same shape)
            return torch.stack(tensors)
    
    def _pad_labels(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """Pad label tensors with special handling."""
        # Determine max length
        max_length = max(len(t) for t in tensors)
        
        # Adjust to pad_to_multiple_of if specified
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // 
                          self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Don't exceed max_length if specified
        if self.max_length and max_length > self.max_length:
            max_length = self.max_length
        
        # Create padded tensor
        padded = torch.full(
            (len(tensors), max_length), 
            self.label_pad_token_id, 
            dtype=tensors[0].dtype
        )
        
        # Fill with actual values
        for i, tensor in enumerate(tensors):
            length = min(len(tensor), max_length)
            if self.padding_side == "right":
                padded[i, :length] = tensor[:length]
            else:
                padded[i, -length:] = tensor[:length]
        
        return padded


def create_dataloader(
    dataset: StreamingDataset,
    batch_size: int = 8,
    num_workers: int = 4,
    collator: Optional[DataCollator] = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2
) -> DataLoader:
    """
    Create a DataLoader for a streaming dataset.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        num_workers: Number of worker processes
        collator: Collator for batching examples
        pin_memory: Whether to pin memory for faster data transfer to GPU
        prefetch_factor: Number of batches to prefetch per worker
        
    Returns:
        DataLoader for the dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    ) 