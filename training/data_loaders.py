import os
import logging
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from typing import Dict, List, Optional, Any, Union
from datasets import load_dataset
import json
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Dataset for causal language modeling"""
    
    def __init__(self, tokenizer, file_path, block_size=1024, use_rwkv_chunking=False, chunk_size=None):
        """
        Create a text dataset for causal language modeling
        
        Args:
            tokenizer: Tokenizer to use
            file_path: Path to data file
            block_size: Maximum sequence length
            use_rwkv_chunking: Whether to use RWKV-specific chunking
            chunk_size: Size of chunks for RWKV processing
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.use_rwkv_chunking = use_rwkv_chunking
        self.chunk_size = chunk_size or block_size
        
        logger.info(f"Loading dataset from {file_path}")
        self.examples = []
        
        # Determine file extension
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Tokenize text
            self.tokenize_and_process_text(text)
        else:
            # Use datasets library for other formats
            dataset = load_dataset("text", data_files=file_path)["train"]
            
            # Tokenize dataset
            for item in dataset:
                self.tokenize_and_process_text(item["text"])
                
        logger.info(f"Created dataset with {len(self.examples)} examples")
    
    def tokenize_and_process_text(self, text):
        """Tokenize and process text into examples"""
        tokenized_text = self.tokenizer.encode(text)
        
        # Process into examples
        for i in range(0, len(tokenized_text) - self.block_size + 1, self.block_size // 2):
            chunk = tokenized_text[i:i + self.block_size]
            self.examples.append(chunk)
            
            # For RWKV chunking, also create smaller chunks
            if self.use_rwkv_chunking and self.chunk_size < self.block_size:
                for j in range(0, len(chunk) - self.chunk_size + 1, self.chunk_size):
                    self.examples.append(chunk[j:j + self.chunk_size])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Convert to tensors
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()
        }


class RLHFDataset(Dataset):
    """Dataset for RLHF training"""
    
    def __init__(self, tokenizer, file_path, block_size=1024, use_rwkv_chunking=False, chunk_size=None):
        """
        Create a dataset for RLHF training
        
        Args:
            tokenizer: Tokenizer to use
            file_path: Path to data file
            block_size: Maximum sequence length
            use_rwkv_chunking: Whether to use RWKV-specific chunking
            chunk_size: Size of chunks for RWKV processing
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.use_rwkv_chunking = use_rwkv_chunking
        self.chunk_size = chunk_size or block_size
        
        logger.info(f"Loading RLHF dataset from {file_path}")
        
        # Load RLHF dataset (expected format: jsonl with 'prompt', 'chosen', 'rejected' fields)
        self.dataset = load_dataset("json", data_files=file_path)["train"]
        
        # Process examples
        self.examples = []
        for item in self.dataset:
            # Process prompt + chosen
            chosen_text = item["prompt"] + item["chosen"]
            chosen_tokens = self.tokenizer.encode(chosen_text)
            
            # Process prompt + rejected
            rejected_text = item["prompt"] + item["rejected"]
            rejected_tokens = self.tokenizer.encode(rejected_text)
            
            # Truncate if needed
            chosen_tokens = chosen_tokens[:self.block_size]
            rejected_tokens = rejected_tokens[:self.block_size]
            
            # Add to examples
            self.examples.append({
                "chosen": chosen_tokens,
                "rejected": rejected_tokens
            })
            
            # For RWKV chunking
            if self.use_rwkv_chunking and self.chunk_size < self.block_size:
                # Create smaller chunks
                for i in range(0, len(chosen_tokens) - self.chunk_size + 1, self.chunk_size // 2):
                    chosen_chunk = chosen_tokens[i:i + self.chunk_size]
                    rejected_chunk = rejected_tokens[i:min(i + self.chunk_size, len(rejected_tokens))]
                    
                    # Only add if rejected chunk is long enough
                    if len(rejected_chunk) >= self.chunk_size // 2:
                        self.examples.append({
                            "chosen": chosen_chunk,
                            "rejected": rejected_chunk
                        })
        
        logger.info(f"Created RLHF dataset with {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Convert to tensors
        chosen = torch.tensor(self.examples[idx]["chosen"], dtype=torch.long)
        rejected = torch.tensor(self.examples[idx]["rejected"], dtype=torch.long)
        
        return {
            "chosen_input_ids": chosen,
            "rejected_input_ids": rejected
        }


def setup_train_dataloader(args, tokenizer, training_config, batch_size=None, use_rwkv_chunking=False, chunk_size=None):
    """
    Set up training data loader
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer
        training_config: Training configuration
        batch_size: Optional batch size override
        use_rwkv_chunking: Whether to use RWKV-specific chunking
        chunk_size: Size of chunks for RWKV processing
        
    Returns:
        Training data loader
    """
    logger.info("Setting up training data loader")
    
    # Get batch size
    batch_size = batch_size or args.batch_size
    
    # Create dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=args.train_file,
        block_size=args.max_seq_length,
        use_rwkv_chunking=use_rwkv_chunking,
        chunk_size=chunk_size
    )
    
    # Create sampler
    sampler = RandomSampler(dataset)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    
    return dataloader


def setup_val_dataloader(args, tokenizer, training_config, batch_size=None, use_rwkv_chunking=False, chunk_size=None):
    """
    Set up validation data loader
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer
        training_config: Training configuration
        batch_size: Optional batch size override
        use_rwkv_chunking: Whether to use RWKV-specific chunking
        chunk_size: Size of chunks for RWKV processing
        
    Returns:
        Validation data loader
    """
    logger.info("Setting up validation data loader")
    
    # Get batch size
    batch_size = batch_size or args.batch_size
    
    # Create dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=args.val_file,
        block_size=args.max_seq_length,
        use_rwkv_chunking=use_rwkv_chunking,
        chunk_size=chunk_size
    )
    
    # Create sampler
    sampler = SequentialSampler(dataset)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    
    return dataloader


def setup_domain_dataloaders(args, tokenizer, training_config, batch_size=None, use_rwkv_chunking=False, chunk_size=None):
    """
    Set up domain-specific data loaders
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer
        training_config: Training configuration
        batch_size: Optional batch size override
        use_rwkv_chunking: Whether to use RWKV-specific chunking
        chunk_size: Size of chunks for RWKV processing
        
    Returns:
        Dictionary of domain data loaders
    """
    logger.info("Setting up domain-specific data loaders")
    
    # Get batch size
    batch_size = batch_size or args.batch_size
    
    # Check if domain files are provided
    if not hasattr(args, 'domain_files') or not args.domain_files:
        logger.warning("No domain files provided, skipping domain data loader setup")
        return None
    
    # Create data loaders
    domain_dataloaders = {}
    for domain_name, file_path in args.domain_files.items():
        logger.info(f"Setting up data loader for domain: {domain_name}")
        
        # Create dataset
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.max_seq_length,
            use_rwkv_chunking=use_rwkv_chunking,
            chunk_size=chunk_size
        )
        
        # Create sampler
        sampler = RandomSampler(dataset)
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=collate_fn
        )
        
        domain_dataloaders[domain_name] = dataloader
    
    return domain_dataloaders


def setup_rlhf_dataloader(args, tokenizer, training_config, batch_size=None, use_rwkv_chunking=False, chunk_size=None):
    """
    Set up RLHF data loader
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer
        training_config: Training configuration
        batch_size: Optional batch size override
        use_rwkv_chunking: Whether to use RWKV-specific chunking
        chunk_size: Size of chunks for RWKV processing
        
    Returns:
        RLHF data loader
    """
    logger.info("Setting up RLHF data loader")
    
    # Get batch size
    batch_size = batch_size or args.batch_size
    
    # Check if RLHF file is provided
    if not hasattr(args, 'rlhf_file') or not args.rlhf_file:
        logger.warning("No RLHF file provided, skipping RLHF data loader setup")
        return None
    
    # Create dataset
    dataset = RLHFDataset(
        tokenizer=tokenizer,
        file_path=args.rlhf_file,
        block_size=args.max_seq_length,
        use_rwkv_chunking=use_rwkv_chunking,
        chunk_size=chunk_size
    )
    
    # Create sampler
    sampler = RandomSampler(dataset)
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=rlhf_collate_fn
    )
    
    return dataloader


def collate_fn(batch):
    """Collate function for standard data loader"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "labels": labels
    }


def rlhf_collate_fn(batch):
    """Collate function for RLHF data loader"""
    chosen_input_ids = torch.stack([item["chosen_input_ids"] for item in batch])
    rejected_input_ids = torch.stack([item["rejected_input_ids"] for item in batch])
    
    return {
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids
    }


class LMDataset(Dataset):
    """Dataset for language model training."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        stride: int = 512,
        overlap: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file
            tokenizer: Tokenizer to use
            max_seq_length: Maximum sequence length
            stride: Stride for sliding window
            overlap: Whether to overlap sequences
            cache_dir: Directory to cache processed data
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.stride = stride
        self.overlap = overlap
        
        # Load and process data
        self.examples = self._load_and_process_data(data_path, cache_dir)
        
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def _load_and_process_data(self, data_path: str, cache_dir: Optional[str] = None) -> List[Dict[str, torch.Tensor]]:
        """Load and process the data."""
        # Check for cached data
        if cache_dir:
            cache_path = os.path.join(cache_dir, f"{os.path.basename(data_path)}.pt")
            if os.path.exists(cache_path):
                logger.info(f"Loading cached data from {cache_path}")
                return torch.load(cache_path)
        
        # Load raw data
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.json'):
                data = json.load(f)
                texts = [item['text'] for item in data]
            else:
                texts = [line.strip() for line in f if line.strip()]
        
        # Process texts into examples
        examples = []
        for text in texts:
            # Tokenize text
            encodings = self.tokenizer(
                text,
                max_length=self.max_seq_length,
                truncation=True,
                return_tensors='pt'
            )
            
            # Create sliding windows if needed
            if len(encodings['input_ids'][0]) > self.max_seq_length:
                input_ids = encodings['input_ids'][0]
                attention_mask = encodings['attention_mask'][0]
                
                # Create sliding windows
                for i in range(0, len(input_ids), self.stride):
                    if not self.overlap and i > 0:
                        i = i - self.max_seq_length + self.stride
                    
                    window_input_ids = input_ids[i:i + self.max_seq_length]
                    window_attention_mask = attention_mask[i:i + self.max_seq_length]
                    
                    # Pad if necessary
                    if len(window_input_ids) < self.max_seq_length:
                        padding_length = self.max_seq_length - len(window_input_ids)
                        window_input_ids = torch.cat([
                            window_input_ids,
                            torch.full((padding_length,), self.tokenizer.pad_token_id)
                        ])
                        window_attention_mask = torch.cat([
                            window_attention_mask,
                            torch.zeros(padding_length, dtype=torch.long)
                        ])
                    
                    examples.append({
                        'input_ids': window_input_ids,
                        'attention_mask': window_attention_mask
                    })
            else:
                examples.append({
                    'input_ids': encodings['input_ids'][0],
                    'attention_mask': encodings['attention_mask'][0]
                })
        
        # Cache processed data if cache_dir is provided
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{os.path.basename(data_path)}.pt")
            torch.save(examples, cache_path)
            logger.info(f"Cached processed data to {cache_path}")
        
        return examples
    
    def __len__(self) -> int:
        """Get the number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an example by index."""
        example = self.examples[idx]
        
        # Create labels (shifted input_ids)
        labels = example['input_ids'].clone()
        labels[:-1] = labels[1:].clone()
        labels[-1] = -100  # Ignore last token in loss calculation
        
        return {
            'input_ids': example['input_ids'],
            'attention_mask': example['attention_mask'],
            'labels': labels
        }

def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset to create loader for
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        drop_last: Whether to drop incomplete batches
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

def load_dataset(data_path: str, config: Any) -> LMDataset:
    """
    Load dataset from path.
    
    Args:
        data_path: Path to the data file
        config: Training configuration
        
    Returns:
        LMDataset instance
    """
    return LMDataset(
        data_path=data_path,
        tokenizer=config.tokenizer,
        max_seq_length=config.max_seq_length
    ) 