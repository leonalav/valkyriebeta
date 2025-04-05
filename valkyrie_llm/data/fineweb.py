"""
Dataset and dataloader for FineWeb dataset from HuggingFace.
"""

import logging
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

logger = logging.getLogger(__name__)

class FineWebDataset(torch.utils.data.Dataset):
    """Dataset for FineWeb from HuggingFace datasets"""
    
    def __init__(self, tokenizer, dataset_name="HuggingFaceFW/fineweb", 
                 name="sample-10BT", block_size=1024, max_examples=None):
        """
        Create a dataset for FineWeb
        
        Args:
            tokenizer: Tokenizer to use
            dataset_name: HuggingFace dataset name
            name: Dataset config name (e.g. "sample-10BT")
            block_size: Maximum sequence length
            max_examples: Maximum number of examples to use (for testing)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        logger.info(f"Loading FineWeb dataset {dataset_name} with config {name}")
        
        # Load dataset in streaming mode
        self.dataset = load_dataset(dataset_name, name=name, split="train", streaming=True)
        
        # Process examples
        self.examples = []
        
        # Load and tokenize a portion of the dataset
        example_count = 0
        for example in self.dataset:
            text = example["text"]
            # Tokenize text
            tokenized_text = self.tokenizer.encode(text)
            
            # Process into examples
            for i in range(0, len(tokenized_text) - self.block_size + 1, self.block_size // 2):
                chunk = tokenized_text[i:i + self.block_size]
                if len(chunk) == self.block_size:  # Only use complete chunks
                    self.examples.append(chunk)
            
            example_count += 1
            if max_examples is not None and example_count >= max_examples:
                break
                
        logger.info(f"Created dataset with {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # Convert to tensors
        input_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone()
        }

def setup_fineweb_dataloader(tokenizer, block_size, batch_size, num_workers=4, 
                             max_examples=None, use_streaming=False, 
                             dataset_name="HuggingFaceFW/fineweb", config_name="sample-10BT"):
    """
    Set up DataLoader for FineWeb dataset from HuggingFace
    
    Args:
        tokenizer: Tokenizer to use
        block_size: Maximum sequence length
        batch_size: Batch size for training
        num_workers: Number of workers for DataLoader
        max_examples: Maximum number of examples (for testing)
        use_streaming: Whether to use streaming mode (for TPU)
        dataset_name: HuggingFace dataset name
        config_name: Dataset config name
        
    Returns:
        DataLoader for FineWeb dataset
    """
    logger.info(f"Setting up FineWeb dataloader from {dataset_name} with config {config_name}")
    
    # Create dataset
    dataset = FineWebDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        name=config_name,
        block_size=block_size,
        max_examples=max_examples
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"Created dataloader with {len(dataset)} examples, {len(dataloader)} batches")
    
    return dataloader 