import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Any
import json
import logging
from .preprocessor import LogicalExample, LogicalDataPreprocessor
import numpy as np

class LogicalReasoningDataset(Dataset):
    def __init__(self, 
                 examples: List[LogicalExample],
                 tokenizer: Any,
                 max_length: int = 512,
                 is_training: bool = True):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.logger = logging.getLogger(__name__)
        
    @classmethod
    def from_file(cls, 
                  file_path: str,
                  tokenizer: Any,
                  config: Any) -> 'LogicalReasoningDataset':
        """Load dataset from preprocessed file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        preprocessor = LogicalDataPreprocessor(tokenizer, config)
        examples = []
        
        for item in data:
            example = LogicalExample(
                text=item['text'],
                logical_tree=item['logical_tree'],
                labels=torch.tensor(item['labels']) if item['labels'] else None,
                metadata=item.get('metadata')
            )
            examples.append(example)
            
        return cls(
            examples=examples,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            is_training=True
        )
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize input text
        encoded = self.tokenizer(
            example.text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }
        
        # Add labels if available
        if example.labels is not None:
            item['labels'] = example.labels
            
        # Add logical tree if available
        if example.logical_tree is not None:
            item['logical_tree'] = example.logical_tree
            
        return item
    
    def get_example_text(self, idx: int) -> str:
        """Get original text for an example"""
        return self.examples[idx].text
    
    def get_logical_tree(self, idx: int) -> Optional[Dict]:
        """Get logical reasoning tree for an example"""
        return self.examples[idx].logical_tree
    
    def create_subset(self, indices: List[int]) -> 'LogicalReasoningDataset':
        """Create a subset of the dataset"""
        subset_examples = [self.examples[i] for i in indices]
        return LogicalReasoningDataset(
            examples=subset_examples,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            is_training=self.is_training
        )

class DynamicBatchingDataset(LogicalReasoningDataset):
    """Dataset with dynamic batching based on sequence length"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length_to_indices = self._group_by_length()
        
    def _group_by_length(self) -> Dict[int, List[int]]:
        """Group examples by sequence length for efficient batching"""
        length_to_indices = {}
        
        for idx, example in enumerate(self.examples):
            length = len(self.tokenizer.encode(example.text))
            length_bucket = 2 ** int(np.ceil(np.log2(length)))
            
            if length_bucket not in length_to_indices:
                length_to_indices[length_bucket] = []
            length_to_indices[length_bucket].append(idx)
            
        return length_to_indices
    
    def get_batch_indices(self, batch_size: int) -> List[List[int]]:
        """Get batched indices grouped by similar lengths"""
        batches = []
        
        for length, indices in self.length_to_indices.items():
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batches.append(batch_indices)
                
        return batches 

class MemoryEfficientDataset(Dataset):
    def __init__(self, data_path, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path):
        # Memory-mapped file reading
        return np.load(data_path, mmap_mode='r')
    
    def __getitem__(self, idx):
        # Load and tokenize on-the-fly
        item = self.data[idx]
        tokenized = self.tokenizer(
            item,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )
        return tokenized
    
    def __len__(self):
        return len(self.data) 