import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any
import json
import logging
from dataclasses import dataclass

@dataclass
class DistillationExample:
    """Data structure for distillation examples"""
    input_text: str
    teacher_output: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DistillationDataset(Dataset):
    def __init__(
        self,
        examples: List[DistillationExample],
        tokenizer: Any,
        max_length: int = 512,
        teacher_max_length: Optional[int] = None
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_max_length = teacher_max_length or max_length
        self.logger = logging.getLogger(__name__)
        
    @classmethod
    def from_file(cls, file_path: str, tokenizer: Any, config: Any) -> 'DistillationDataset':
        """Load dataset from preprocessed file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        examples = []
        for item in data:
            example = DistillationExample(
                input_text=item['input_text'],
                teacher_output=item.get('teacher_output'),
                metadata=item.get('metadata')
            )
            examples.append(example)
            
        return cls(
            examples=examples,
            tokenizer=tokenizer,
            max_length=config.max_seq_length,
            teacher_max_length=getattr(config, 'teacher_max_length', None)
        )
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize input text
        input_encoding = self.tokenizer(
            example.input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'text': example.input_text
        }
        
        # Add teacher outputs if available
        if example.teacher_output is not None:
            teacher_encoding = self.tokenizer(
                example.teacher_output,
                max_length=self.teacher_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            item['teacher_input_ids'] = teacher_encoding['input_ids'].squeeze(0)
            item['teacher_attention_mask'] = teacher_encoding['attention_mask'].squeeze(0)
        
        return item
