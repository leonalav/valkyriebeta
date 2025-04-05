from typing import List, Dict, Optional, Any, Union
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os

from .distillation_dataset import DistillationDataset, DistillationExample
from .dataloader_factory import DataLoaderFactory
from ..model.api_distillation import APITeacherModel

class DistillationPipeline:
    """Pipeline for processing and preparing distillation data"""
    
    def __init__(
        self,
        tokenizer: Any,
        teacher_model: APITeacherModel,
        cache_dir: str = ".cache/distillation",
        output_dir: str = "processed_data/distillation",
        config: Optional[Any] = None
    ):
        self.tokenizer = tokenizer
        self.teacher_model = teacher_model
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
    def process_examples(
        self,
        input_texts: List[str],
        batch_size: int = 32,
        max_length: Optional[int] = None
    ) -> List[DistillationExample]:
        """Process input examples and get teacher outputs"""
        examples = []
        max_length = max_length or self.config.max_seq_length
        
        # Process in batches
        for i in tqdm(range(0, len(input_texts), batch_size), desc="Getting teacher outputs"):
            batch_texts = input_texts[i:i + batch_size]
            
            # Get teacher outputs for batch
            teacher_outputs = []
            for text in batch_texts:
                output = self.teacher_model.get_teacher_response([
                    {"role": "user", "content": text}
                ])
                teacher_outputs.append(output)
            
            # Create examples
            for text, teacher_output in zip(batch_texts, teacher_outputs):
                if teacher_output is not None:
                    example = DistillationExample(
                        input_text=text,
                        teacher_output=teacher_output
                    )
                    examples.append(example)
                
        return examples
    
    def create_dataset(
        self,
        examples: List[DistillationExample],
        output_file: Optional[str] = None
    ) -> DistillationDataset:
        """Create dataset from processed examples"""
        if output_file:
            # Save examples to file
            data_to_save = [{
                'input_text': ex.input_text,
                'teacher_output': ex.teacher_output,
                'metadata': ex.metadata
            } for ex in examples]
            
            with open(output_file, 'w') as f:
                json.dump(data_to_save, f)
                
        return DistillationDataset(
            examples=examples,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length
        )
    
    def create_data_loader(
        self,
        dataset: DistillationDataset,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Create data loader from dataset"""
        return DataLoaderFactory.create_loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
