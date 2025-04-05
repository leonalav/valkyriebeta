import torch
from typing import Dict, List, Union, Optional, Any
import numpy as np
from dataclasses import dataclass
import json
import logging
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
try:
    from config.model_config import ModelConfig
except ImportError:
    # Fallback for when the module is not in the path
    ModelConfig = None
from validators.security_validator import SecurityValidator

class DataLoaderFactory:
    @staticmethod
    def create_loader(data_files, tokenizer, batch_size, 
                     enable_memory_mapping=True,
                     enable_prefetch=True,
                     enable_caching=True,
                     is_inference=False):
        # Basic implementation - you can expand this based on your needs
        return DataLoader(
            dataset=data_files,
            batch_size=batch_size,
            shuffle=not is_inference,
            pin_memory=enable_memory_mapping
        )

@dataclass
class LogicalExample:
    """Data structure for logical reasoning examples"""
    text: str
    logical_tree: Optional[Dict[str, Any]] = None
    labels: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None

class LogicalDataPreprocessor:
    """Preprocessor for logical reasoning examples"""
    def __init__(self, tokenizer_name: str, config: ModelConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def preprocess_logical_example(self, 
                             text: str, 
                             logical_tree: Optional[Dict] = None,
                             labels: Optional[List[int]] = None) -> LogicalExample:
        """Clean and process a logical reasoning example"""
        text = self._clean_text(text)
        
        if logical_tree is not None:
            logical_tree = self._process_logical_tree(logical_tree)
            
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.long)
            
        return LogicalExample(
            text=text,
            logical_tree=logical_tree,
            labels=labels,
            metadata={'source': 'preprocessed'}
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Add space around logical operators
        logical_operators = ['AND', 'OR', 'NOT', 'IMPLIES', 'IFF']
        for op in logical_operators:
            text = text.replace(op, f' {op} ')
        return text.strip()
    
    def _process_logical_tree(self, tree: Dict) -> Dict:
        """Process and validate logical reasoning tree"""
        processed_tree = {}
        
        def validate_node(node):
            if not isinstance(node, dict):
                return node
            
            required_keys = {'operation', 'arguments'}
            if not all(key in node for key in required_keys):
                raise ValueError(f"Invalid logical tree node: {node}")
            
            return {
                'operation': node['operation'],
                'arguments': [validate_node(arg) for arg in node['arguments']]
            }
        
        try:
            processed_tree = validate_node(tree)
        except Exception as e:
            self.logger.warning(f"Error processing logical tree: {e}")
            return None
            
        return processed_tree
    
    def batch_preprocess(self, 
                        examples: List[Dict[str, Any]], 
                        max_length: Optional[int] = None) -> List[LogicalExample]:
        """Preprocess a batch of examples"""
        processed_examples = []
        max_length = max_length or self.config.max_seq_length
        
        for example in tqdm(examples, desc="Preprocessing examples"):
            try:
                processed = self.preprocess_logical_example(
                    text=example['text'],
                    logical_tree=example.get('logical_tree'),
                    labels=example.get('labels')
                )
                processed_examples.append(processed)
            except Exception as e:
                self.logger.warning(f"Error processing example: {e}")
                continue
                
        return processed_examples
    
    def save_preprocessed(self, 
                         examples: List[LogicalExample], 
                         output_path: str):
        """Save preprocessed examples to disk"""
        # Validate path for security
        validation_result = SecurityValidator.validate_paths([output_path])
        if not validation_result.is_valid:
            error_msg = "; ".join(validation_result.errors)
            self.logger.error(f"Path validation failed: {error_msg}")
            raise ValueError(f"Invalid output path: {output_path}")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
        data_to_save = []
        for example in examples:
            data_to_save.append({
                'text': example.text,
                'logical_tree': example.logical_tree,
                'labels': example.labels.tolist() if example.labels is not None else None,
                'metadata': example.metadata
            })
            
        with open(output_path, 'w') as f:
            json.dump(data_to_save, f)
        
        self.logger.info(f"Saved {len(examples)} preprocessed examples to {output_path}")
    
    def preprocess(self, text: str) -> Dict[str, Any]:
        """Preprocess a single text input."""
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
        }