from transformers import PreTrainedTokenizer
from typing import List, Dict, Optional
import torch
import logging

class LogicalTokenizer:
    def __init__(self, 
                 base_tokenizer: PreTrainedTokenizer,
                 special_tokens: Optional[List[str]] = None):
        self.base_tokenizer = base_tokenizer
        self.logger = logging.getLogger(__name__)
        
        # Handle Gemma tokenizer specifics
        if "gemma" in str(base_tokenizer.__class__):
            if base_tokenizer.pad_token is None:
                base_tokenizer.pad_token = base_tokenizer.eos_token
            base_tokenizer.padding_side = "left"
            base_tokenizer.truncation_side = "left"
        
        # Add special tokens for logical operations
        default_special_tokens = [
            '[AND]', '[OR]', '[NOT]', '[IMPLIES]', '[IFF]',
            '[PREMISE]', '[CONCLUSION]', '[TREE]', '[/TREE]'
        ]
        special_tokens = special_tokens or default_special_tokens
        
        # Add special tokens to tokenizer
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        self.base_tokenizer.add_special_tokens(special_tokens_dict)
        
    def tokenize(self, 
                 text: str,
                 max_length: Optional[int] = None,
                 padding: str = 'max_length',
                 **kwargs) -> Dict[str, torch.Tensor]:
        """Tokenize input text"""
        # Process logical operators
        text = self._process_logical_operators(text)
        
        # Tokenize using base tokenizer
        encoded = self.base_tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=True,
            return_tensors='pt',
            **kwargs
        )
        
        return encoded
    
    def _process_logical_operators(self, text: str) -> str:
        """Process logical operators in text"""
        # Replace logical operators with special tokens
        replacements = {
            'AND': '[AND]',
            'OR': '[OR]',
            'NOT': '[NOT]',
            'IMPLIES': '[IMPLIES]',
            'IFF': '[IFF]'
        }
        
        for op, special_token in replacements.items():
            text = text.replace(f' {op} ', f' {special_token} ')
            
        return text
    
    def decode(self, 
               token_ids: torch.Tensor,
               skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        return self.base_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def encode_logical_tree(self, tree: Dict) -> str:
        """Encode logical reasoning tree as text"""
        def _encode_node(node):
            if isinstance(node, str):
                return node
            
            operation = node['operation']
            arguments = node['arguments']
            
            encoded_args = [_encode_node(arg) for arg in arguments]
            return f"[TREE] [{operation}] {' '.join(encoded_args)} [/TREE]"
        
        return _encode_node(tree)
    
    @property
    def vocab_size(self) -> int:
        return len(self.base_tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        return self.base_tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self.base_tokenizer.eos_token_id