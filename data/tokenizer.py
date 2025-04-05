import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import re
import math
from pathlib import Path

# Import the EnhancedTokenizer from model.nlp
from model.nlp.enhanced_tokenizer import EnhancedTokenizer

class Tokenizer:
    """
    Basic tokenizer class for the Valkyrie LLM.
    Implements simple tokenization capabilities.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 2048,
        add_special_tokens: bool = True,
        vocab_file: Optional[str] = None
    ):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens
            vocab_file: Optional path to vocabulary file
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.mask_token = "[MASK]"
        
        # Special token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.mask_token_id = 4
        
        # Initialize vocabulary
        self._init_vocab(vocab_file)
        
    def _init_vocab(self, vocab_file: Optional[str] = None):
        """Initialize vocabulary"""
        if vocab_file and Path(vocab_file).exists():
            # Load vocabulary from file
            self.vocab = {}
            self.ids_to_tokens = {}
            
            with open(vocab_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    token = line.strip()
                    self.vocab[token] = i
                    self.ids_to_tokens[i] = token
        else:
            # Initialize with special tokens
            self.vocab = {
                self.pad_token: self.pad_token_id,
                self.unk_token: self.unk_token_id,
                self.bos_token: self.bos_token_id,
                self.eos_token: self.eos_token_id,
                self.mask_token: self.mask_token_id
            }
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens
        
        Args:
            text: Input text
            
        Returns:
            tokens: List of tokens
        """
        # Simple whitespace tokenization
        tokens = text.split()
        
        # Add special tokens if needed
        if self.add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
            
        return tokens
    
    def encode(
        self, 
        text: str, 
        max_length: Optional[int] = None,
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text into token IDs
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length' or 'longest')
            truncation: Whether to truncate sequences
            return_tensors: Return format ('pt' for PyTorch tensors)
            
        Returns:
            Dict containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
        """
        # Set default max length if not provided
        if max_length is None:
            max_length = self.max_length
            
        # Tokenize text
        tokens = self.tokenize(text)
        
        # Convert tokens to IDs
        input_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Truncate if needed
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad if needed
        if padding == 'max_length':
            padding_length = max_length - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                
        # Convert to tensors if needed
        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            text: Decoded text
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # Convert IDs to tokens
        tokens = [self.ids_to_tokens.get(id, self.unk_token) for id in token_ids]
        
        # Skip special tokens if needed
        if skip_special_tokens:
            special_tokens = {self.pad_token, self.unk_token, self.bos_token, self.eos_token, self.mask_token}
            tokens = [token for token in tokens if token not in special_tokens]
            
        # Join tokens into text
        text = ' '.join(tokens)
        
        return text
    
    def batch_encode_plus(
        self, 
        texts: List[str], 
        max_length: Optional[int] = None,
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of texts
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length' or 'longest')
            truncation: Whether to truncate sequences
            return_tensors: Return format ('pt' for PyTorch tensors)
            
        Returns:
            Dict containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
        """
        # Encode each text
        encodings = [self.encode(text, max_length=max_length, padding='do_not_pad', 
                                truncation=truncation, return_tensors='') for text in texts]
        
        # Get input IDs and attention masks
        input_ids = [enc['input_ids'] for enc in encodings]
        attention_masks = [enc['attention_mask'] for enc in encodings]
        
        # Determine max length for padding
        if padding == 'longest':
            max_len = max(len(ids) for ids in input_ids)
        else:
            max_len = max_length if max_length is not None else self.max_length
            
        # Pad sequences
        padded_input_ids = []
        padded_attention_masks = []
        
        for ids, mask in zip(input_ids, attention_masks):
            padding_length = max_len - len(ids)
            
            if padding_length > 0:
                padded_ids = ids + [self.pad_token_id] * padding_length
                padded_mask = mask + [0] * padding_length
            else:
                padded_ids = ids
                padded_mask = mask
                
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
            
        # Convert to tensors if needed
        if return_tensors == 'pt':
            padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
            padded_attention_masks = torch.tensor(padded_attention_masks, dtype=torch.long)
            
        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks
        }
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return len(self.vocab) 