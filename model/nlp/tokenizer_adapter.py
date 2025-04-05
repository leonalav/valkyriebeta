"""
Tokenizer Adapter for Neural EnhancedTokenizer

This module bridges the gap between the tokenization class (text → IDs) and 
the neural tokenizer module (IDs → embeddings) providing a seamless interface.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Tuple

from model.nlp.tokenization import EnhancedTokenizer as TokenizerProcessor
from model.nlp.enhanced_tokenizer import EnhancedTokenizer as TokenizerModule

class TokenizerAdapter:
    """
    Adapter that connects the tokenization processor with the neural tokenizer module.
    
    This class provides a unified interface for:
    1. Text preprocessing (tokenization.py)
    2. Neural embedding generation (enhanced_tokenizer.py)
    3. Contextual processing of embeddings
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1,
        use_factorized_embeddings: bool = False,
        config: Optional[Any] = None
    ):
        """
        Initialize the tokenizer adapter.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Size of hidden states
            max_position_embeddings: Maximum sequence length
            dropout: Dropout probability
            use_factorized_embeddings: Whether to use factorized embeddings
            config: Optional model configuration
        """
        # Create tokenizer processor (text → IDs)
        self.processor = TokenizerProcessor(
            vocab_size=vocab_size,
            model_max_length=max_position_embeddings
        )
        
        # Create neural tokenizer module (IDs → embeddings)
        self.module = TokenizerModule(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            use_factorized_embeddings=use_factorized_embeddings,
            config=config
        )
        
        # Store configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        
    def process_text(
        self, 
        text: Union[str, List[str]],
        return_tensors: str = "pt",
        padding: Union[bool, str] = "max_length",
        truncation: bool = True,
        add_special_tokens: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Process text from raw input to prepared tensors ready for the model.
        
        Args:
            text: Input text or list of texts
            return_tensors: Return format ('pt' for PyTorch tensors)
            padding: Whether to pad sequences (bool or 'max_length')
            truncation: Whether to truncate sequences
            add_special_tokens: Whether to add BOS/EOS tokens
            **kwargs: Additional arguments for the tokenizer
            
        Returns:
            Dictionary containing input tensors
        """
        # Convert single text to list for consistent handling
        if isinstance(text, str):
            text = [text]
            
        # Process each text in the list
        encodings = []
        for t in text:
            # Encode text to token IDs
            token_ids = self.processor.encode(
                t, 
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=self.max_position_embeddings,
                return_tensors=None,  # We'll handle tensor conversion later
                **kwargs
            )
            encodings.append(token_ids)
            
        # Pad to max length in batch
        max_length = max(len(ids) for ids in encodings)
        padded_encodings = []
        attention_masks = []
        
        for ids in encodings:
            # Create attention mask
            attention_mask = [1] * len(ids) + [0] * (max_length - len(ids))
            
            # Pad token IDs
            padded_ids = ids + [self.processor.pad_token_id] * (max_length - len(ids))
            
            padded_encodings.append(padded_ids)
            attention_masks.append(attention_mask)
            
        # Convert to tensors
        input_ids = torch.tensor(padded_encodings, dtype=torch.long)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def encode_plus(
        self, 
        text: Union[str, List[str]], 
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text and return a dictionary of tensors.
        
        Args:
            text: Input text or list of texts
            **kwargs: Additional arguments for process_text
            
        Returns:
            Dictionary containing input tensors
        """
        return self.process_text(text, **kwargs)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Call method for easy processing of text.
        
        Args:
            text: Input text or list of texts
            **kwargs: Additional arguments for process_text
            
        Returns:
            Dictionary containing input tensors
        """
        return self.process_text(text, **kwargs)
    
    def embed_tokens(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_position_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert token IDs to neural embeddings.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            return_position_embeddings: Whether to return position embeddings separately
            
        Returns:
            Embedded tokens or tuple of (embedded tokens, position embeddings)
        """
        return self.module(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_position_embeddings=return_position_embeddings
        )
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments for the decoder
            
        Returns:
            Decoded text
        """
        return self.processor.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
    
    def batch_decode(
        self,
        sequences: List[Union[List[int], torch.Tensor]],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Decode a batch of token IDs back to texts.
        
        Args:
            sequences: Batch of token ID sequences
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments for the decoder
            
        Returns:
            List of decoded texts
        """
        return [self.decode(seq, skip_special_tokens=skip_special_tokens, **kwargs) for seq in sequences]
    
    def get_module(self) -> nn.Module:
        """
        Get the neural tokenizer module.
        
        Returns:
            Neural tokenizer module
        """
        return self.module 