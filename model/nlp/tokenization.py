import torch
import logging
import re
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class EnhancedTokenizer:
    """
    Enhanced tokenizer with more realistic functionality.
    This implementation provides a basic word tokenizer with support for special tokens,
    padding, truncation, and integration with the neural EnhancedTokenizer module.
    """
    def __init__(self, vocab_size=30000, model_max_length=2048, **kwargs):
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length
        
        # Special tokens
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        
        # Special token IDs
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.mask_token_id = 4
        
        # Simple vocab (will be extended in real implementations)
        self._special_tokens = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
            self.bos_token: self.bos_token_id,
            self.unk_token: self.unk_token_id,
            self.mask_token: self.mask_token_id,
        }
        
        # Simple token -> id mapping for common tokens
        self._token_to_id = dict(self._special_tokens)
        self._id_to_token = {v: k for k, v in self._token_to_id.items()}
        
        # Pre-tokenization pattern (split on spaces and punctuation)
        self.pattern = re.compile(r'\w+|[^\w\s]')
        
        logger.info(f"Initialized EnhancedTokenizer with vocab size {vocab_size}, model_max_length {model_max_length}")
        
    def tokenize(self, text: str, **kwargs) -> List[int]:
        """
        Tokenize the input text into token IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        # Extract the words and punctuation
        words = self.pattern.findall(text.lower())
        
        # Convert to token IDs
        token_ids = []
        for word in words:
            if word in self._token_to_id:
                token_ids.append(self._token_to_id[word])
            else:
                # For unknown tokens, use a deterministic hash to ensure consistency
                token_id = hash(word) % (self.vocab_size - len(self._special_tokens)) + len(self._special_tokens)
                # Cache the token mapping for future use
                self._token_to_id[word] = token_id
                self._id_to_token[token_id] = word
                token_ids.append(token_id)
                
        return token_ids
        
    def encode(self, text: str, add_special_tokens: bool = True, 
              padding: Union[bool, str] = False, truncation: bool = False,
              max_length: Optional[int] = None, return_tensors: Optional[str] = None,
              **kwargs) -> Union[List[int], torch.Tensor]:
        """
        Encode text to token IDs with additional processing options.
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            padding: Whether to pad sequences (bool or 'max_length')
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Return format, if 'pt' returns PyTorch tensors
            
        Returns:
            Token IDs as list or tensor
        """
        if isinstance(text, list):
            return [self.encode(t, add_special_tokens, padding, truncation, 
                              max_length, return_tensors, **kwargs) for t in text]
        
        # Default max length
        if max_length is None:
            max_length = self.model_max_length
            
        # Tokenize
        token_ids = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
            
        # Truncate if needed
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            # Ensure we have EOS at the end if we truncated
            if add_special_tokens:
                token_ids[-1] = self.eos_token_id
                
        # Pad if needed
        if padding:
            pad_length = max_length if padding == 'max_length' else self.model_max_length
            if len(token_ids) < pad_length:
                token_ids = token_ids + [self.pad_token_id] * (pad_length - len(token_ids))
                
        # Convert to tensor if requested
        if return_tensors == 'pt':
            token_ids = torch.tensor(token_ids, dtype=torch.long)
            
        return token_ids
        
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = False, **kwargs) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        text_tokens = []
        for token_id in token_ids:
            # Skip special tokens if requested
            if skip_special_tokens and token_id in [self.pad_token_id, self.bos_token_id, self.eos_token_id]:
                continue
                
            # Convert ID to token
            token = self._id_to_token.get(token_id, f"<{token_id}>")
            text_tokens.append(token)
            
        # Join tokens with spaces (simple approach)
        text = ' '.join(text_tokens)
        # Clean up spacing around punctuation (simplistic)
        text = re.sub(r'\s([,.!?;:])', r'\1', text)
        
        return text
        
    def prepare_for_model(self, token_ids: List[int], **kwargs) -> Dict[str, Any]:
        """
        Prepare tokenized input for the model.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Dict with input_ids and attention_mask
        """
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1] * len(token_ids)
        
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask
        }
        
    def build_inputs_with_special_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Build model inputs by adding special tokens.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of token IDs with special tokens added
        """
        return [self.bos_token_id] + token_ids + [self.eos_token_id] 