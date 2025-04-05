from typing import List, Dict, Optional, Union
from transformers import PreTrainedTokenizerFast
import torch

class EfficientTokenizer:
    def __init__(self, base_tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = base_tokenizer
        self.vocab_size = len(base_tokenizer)

class Tokenizer:
    """Simple placeholder tokenizer for example usage"""
    
    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        
        # Simple character-level tokenization for demonstration
        self.char_to_id = {chr(i + 32): i + 3 for i in range(95)}  # ASCII printable chars
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
        # Add special tokens
        self.id_to_char[self.pad_token_id] = "[PAD]"
        self.id_to_char[self.eos_token_id] = "[EOS]"
        self.id_to_char[self.bos_token_id] = "[BOS]"
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids"""
        if not text:
            return []
            
        # Character-level tokenization
        tokens = [self.char_to_id.get(c, self.vocab_size - 1) for c in text]
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
            
        return tokens
        
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text"""
        if not token_ids:
            return ""
            
        # Filter special tokens if requested
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in [self.pad_token_id, self.eos_token_id, self.bos_token_id]]
            
        # Convert ids to characters with a placeholder for unknown tokens
        text = ""
        for t in token_ids:
            if t in self.id_to_char:
                text += self.id_to_char[t]
            else:
                text += "\uFFFD"  # Unicode replacement character for unknown tokens
        
        return text
        
    def batch_encode(self, texts: List[str], padding: bool = True, max_length: int = None) -> Dict[str, torch.Tensor]:
        """Encode a batch of texts"""
        # Encode each text
        encoded = [self.encode(text) for text in texts]
        
        # Determine max length
        if max_length is None:
            max_length = max(len(e) for e in encoded)
            
        # Pad sequences
        if padding:
            encoded = [e + [self.pad_token_id] * (max_length - len(e)) for e in encoded]
            attention_mask = [[1] * len(e) + [0] * (max_length - len(e)) for e in encoded]
        else:
            attention_mask = [[1] * len(e) for e in encoded]
            
        # Convert to tensors
        input_ids = torch.tensor(encoded)
        attention_mask = torch.tensor(attention_mask)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
    def __call__(self, texts: Union[str, List[str]], padding: bool = True, max_length: int = None) -> Dict[str, torch.Tensor]:
        """Convenience method for encoding"""
        if isinstance(texts, str):
            texts = [texts]
            
        return self.batch_encode(texts, padding=padding, max_length=max_length)
