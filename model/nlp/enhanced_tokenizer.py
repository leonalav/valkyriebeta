import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import re
import math

class EnhancedTokenizer(nn.Module):
    """
    Enhanced tokenizer module for the Valkyrie LLM.
    Implements advanced tokenization capabilities with learnable embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1,
        use_positional_embeddings: bool = True,
        use_token_type_embeddings: bool = True,
        num_token_types: int = 2,
        use_factorized_embeddings: bool = False,
        factorized_dim: int = 384,
        config = None
    ):
        """
        Initialize enhanced tokenizer.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Size of hidden states
            max_position_embeddings: Maximum sequence length
            dropout: Dropout probability
            use_positional_embeddings: Whether to use positional embeddings
            use_token_type_embeddings: Whether to use token type embeddings
            num_token_types: Number of token types
            use_factorized_embeddings: Whether to use factorized embeddings
            factorized_dim: Dimension of factorized embeddings
            config: Optional model configuration
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        self.vocab_size = vocab_size if config is None else getattr(config, 'vocab_size', vocab_size)
        self.hidden_size = hidden_size if config is None else getattr(config, 'hidden_size', hidden_size)
        self.max_position_embeddings = max_position_embeddings if config is None else getattr(config, 'max_seq_len', max_position_embeddings)
        self.use_positional_embeddings = use_positional_embeddings
        self.use_token_type_embeddings = use_token_type_embeddings
        self.num_token_types = num_token_types
        self.use_factorized_embeddings = use_factorized_embeddings
        self.factorized_dim = factorized_dim
        
        # Word embeddings
        if use_factorized_embeddings:
            # Factorized embeddings for large vocabularies
            self.word_embeddings_1 = nn.Embedding(vocab_size, factorized_dim)
            self.word_embeddings_2 = nn.Linear(factorized_dim, hidden_size)
        else:
            # Standard embeddings
            self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Positional embeddings
        if use_positional_embeddings:
            self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
            
        # Token type embeddings
        if use_token_type_embeddings:
            self.token_type_embeddings = nn.Embedding(num_token_types, hidden_size)
            
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Special token handling
        self.special_token_embeddings = nn.ParameterDict({
            'pad_token': nn.Parameter(torch.zeros(hidden_size)),
            'unk_token': nn.Parameter(torch.zeros(hidden_size)),
            'bos_token': nn.Parameter(torch.zeros(hidden_size)),
            'eos_token': nn.Parameter(torch.zeros(hidden_size)),
            'mask_token': nn.Parameter(torch.zeros(hidden_size))
        })
        
        # Subword handling
        self.subword_combiner = nn.Linear(hidden_size * 2, hidden_size)
        
        # Initialize state
        self.is_initialized = False
        
        # Entity patterns (simplified)
        self.entity_patterns = {
            'DATE': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            'EMAIL': r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
            'URL': r'https?://[^\s]+',
            'NUMBER': r'\b\d+(?:\.\d+)?\b'
        }
        
    def initialize(self):
        """Initialize tokenizer components"""
        if not self.is_initialized:
            # Initialize embeddings
            if self.use_factorized_embeddings:
                nn.init.normal_(self.word_embeddings_1.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.word_embeddings_2.weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.word_embeddings_2.bias)
            else:
                nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
                
            if self.use_positional_embeddings:
                nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
                
            if self.use_token_type_embeddings:
                nn.init.normal_(self.token_type_embeddings.weight, mean=0.0, std=0.02)
                
            # Initialize special token embeddings
            for param in self.special_token_embeddings.values():
                nn.init.normal_(param, mean=0.0, std=0.02)
                
            # Initialize subword combiner
            nn.init.normal_(self.subword_combiner.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.subword_combiner.bias)
                
            self.is_initialized = True
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
        subword_mask: Optional[torch.Tensor] = None
    ):
        """
        Convert input IDs to embeddings.
        
        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            token_type_ids: Optional token type IDs of shape [batch_size, seq_len]
            position_ids: Optional position IDs of shape [batch_size, seq_len]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            special_tokens_mask: Optional mask for special tokens of shape [batch_size, seq_len]
            subword_mask: Optional mask for subword tokens of shape [batch_size, seq_len]
            
        Returns:
            embeddings: Token embeddings of shape [batch_size, seq_len, hidden_size]
        """
        # Ensure initialized
        if not self.is_initialized:
            self.initialize()
            
        # Get input shape
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Get word embeddings
        if self.use_factorized_embeddings:
            # Apply factorized embeddings
            embeddings = self.word_embeddings_1(input_ids)
            embeddings = self.word_embeddings_2(embeddings)
        else:
            # Apply standard embeddings
            embeddings = self.word_embeddings(input_ids)
        
        # Add positional embeddings if enabled
        if self.use_positional_embeddings:
            # Create position IDs if not provided
            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
                
            # Add positional embeddings
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
            
        # Add token type embeddings if enabled
        if self.use_token_type_embeddings and token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
            
        # Handle special tokens if mask provided
        if special_tokens_mask is not None:
            # For each special token type
            for token_name, token_embedding in self.special_token_embeddings.items():
                # Create token-specific mask based on token IDs
                if token_name == 'pad_token' and hasattr(self, 'pad_token_id'):
                    token_mask = (input_ids == self.pad_token_id).float()
                elif token_name == 'unk_token' and hasattr(self, 'unk_token_id'):
                    token_mask = (input_ids == self.unk_token_id).float()
                elif token_name == 'bos_token' and hasattr(self, 'bos_token_id'):
                    token_mask = (input_ids == self.bos_token_id).float()
                elif token_name == 'eos_token' and hasattr(self, 'eos_token_id'):
                    token_mask = (input_ids == self.eos_token_id).float()
                elif token_name == 'mask_token' and hasattr(self, 'mask_token_id'):
                    token_mask = (input_ids == self.mask_token_id).float()
                else:
                    # Skip if token ID not defined
                    continue
                    
                # Apply special token embedding
                token_mask = token_mask.unsqueeze(-1)
                special_embedding = token_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)
                embeddings = embeddings * (1 - token_mask) + special_embedding * token_mask
        
        # Handle subword tokens if mask provided
        if subword_mask is not None:
            # Identify subword tokens
            subword_positions = torch.nonzero(subword_mask, as_tuple=True)
            
            if len(subword_positions[0]) > 0:
                # Get previous token embeddings for subwords
                prev_positions = (subword_positions[0], subword_positions[1] - 1)
                
                # Only process valid previous positions
                valid_mask = prev_positions[1] >= 0
                if valid_mask.any():
                    valid_batch_idx = subword_positions[0][valid_mask]
                    valid_prev_pos = prev_positions[1][valid_mask]
                    valid_curr_pos = subword_positions[1][valid_mask]
                    
                    # Get embeddings for current and previous tokens
                    prev_embeddings = embeddings[valid_batch_idx, valid_prev_pos]
                    curr_embeddings = embeddings[valid_batch_idx, valid_curr_pos]
                    
                    # Combine embeddings
                    combined = torch.cat([prev_embeddings, curr_embeddings], dim=-1)
                    new_embeddings = self.subword_combiner(combined)
                    
                    # Update embeddings
                    embeddings[valid_batch_idx, valid_curr_pos] = new_embeddings
        
        # Apply layer normalization and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        max_length: Optional[int] = None,
        padding: str = 'max_length',
        truncation: bool = True,
        return_tensors: str = 'pt'
    ):
        """
        Encode text into token IDs.
        This is a placeholder method that should be implemented by a proper tokenizer.
        
        Args:
            texts: Input text or list of texts
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length' or 'longest')
            truncation: Whether to truncate sequences
            return_tensors: Return format ('pt' for PyTorch tensors)
            
        Returns:
            Dict containing:
                - input_ids: Token IDs
                - attention_mask: Attention mask
        """
        # This is a placeholder - in a real implementation, this would use a proper tokenizer
        if isinstance(texts, str):
            texts = [texts]
            
        # Set default max length if not provided
        if max_length is None:
            max_length = self.max_position_embeddings
            
        # Create dummy token IDs and attention mask
        batch_size = len(texts)
        input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        
        # Return dummy tensors
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def tokenize(self, text: str, context: Optional[str] = None, **kwargs):
        """
        Tokenize text with enhanced capabilities
        
        Args:
            text: Text to tokenize
            context: Optional context for contextual tokenization
            **kwargs: Additional arguments for base tokenizer
            
        Returns:
            tokens: Tokenized output
            entity_info: Information about detected entities
        """
        # Ensure initialized
        if not self.is_initialized:
            self.initialize()
            
        # Entity detection
        entity_info = {}
        if self.handle_entities:
            text, entity_info = self._detect_entities(text)
            
        # Base tokenization
        tokens = self.base_tokenizer.tokenize(text, **kwargs)
        
        # Add special tokens if needed
        if self.add_special_tokens and 'add_special_tokens' not in kwargs:
            # Check if special tokens are already added
            if not (tokens and (tokens[0] == self.base_tokenizer.cls_token or 
                               tokens[0] == self.base_tokenizer.bos_token)):
                # Add special tokens based on tokenizer type
                if hasattr(self.base_tokenizer, 'cls_token') and self.base_tokenizer.cls_token:
                    tokens = [self.base_tokenizer.cls_token] + tokens + [self.base_tokenizer.sep_token]
                elif hasattr(self.base_tokenizer, 'bos_token') and self.base_tokenizer.bos_token:
                    tokens = [self.base_tokenizer.bos_token] + tokens + [self.base_tokenizer.eos_token]
                    
        # Apply contextual tokenization if context is provided
        if self.use_contextual_tokenization and context is not None:
            tokens = self._apply_context(tokens, context)
            
        return tokens, entity_info
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True, **kwargs):
        """
        Decode token IDs back to text
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments for base tokenizer
            
        Returns:
            text: Decoded text
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        # Decode using base tokenizer
        text = self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        
        # Post-process decoded text (e.g., restore entities)
        # This would require tracking entity positions during encoding
        
        return text
    
    def batch_encode_plus(self, texts: List[str], contexts: Optional[List[str]] = None, **kwargs):
        """
        Encode a batch of texts
        
        Args:
            texts: List of texts to encode
            contexts: Optional list of contexts for contextual encoding
            **kwargs: Additional arguments for base tokenizer
            
        Returns:
            batch_encoding: Batch of encoded inputs
        """
        # Process each text
        batch_encodings = []
        batch_entity_info = []
        
        for i, text in enumerate(texts):
            context = contexts[i] if contexts and i < len(contexts) else None
            encoding = self.encode(text, context, **kwargs)
            batch_encodings.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            })
            batch_entity_info.append(encoding['entity_info'])
            
        # Combine into batch
        batch_input_ids = [enc['input_ids'] for enc in batch_encodings]
        batch_attention_mask = [enc['attention_mask'] for enc in batch_encodings]
        
        # Pad to max length in batch
        max_length = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        
        for input_ids, attention_mask in zip(batch_input_ids, batch_attention_mask):
            padding_length = max_length - len(input_ids)
            
            if padding_length > 0:
                # Get padding token ID
                pad_token_id = self.base_tokenizer.pad_token_id if hasattr(self.base_tokenizer, 'pad_token_id') else 0
                
                # Pad input IDs and attention mask
                padded_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype)
                ])
                padded_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=attention_mask.dtype)
                ])
            else:
                padded_ids = input_ids
                padded_mask = attention_mask
                
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            
        # Stack into tensors
        batch_encoding = {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask),
            'entity_info': batch_entity_info
        }
        
        return batch_encoding
    
    def _detect_entities(self, text: str):
        """
        Detect entities in text
        
        Args:
            text: Input text
            
        Returns:
            processed_text: Text with entity markers
            entity_info: Information about detected entities
        """
        entity_info = {
            'entities': [],
            'positions': []
        }
        
        # Detect entities using patterns
        for entity_type, pattern in self.entity_patterns.items():
            for match in re.finditer(pattern, text):
                start, end = match.span()
                entity_value = match.group()
                
                entity_info['entities'].append({
                    'type': entity_type,
                    'value': entity_value,
                    'start': start,
                    'end': end
                })
                entity_info['positions'].append((start, end, entity_type))
                
        # Sort positions by start index
        entity_info['positions'].sort()
        
        # Process text with entity markers (optional)
        # This is a simplified approach - in a real implementation,
        # you might want to replace entities with special tokens
        
        return text, entity_info
    
    def _apply_context(self, tokens: List[str], context: str):
        """
        Apply context to tokenization
        
        Args:
            tokens: Tokenized text
            context: Context information
            
        Returns:
            contextualized_tokens: Tokens with context applied
        """
        # This is a placeholder for contextual tokenization
        # In a real implementation, this would modify tokenization based on context
        
        return tokens
    
    def __len__(self):
        """Return vocabulary size of base tokenizer"""
        if hasattr(self.base_tokenizer, '__len__'):
            return len(self.base_tokenizer)
        elif hasattr(self.base_tokenizer, 'vocab_size'):
            return self.base_tokenizer.vocab_size
        else:
            return 0 