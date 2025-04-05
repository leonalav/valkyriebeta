"""
Transformer models implementation.

This module provides various transformer architectures, including:
- Standard transformer encoder
- Efficient transformer with optimized attention
- Enhanced transformer with additional capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

class TransformerConfig:
    """Configuration class for transformer models"""
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_linear_attention: bool = False,
        linear_attention_feature_dim: int = 16,
        **kwargs
    ):
        """
        Initialize transformer configuration.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Size of hidden layers
            num_hidden_layers: Number of hidden layers
            num_attention_heads: Number of attention heads
            intermediate_size: Size of intermediate layers
            hidden_act: Activation function for hidden layers
            hidden_dropout_prob: Dropout probability for hidden layers
            attention_probs_dropout_prob: Dropout probability for attention
            max_position_embeddings: Maximum sequence length
            type_vocab_size: Number of token types
            initializer_range: Range for weight initialization
            layer_norm_eps: Epsilon for layer normalization
            use_linear_attention: Whether to use linear attention
            linear_attention_feature_dim: Feature dimension for linear attention
            **kwargs: Additional arguments
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_linear_attention = use_linear_attention
        self.linear_attention_feature_dim = linear_attention_feature_dim
        
        # Add any additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer.
    
    This is a single layer of the transformer encoder, containing:
    - Multi-head self-attention
    - Position-wise feed-forward network
    - Layer normalization and dropout
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        use_linear_attention: bool = False,
        linear_attention_dim: int = 16
    ):
        """
        Initialize transformer encoder layer.
        
        Args:
            hidden_size: Size of hidden layers
            num_attention_heads: Number of attention heads
            intermediate_size: Size of intermediate feed-forward layer
            dropout: Dropout probability
            activation: Activation function
            layer_norm_eps: Epsilon for layer normalization
            use_linear_attention: Whether to use linear attention
            linear_attention_dim: Feature dimension for linear attention
        """
        super().__init__()
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward layers
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation function: {activation}")
            
        # Store configuration
        self.use_linear_attention = use_linear_attention
        self.linear_attention_dim = linear_attention_dim
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for transformer encoder layer.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        # Prepare attention mask for PyTorch's MultiheadAttention
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] -> [batch_size, seq_len, seq_len]
            # with 1s to attend and 0s to mask
            extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, attention_mask.size(1), -1)
            
            # Convert from 1s for attend, 0s for mask to 0s for attend, -inf for mask
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None
        
        # Self-attention with residual connection and layer norm
        attn_output, attn_weights = self.self_attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=extended_attention_mask,
            need_weights=output_attentions
        )
        hidden_states = hidden_states + self.dropout1(attn_output)
        hidden_states = self.norm1(hidden_states)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(hidden_states))))
        hidden_states = hidden_states + self.dropout2(ff_output)
        hidden_states = self.norm2(hidden_states)
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
            
        return outputs

class Transformer(nn.Module):
    """
    Transformer encoder model.
    
    This is a transformer encoder model, consisting of:
    - Embedding layers for tokens and positions
    - Multiple transformer encoder layers
    - Output pooling
    """
    
    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1,
        max_position_embeddings: int = 512,
        **kwargs
    ):
        """
        Initialize transformer model.
        
        Args:
            config: Transformer configuration
            vocab_size: Size of vocabulary
            hidden_size: Size of hidden layers
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            intermediate_size: Size of intermediate feed-forward layer
            dropout: Dropout probability
            max_position_embeddings: Maximum sequence length
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Use provided config or create one from arguments
        self.config = config or TransformerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_position_embeddings,
            **kwargs
        )
        
        # Store configuration parameters for easy access
        self.vocab_size = self.config.vocab_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        
        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Embedding(self.config.max_position_embeddings, self.hidden_size)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_heads,
                intermediate_size=self.config.intermediate_size,
                dropout=self.config.hidden_dropout_prob,
                activation=self.config.hidden_act,
                layer_norm_eps=self.config.layer_norm_eps,
                use_linear_attention=self.config.use_linear_attention,
                linear_attention_dim=self.config.linear_attention_feature_dim
            )
            for _ in range(self.num_layers)
        ])
        
        # Layer normalization and dropout
        self.ln_f = nn.LayerNorm(self.hidden_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized Transformer with {self.num_layers} layers, {self.num_heads} heads")
    
    def _init_weights(self, module):
        """Initialize weights for transformer modules"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Standard Transformer initialization
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for transformer model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings (without token type embeddings for simplicity)
        hidden_states = token_embeds + position_embeds
        
        hidden_states = self.dropout(hidden_states)
        
        # Initialize lists for hidden states and attentions if requested
        all_hidden_states = [hidden_states] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # Apply encoder layers
        for layer_module in self.layers:
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions.append(layer_outputs[1])
                
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Final layer normalization
        hidden_states = self.ln_f(hidden_states)
        
        # Prepare outputs
        outputs = {
            "hidden_states": hidden_states,
            "last_hidden_state": hidden_states,
        }
        
        if output_hidden_states:
            outputs["all_hidden_states"] = all_hidden_states
            
        if output_attentions:
            outputs["attentions"] = all_attentions
            
        return outputs

class EfficientTransformerEnhanced(Transformer):
    """
    Enhanced transformer model with efficient attention and improved scaling.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        activation: str = "gelu",
        config = None
    ):
        """
        Initialize the transformer.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Size of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads in each layer
            intermediate_size: Size of intermediate feed-forward layer
            max_position_embeddings: Maximum sequence length
            dropout: Dropout probability
            layer_norm_eps: Epsilon for layer normalization
            activation: Activation function ("gelu" or "relu")
            config: Optional model configuration
        """
        super().__init__(config=config)
        
        # Use config values if provided
        if config is not None:
            if hasattr(config, 'vocab_size'):
                vocab_size = config.vocab_size
            if hasattr(config, 'hidden_size'):
                hidden_size = config.hidden_size
            if hasattr(config, 'num_layers'):
                num_layers = config.num_layers
            if hasattr(config, 'num_heads'):
                num_heads = config.num_heads
            if hasattr(config, 'intermediate_size'):
                intermediate_size = config.intermediate_size
            if hasattr(config, 'max_position_embeddings'):
                max_position_embeddings = config.max_position_embeddings
            if hasattr(config, 'dropout'):
                dropout = config.dropout
        
        # Store configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout_prob = dropout  # Store as dropout_prob to avoid confusion
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Position embedding
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Dropout
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        logger.info(f"Initialized EfficientTransformerEnhanced with {num_layers} layers, {num_heads} heads")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the transformer.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            output_hidden_states: Whether to output all hidden states
            output_attentions: Whether to output attention weights
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of outputs
        """
        # Get basic dimensions
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device)
        
        # Convert attention mask to additive mask for attention
        # (1.0 for position to attend, 0.0 for positions to mask)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get token embeddings
        token_embeddings = self.token_embedding(input_ids)
        
        # Get position embeddings
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings (without token type embeddings for simplicity)
        embeddings = token_embeddings + position_embeddings
        
        # Apply dropout
        hidden_states = self.embedding_dropout(embeddings)
        
        # Store all hidden states if requested
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        # Process through transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
                
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                output_attention=output_attentions
            )
            
            hidden_states = layer_outputs["hidden_states"]
            
            if output_attentions and "attention" in layer_outputs:
                all_attentions.append(layer_outputs["attention"])
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Add final hidden states if requested
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Prepare outputs
        outputs = {
            "hidden_states": hidden_states
        }
        
        if output_hidden_states:
            outputs["all_hidden_states"] = all_hidden_states
            
        if output_attentions:
            outputs["attentions"] = all_attentions
        
        return outputs

class TransformerLayer(nn.Module):
    """
    A single transformer layer with attention and feed-forward networks.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
        activation: str = "gelu"
    ):
        """
        Initialize the transformer layer.
        
        Args:
            hidden_size: Size of hidden layers
            num_heads: Number of attention heads
            intermediate_size: Size of intermediate feed-forward layer
            dropout: Dropout probability
            layer_norm_eps: Epsilon for layer normalization
            activation: Activation function ("gelu" or "relu")
        """
        super().__init__()
        
        # Self-attention
        self.self_attention = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer norms
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
            activation=activation
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the transformer layer.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, 1, 1, seq_len]
            output_attention: Whether to output attention weights
            
        Returns:
            Dictionary of outputs
        """
        # Self-attention
        attention_outputs = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attention=output_attention
        )
        
        attention_output = attention_outputs["hidden_states"]
        
        # Residual connection and layer norm
        hidden_states = self.attention_layer_norm(hidden_states + self.dropout(attention_output))
        
        # Feed-forward network
        ffn_output = self.ffn(hidden_states)
        
        # Residual connection and layer norm
        hidden_states = self.ffn_layer_norm(hidden_states + self.dropout(ffn_output))
        
        # Prepare outputs
        outputs = {
            "hidden_states": hidden_states
        }
        
        if output_attention:
            outputs["attention"] = attention_outputs.get("attention")
        
        return outputs

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        """
        Initialize the multi-head attention.
        
        Args:
            hidden_size: Size of hidden layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        # Query, key, value projections
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape input tensor for multi-head attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Reshaped tensor [batch_size, num_heads, seq_len, head_size]
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the multi-head attention.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, 1, 1, seq_len]
            output_attention: Whether to output attention weights
            
        Returns:
            Dictionary of outputs
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project query, key, value
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = self.transpose_for_scores(query)  # [batch, num_heads, seq_len, head_size]
        key = self.transpose_for_scores(key)      # [batch, num_heads, seq_len, head_size]
        value = self.transpose_for_scores(value)  # [batch, num_heads, seq_len, head_size]
        
        # Compute scaled dot-product attention
        # (batch, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (self.head_size ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)  # [batch, num_heads, seq_len, head_size]
        
        # Reshape back to original size
        context = context.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, num_heads, head_size]
        context = context.view(batch_size, seq_len, self.hidden_size)  # [batch, seq_len, hidden_size]
        
        # Apply output projection
        output = self.output(context)
        
        # Prepare outputs
        outputs = {
            "hidden_states": output
        }
        
        if output_attention:
            outputs["attention"] = attention_probs
        
        return outputs

class FeedForwardNetwork(nn.Module):
    """
    Feed-forward network used in transformer layers.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize the feed-forward network.
        
        Args:
            hidden_size: Size of hidden layers
            intermediate_size: Size of intermediate layer
            dropout: Dropout probability
            activation: Activation function ("gelu" or "relu")
        """
        super().__init__()
        
        # Layers
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        
        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feed-forward network.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Output hidden states [batch_size, seq_len, hidden_size]
        """
        # First linear layer + activation
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.activation(intermediate_output)
        
        # Second linear layer
        output = self.output(intermediate_output)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output