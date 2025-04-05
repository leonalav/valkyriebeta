"""
Mock implementation of ValkyrieLLM for integration testing.

This provides a minimal implementation of the ValkyrieLLM model to allow
the integration examples to run without needing the full model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from .transformer import Transformer, EfficientTransformerEnhanced, TransformerConfig

logger = logging.getLogger(__name__)

class ValkyrieLLM(nn.Module):
    """
    Mock implementation of ValkyrieLLM for integration testing.
    
    This is a simplified version that implements the core interfaces needed
    for the integration examples.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: Optional[int] = None,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
        use_memory: bool = False,
        use_tree_reasoning: bool = False,
        embedding_type: str = "standard",
        **kwargs
    ):
        """
        Initialize a mock ValkyrieLLM model.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Size of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            intermediate_size: Size of intermediate feed-forward layer
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            use_memory: Whether to use memory
            use_tree_reasoning: Whether to use tree reasoning
            embedding_type: Type of embedding to use
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.embedding_type = embedding_type
        
        # Create transformer config
        self.config = TransformerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_seq_length,
            **kwargs
        )
        
        # Create transformer backbone
        self.transformer = EfficientTransformerEnhanced(
            config=self.config,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=self.intermediate_size,
            dropout=dropout,
            max_position_embeddings=max_seq_length,
            **kwargs
        )
        
        # Embeddings from transformer
        self.token_embedding = self.transformer.token_embedding
        
        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights between token embedding and LM head
        self.lm_head.weight = self.token_embedding.weight
        
        # Capability flags
        self.use_memory = use_memory
        self.use_tree_reasoning = use_tree_reasoning
        self._reasoning_enabled = False
        
        logger.info(f"Initialized mock ValkyrieLLM with {num_layers} layers, {num_heads} heads")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ValkyrieLLM.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights
            return_dict: Whether to return outputs as a dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of model outputs
        """
        # Run transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs
        )
        
        # Get hidden states
        hidden_states = transformer_outputs["hidden_states"]
        
        # Apply language modeling head
        logits = self.lm_head(hidden_states)
        
        # Prepare outputs
        outputs = {
            "logits": logits,
            "hidden_states": hidden_states,
        }
        
        if output_hidden_states and "all_hidden_states" in transformer_outputs:
            outputs["all_hidden_states"] = transformer_outputs["all_hidden_states"]
            
        if output_attentions and "attentions" in transformer_outputs:
            outputs["attentions"] = transformer_outputs["attentions"]
            
        # Calculate loss if labels are provided
        if "labels" in kwargs:
            labels = kwargs["labels"]
            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            outputs["loss"] = loss
            
        return outputs
    
    def enable_reasoning(self, reasoning_type: str = "tree"):
        """
        Enable reasoning capabilities.
        In this mock version, this only sets a flag.
        
        Args:
            reasoning_type: Type of reasoning to enable
        """
        logger.info(f"Mock enabling reasoning: {reasoning_type}")
        self._reasoning_enabled = True
        self._reasoning_type = reasoning_type 