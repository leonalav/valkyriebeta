import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

class TextClassifier(nn.Module):
    """
    Text classification module for the Valkyrie LLM.
    Implements classification capabilities on top of transformer hidden states.
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = 2,
        dropout: float = 0.1,
        pooling_type: str = 'mean',
        use_layer_norm: bool = True,
        config = None
    ):
        """
        Initialize text classifier.
        
        Args:
            hidden_size: Size of hidden states
            num_classes: Number of output classes
            dropout: Dropout probability
            pooling_type: Type of pooling to use ('mean', 'max', 'first', 'last')
            use_layer_norm: Whether to use layer normalization
            config: Optional model configuration
        """
        super().__init__()
        
        # Store configuration
        self.config = config
        self.hidden_size = hidden_size if config is None else getattr(config, 'hidden_size', hidden_size)
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        
        if use_layer_norm:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, num_classes)
            )
        
        # Initialize state
        self.is_initialized = False
        
    def initialize(self):
        """Initialize classifier components"""
        if not self.is_initialized:
            # Initialize weights
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                        
            self.is_initialized = True
    
    def pool_hidden_states(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Pool hidden states based on pooling type.
        
        Args:
            hidden_states: Hidden states of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            
        Returns:
            pooled: Pooled hidden states of shape [batch_size, hidden_size]
        """
        if self.pooling_type == 'mean':
            # Mean pooling with attention mask
            if attention_mask is not None:
                # Expand mask to match hidden states dimensions
                expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                # Apply mask and compute mean
                sum_hidden = torch.sum(hidden_states * expanded_mask, dim=1)
                # Avoid division by zero
                sum_mask = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                # Simple mean pooling
                pooled = hidden_states.mean(dim=1)
                
        elif self.pooling_type == 'max':
            # Max pooling with attention mask
            if attention_mask is not None:
                # Create a mask for padding tokens
                mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                # Replace padding with large negative value
                masked_hidden = hidden_states.masked_fill(mask == 0, -1e9)
                # Max pooling
                pooled, _ = torch.max(masked_hidden, dim=1)
            else:
                # Simple max pooling
                pooled, _ = torch.max(hidden_states, dim=1)
                
        elif self.pooling_type == 'first':
            # Use first token (like CLS in BERT)
            pooled = hidden_states[:, 0, :]
            
        elif self.pooling_type == 'last':
            # Use last token based on attention mask
            if attention_mask is not None:
                # Get the last token index for each sequence
                last_indexes = attention_mask.sum(dim=1) - 1
                # Clamp to ensure valid index
                last_indexes = torch.clamp(last_indexes, min=0)
                # Get batch indices
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                # Select the last token for each sequence
                pooled = hidden_states[batch_indices, last_indexes]
            else:
                # Use the last token
                pooled = hidden_states[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
            
        return pooled
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Forward pass through the classifier.
        
        Args:
            hidden_states: Hidden states from transformer of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask of shape [batch_size, seq_len]
            labels: Optional labels for computing loss
            
        Returns:
            Dict containing:
                - logits: Classification logits
                - loss: Classification loss (if labels provided)
                - predictions: Predicted class indices
                - probabilities: Class probabilities
        """
        # Ensure initialized
        if not self.is_initialized:
            self.initialize()
            
        # Pool hidden states
        pooled = self.pool_hidden_states(hidden_states, attention_mask)
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # Get logits
        logits = self.classifier(pooled)
        
        # Get predictions and probabilities
        predictions = torch.argmax(logits, dim=-1)
        probabilities = F.softmax(logits, dim=-1)
        
        # Prepare outputs
        outputs = {
            'logits': logits,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            outputs['loss'] = loss
            
        return outputs
    
    def classify(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Classify text based on hidden states.
        
        Args:
            hidden_states: Hidden states from transformer
            attention_mask: Optional attention mask
            
        Returns:
            predictions: Predicted class indices
        """
        outputs = self.forward(hidden_states, attention_mask)
        return outputs['predictions']

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Compute classification loss
        
        Args:
            logits: Classification logits
            labels: Ground truth labels
            
        Returns:
            loss: Classification loss
        """
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        return loss 