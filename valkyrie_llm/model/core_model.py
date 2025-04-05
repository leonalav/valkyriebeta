"""
Core model implementation for ValkyrieLLM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CoreModel(nn.Module):
    """
    Core model for ValkyrieLLM.
    """
    
    def __init__(self, vocab_size=30000, hidden_size=768, num_layers=12, num_heads=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Basic components
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(1024, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
            for _ in range(num_layers)
        ])
        
        # Output head
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for the model.
        """
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        token_embeddings = self.token_embedding(input_ids)
        
        # Get position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(positions)
        
        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        # Get logits
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states
        } 