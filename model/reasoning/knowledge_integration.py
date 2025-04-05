import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class KnowledgeIntegrationModule(nn.Module):
    """Module for integrating external knowledge with reasoning"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.knowledge_size = config.knowledge_size
        
        # Knowledge components
        self.knowledge_bank = nn.Parameter(
            torch.randn(config.num_knowledge_rules, config.hidden_size)
        )
        
        # Knowledge attention
        self.knowledge_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Knowledge gate and update
        self.knowledge_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.knowledge_updater = nn.GRUCell(
            config.hidden_size,
            config.hidden_size
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Expand knowledge bank for batch processing
        knowledge = self.knowledge_bank.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply knowledge attention
        knowledge_output, _ = self.knowledge_attention(
            x.transpose(0, 1),
            knowledge.transpose(0, 1),
            knowledge.transpose(0, 1)
        )
        knowledge_output = knowledge_output.transpose(0, 1)
        
        # Compute knowledge gate
        gate = self.knowledge_gate(
            torch.cat([x, knowledge_output], dim=-1)
        )
        
        # Update knowledge bank
        if self.training:
            with torch.no_grad():
                # Update knowledge using mean of batch
                knowledge_update = self.knowledge_updater(
                    x.mean(dim=1),
                    self.knowledge_bank
                )
                self.knowledge_bank.data = knowledge_update
        
        # Combine knowledge with input
        output = x + gate * knowledge_output
        
        # Apply output transformations
        output = self.output_proj(output)
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output 