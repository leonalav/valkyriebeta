import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ContrastiveLearningModule(nn.Module):
    """Module for contrastive learning with logical reasoning"""
    def __init__(self, config):
        super().__init__()
        self.temperature = config.cl_temperature
        
        # Projection heads
        self.base_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.projection_size)
        )
        
        self.logical_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.projection_size)
        )
        
        self.reasoning_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.projection_size)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(config.projection_size)
        
    def forward(self, 
                base_embeddings: torch.Tensor,
                logical_embeddings: torch.Tensor,
                reasoning_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project embeddings
        z1 = self.layer_norm(F.normalize(self.base_projection(base_embeddings), dim=-1))
        z2 = self.layer_norm(F.normalize(self.logical_projection(logical_embeddings), dim=-1))
        
        # Add reasoning states if available
        if reasoning_states is not None:
            z3 = self.layer_norm(F.normalize(self.reasoning_projection(reasoning_states), dim=-1))
            z1 = torch.cat([z1, z3], dim=0)
            
        # Compute similarities
        similarities = torch.matmul(z1, z2.t()) / self.temperature
        
        # Create labels for contrastive loss
        batch_size = base_embeddings.size(0)
        labels = torch.arange(
            batch_size * (2 if reasoning_states is not None else 1),
            device=similarities.device
        )
        
        # Compute NT-Xent loss
        loss = F.cross_entropy(similarities, labels)
        
        return loss 