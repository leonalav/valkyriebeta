import torch
import torch.nn as nn
from typing import Optional

class LogicalReasoningLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        self.logical_transform = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(config.hidden_dropout)
        )
        
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.logical_transform(hidden_states)
        return residual + hidden_states
