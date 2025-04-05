import torch
import torch.nn as nn
import torch.nn.functional as F

class ReasoningAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.down_project = nn.Linear(config.hidden_size, config.adapter_size)
        self.up_project = nn.Linear(config.adapter_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Reasoning-specific components
        self.reasoning_gate = nn.Linear(config.hidden_size, 3)  # logical, analytical, creative
        self.reasoning_paths = nn.ModuleList([
            nn.Linear(config.adapter_size, config.adapter_size)
            for _ in range(3)
        ])
        
    def forward(self, x):
        # Gate different reasoning paths
        gates = F.softmax(self.reasoning_gate(x), dim=-1)
        
        # Down project
        hidden = self.down_project(x)
        
        # Apply different reasoning paths
        reasoned = torch.zeros_like(hidden)
        for i, path in enumerate(self.reasoning_paths):
            reasoned += gates[:, :, i:i+1] * path(hidden)
        
        # Up project and residual
        return x + self.up_project(reasoned) 