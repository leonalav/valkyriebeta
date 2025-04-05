import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficient_layers import MemoryEfficientLinear

class EfficientFeedForward(nn.Module):
    def __init__(self, config, linear_class=MemoryEfficientLinear):
        super().__init__()
        self.w1 = linear_class(config.hidden_size, 4 * config.hidden_size)
        self.w2 = linear_class(4 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))

class ParallelFeedForward(nn.Module):
    def __init__(self, config, linear_class=MemoryEfficientLinear):
        super().__init__()
        self.w1 = linear_class(config.hidden_size, 4 * config.hidden_size)
        self.w2 = linear_class(4 * config.hidden_size, config.hidden_size)
        self.w3 = linear_class(config.hidden_size, 4 * config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w2(self.dropout(F.gelu(self.w1(x))))
        x2 = self.w2(self.dropout(F.gelu(self.w3(x))))
        return x1 + x2
