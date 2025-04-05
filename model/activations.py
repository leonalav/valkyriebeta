import torch.nn as nn
import torch.nn.functional as F
class SwiGLU(nn.Module):
    """More efficient alternative to GELU"""
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w2 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w3 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, x):
        swish = F.silu(self.w1(x))
        linear = self.w2(x)
        return self.w3(swish * linear) 