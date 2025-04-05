import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation (LoRA)"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights

        # Main weights
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LoRA components
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.scaling = self.lora_alpha / self.r
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merge_weights and self.training:
            return F.linear(x, self.weight + self.scaling * self.lora_B @ self.lora_A, self.bias)
            
        # Regular forward + LoRA path
        main_output = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_output = (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            return main_output + lora_output
        return main_output

    def merge_lora_weights(self):
        """Merge LoRA weights with the main weights"""
        if self.r > 0:
            self.weight.data += self.scaling * self.lora_B @ self.lora_A
            # Reset LoRA weights
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)