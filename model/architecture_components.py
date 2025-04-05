import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math

@dataclass
class ArchitectureConfig:
    """Configuration for specialized architecture components"""
    use_mixture_of_experts: bool = True
    num_experts: int = 8
    expert_capacity: int = 32
    use_dynamic_routing: bool = True
    use_hierarchical_layers: bool = True
    num_hierarchical_layers: int = 3
    use_skip_connections: bool = True

class HierarchicalLayer(nn.Module):
    """Hierarchical processing layer for capturing multi-level patterns"""
    def __init__(self, config):
        super().__init__()
        self.num_levels = config.num_hierarchical_layers
        
        # Create processing blocks for each level
        self.level_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU()
            ) for _ in range(self.num_levels)
        ])
        
        # Level-wise attention for combining hierarchical features
        self.level_attention = nn.Linear(config.hidden_size, self.num_levels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_size = x.shape
        level_outputs = []
        
        for level, block in enumerate(self.level_blocks):
            # Adjust receptive field for each level
            stride = 2 ** level
            level_input = x[:, ::stride, :]
            level_output = block(level_input)
            
            # Interpolate back to original sequence length
            if stride > 1:
                level_output = F.interpolate(
                    level_output.transpose(1, 2),
                    size=seq_length,
                    mode='linear'
                ).transpose(1, 2)
                
            level_outputs.append(level_output)
            
        # Combine levels using attention weights
        stacked_outputs = torch.stack(level_outputs, dim=-1)
        attention_weights = F.softmax(self.level_attention(x), dim=-1)
        combined_output = torch.sum(
            stacked_outputs * attention_weights.unsqueeze(-2),
            dim=-1
        )
        
        return combined_output

class DynamicRouter(nn.Module):
    """Routes inputs to appropriate experts based on content"""
    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_experts)
        self.expert_capacity = config.expert_capacity
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute routing probabilities
        routing_logits = self.router(x)  # [batch, seq_len, num_experts]
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(
            routing_probs, k=2, dim=-1
        )
        
        # Normalize probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        return top_k_probs, top_k_indices, routing_logits

class ExpertGating(nn.Module):
    """Gating mechanism for expert mixture"""
    def __init__(self, config):
        super().__init__()
        self.w_gate = nn.Parameter(
            torch.randn(config.hidden_size, config.num_experts) / math.sqrt(config.hidden_size)
        )
        self.w_noise = nn.Parameter(torch.zeros(config.num_experts))
        
    def forward(self, x: torch.Tensor, router_outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        top_k_probs, top_k_indices, _ = router_outputs
        
        # Add noise during training for better load balancing
        if self.training:
            noise = torch.randn_like(self.w_noise) * F.softplus(self.w_noise)
            gates = top_k_probs + noise
        else:
            gates = top_k_probs
            
        return gates

class SkipManager(nn.Module):
    """Manages dynamic skip connections"""
    def __init__(self, config):
        super().__init__()
        self.skip_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.skip_gate = nn.Linear(config.hidden_size, 1)
        
    def forward(self, x: torch.Tensor, skip_candidates: List[torch.Tensor]) -> torch.Tensor:
        if not skip_candidates:
            return x
            
        # Compute skip connection weights
        skip_weights = [
            torch.sigmoid(self.skip_gate(candidate))
            for candidate in skip_candidates
        ]
        
        # Apply weighted skip connections
        skip_tensors = [
            weight * self.skip_proj(candidate)
            for weight, candidate in zip(skip_weights, skip_candidates)
        ]
        
        return x + sum(skip_tensors)
