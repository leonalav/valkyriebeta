import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class LogicalAttention(nn.Module):
    """Attention mechanism specialized for logical reasoning"""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_reasoning_heads
        self.head_dim = config.hidden_size // config.num_reasoning_heads
        
        # Logical attention components
        self.predicate_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.rule_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.consistency_gate = nn.Linear(config.hidden_size, 1)
        
        # Rule-based attention
        self.rule_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=self.num_heads,
            dropout=config.dropout
        )
        self.rule_bank = nn.Parameter(
            torch.randn(config.num_logical_rules, config.hidden_size)
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = x.shape[:2]
        
        # Project input into predicate space
        predicates = self.predicate_proj(x)
        
        # Expand rule bank for batch processing
        rules = self.rule_bank.unsqueeze(0).expand(batch_size, -1, -1)
        rules = self.rule_proj(rules)
        
        # Apply rule-based attention
        if mask is not None:
            # Extend mask for rules
            rule_mask = torch.ones(
                (batch_size, rules.size(1)),
                dtype=torch.bool,
                device=mask.device
            )
            mask = torch.cat([mask, rule_mask], dim=1)
            
        # Combine predicates and rules
        combined = torch.cat([predicates, rules], dim=1)
        
        # Apply multi-head attention
        attn_output, _ = self.rule_attention(
            predicates.transpose(0, 1),
            combined.transpose(0, 1),
            combined.transpose(0, 1),
            key_padding_mask=mask if mask is not None else None
        )
        attn_output = attn_output.transpose(0, 1)
        
        # Check logical consistency
        consistency_scores = torch.sigmoid(self.consistency_gate(attn_output))
        
        # Apply output projection and dropout
        output = self.output_proj(attn_output)
        output = self.dropout(output)
        
        # Combine with original input using consistency scores
        return x + consistency_scores * output 