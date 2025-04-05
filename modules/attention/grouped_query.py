import torch
import torch.nn as nn
from typing import Optional
import math

class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_query_groups = config.num_query_groups
        self.head_dim = config.hidden_size // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, 
                               config.hidden_size * self.num_query_groups // self.num_heads)
        self.v_proj = nn.Linear(config.hidden_size,
                               config.hidden_size * self.num_query_groups // self.num_heads)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_length, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_query_groups, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_query_groups, self.head_dim)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        output = torch.matmul(attention_probs, v)
        output = output.view(batch_size, seq_length, -1)
        return self.o_proj(output)
