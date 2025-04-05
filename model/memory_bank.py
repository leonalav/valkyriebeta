import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MemoryBank(nn.Module):
    def __init__(self, memory_size, hidden_size, num_heads):
        super().__init__()
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Memory slots
        self.memory = nn.Parameter(torch.randn(1, memory_size, hidden_size))
        
        # Attention for memory access
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer norm for memory and input
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Normalize input and memory
        x = self.input_norm(x)
        memory = self.memory_norm(self.memory)
        
        # Project queries, keys, and values
        queries = self.query_proj(x)  # [batch, seq_len, hidden]
        keys = self.key_proj(memory)  # [1, mem_size, hidden]
        values = self.value_proj(memory)  # [1, mem_size, hidden]
        
        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1))  # [batch, seq_len, mem_size]
        scores = scores / math.sqrt(self.hidden_size)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Get memory output
        memory_out = torch.matmul(attn_weights, values)  # [batch, seq_len, hidden]
        
        # Project output
        output = self.output_proj(memory_out)  # [batch, seq_len, hidden]
        
        return output  # Return [batch, seq_len, hidden_size] 