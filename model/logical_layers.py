import torch
import torch.nn as nn
from typing import Tuple, Optional, List

class TreeLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input transformations
        self.ioux = nn.Linear(input_size, 3 * hidden_size)
        self.iouh = nn.Linear(hidden_size, 3 * hidden_size)
        
        # Forget gate transformations (one for each child)
        self.fx = nn.Linear(input_size, hidden_size)
        self.fh = nn.Linear(hidden_size, hidden_size)
        
        # Cell output transformation
        self.cell_out = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, 
                x: torch.Tensor, 
                child_states: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Number of children
        num_children = len(child_states)
        
        # Calculate input, output, and update gates
        iou = self.ioux(x)
        i, o, u = torch.split(iou, self.hidden_size, dim=-1)
        
        # Initialize the new cell state
        c_new = torch.zeros_like(i)
        
        if num_children > 0:
            # Collect child hidden states and cell states
            child_h = torch.stack([h for h, _ in child_states])
            child_c = torch.stack([c for _, c in child_states])
            
            # Calculate forget gates for each child
            f = torch.sigmoid(self.fx(x).unsqueeze(0) + self.fh(child_h))
            
            # Update cell state using children
            c_new = c_new + torch.sum(f * child_c, dim=0)
        
        # Update cell state with input
        i = torch.sigmoid(i)
        u = torch.tanh(u)
        c_new = c_new + i * u
        
        # Calculate output
        o = torch.sigmoid(o)
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new

class MemoryNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Reduce memory size and use gradient checkpointing
        self.chunk_size = 512  # Fixed chunk size
        self.memory_size = min(1024, config.memory_size)  # Cap memory size
        self.temporal_weight = nn.Parameter(torch.ones(self.chunk_size, 1))  # Match chunk size
        
        # Memory efficient attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=4,  # Reduced number of heads
            dropout=0.1,
            batch_first=True
        )
        
        # Efficient processing
        self.memory_update = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size)
        )
    
    def forward(self, hidden_states):
        batch_size = hidden_states.size(0)
        outputs = []
        
        # Process in chunks
        for i in range(0, hidden_states.size(1), self.chunk_size):
            chunk = hidden_states[:, i:i+self.chunk_size]
            
            # Pad chunk if needed
            if chunk.size(1) < self.chunk_size:
                pad_size = self.chunk_size - chunk.size(1)
                chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad_size))
            
            # Compute attention
            with torch.cuda.amp.autocast(enabled=True):
                attention_output, attention_weights = self.attention(
                    chunk, chunk, chunk,
                    need_weights=True
                )
            
            # Update temporal weights (now dimensions match)
            with torch.no_grad():
                weight_update = 1 - attention_weights.mean(dim=0).unsqueeze(-1)
                self.temporal_weight.data = weight_update.detach()
            
            # Process chunk
            chunk_output = self.memory_update(attention_output)
            
            # Remove padding if added
            if i + self.chunk_size > hidden_states.size(1):
                chunk_output = chunk_output[:, :hidden_states.size(1)-i]
            
            outputs.append(chunk_output)
        
        # Combine chunks
        return torch.cat(outputs, dim=1)

class LogicalReasoningModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.memory_network = MemoryNetwork(config)
        
        # Memory efficient processing
        self.reasoning_ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, config.hidden_size)
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, hidden_states):
        # Process with gradient checkpointing
        if self.training:
            memory_output = torch.utils.checkpoint.checkpoint(
                self.memory_network,
                hidden_states
            )
        else:
            memory_output = self.memory_network(hidden_states)
        
        # Efficient reasoning
        reasoning_output = self.reasoning_ffn(memory_output)
        
        # Residual connection and norm
        output = self.layer_norm(hidden_states + reasoning_output)
        
        return output 