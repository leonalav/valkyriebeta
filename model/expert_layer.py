import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Expert-specific processing
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Expert-specific reasoning components
        self.expert_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads // 2,  # Smaller attention for efficiency
            batch_first=True
        )
        
        # Expert specialization layer
        self.specialization = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout)
        
    def forward(self, x):
        # Expert attention processing
        attn_out, _ = self.expert_attention(x, x, x)
        x = x + self.dropout(attn_out)
        
        # Expert feed-forward processing
        intermediate = self.dense1(x)
        intermediate = self.activation(intermediate)
        intermediate = self.dropout(intermediate)
        output = self.dense2(intermediate)
        
        # Expert specialization
        specialized = self.specialization(output)
        
        # Residual connection and layer norm
        return self.layer_norm(x + specialized) 