import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaReasoningLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.meta_controller = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        self.strategy_selector = nn.Linear(config.hidden_size, config.num_strategies)
        self.reasoning_strategies = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(config.num_strategies)
        ])
        
    def forward(self, x, task_embedding):
        # Meta-control signal
        meta_out, _ = self.meta_controller(x)
        
        # Select reasoning strategy
        strategy_weights = F.softmax(self.strategy_selector(meta_out), dim=-1)
        
        # Apply selected strategies
        output = torch.zeros_like(x)
        for i, strategy in enumerate(self.reasoning_strategies):
            output += strategy_weights[:, :, i:i+1] * strategy(x)
        
        return output 