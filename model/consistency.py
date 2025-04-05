import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyChecker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.verify_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.consistency_score = nn.Linear(config.hidden_size, 1)
        
        # Contradiction detection
        self.contradiction_check = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, previous_states):
        # Verify current reasoning
        verified = self.verify_layer(x)
        
        # Check consistency with previous states
        consistency_scores = []
        for prev_state in previous_states:
            combined = torch.cat([verified, prev_state], dim=-1)
            score = self.contradiction_check(combined)
            consistency_scores.append(score)
        
        # Adjust based on consistency
        consistency = torch.stack(consistency_scores).mean(dim=0)
        return x * consistency 