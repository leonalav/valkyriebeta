import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeDistillation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temperature = config.distillation_temperature
        self.attention_distill = nn.Linear(config.hidden_size, config.hidden_size)
        self.knowledge_compress = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
    def forward(self, student_features, teacher_features):
        if teacher_features is None:
            return student_features
            
        # Compress teacher knowledge
        teacher_knowledge = self.attention_distill(teacher_features)
        
        # Combine with student features
        combined = torch.cat([student_features, teacher_knowledge], dim=-1)
        distilled = self.knowledge_compress(combined)
        
        return distilled / self.temperature 