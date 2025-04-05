import torch
from typing import Optional
from dataclasses import dataclass

@dataclass
class MemoryConfig:
    use_checkpointing: bool = True
    max_memory_gb: float = 24
    empty_cache_freq: int = 16
    
class MemoryOptimizer:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.step_counter = 0
        
    def optimize_memory(self, model: torch.nn.Module):
        """Apply memory optimizations to model"""
        if self.config.use_checkpointing:
            self.apply_checkpointing(model)
            
        if self.step_counter % self.config.empty_cache_freq == 0:
            torch.cuda.empty_cache()
            
        self.step_counter += 1
        
    @staticmethod
    def apply_checkpointing(model: torch.nn.Module):
        """Apply gradient checkpointing to applicable layers"""
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
