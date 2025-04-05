import torch
from typing import Optional
from dataclasses import dataclass
import gc

@dataclass
class ResourceConfig:
    max_batch_size: int = 32
    enable_cuda_graphs: bool = True
    enable_amp: bool = True
    memory_efficient_attention: bool = True

class ResourceManager:
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model = model.to(self.device)
        if self.config.enable_amp:
            model = torch.cuda.amp.autocast()(model)
        return model

    def clear_cache(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_optimal_batch_size(self, input_size: int) -> int:
        available_memory = torch.cuda.get_device_properties(0).total_memory
        return min(self.config.max_batch_size, available_memory // (input_size * 4))
