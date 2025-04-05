from typing import Dict, Any, Optional
import torch
from dataclasses import dataclass

@dataclass
class CustomizationConfig:
    learning_rate: float = 1e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation: int = 1

class ModelCustomizer:
    def __init__(self, config: CustomizationConfig):
        self.config = config

    def create_fine_tuning_setup(self, model: torch.nn.Module) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs
        )

        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "grad_accumulation": self.config.gradient_accumulation
        }

    def load_custom_weights(self, model: torch.nn.Module, weights_path: str):
        custom_weights = torch.load(weights_path)
        model.load_state_dict(custom_weights, strict=False)
        return model
