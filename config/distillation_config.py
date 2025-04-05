from dataclasses import dataclass
from typing import Optional

@dataclass
class DistillationConfig:
    """Configuration for API-based knowledge distillation"""
    # API Configuration
    api_key: str
    site_url: str
    site_name: str
    teacher_model: str = "deepseek/deepseek-r1:free"
    
    # Model Configuration
    hidden_size: int = 768
    temperature: float = 0.85
    distillation_temperature: float = 2.0
    top_p: float = 1.0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    repetition_penalty: float = 1
    top_k: int = 0
    
    # Training Configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    alpha: float = 0.5  # Weight for distillation loss
    
    # Monitoring
    use_wandb: bool = False
    wandb_project: str = "api_distillation"
    
    # Output
    output_dir: str = "./checkpoints"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "DistillationConfig":
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        })
