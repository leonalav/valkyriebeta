import torch
import torch.nn as nn
import math
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple, Callable
from dataclasses import dataclass
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""
    
    # Learning rate settings
    learning_rate: float = 5e-5
    min_learning_rate: float = 1e-7
    warmup_steps: int = 1000
    warmup_ratio: float = 0.1
    max_steps: int = 100000
    
    # Optimizer settings
    optimizer_type: str = "adam"  # "adam", "adamw", "adafactor", "sgd", "rmsprop", "lamb"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Advanced optimizer settings
    fused_optimizer: bool = True
    fused_adam: bool = True
    fused_lamb: bool = True
    use_8bit_adam: bool = False
    
    # Scheduler settings
    scheduler_type: str = "cosine"  # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    lr_decay_power: float = 1.0
    end_learning_rate: float = 0.0
    lr_decay_steps: Optional[int] = None
    
    # Gradient settings
    max_grad_norm: float = 1.0
    grad_accumulation_steps: int = 1
    
    # Additional settings
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_update_every: int = 1
    
    def __post_init__(self):
        """Validate configuration."""
        if self.warmup_steps < 0:
            logger.warning("warmup_steps < 0, setting to 0")
            self.warmup_steps = 0
        
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            logger.warning("warmup_ratio must be between 0 and 1, setting to 0.1")
            self.warmup_ratio = 0.1
        
        if self.lr_decay_steps is None:
            self.lr_decay_steps = self.max_steps
            
        # Verify fused optimizers are available
        if self.fused_optimizer:
            try:
                from apex.optimizers import FusedAdam, FusedLAMB
                has_apex = True
            except ImportError:
                has_apex = False
                
            if not has_apex:
                logger.warning("apex not available, disabling fused optimizers")
                self.fused_optimizer = False
                self.fused_adam = False
                self.fused_lamb = False
        
        # Verify 8-bit optimizer is available
        if self.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                has_bnb = True
            except ImportError:
                has_bnb = False
                
            if not has_bnb:
                logger.warning("bitsandbytes not available, disabling 8-bit Adam")
                self.use_8bit_adam = False


class OptimizationManager:
    """
    Manages optimization for model training.
    
    This class handles the creation and configuration of optimizers,
    learning rate schedulers, and gradient clipping.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimization manager.
        
        Args:
            config: Configuration for optimization
        """
        self.config = config or OptimizationConfig()
    
    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Create optimizer based on configuration.
        
        Args:
            model: PyTorch model
            
        Returns:
            Configured optimizer
        """
        # Prepare parameter groups with weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        
        # Create optimizer based on type
        optimizer_type = self.config.optimizer_type.lower()
        
        if optimizer_type == "adam":
            # Check if fused or 8-bit Adam is requested
            if self.config.use_8bit_adam:
                try:
                    import bitsandbytes as bnb
                    logger.info("Using 8-bit Adam optimizer")
                    optimizer = bnb.optim.Adam8bit(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        betas=(self.config.adam_beta1, self.config.adam_beta2),
                        eps=self.config.adam_epsilon
                    )
                except ImportError:
                    logger.warning("bitsandbytes not found, falling back to regular Adam")
                    optimizer = torch.optim.Adam(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        betas=(self.config.adam_beta1, self.config.adam_beta2),
                        eps=self.config.adam_epsilon
                    )
            elif self.config.fused_optimizer and self.config.fused_adam:
                try:
                    from apex.optimizers import FusedAdam
                    logger.info("Using Fused Adam optimizer")
                    optimizer = FusedAdam(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        betas=(self.config.adam_beta1, self.config.adam_beta2),
                        eps=self.config.adam_epsilon
                    )
                except ImportError:
                    logger.warning("apex not found, falling back to regular Adam")
                    optimizer = torch.optim.Adam(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        betas=(self.config.adam_beta1, self.config.adam_beta2),
                        eps=self.config.adam_epsilon
                    )
            else:
                optimizer = torch.optim.Adam(
                    optimizer_grouped_parameters,
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon
                )
        
        elif optimizer_type == "adamw":
            if self.config.use_8bit_adam:
                try:
                    import bitsandbytes as bnb
                    logger.info("Using 8-bit AdamW optimizer")
                    optimizer = bnb.optim.AdamW8bit(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        betas=(self.config.adam_beta1, self.config.adam_beta2),
                        eps=self.config.adam_epsilon
                    )
                except ImportError:
                    logger.warning("bitsandbytes not found, falling back to regular AdamW")
                    optimizer = torch.optim.AdamW(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        betas=(self.config.adam_beta1, self.config.adam_beta2),
                        eps=self.config.adam_epsilon
                    )
            elif self.config.fused_optimizer and self.config.fused_adam:
                try:
                    from apex.optimizers import FusedAdam
                    logger.info("Using Fused Adam optimizer with weight decay correction")
                    optimizer = FusedAdam(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        betas=(self.config.adam_beta1, self.config.adam_beta2),
                        eps=self.config.adam_epsilon,
                        adam_w_mode=True
                    )
                except ImportError:
                    logger.warning("apex not found, falling back to regular AdamW")
                    optimizer = torch.optim.AdamW(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        betas=(self.config.adam_beta1, self.config.adam_beta2),
                        eps=self.config.adam_epsilon
                    )
            else:
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon
                )
        
        elif optimizer_type == "adafactor":
            try:
                from transformers.optimization import Adafactor
                logger.info("Using Adafactor optimizer")
                optimizer = Adafactor(
                    optimizer_grouped_parameters,
                    lr=self.config.learning_rate,
                    eps=(1e-30, 1e-3),
                    clip_threshold=1.0,
                    decay_rate=-0.8,
                    beta1=None,
                    weight_decay=self.config.weight_decay,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False
                )
            except ImportError:
                logger.warning("transformers.optimization not found, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon
                )
        
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        
        elif optimizer_type == "rmsprop":
            optimizer = torch.optim.RMSprop(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                alpha=0.9,
                eps=1e-8,
                weight_decay=self.config.weight_decay,
                momentum=0.0,
                centered=False
            )
        
        elif optimizer_type == "lamb":
            if self.config.fused_optimizer and self.config.fused_lamb:
                try:
                    from apex.optimizers import FusedLAMB
                    logger.info("Using Fused LAMB optimizer")
                    optimizer = FusedLAMB(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        betas=(self.config.adam_beta1, self.config.adam_beta2),
                        eps=self.config.adam_epsilon,
                        weight_decay=self.config.weight_decay
                    )
                except ImportError:
                    logger.warning("apex not found, falling back to regular AdamW")
                    optimizer = torch.optim.AdamW(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        betas=(self.config.adam_beta1, self.config.adam_beta2),
                        eps=self.config.adam_epsilon
                    )
            else:
                try:
                    from pl_bolts.optimizers import LAMB
                    logger.info("Using LAMB optimizer")
                    optimizer = LAMB(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        betas=(self.config.adam_beta1, self.config.adam_beta2),
                        eps=self.config.adam_epsilon,
                        weight_decay=self.config.weight_decay
                    )
                except ImportError:
                    logger.warning("pl_bolts not found, falling back to AdamW")
                    optimizer = torch.optim.AdamW(
                        optimizer_grouped_parameters,
                        lr=self.config.learning_rate,
                        betas=(self.config.adam_beta1, self.config.adam_beta2),
                        eps=self.config.adam_epsilon
                    )
        
        else:
            logger.warning(f"Unknown optimizer: {optimizer_type}, using AdamW")
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
        
        return optimizer
    
    def create_scheduler(self, 
                        optimizer: torch.optim.Optimizer, 
                        num_training_steps: Optional[int] = None) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create learning rate scheduler based on configuration.
        
        Args:
            optimizer: PyTorch optimizer
            num_training_steps: Total number of training steps
            
        Returns:
            Configured scheduler
        """
        num_training_steps = num_training_steps or self.config.max_steps
        
        if self.config.warmup_ratio > 0:
            warmup_steps = int(num_training_steps * self.config.warmup_ratio)
            logger.info(f"Warmup ratio of {self.config.warmup_ratio} gives {warmup_steps} warmup steps")
        else:
            warmup_steps = self.config.warmup_steps
            
        logger.info(f"Creating {self.config.scheduler_type} scheduler with {warmup_steps} warmup steps out of {num_training_steps} total steps")
        
        scheduler_type = self.config.scheduler_type.lower()
        
        if scheduler_type == "linear":
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0,
                    float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
                )
            
            scheduler = LambdaLR(optimizer, lr_lambda)
        
        elif scheduler_type == "cosine":
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            scheduler = LambdaLR(optimizer, lr_lambda)
        
        elif scheduler_type == "cosine_with_restarts":
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                
                # Implement cosine with hard restarts
                cycles = 3  # number of restarts
                cycle_length = (num_training_steps - warmup_steps) / cycles
                cycle_progress = (current_step - warmup_steps) % cycle_length / cycle_length
                
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cycle_progress)))
            
            scheduler = LambdaLR(optimizer, lr_lambda)
        
        elif scheduler_type == "polynomial":
            decay_steps = self.config.lr_decay_steps or num_training_steps
            power = self.config.lr_decay_power
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                elif current_step > decay_steps:
                    return self.config.end_learning_rate / self.config.learning_rate
                
                # Polynomial decay
                progress = float(current_step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
                remaining = (1 - progress) ** power
                return max(
                    self.config.end_learning_rate / self.config.learning_rate,
                    remaining
                )
            
            scheduler = LambdaLR(optimizer, lr_lambda)
        
        elif scheduler_type == "constant":
            scheduler = LambdaLR(optimizer, lambda _: 1.0)
        
        elif scheduler_type == "constant_with_warmup":
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            
            scheduler = LambdaLR(optimizer, lr_lambda)
        
        else:
            logger.warning(f"Unknown scheduler: {scheduler_type}, using linear")
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0,
                    float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps))
                )
            
            scheduler = LambdaLR(optimizer, lr_lambda)
        
        return scheduler
    
    def clip_gradients(self, model: nn.Module) -> Optional[float]:
        """
        Clip gradients to prevent exploding gradients.
        
        Args:
            model: PyTorch model
            
        Returns:
            Global norm of gradients after clipping, or None if no clipping is applied
        """
        max_norm = self.config.max_grad_norm
        
        if max_norm <= 0:
            return None
        
        # Compute gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Apply gradient clipping
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def get_learning_rate(self, optimizer: torch.optim.Optimizer) -> float:
        """
        Get current learning rate from optimizer.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            Current learning rate
        """
        return optimizer.param_groups[0]["lr"]


class EMA:
    """
    Exponential Moving Average for model parameters.
    
    This class maintains a moving average of the model parameters,
    which can be used for evaluation or as a final model.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, update_every: int = 1):
        """
        Initialize EMA.
        
        Args:
            model: PyTorch model
            decay: Decay rate for EMA
            update_every: Update EMA every N steps
        """
        self.model = model
        self.decay = decay
        self.update_every = update_every
        self.shadow = {}
        self.backup = {}
        self.steps = 0
        
        # Initialize EMA parameters
        self.register()
    
    def register(self):
        """Register model parameters for EMA tracking."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        self.steps += 1
        
        # Only update according to schedule
        if self.steps % self.update_every != 0:
            return
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_average.clone()
                else:
                    # Register if not already registered
                    self.shadow[name] = param.data.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model (and store current parameters)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    self.backup[name] = param.data.clone()
                    param.data = self.shadow[name].clone()
    
    def restore(self):
        """Restore original parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.backup:
                    param.data = self.backup[name].clone()
        
        # Clear backup to free memory
        self.backup = {}
    
    def state_dict(self) -> Dict[str, Any]:
        """Get EMA state for saving."""
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "steps": self.steps
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load EMA state."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
        self.steps = state_dict["steps"] 