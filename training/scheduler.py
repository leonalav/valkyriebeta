import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, _LRScheduler, LambdaLR
from typing import Dict, Any, Optional, Union, List, Callable
import math

class AdaptiveCosineScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        num_cycles: float = 0.5,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.num_cycles = num_cycles
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, self.num_warmup_steps))
            
        progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        
        # Cosine decay with adaptive minimum
        min_lr = self.min_lr_ratio * self.base_lrs[0]
        return max(
            min_lr,
            ((1 + math.cos(math.pi * self.num_cycles * 2 * progress)) / 2) * (1 - min_lr) + min_lr
        )

    def get_lr(self):
        return [self.lr_lambda(self.last_epoch) * base_lr for base_lr in self.base_lrs]

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    num_warmup_steps: Optional[int] = None,
    **kwargs
) -> _LRScheduler:
    """
    Get a learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        **kwargs: Additional arguments for specific schedulers
        
    Returns:
        Learning rate scheduler
    """
    # Set default warmup steps if not provided
    if num_warmup_steps is None:
        num_warmup_steps = max(100, int(num_training_steps * 0.1))
    
    # Create scheduler based on type
    if scheduler_type == "linear":
        return LinearLRScheduler(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    elif scheduler_type == "cosine":
        return CosineLRScheduler(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    elif scheduler_type == "constant":
        return ConstantLRScheduler(
            optimizer, num_warmup_steps, **kwargs
        )
    elif scheduler_type == "polynomial":
        return PolynomialLRScheduler(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    elif scheduler_type == "cosine_with_restarts":
        return CosineWithRestartsLRScheduler(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    elif scheduler_type == "rwkv_scheduler":
        return RWKVHybridScheduler(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    elif scheduler_type == "cosine_with_warmup":
        return CosineWithWarmupScheduler(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

class LinearLRScheduler(_LRScheduler):
    """Linear learning rate scheduler with warmup"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
        min_lr_ratio: float = 0.0
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Warmup phase: linear increase
            return [
                base_lr * self.last_epoch / max(1, self.num_warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Decay phase: linear decay
            return [
                base_lr * max(
                    self.min_lr_ratio,
                    (self.num_training_steps - self.last_epoch) / 
                    max(1, self.num_training_steps - self.num_warmup_steps)
                )
                for base_lr in self.base_lrs
            ]

class CosineLRScheduler(_LRScheduler):
    """Cosine learning rate scheduler with warmup"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        min_lr_ratio: float = 0.0
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Warmup phase: linear increase
            return [
                base_lr * self.last_epoch / max(1, self.num_warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Decay phase: cosine decay
            progress = (self.last_epoch - self.num_warmup_steps) / max(
                1, self.num_training_steps - self.num_warmup_steps
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
            return [
                base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor)
                for base_lr in self.base_lrs
            ]

class CosineWithWarmupScheduler(_LRScheduler):
    """
    Cosine scheduler with linear warmup, matching the implementation in optimization.py
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        min_lr_ratio: float = 0.0,
        lr: float = 1e-4  # Added to match optimization.py signature
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Warmup phase: linear increase
            return [
                base_lr * float(self.last_epoch) / float(max(1, self.num_warmup_steps))
                for base_lr in self.base_lrs
            ]
        
        # Cosine decay phase
        progress = float(self.last_epoch - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
        
        return [
            base_lr * max(self.min_lr_ratio, cosine_decay) 
            for base_lr in self.base_lrs
        ]

class ConstantLRScheduler(_LRScheduler):
    """Constant learning rate scheduler with warmup"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        last_epoch: int = -1
    ):
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Warmup phase: linear increase
            return [
                base_lr * self.last_epoch / max(1, self.num_warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Constant phase
            return self.base_lrs

class PolynomialLRScheduler(_LRScheduler):
    """Polynomial decay learning rate scheduler with warmup"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        power: float = 1.0,
        last_epoch: int = -1,
        min_lr_ratio: float = 0.0
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.power = power
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Warmup phase: linear increase
            return [
                base_lr * self.last_epoch / max(1, self.num_warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Decay phase: polynomial decay
            progress = (self.last_epoch - self.num_warmup_steps) / max(
                1, self.num_training_steps - self.num_warmup_steps
            )
            polynomial_factor = (1 - progress) ** self.power
            return [
                base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * polynomial_factor)
                for base_lr in self.base_lrs
            ]

class CosineWithRestartsLRScheduler(_LRScheduler):
    """Cosine learning rate scheduler with warmup and hard restarts"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: int = 3,
        last_epoch: int = -1,
        min_lr_ratio: float = 0.0
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Warmup phase: linear increase
            return [
                base_lr * self.last_epoch / max(1, self.num_warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            # Decay phase with restarts
            steps_since_warmup = self.last_epoch - self.num_warmup_steps
            total_decay_steps = max(1, self.num_training_steps - self.num_warmup_steps)
            
            # Calculate cycle length
            cycle_length = total_decay_steps / self.num_cycles
            
            # Calculate current cycle and progress within cycle
            current_cycle = min(self.num_cycles - 1, int(steps_since_warmup / cycle_length))
            cycle_progress = (steps_since_warmup - current_cycle * cycle_length) / cycle_length
            
            # Cosine factor with restarts
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
            
            return [
                base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor)
                for base_lr in self.base_lrs
            ]

class RWKVHybridScheduler(_LRScheduler):
    """
    Specialized scheduler for hybrid RWKV-Transformer model.
    Uses different schedules for RWKV and Transformer parameters.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
        min_lr_ratio: float = 0.0,
        rwkv_param_names: List[str] = None,
        transformer_param_names: List[str] = None,
        rwkv_lr_factor: float = 1.2,  # RWKV layers often need slightly higher LR
        transformer_schedule: str = "cosine",
        rwkv_schedule: str = "cosine_with_restarts"
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        self.rwkv_lr_factor = rwkv_lr_factor
        
        # Default parameter group names if not provided
        self.rwkv_param_names = rwkv_param_names or ["rwkv", "time_mix", "time_decay", "time_first"]
        self.transformer_param_names = transformer_param_names or ["attention", "mlp", "norm"]
        
        # Create parameter group indices
        self.rwkv_param_indices = []
        self.transformer_param_indices = []
        self.shared_param_indices = []
        
        # Map parameter groups to their respective indices
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get("name", "")
            is_rwkv = any(name in group_name.lower() for name in self.rwkv_param_names)
            is_transformer = any(name in group_name.lower() for name in self.transformer_param_names)
            
            if is_rwkv and not is_transformer:
                self.rwkv_param_indices.append(i)
            elif is_transformer and not is_rwkv:
                self.transformer_param_indices.append(i)
            else:
                self.shared_param_indices.append(i)
        
        # Create sub-schedulers
        self.transformer_scheduler_fn = self._get_scheduler_fn(transformer_schedule)
        self.rwkv_scheduler_fn = self._get_scheduler_fn(rwkv_schedule)
        
        super().__init__(optimizer, last_epoch)
    
    def _get_scheduler_fn(self, scheduler_type: str) -> Callable:
        """Get scheduler function based on type"""
        if scheduler_type == "linear":
            return lambda step, base_lr: self._linear_schedule(step, base_lr)
        elif scheduler_type == "cosine":
            return lambda step, base_lr: self._cosine_schedule(step, base_lr)
        elif scheduler_type == "cosine_with_restarts":
            return lambda step, base_lr: self._cosine_with_restarts_schedule(step, base_lr)
        else:
            # Default to cosine
            return lambda step, base_lr: self._cosine_schedule(step, base_lr)
    
    def _linear_schedule(self, step: int, base_lr: float) -> float:
        """Linear learning rate schedule with warmup"""
        if step < self.num_warmup_steps:
            return base_lr * step / max(1, self.num_warmup_steps)
        else:
            return base_lr * max(
                self.min_lr_ratio,
                (self.num_training_steps - step) / 
                max(1, self.num_training_steps - self.num_warmup_steps)
            )
    
    def _cosine_schedule(self, step: int, base_lr: float) -> float:
        """Cosine learning rate schedule with warmup"""
        if step < self.num_warmup_steps:
            return base_lr * step / max(1, self.num_warmup_steps)
        else:
            progress = (step - self.num_warmup_steps) / max(
                1, self.num_training_steps - self.num_warmup_steps
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor)
    
    def _cosine_with_restarts_schedule(self, step: int, base_lr: float) -> float:
        """Cosine learning rate schedule with warmup and hard restarts"""
        if step < self.num_warmup_steps:
            return base_lr * step / max(1, self.num_warmup_steps)
        else:
            steps_since_warmup = step - self.num_warmup_steps
            total_decay_steps = max(1, self.num_training_steps - self.num_warmup_steps)
            
            # Calculate cycle length (3 cycles)
            cycle_length = total_decay_steps / 3
            
            # Calculate current cycle and progress within cycle
            current_cycle = min(2, int(steps_since_warmup / cycle_length))
            cycle_progress = (steps_since_warmup - current_cycle * cycle_length) / cycle_length
            
            # Cosine factor with restarts
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
            
            return base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor)
    
    def get_lr(self):
        """Get learning rates for all parameter groups"""
        lrs = []
        
        # Apply appropriate schedule based on parameter group
        for i, base_lr in enumerate(self.base_lrs):
            if i in self.rwkv_param_indices:
                # Apply RWKV-specific schedule with adjusted base LR
                adjusted_base_lr = base_lr * self.rwkv_lr_factor
                lrs.append(self.rwkv_scheduler_fn(self.last_epoch, adjusted_base_lr))
            elif i in self.transformer_param_indices:
                # Apply Transformer-specific schedule
                lrs.append(self.transformer_scheduler_fn(self.last_epoch, base_lr))
            else:
                # Apply default schedule for shared parameters
                lrs.append(self._cosine_schedule(self.last_epoch, base_lr))
        
        return lrs
