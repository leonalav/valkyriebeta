"""
Adapter module to integrate CoreModel with existing training code.

This module provides utility functions to integrate our CoreModel 
with existing training scripts without modifying them extensively.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .core_model import CoreModel

logger = logging.getLogger(__name__)

def get_training_model(config: Optional[Dict[str, Any]] = None, 
                      use_3b: bool = True,
                      **kwargs) -> nn.Module:
    """
    Get a model suitable for training.
    
    Args:
        config: Optional configuration dictionary for the model
        use_3b: Whether to use the 3B parameter configuration
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        A model suitable for training
    """
    if use_3b:
        logger.info("Creating 3B parameter model for training")
        model = CoreModel.from_3b_config()
    else:
        # Create model using config or default parameters
        if config is not None:
            model = CoreModel(
                vocab_size=config.get('vocab_size', 50257),
                hidden_size=config.get('hidden_size', 768),
                num_layers=config.get('num_layers', 12),
                num_heads=config.get('num_heads', 12),
                max_seq_length=config.get('max_seq_length', 2048),
                dropout=config.get('dropout', 0.1),
                **kwargs
            )
        else:
            model = CoreModel(**kwargs)
    
    # Enable memory optimization for large models
    if use_3b:
        logger.info("Enabling gradient checkpointing for memory efficiency")
        model.gradient_checkpointing_enable()
    
    return model

def prepare_model_for_training(model: nn.Module, 
                              mixed_precision: bool = True, 
                              distributed: bool = False) -> nn.Module:
    """
    Prepare model for training with various optimizations.
    
    Args:
        model: The model to prepare
        mixed_precision: Whether to use mixed precision training
        distributed: Whether to use distributed training
        
    Returns:
        The prepared model
    """
    # Enable training mode
    model.train()
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Prepare for distributed training if needed
    if distributed and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for distributed training")
        model = nn.parallel.DistributedDataParallel(model)
    
    return model

def get_optimizer_and_scheduler(model: nn.Module, 
                               config: Dict[str, Any],
                               num_training_steps: int) -> tuple:
    """
    Get optimizer and learning rate scheduler for training.
    
    Args:
        model: The model to optimize
        config: Training configuration
        num_training_steps: Number of training steps
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Get optimizer parameters from config
    learning_rate = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    adam_beta1 = config.get('adam_beta1', 0.9)
    adam_beta2 = config.get('adam_beta2', 0.999)
    adam_epsilon = config.get('adam_epsilon', 1e-8)
    
    # Prepare optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon
    )
    
    # Create scheduler
    warmup_steps = config.get('warmup_steps', int(0.1 * num_training_steps))
    scheduler = get_scheduler(
        config.get('lr_scheduler_type', 'cosine'),
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler

def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    """
    Get learning rate scheduler.
    
    Args:
        name: Type of scheduler ('linear', 'cosine', 'constant', etc.)
        optimizer: Optimizer to use
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        
    Returns:
        Scheduler
    """
    # Simple implementation of common schedulers
    if name == "linear":
        return LinearScheduler(optimizer, num_warmup_steps, num_training_steps)
    elif name == "cosine":
        return CosineScheduler(optimizer, num_warmup_steps, num_training_steps)
    else:
        return ConstantScheduler(optimizer, num_warmup_steps)

class LinearScheduler:
    """Linear learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, num_warmup_steps, num_training_steps):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._get_lr()
    
    def _get_lr(self):
        if self.current_step < self.num_warmup_steps:
            return float(self.current_step) / float(max(1, self.num_warmup_steps))
        
        return max(
            0.0,
            float(self.num_training_steps - self.current_step) / 
            float(max(1, self.num_training_steps - self.num_warmup_steps))
        )

class CosineScheduler:
    """Cosine learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, num_warmup_steps, num_training_steps):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._get_lr()
    
    def _get_lr(self):
        import math
        
        if self.current_step < self.num_warmup_steps:
            return float(self.current_step) / float(max(1, self.num_warmup_steps))
        
        progress = float(self.current_step - self.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * progress))
        )

class ConstantScheduler:
    """Constant learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, num_warmup_steps):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self._get_lr()
    
    def _get_lr(self):
        if self.current_step < self.num_warmup_steps:
            return float(self.current_step) / float(max(1, self.num_warmup_steps))
        
        return 1.0 