"""
Advanced learning rate schedulers for ValkyrieLLM.

This module provides specialized learning rate scheduling techniques that go beyond
the standard schedulers available in PyTorch, designed specifically for LLM training.
"""

import math
import warnings
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable
import torch
from torch.optim.lr_scheduler import _LRScheduler


class OneCycleLR(_LRScheduler):
    """
    One Cycle Learning Rate Scheduler.
    
    Implements the 1cycle learning rate policy, including the three phases:
    1. Linear warmup phase (from base_lr to max_lr)
    2. Cosine annealing phase (from max_lr to min_lr)
    3. Optional cooldown phase (at min_lr)
    
    Based on the paper: "Super-Convergence: Very Fast Training of Neural Networks
    Using Large Learning Rates" by Leslie N. Smith and Nicholay Topin.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle.
        total_steps (int): The total number of steps in the training process.
        pct_start (float): Percentage of the cycle spent increasing the learning rate.
        pct_cooldown (float): Percentage of the cycle spent at min_lr after annealing.
        div_factor (float): Determines the initial learning rate via initial_lr = max_lr/div_factor
        final_div_factor (float): Determines the minimum learning rate via min_lr = initial_lr/final_div_factor
        anneal_strategy (str): {'cos', 'linear'}, specifies the annealing strategy.
        three_phase (bool): If True, use three phases (warmup, annealing, cooldown).
    """
    
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, pct_cooldown=0.1,
                 div_factor=25.0, final_div_factor=10000.0, anneal_strategy='cos',
                 three_phase=True, last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.pct_cooldown = pct_cooldown
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.anneal_strategy = anneal_strategy
        self.three_phase = three_phase
        
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError(f"anneal_strategy must be 'cos' or 'linear', got {anneal_strategy}")
        
        # Calculate steps for each phase
        self.warmup_steps = int(total_steps * pct_start)
        if three_phase:
            self.cooldown_steps = int(total_steps * pct_cooldown)
            self.anneal_steps = total_steps - self.warmup_steps - self.cooldown_steps
        else:
            self.cooldown_steps = 0
            self.anneal_steps = total_steps - self.warmup_steps
        
        # Calculate learning rates for each phase
        if not isinstance(max_lr, list) and not isinstance(max_lr, tuple):
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        else:
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(f"Expected {len(optimizer.param_groups)} max_lr, got {len(max_lr)}")
            self.max_lrs = list(max_lr)
        
        # Initial learning rates (start of warmup)
        self.init_lrs = [max_lr / self.div_factor for max_lr in self.max_lrs]
        
        # Minimum learning rates (end of annealing)
        self.min_lrs = [init_lr / self.final_div_factor for init_lr in self.init_lrs]
        
        super(OneCycleLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Warmup phase: linear increase from init_lr to max_lr
            alpha = step / self.warmup_steps
            return [init_lr + alpha * (max_lr - init_lr) 
                   for init_lr, max_lr in zip(self.init_lrs, self.max_lrs)]
        
        elif step < self.warmup_steps + self.anneal_steps:
            # Annealing phase: cosine or linear decrease from max_lr to min_lr
            alpha = (step - self.warmup_steps) / self.anneal_steps
            
            if self.anneal_strategy == 'cos':
                # Cosine annealing
                cos_factor = (1 + math.cos(math.pi * alpha)) / 2
                return [min_lr + cos_factor * (max_lr - min_lr) 
                       for min_lr, max_lr in zip(self.min_lrs, self.max_lrs)]
            else:
                # Linear annealing
                return [max_lr - alpha * (max_lr - min_lr) 
                       for min_lr, max_lr in zip(self.min_lrs, self.max_lrs)]
        
        else:
            # Cooldown phase: constant min_lr
            return self.min_lrs


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine Annealing with Warm Restarts and initial warmup.
    
    First cycle is extended to include a warmup period. All subsequent
    cycles follow the standard cosine annealing with warm restarts pattern.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult (float): Cycle steps magnification factor.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        warmup_steps (int): Number of warmup steps.
        gamma (float): Decrease rate of max learning rate per cycle.
        last_epoch (int): The index of the last epoch.
    """
    
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, max_lr=0.1,
                 min_lr=1e-7, warmup_steps=0, gamma=1.0, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr if isinstance(max_lr, list) else [max_lr] * len(optimizer.param_groups)
        self.min_lr = min_lr if isinstance(min_lr, list) else [min_lr] * len(optimizer.param_groups)
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cycle = 0
        self.cycle_steps = first_cycle_steps
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # Initialize step counts and learning rates
        self.initial_lrs = []
        
        for param_group, max_lr in zip(self.optimizer.param_groups, self.max_lr):
            param_group['initial_lr'] = max_lr
            self.initial_lrs.append(max_lr)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        
        lrs = []
        step = self.last_epoch
        
        # Calculate current cycle and steps within cycle
        if step < self.warmup_steps:
            # Warmup phase
            alpha = step / self.warmup_steps
            for initial_lr, max_lr in zip(self.initial_lrs, self.max_lr):
                lr = initial_lr + alpha * (max_lr - initial_lr)
                lrs.append(lr)
            return lrs
        
        # Determine cycle and step within cycle
        cycle_step = step - self.warmup_steps
        cycles_completed = 0
        cycle_step_sum = 0
        
        cycle_steps = self.first_cycle_steps
        while cycle_step > cycle_step_sum + cycle_steps:
            cycle_step_sum += cycle_steps
            cycles_completed += 1
            cycle_steps = int(cycle_steps * self.cycle_mult)
        
        cycle_step = cycle_step - cycle_step_sum
        
        # Calculate current cycle's max_lr based on decay
        current_max_lrs = [base_max_lr * (self.gamma ** cycles_completed) 
                          for base_max_lr in self.max_lr]
        
        # Calculate and return this step's learning rates
        for i, (min_lr, max_lr) in enumerate(zip(self.min_lr, current_max_lrs)):
            # Cosine formula
            cos_factor = 0.5 * (1 + math.cos(math.pi * cycle_step / cycle_steps))
            lr = min_lr + (max_lr - min_lr) * cos_factor
            lrs.append(lr)
        
        return lrs


class AdaptiveGradualWarmupScheduler(_LRScheduler):
    """
    Gradually warm-up learning rate and then apply another scheduler.
    
    This scheduler adjusts its warmup length based on validation loss trends,
    extending warmup if loss spikes occur, ensuring more stable training.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_scheduler (_LRScheduler): Scheduler to wrap.
        warmup_steps (int): Target number of warmup steps.
        min_warmup_steps (int): Minimum number of warmup steps to perform.
        adaptive_threshold (float): Loss improvement threshold for warmup.
        last_epoch (int): The index of the last epoch.
    """
    
    def __init__(self, optimizer, base_scheduler, warmup_steps=1000, 
                 min_warmup_steps=100, adaptive_threshold=0.1, last_epoch=-1):
        self.base_scheduler = base_scheduler
        self.warmup_steps = warmup_steps
        self.min_warmup_steps = min_warmup_steps
        self.adaptive_threshold = adaptive_threshold
        self.current_warmup_steps = warmup_steps
        self.finished_warmup = False
        self.previous_losses = []
        
        # Get initial learning rate
        for param_group in optimizer.param_groups:
            param_group.setdefault('initial_lr', param_group['lr'])
        
        # Save initial learning rates
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        
        super(AdaptiveGradualWarmupScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")
        
        if self.last_epoch < self.current_warmup_steps:
            # Still in warmup phase
            alpha = self.last_epoch / self.current_warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        
        # Warmup is finished, use the base scheduler
        if not self.finished_warmup:
            self.base_scheduler.base_lrs = self.base_lrs
            self.finished_warmup = True
        
        return self.base_scheduler.get_lr()
    
    def step(self, epoch=None, metrics=None):
        """
        Step the scheduler with optional validation metrics.
        
        Args:
            epoch (int, optional): The current epoch.
            metrics (float, optional): The validation loss or other metric.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        
        self.last_epoch = epoch
        
        # If metrics provided, potentially adjust warmup steps
        if metrics is not None and not self.finished_warmup:
            self.previous_losses.append(metrics)
            
            if len(self.previous_losses) >= 3:
                # Check if loss is increasing
                if self.previous_losses[-1] > self.previous_losses[-2] * (1 + self.adaptive_threshold):
                    # Loss is increasing significantly, extend warmup
                    additional_steps = min(500, self.current_warmup_steps // 2)
                    self.current_warmup_steps += additional_steps
                    print(f"Loss increased, extending warmup by {additional_steps} steps")
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer for better generalization.
    
    Implements the SAM optimization technique as described in the paper:
    "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    by Foret et al.
    
    Args:
        params: Model parameters
        base_optimizer: Base optimizer like SGD or Adam
        rho: Size of the neighborhood for computing sharpness
        adaptive: Whether to use adaptive SAM
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        # Initialize the base optimizer
        self.base_optimizer = base_optimizer
        
        # Check if the base_optimizer is initialized properly
        if not isinstance(self.base_optimizer, torch.optim.Optimizer):
            raise TypeError("base_optimizer must be an instance of torch.optim.Optimizer")
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Compute and apply the perturbation for SAM."""
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                if group["adaptive"]:
                    # Adaptive SAM
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale
                else:
                    # Standard SAM
                    e_w = p.grad * scale
                
                # Store the perturbation and apply it
                self.state[p]["e_w"] = e_w
                p.add_(e_w)
        
        # Zero gradients for the second step
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Revert the perturbation and apply the original optimization step."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                
                # Remove the perturbation
                p.sub_(self.state[p]["e_w"])
                
                # Clean up for the next iteration
                self.state[p].pop("e_w")
        
        # Apply the base optimizer's step
        self.base_optimizer.step()
        
        # Zero gradients for the next step
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        """Compute the gradient norm for all parameters."""
        norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach())
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ])
        )
        return norm
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        super().load_state_dict(state_dict)
        self.base_optimizer.load_state_dict(state_dict.get('base_optimizer', {}))
    
    def state_dict(self):
        """Save optimizer state."""
        state_dict = super().state_dict()
        state_dict['base_optimizer'] = self.base_optimizer.state_dict()
        return state_dict


class CurriculumScheduler:
    """
    Implements curriculum learning by gradually increasing task difficulty.
    
    This scheduler adjusts training parameters based on model performance,
    gradually increasing the difficulty of the training data or tasks.
    
    Args:
        initial_difficulty (float): Initial difficulty level (0.0 to 1.0).
        stages (int): Number of curriculum stages.
        metrics_threshold (float): Threshold to advance to next difficulty level.
        patience (int): Number of steps to wait before increasing difficulty.
        max_difficulty (float): Maximum difficulty level.
    """
    
    def __init__(self, initial_difficulty=0.1, stages=5, metrics_threshold=0.8, 
                 patience=100, max_difficulty=1.0):
        self.initial_difficulty = initial_difficulty
        self.current_difficulty = initial_difficulty
        self.stages = stages
        self.metrics_threshold = metrics_threshold
        self.patience = patience
        self.max_difficulty = max_difficulty
        
        # Track metrics for curriculum advancement
        self.steps_at_current_stage = 0
        self.best_metric = float('inf')
        self.stage = 0
        
        # Calculate difficulty increment
        self.difficulty_increment = (max_difficulty - initial_difficulty) / (stages - 1) if stages > 1 else 0
    
    def step(self, metrics=None):
        """
        Update the curriculum stage based on performance metrics.
        
        Args:
            metrics (float): Performance metric (lower is better).
            
        Returns:
            float: Current difficulty level.
        """
        self.steps_at_current_stage += 1
        
        # Update best metric if improved
        if metrics is not None and metrics < self.best_metric:
            self.best_metric = metrics
        
        # Check if we should advance to next stage
        if (self.steps_at_current_stage >= self.patience and 
            metrics is not None and 
            metrics <= self.metrics_threshold and 
            self.stage < self.stages - 1):
            
            # Advance to next stage
            self.stage += 1
            self.current_difficulty = min(
                self.initial_difficulty + self.stage * self.difficulty_increment,
                self.max_difficulty
            )
            
            # Reset counters
            self.steps_at_current_stage = 0
            self.best_metric = float('inf')
            
            print(f"Advancing to curriculum stage {self.stage}/{self.stages-1} "
                  f"with difficulty {self.current_difficulty:.2f}")
        
        return self.current_difficulty
    
    def get_difficulty(self):
        """Get the current difficulty level."""
        return self.current_difficulty
    
    def state_dict(self):
        """Get state for saving."""
        return {
            'current_difficulty': self.current_difficulty,
            'stage': self.stage,
            'steps_at_current_stage': self.steps_at_current_stage,
            'best_metric': self.best_metric
        }
    
    def load_state_dict(self, state_dict):
        """Load saved state."""
        self.current_difficulty = state_dict.get('current_difficulty', self.initial_difficulty)
        self.stage = state_dict.get('stage', 0)
        self.steps_at_current_stage = state_dict.get('steps_at_current_stage', 0)
        self.best_metric = state_dict.get('best_metric', float('inf'))


class LearningRateMonitor:
    """
    Monitors learning rates across all parameter groups.
    
    This utility class tracks learning rates throughout training and
    provides functionality to log, plot, and analyze learning rate trends.
    
    Args:
        log_interval (int): Interval for logging learning rates.
    """
    
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.lrs_history = []
        self.step_history = []
        self.current_step = 0
    
    def step(self, optimizer):
        """
        Record learning rates for the current step.
        
        Args:
            optimizer: The optimizer to monitor.
        """
        self.current_step += 1
        
        if self.current_step % self.log_interval == 0:
            current_lrs = []
            
            # Get learning rates from all parameter groups
            for param_group in optimizer.param_groups:
                current_lrs.append(param_group['lr'])
            
            # Record learning rates and step
            self.lrs_history.append(current_lrs)
            self.step_history.append(self.current_step)
    
    def get_lr_history(self, param_group_idx=0):
        """
        Get learning rate history for a specific parameter group.
        
        Args:
            param_group_idx (int): Parameter group index.
            
        Returns:
            List of learning rates and steps.
        """
        lrs = [group_lrs[param_group_idx] for group_lrs in self.lrs_history]
        return self.step_history, lrs
    
    def reset(self):
        """Reset the monitor."""
        self.lrs_history = []
        self.step_history = []
        self.current_step = 0 