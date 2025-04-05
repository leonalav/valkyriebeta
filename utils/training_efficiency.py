import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import logging
from contextlib import contextmanager
import psutil
import threading
from collections import deque

@dataclass
class ThroughputStats:
    """Training throughput statistics"""
    samples_per_second: float
    tokens_per_second: float
    gpu_utilization: float
    cpu_utilization: float
    memory_utilization: float
    batch_time: float
    forward_time: float
    backward_time: float
    optimizer_time: float

@dataclass
class LossStats:
    """Loss computation statistics"""
    loss_value: float
    loss_scale: float
    grad_norm: float
    num_inf_nan: int

class TrainingEfficiencyManager:
    """Manages training efficiency optimizations"""
    def __init__(self, config):
        self.config = config
        
        # Initialize mixed precision training
        self.use_amp = config.use_amp
        self.amp_dtype = getattr(torch, config.amp_dtype)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Initialize throughput tracking
        self.throughput_window = deque(maxlen=100)  # Track last 100 steps
        self.step_times = deque(maxlen=100)
        self.start_time = time.time()
        
        # Initialize loss tracking
        self.loss_history = deque(maxlen=1000)
        self.grad_norm_history = deque(maxlen=1000)
        
        # Initialize resource monitoring
        self.monitor_thread = None
        self.monitoring = False
        self.resource_stats = {
            'gpu_util': deque(maxlen=100),
            'cpu_util': deque(maxlen=100),
            'memory_util': deque(maxlen=100)
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Dynamic optimization parameters
        self.current_loss_scale = 1.0
        self.grad_clip_threshold = config.max_grad_norm
        self.learning_rates = []
        
    def start_monitoring(self):
        """Start resource monitoring thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_resources(self):
        """Monitor system resources"""
        while self.monitoring:
            if torch.cuda.is_available():
                # GPU utilization
                gpu_util = torch.cuda.utilization()
                self.resource_stats['gpu_util'].append(gpu_util)
                
            # CPU utilization
            cpu_util = psutil.cpu_percent()
            self.resource_stats['cpu_util'].append(cpu_util)
            
            # Memory utilization
            memory = psutil.virtual_memory()
            self.resource_stats['memory_util'].append(memory.percent)
            
            time.sleep(0.1)  # Sample every 100ms
            
    @contextmanager
    def train_step_context(self):
        """Context manager for tracking training step efficiency"""
        step_start = time.time()
        try:
            yield
        finally:
            step_time = time.time() - step_start
            self.step_times.append(step_time)
            
    def optimize_forward_backward(self, 
                                model: nn.Module,
                                batch: Dict[str, torch.Tensor],
                                optimizer: torch.optim.Optimizer) -> LossStats:
        """Optimized forward and backward pass"""
        forward_start = time.time()
        
        # Mixed precision forward pass
        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            outputs = model(**batch)
            loss = outputs.loss
            
        forward_time = time.time() - forward_start
        
        # Track original loss for monitoring
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        # Efficient backward pass
        backward_start = time.time()
        if self.scaler:
            # Scale loss and run backward pass
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            
            # Unscale gradients for clipping
            self.scaler.unscale_(optimizer)
            
            # Smart gradient clipping
            grad_norm = self._smart_clip_gradients(model)
            
            # Step optimizer with scaling
            self.scaler.step(optimizer)
            scale = self.scaler.get_scale()
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = self._smart_clip_gradients(model)
            optimizer.step()
            scale = 1.0
            
        backward_time = time.time() - backward_start
        
        # Count inf/nan values
        num_inf_nan = self._count_inf_nan_gradients(model)
        
        # Update throughput stats
        self._update_throughput_stats(batch, forward_time, backward_time)
        
        return LossStats(
            loss_value=loss_value,
            loss_scale=scale,
            grad_norm=grad_norm,
            num_inf_nan=num_inf_nan
        )
        
    def _smart_clip_gradients(self, model: nn.Module) -> float:
        """Smart gradient clipping with dynamic thresholds"""
        # Calculate gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.grad_clip_threshold
        )
        
        self.grad_norm_history.append(grad_norm.item())
        
        # Dynamically adjust clipping threshold
        if len(self.grad_norm_history) >= 100:
            median_norm = np.median(self.grad_norm_history)
            if median_norm < self.grad_clip_threshold / 2:
                self.grad_clip_threshold *= 0.9
            elif median_norm > self.grad_clip_threshold * 0.8:
                self.grad_clip_threshold *= 1.1
                
        return grad_norm.item()
        
    def _count_inf_nan_gradients(self, model: nn.Module) -> int:
        """Count number of inf/nan gradients"""
        num_inf_nan = 0
        for param in model.parameters():
            if param.grad is not None:
                num_inf_nan += torch.isfinite(param.grad).logical_not().sum().item()
        return num_inf_nan
        
    def _update_throughput_stats(self, 
                                batch: Dict[str, torch.Tensor],
                                forward_time: float,
                                backward_time: float):
        """Update throughput statistics"""
        # Calculate samples and tokens processed
        batch_size = next(iter(batch.values())).size(0)
        seq_length = next(iter(batch.values())).size(1)
        num_tokens = batch_size * seq_length
        
        step_time = self.step_times[-1]
        samples_per_second = batch_size / step_time
        tokens_per_second = num_tokens / step_time
        
        self.throughput_window.append({
            'samples_per_second': samples_per_second,
            'tokens_per_second': tokens_per_second,
            'forward_time': forward_time,
            'backward_time': backward_time
        })
        
    def get_throughput_stats(self) -> ThroughputStats:
        """Get current throughput statistics"""
        if not self.throughput_window:
            return None
            
        # Calculate averages
        avg_samples = np.mean([x['samples_per_second'] for x in self.throughput_window])
        avg_tokens = np.mean([x['tokens_per_second'] for x in self.throughput_window])
        avg_forward = np.mean([x['forward_time'] for x in self.throughput_window])
        avg_backward = np.mean([x['backward_time'] for x in self.throughput_window])
        
        # Get resource utilization
        gpu_util = np.mean(self.resource_stats['gpu_util']) if self.resource_stats['gpu_util'] else 0
        cpu_util = np.mean(self.resource_stats['cpu_util'])
        memory_util = np.mean(self.resource_stats['memory_util'])
        
        return ThroughputStats(
            samples_per_second=avg_samples,
            tokens_per_second=avg_tokens,
            gpu_utilization=gpu_util,
            cpu_utilization=cpu_util,
            memory_utilization=memory_util,
            batch_time=np.mean(self.step_times),
            forward_time=avg_forward,
            backward_time=avg_backward,
            optimizer_time=np.mean(self.step_times) - avg_forward - avg_backward
        )
        
    def update_learning_rate(self, 
                           optimizer: torch.optim.Optimizer,
                           scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                           metrics: Optional[Dict[str, float]] = None):
        """Dynamic learning rate adjustment"""
        current_lr = optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)
        
        if scheduler is not None:
            # Use scheduler if provided
            scheduler.step()
        elif metrics is not None and len(self.learning_rates) >= 2:
            # Dynamic adjustment based on metrics
            if metrics.get('loss', float('inf')) > np.mean(self.loss_history):
                # Reduce learning rate if loss is increasing
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95
            elif metrics.get('loss', float('inf')) < np.mean(self.loss_history) * 0.95:
                # Increase learning rate if loss is decreasing significantly
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 1.05
                    
    def get_training_efficiency_stats(self) -> Dict[str, Any]:
        """Get comprehensive training efficiency statistics"""
        throughput = self.get_throughput_stats()
        
        return {
            'throughput': {
                'samples_per_second': throughput.samples_per_second,
                'tokens_per_second': throughput.tokens_per_second,
                'batch_time_ms': throughput.batch_time * 1000,
                'forward_time_ms': throughput.forward_time * 1000,
                'backward_time_ms': throughput.backward_time * 1000,
                'optimizer_time_ms': throughput.optimizer_time * 1000
            },
            'resource_utilization': {
                'gpu_util': throughput.gpu_utilization,
                'cpu_util': throughput.cpu_utilization,
                'memory_util': throughput.memory_utilization
            },
            'training_stats': {
                'loss_mean': np.mean(self.loss_history) if self.loss_history else 0,
                'loss_std': np.std(self.loss_history) if self.loss_history else 0,
                'grad_norm_mean': np.mean(self.grad_norm_history) if self.grad_norm_history else 0,
                'grad_clip_threshold': self.grad_clip_threshold,
                'learning_rate': self.learning_rates[-1] if self.learning_rates else 0
            }
        }
        
    def print_efficiency_stats(self):
        """Print formatted efficiency statistics"""
        stats = self.get_training_efficiency_stats()
        
        self.logger.info("\nTraining Efficiency Statistics:")
        self.logger.info(f"Throughput:")
        self.logger.info(f"  Samples/second: {stats['throughput']['samples_per_second']:.2f}")
        self.logger.info(f"  Tokens/second: {stats['throughput']['tokens_per_second']:.2f}")
        self.logger.info(f"  Batch time: {stats['throughput']['batch_time_ms']:.2f}ms")
        self.logger.info(f"\nResource Utilization:")
        self.logger.info(f"  GPU: {stats['resource_utilization']['gpu_util']:.1f}%")
        self.logger.info(f"  CPU: {stats['resource_utilization']['cpu_util']:.1f}%")
        self.logger.info(f"  Memory: {stats['resource_utilization']['memory_util']:.1f}%")
        self.logger.info(f"\nTraining Stats:")
        self.logger.info(f"  Loss: {stats['training_stats']['loss_mean']:.4f} ± {stats['training_stats']['loss_std']:.4f}")
        self.logger.info(f"  Grad norm: {stats['training_stats']['grad_norm_mean']:.4f}")
        self.logger.info(f"  Learning rate: {stats['training_stats']['learning_rate']:.6f}")
        
    def cleanup(self):
        """Cleanup manager resources"""
        self.stop_monitoring()
        self.throughput_window.clear()
        self.step_times.clear()
        self.loss_history.clear()
        self.grad_norm_history.clear()
        for stats in self.resource_stats.values():
            stats.clear() 