import torch
import gc
import psutil
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

@dataclass
class MemoryStats:
    gpu_allocated: float
    gpu_cached: float
    cpu_used: float
    gpu_peak: float
    cpu_peak: float

class MemoryManager:
    """Manages memory optimizations dynamically during training and inference"""
    def __init__(self, config):
        self.config = config
        # Add compatibility with enhanced memory manager interface
        self.profiler = self.config.profiler if hasattr(self.config, 'profiler') else None
        self.min_batch_size = self.config.min_batch_size if hasattr(self.config, 'min_batch_size') else 1
        self.max_batch_size = self.config.max_batch_size if hasattr(self.config, 'max_batch_size') else 32
        self.gpu_memory_peak = 0
        self.cpu_memory_peak = 0
        self.current_batch_size = config.batch_size
        
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**2
            gpu_cached = torch.cuda.memory_reserved() / 1024**2
        else:
            gpu_allocated = gpu_cached = 0
            
        cpu_used = psutil.Process().memory_info().rss / 1024**2
        
        # Update peaks
        self.gpu_memory_peak = max(self.gpu_memory_peak, gpu_allocated)
        self.cpu_memory_peak = max(self.cpu_memory_peak, cpu_used)
        
        return MemoryStats(
            gpu_allocated=gpu_allocated,
            gpu_cached=gpu_cached,
            cpu_used=cpu_used,
            gpu_peak=self.gpu_memory_peak,
            cpu_peak=self.cpu_memory_peak
        )
        
    def optimize_batch_size(self, current_memory_usage: float) -> int:
        """Dynamically adjust batch size based on memory usage"""
        if not self.config.dynamic_batch_size:
            return self.current_batch_size
            
        target = self.config.target_memory_usage
        if current_memory_usage > target * 1.1:  # Over target by 10%
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
        elif current_memory_usage < target * 0.8:  # Under target by 20%
            self.current_batch_size = min(
                self.config.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
        return self.current_batch_size
        
    def should_offload_to_cpu(self, tensor_size: int) -> bool:
        """Determine if a tensor should be offloaded to CPU"""
        if not self.config.use_cpu_offload:
            return False
            
        stats = self.get_memory_stats()
        gpu_usage = stats.gpu_allocated / (torch.cuda.get_device_properties(0).total_memory / 1024**2)
        return gpu_usage > self.config.cpu_offload_threshold
        
    def clear_memory(self, aggressive: bool = False):
        """Clear unused memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if aggressive:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
    def optimize_attention_pattern(self, seq_length: int) -> Dict[str, any]:
        """Determine optimal attention pattern based on sequence length"""
        if seq_length <= 512:
            return {"pattern": "full"}
        elif seq_length <= 2048:
            return {
                "pattern": "sliding_window",
                "window_size": min(self.config.window_size, seq_length // 4)
            }
        else:
            return {
                "pattern": "local_global",
                "local_size": 512,
                "global_tokens": 256
            }
            
    def get_activation_checkpoint_layers(self) -> Tuple[int, int]:
        """Determine which layers to apply activation checkpointing"""
        if not self.config.use_activation_checkpointing:
            return (0, 0)
            
        num_layers = self.config.num_layers
        checkpoint_ratio = self.config.activation_checkpointing_ratio
        start_layer = int(num_layers * (1 - checkpoint_ratio))
        return (start_layer, num_layers)
        
    def optimize_linear_patterns(self) -> Dict[str, any]:
        """Determine optimal linear layer patterns"""
        stats = self.get_memory_stats()
        gpu_usage = stats.gpu_allocated / (torch.cuda.get_device_properties(0).total_memory / 1024**2)
        
        if gpu_usage > 0.9:  # High memory pressure
            return {
                "use_sparse": True,
                "sparsity_ratio": 0.5,
                "quantize": True,
                "bits": 4
            }
        elif gpu_usage > 0.7:  # Medium memory pressure
            return {
                "use_sparse": True,
                "sparsity_ratio": 0.3,
                "quantize": True,
                "bits": 8
            }
        else:  # Low memory pressure
            return {
                "use_sparse": False,
                "quantize": True,
                "bits": 8
            }
            
    def optimize_embedding_pattern(self) -> Dict[str, any]:
        """Determine optimal embedding pattern"""
        stats = self.get_memory_stats()
        gpu_usage = stats.gpu_allocated / (torch.cuda.get_device_properties(0).total_memory / 1024**2)
        
        if gpu_usage > 0.8:
            return {
                "factorize": True,
                "rank": self.config.hidden_size // 8,
                "tie_weights": True
            }
        else:
            return {
                "factorize": True,
                "rank": self.config.hidden_size // 4,
                "tie_weights": True
            }
            
    @staticmethod
    def get_optimal_dtype(current_memory_usage: float) -> torch.dtype:
        """Determine optimal dtype based on memory usage"""
        if current_memory_usage > 0.9:
            return torch.float16
        elif current_memory_usage > 0.7:
            return torch.bfloat16
        else:
            return torch.float32