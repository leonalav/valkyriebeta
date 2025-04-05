import torch
import time
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class ProfileStats:
    """Statistics collected during profiling"""
    forward_time: float = 0.0
    backward_time: float = 0.0
    optimizer_time: float = 0.0
    memory_allocated: float = 0.0
    memory_cached: float = 0.0
    flops: int = 0
    throughput: float = 0.0

class PerformanceProfiler:
    """Profiles model performance and resource usage"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stats: List[ProfileStats] = []
        self.current_stats = ProfileStats()
        self.start_time = time.time()
        
    @contextmanager
    def profile_forward(self, batch_size: int):
        """Profile forward pass"""
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        try:
            yield
        finally:
            self.current_stats.forward_time = time.time() - start
            self.current_stats.memory_allocated = torch.cuda.max_memory_allocated() / 1e9
            self.current_stats.throughput = batch_size / self.current_stats.forward_time
            
    @contextmanager
    def profile_backward(self):
        """Profile backward pass"""
        start = time.time()
        try:
            yield
        finally:
            self.current_stats.backward_time = time.time() - start
            
    def update_stats(self):
        """Update profiling statistics"""
        self.stats.append(self.current_stats)
        self.current_stats = ProfileStats()
        
    def get_average_stats(self) -> ProfileStats:
        """Get averaged statistics"""
        if not self.stats:
            return ProfileStats()
            
        avg_stats = ProfileStats()
        n = len(self.stats)
        
        for stat in self.stats:
            avg_stats.forward_time += stat.forward_time
            avg_stats.backward_time += stat.backward_time
            avg_stats.optimizer_time += stat.optimizer_time
            avg_stats.memory_allocated += stat.memory_allocated
            avg_stats.memory_cached += stat.memory_cached
            avg_stats.flops += stat.flops
            avg_stats.throughput += stat.throughput
            
        # Average all fields
        avg_stats.forward_time /= n
        avg_stats.backward_time /= n
        avg_stats.optimizer_time /= n
        avg_stats.memory_allocated /= n
        avg_stats.memory_cached /= n
        avg_stats.flops /= n
        avg_stats.throughput /= n
        
        return avg_stats
        
    def print_summary(self):
        """Print profiling summary"""
        avg_stats = self.get_average_stats()
        
        self.logger.info("\n=== Performance Profile ===")
        self.logger.info(f"Forward Time: {avg_stats.forward_time*1000:.2f}ms")
        self.logger.info(f"Backward Time: {avg_stats.backward_time*1000:.2f}ms")
        self.logger.info(f"Memory Allocated: {avg_stats.memory_allocated:.2f}GB")
        self.logger.info(f"Throughput: {avg_stats.throughput:.2f} samples/sec")
