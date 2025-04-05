"""
Profiling modules for LLM training performance analysis.
Provides tools to monitor memory usage, communication patterns, and training dynamics.
"""

import torch
import time
import threading
import logging
import psutil
import os
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class MemoryProfiler:
    """
    Profiles memory usage during model training.
    Tracks GPU and CPU memory consumption over time.
    """
    
    def __init__(
        self,
        interval: int = 100,
        log_to_file: bool = True,
        log_path: str = "memory_profile.jsonl",
        verbose: bool = False
    ):
        """
        Initialize memory profiler.
        
        Args:
            interval: Number of steps between memory profiling snapshots
            log_to_file: Whether to log profiling data to a file
            log_path: Path to log file
            verbose: Whether to print memory stats to console
        """
        self.interval = interval
        self.log_to_file = log_to_file
        self.log_path = log_path
        self.verbose = verbose
        
        self.running = False
        self.thread = None
        self.step = 0
        self.memory_stats = []
        self.cpu_process = psutil.Process(os.getpid())
        
        if log_to_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
    
    def start(self):
        """Start memory profiling thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._profile_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started memory profiling with interval {self.interval} steps")
    
    def stop(self):
        """Stop memory profiling thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        # Save final memory stats
        if self.log_to_file and self.memory_stats:
            self._save_stats()
            
        logger.info("Stopped memory profiling")
    
    def _profile_loop(self):
        """Main profiling loop that runs in a separate thread."""
        while self.running:
            # Capture memory snapshot
            self._capture_snapshot()
            
            # Sleep for a bit to avoid excessive profiling
            time.sleep(0.1)
            
            # Increment internal step counter
            self.step += 1
            
            # Log to file periodically
            if self.log_to_file and self.step % 10 == 0:
                self._save_stats()
    
    def _capture_snapshot(self):
        """Capture a memory usage snapshot."""
        if self.step % self.interval != 0:
            return
            
        # Collect GPU memory stats if available
        gpu_stats = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
                gpu_stats[f"gpu_{i}"] = {
                    "allocated_gb": allocated,
                    "reserved_gb": reserved
                }
        
        # Collect CPU memory stats
        cpu_percent = self.cpu_process.cpu_percent()
        memory_info = self.cpu_process.memory_info()
        cpu_stats = {
            "rss_gb": memory_info.rss / (1024 ** 3),  # GB
            "vms_gb": memory_info.vms / (1024 ** 3),  # GB
            "cpu_percent": cpu_percent
        }
        
        # Collect timestamp
        timestamp = time.time()
        
        # Create snapshot
        snapshot = {
            "step": self.step,
            "timestamp": timestamp,
            "gpu": gpu_stats,
            "cpu": cpu_stats
        }
        
        # Add to memory stats
        self.memory_stats.append(snapshot)
        
        # Log to console if verbose
        if self.verbose:
            gpu_mem = sum(stats["allocated_gb"] for stats in gpu_stats.values()) if gpu_stats else 0
            logger.info(f"Step {self.step}: GPU Memory: {gpu_mem:.2f} GB, CPU Memory: {cpu_stats['rss_gb']:.2f} GB")
    
    def _save_stats(self):
        """Save memory stats to log file."""
        if not self.memory_stats:
            return
            
        with open(self.log_path, "a") as f:
            for snapshot in self.memory_stats:
                f.write(json.dumps(snapshot) + "\n")
        
        # Clear in-memory stats after saving
        self.memory_stats = []


class CommunicationProfiler:
    """
    Profiles communication patterns in distributed training.
    Tracks volume, frequency, and latency of communication operations.
    """
    
    def __init__(
        self,
        interval: int = 100,
        log_to_file: bool = True,
        log_path: str = "communication_profile.jsonl",
        track_bandwidth: bool = True,
        track_latency: bool = True
    ):
        """
        Initialize communication profiler.
        
        Args:
            interval: Number of steps between profiling snapshots
            log_to_file: Whether to log profiling data to a file
            log_path: Path to log file
            track_bandwidth: Whether to track communication bandwidth
            track_latency: Whether to track communication latency
        """
        self.interval = interval
        self.log_to_file = log_to_file
        self.log_path = log_path
        self.track_bandwidth = track_bandwidth
        self.track_latency = track_latency
        
        self.running = False
        self.thread = None
        self.step = 0
        self.comm_stats = []
        
        # Communication operation tracking
        self.op_counters = defaultdict(int)
        self.op_sizes = defaultdict(list)
        self.op_latencies = defaultdict(list)
        
        # Register PyTorch distributed hooks if available
        self._register_dist_hooks()
        
        if log_to_file:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
    
    def _register_dist_hooks(self):
        """Register hooks on PyTorch distributed operations."""
        if not torch.distributed.is_available():
            logger.warning("PyTorch distributed not available, communication profiling will be limited")
            return
            
        if not torch.distributed.is_initialized():
            logger.warning("PyTorch distributed not initialized, communication profiling will be limited")
            return
            
        # Try to monkey patch distributed operations to track communication
        original_all_reduce = torch.distributed.all_reduce
        
        def patched_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
            """Patched all_reduce to track communication."""
            start_time = time.time()
            size_bytes = tensor.element_size() * tensor.numel()
            
            # Call original function
            result = original_all_reduce(tensor, op, group, async_op)
            
            # Track operation
            if not async_op:
                latency = time.time() - start_time
                self._record_operation("all_reduce", size_bytes, latency)
            
            return result
        
        # Apply patch
        try:
            torch.distributed.all_reduce = patched_all_reduce
            logger.info("Successfully patched torch.distributed.all_reduce for profiling")
        except Exception as e:
            logger.warning(f"Failed to patch distributed operations: {e}")
    
    def _record_operation(self, op_name: str, size_bytes: int, latency: float):
        """Record a communication operation."""
        self.op_counters[op_name] += 1
        self.op_sizes[op_name].append(size_bytes)
        self.op_latencies[op_name].append(latency)
    
    def start(self):
        """Start communication profiling thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._profile_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started communication profiling with interval {self.interval} steps")
    
    def stop(self):
        """Stop communication profiling thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        # Save final stats
        if self.log_to_file and self.comm_stats:
            self._save_stats()
            
        logger.info("Stopped communication profiling")
    
    def _profile_loop(self):
        """Main profiling loop that runs in a separate thread."""
        while self.running:
            # Capture communication snapshot
            self._capture_snapshot()
            
            # Sleep for a bit to avoid excessive profiling
            time.sleep(0.1)
            
            # Increment internal step counter
            self.step += 1
            
            # Log to file periodically
            if self.log_to_file and self.step % 10 == 0:
                self._save_stats()
    
    def _capture_snapshot(self):
        """Capture a communication snapshot."""
        if self.step % self.interval != 0:
            return
            
        # Create operation stats
        op_stats = {}
        for op_name, count in self.op_counters.items():
            sizes = self.op_sizes[op_name]
            latencies = self.op_latencies[op_name]
            
            op_stats[op_name] = {
                "count": count,
                "total_bytes": sum(sizes),
                "avg_size_kb": np.mean(sizes) / 1024 if sizes else 0,
                "avg_latency_ms": np.mean(latencies) * 1000 if latencies else 0,
                "bandwidth_mbps": sum(sizes) / sum(latencies) / (1024 * 1024) if latencies and sum(latencies) > 0 else 0
            }
        
        # Create snapshot
        snapshot = {
            "step": self.step,
            "timestamp": time.time(),
            "operations": op_stats
        }
        
        # Add to stats
        self.comm_stats.append(snapshot)
        
        # Reset counters for next interval
        self.op_counters = defaultdict(int)
        self.op_sizes = defaultdict(list)
        self.op_latencies = defaultdict(list)
    
    def _save_stats(self):
        """Save communication stats to log file."""
        if not self.comm_stats:
            return
            
        with open(self.log_path, "a") as f:
            for snapshot in self.comm_stats:
                f.write(json.dumps(snapshot) + "\n")
        
        # Clear in-memory stats after saving
        self.comm_stats = []


class PerformanceProfiler:
    """
    Profiles model forward and backward pass performance.
    Tracks computation time for different model components.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        warm_up: int = 10,
        profile_steps: int = 100,
        granularity: str = "layer"  # "layer", "block", or "operation"
    ):
        """
        Initialize performance profiler.
        
        Args:
            model: Model to profile
            warm_up: Number of warm-up steps before profiling
            profile_steps: Number of steps to profile
            granularity: Profiling granularity level
        """
        self.model = model
        self.warm_up = warm_up
        self.profile_steps = profile_steps
        self.granularity = granularity
        
        self.step = 0
        self.forward_times = defaultdict(list)
        self.backward_times = defaultdict(list)
        self.hooks = []
        
        # Register hooks based on granularity
        self._register_hooks()
    
    def _register_hooks(self):
        """Register profiling hooks on model components."""
        if self.granularity == "layer":
            # Register hooks on top-level modules
            for name, module in self.model.named_children():
                self._register_module_hooks(name, module)
        
        elif self.granularity == "block":
            # Register hooks on blocks of interest (transformer layers, etc.)
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
                for i, layer in enumerate(self.model.transformer.layers):
                    self._register_module_hooks(f"transformer.layer_{i}", layer)
        
        elif self.granularity == "operation":
            # Register hooks on individual operations
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Dropout)):
                    self._register_module_hooks(name, module)
    
    def _register_module_hooks(self, name: str, module: torch.nn.Module):
        """Register forward and backward hooks on a module."""
        module_times = {}
        
        def forward_hook(module, inputs, outputs):
            """Forward pass timing hook."""
            if self.step < self.warm_up:
                return
                
            if self.step >= self.warm_up + self.profile_steps:
                return
                
            module_times["start"] = time.time()
        
        def forward_hook_end(module, inputs, outputs):
            """Forward pass end timing hook."""
            if self.step < self.warm_up:
                return
                
            if self.step >= self.warm_up + self.profile_steps:
                return
                
            if "start" in module_times:
                forward_time = time.time() - module_times.pop("start")
                self.forward_times[name].append(forward_time)
        
        def backward_hook(module, grad_input, grad_output):
            """Backward pass timing hook."""
            if self.step < self.warm_up:
                return
                
            if self.step >= self.warm_up + self.profile_steps:
                return
                
            module_times["backward_start"] = time.time()
        
        def backward_hook_end(module, grad_input, grad_output):
            """Backward pass end timing hook."""
            if self.step < self.warm_up:
                return
                
            if self.step >= self.warm_up + self.profile_steps:
                return
                
            if "backward_start" in module_times:
                backward_time = time.time() - module_times.pop("backward_start")
                self.backward_times[name].append(backward_time)
        
        # Register hooks
        handle1 = module.register_forward_pre_hook(forward_hook)
        handle2 = module.register_forward_hook(forward_hook_end)
        handle3 = module.register_backward_hook(backward_hook)
        
        self.hooks.extend([handle1, handle2, handle3])
    
    def step_completed(self):
        """Mark a training step as completed."""
        self.step += 1
        
        # Log summary after profiling is complete
        if self.step == self.warm_up + self.profile_steps:
            self._log_profiling_summary()
    
    def _log_profiling_summary(self):
        """Log profiling summary statistics."""
        logger.info("Performance profiling summary:")
        
        # Forward pass stats
        logger.info("Forward pass timing (ms):")
        for name, times in sorted(self.forward_times.items()):
            avg_time_ms = np.mean(times) * 1000
            logger.info(f"  {name}: {avg_time_ms:.2f} ms")
        
        # Backward pass stats
        logger.info("Backward pass timing (ms):")
        for name, times in sorted(self.backward_times.items()):
            avg_time_ms = np.mean(times) * 1000
            logger.info(f"  {name}: {avg_time_ms:.2f} ms")
        
        # Total time
        total_forward = sum(np.mean(times) for times in self.forward_times.values())
        total_backward = sum(np.mean(times) for times in self.backward_times.values())
        logger.info(f"Total forward time: {total_forward * 1000:.2f} ms")
        logger.info(f"Total backward time: {total_backward * 1000:.2f} ms")
        logger.info(f"Total step time: {(total_forward + total_backward) * 1000:.2f} ms")
    
    def remove_hooks(self):
        """Remove all profiling hooks from the model."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = [] 