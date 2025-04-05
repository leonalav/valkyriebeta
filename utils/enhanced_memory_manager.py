import torch
import gc
import os
import logging
import psutil
import threading
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """Statistics about memory usage"""
    timestamp: float
    cuda_allocated_gb: float = 0.0
    cuda_reserved_gb: float = 0.0
    cuda_max_allocated_gb: float = 0.0
    system_used_gb: float = 0.0
    system_available_gb: float = 0.0
    batch_size: int = 0
    sequence_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp,
            "cuda_allocated_gb": self.cuda_allocated_gb,
            "cuda_reserved_gb": self.cuda_reserved_gb,
            "cuda_max_allocated_gb": self.cuda_max_allocated_gb,
            "system_used_gb": self.system_used_gb,
            "system_available_gb": self.system_available_gb,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length
        }

class EnhancedMemoryManager:
    """Advanced memory manager for LLM training and inference
    
    Features:
    - Dynamic batch size adjustment
    - Memory usage tracking
    - Automatic garbage collection
    - CUDA cache management
    - CPU offloading
    """
    
    def __init__(
        self,
        target_gpu_utilization: float = 0.9,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
        gpu_memory_buffer_gb: float = 1.0,
        monitoring_interval: float = 10.0,
        auto_gc_threshold: float = 0.95,
        enable_monitoring: bool = True
    ):
        """Initialize the memory manager
        
        Args:
            target_gpu_utilization: Target GPU memory utilization ratio (0.0 to 1.0)
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            gpu_memory_buffer_gb: GPU memory buffer in GB to keep free
            monitoring_interval: Memory monitoring interval in seconds
            auto_gc_threshold: Threshold for triggering automatic garbage collection
            enable_monitoring: Whether to enable background monitoring
        """
        self.target_gpu_utilization = target_gpu_utilization
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.gpu_memory_buffer_gb = gpu_memory_buffer_gb
        self.monitoring_interval = monitoring_interval
        self.auto_gc_threshold = auto_gc_threshold
        
        # Memory profiling data
        self.current_batch_size = max_batch_size
        self.history: List[MemoryStats] = []
        self.oom_events = 0
        
        # Threading for background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Start monitoring if enabled
        if enable_monitoring:
            self.start_monitoring()
            
    def start_monitoring(self):
        """Start background memory monitoring"""
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Memory monitoring is already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="memory-monitor"
        )
        self.monitoring_thread.start()
        logger.info("Started memory monitoring")
        
    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        logger.info("Stopped memory monitoring")
        
    def _monitoring_loop(self):
        """Background thread for monitoring memory usage"""
        while self.monitoring_active:
            try:
                # Collect memory stats
                stats = self.get_memory_stats()
                self.history.append(stats)
                
                # Keep history to a reasonable size
                if len(self.history) > 1000:
                    self.history = self.history[-1000:]
                    
                # Check if we need to perform garbage collection
                if stats.cuda_allocated_gb / stats.cuda_reserved_gb > self.auto_gc_threshold:
                    logger.info(f"Auto GC triggered: {stats.cuda_allocated_gb:.2f}GB/{stats.cuda_reserved_gb:.2f}GB")
                    self.release_memory()
                    
                # Wait for next monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {str(e)}")
                time.sleep(self.monitoring_interval)
                
    def get_memory_stats(self, batch_size: int = 0, sequence_length: int = 0) -> MemoryStats:
        """Get current memory statistics
        
        Args:
            batch_size: Current batch size (if known)
            sequence_length: Current sequence length (if known)
            
        Returns:
            MemoryStats with current memory usage information
        """
        stats = MemoryStats(
            timestamp=time.time(),
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        # System memory
        try:
            system_memory = psutil.virtual_memory()
            stats.system_used_gb = system_memory.used / (1024 ** 3)
            stats.system_available_gb = system_memory.available / (1024 ** 3)
        except Exception as e:
            logger.warning(f"Failed to get system memory: {str(e)}")
            
        # CUDA memory
        if torch.cuda.is_available():
            try:
                stats.cuda_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                stats.cuda_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                stats.cuda_max_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            except Exception as e:
                logger.warning(f"Failed to get CUDA memory: {str(e)}")
                
        return stats
        
    def track_batch(self, batch_size: int, sequence_length: int):
        """Track a batch for memory usage analysis
        
        Args:
            batch_size: Current batch size
            sequence_length: Current sequence length
        """
        stats = self.get_memory_stats(batch_size, sequence_length)
        self.history.append(stats)
        
    def compute_optimal_batch_size(
        self, 
        current_batch_size: int, 
        sequence_length: int,
        gpu_mem_per_sample: Optional[float] = None
    ) -> int:
        """Compute optimal batch size based on memory constraints
        
        Args:
            current_batch_size: Current batch size
            sequence_length: Current sequence length
            gpu_mem_per_sample: GPU memory per sample in GB (if known)
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return current_batch_size
            
        # Get available GPU memory
        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
        
        # Calculate available memory (accounting for buffer)
        available_memory = total_gpu_memory - allocated_memory - self.gpu_memory_buffer_gb
        
        # If we don't know gpu_mem_per_sample, estimate it from history
        if gpu_mem_per_sample is None:
            relevant_history = [
                stat for stat in self.history 
                if stat.batch_size > 0 and stat.sequence_length > 0
            ]
            
            if relevant_history:
                # Find similar sequence lengths
                similar_length_stats = [
                    stat for stat in relevant_history
                    if 0.8 <= stat.sequence_length / sequence_length <= 1.2
                ]
                
                if similar_length_stats:
                    # Estimate memory per sample based on similar batches
                    memory_per_samples = [
                        stat.cuda_allocated_gb / stat.batch_size 
                        for stat in similar_length_stats
                    ]
                    gpu_mem_per_sample = sum(memory_per_samples) / len(memory_per_samples)
                else:
                    # Fallback: use a simple scaling assumption
                    gpu_mem_per_sample = 0.5 * (sequence_length / 1024)  # Example scaling
            else:
                # No history yet, use a conservative estimate
                gpu_mem_per_sample = 0.5  # GB per sample
                
        # Calculate how many samples we can fit
        if gpu_mem_per_sample > 0:
            max_samples = int(available_memory / gpu_mem_per_sample)
            optimal_batch_size = max(self.min_batch_size, min(max_samples, self.max_batch_size))
            
            # If current batch size is working, don't reduce unless we need to
            if current_batch_size <= optimal_batch_size:
                return current_batch_size
                
            return optimal_batch_size
            
        # If we can't estimate, return current batch size
        return current_batch_size
        
    def release_memory(self, full_release: bool = False):
        """Release unused memory
        
        Args:
            full_release: Whether to perform a full memory release (more aggressive)
        """
        # Run garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Full release performs more aggressive cleanup
        if full_release:
            # Reset peak memory stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
            # Force a full garbage collection
            for _ in range(3):
                gc.collect()
        
        # Log memory after release
        if logger.isEnabledFor(logging.DEBUG):
            stats = self.get_memory_stats()
            logger.debug(f"Memory after release: {stats.cuda_allocated_gb:.2f}GB allocated, "
                        f"{stats.cuda_reserved_gb:.2f}GB reserved")
                        
    def handle_oom(self, batch_size: int, sequence_length: int) -> int:
        """Handle an out-of-memory error by reducing batch size
        
        Args:
            batch_size: Batch size that caused OOM
            sequence_length: Sequence length that caused OOM
            
        Returns:
            New suggested batch size
        """
        self.oom_events += 1
        
        # Log OOM event
        logger.warning(f"OOM detected with batch_size={batch_size}, seq_length={sequence_length}")
        
        # Perform full memory release
        self.release_memory(full_release=True)
        
        # Calculate new batch size (more conservative)
        new_batch_size = max(self.min_batch_size, batch_size // 2)
        logger.info(f"Reducing batch size from {batch_size} to {new_batch_size}")
        
        return new_batch_size
        
    def offload_tensors(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Offload tensors to CPU to free GPU memory
        
        Args:
            tensors: List of tensors to offload
            
        Returns:
            List of offloaded tensors (on CPU)
        """
        cpu_tensors = []
        for tensor in tensors:
            if tensor.device.type == 'cuda':
                cpu_tensors.append(tensor.detach().cpu())
            else:
                cpu_tensors.append(tensor)
                
        return cpu_tensors
        
    def reload_tensors(self, tensors: List[torch.Tensor], device: torch.device) -> List[torch.Tensor]:
        """Reload tensors to specified device
        
        Args:
            tensors: List of tensors to reload
            device: Target device
            
        Returns:
            List of reloaded tensors on target device
        """
        return [tensor.to(device) for tensor in tensors]
        
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of memory usage statistics
        
        Returns:
            Dictionary with memory usage summary
        """
        if not self.history:
            return {
                "samples": 0,
                "oom_events": self.oom_events
            }
            
        # Calculate statistics
        allocated = [stat.cuda_allocated_gb for stat in self.history if hasattr(stat, 'cuda_allocated_gb')]
        reserved = [stat.cuda_reserved_gb for stat in self.history if hasattr(stat, 'cuda_reserved_gb')]
        batch_sizes = [stat.batch_size for stat in self.history if stat.batch_size > 0]
        
        avg_allocated = sum(allocated) / len(allocated) if allocated else 0
        max_allocated = max(allocated) if allocated else 0
        avg_reserved = sum(reserved) / len(reserved) if reserved else 0
        max_reserved = max(reserved) if reserved else 0
        avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
        
        # Get latest stats
        latest = self.history[-1] if self.history else None
        
        return {
            "samples": len(self.history),
            "oom_events": self.oom_events,
            "avg_allocated_gb": avg_allocated,
            "max_allocated_gb": max_allocated,
            "avg_reserved_gb": avg_reserved,
            "max_reserved_gb": max_reserved,
            "avg_batch_size": avg_batch_size,
            "latest": latest.to_dict() if latest else None
        }
        
    def __del__(self):
        """Cleanup when the manager is deleted"""
        self.stop_monitoring()