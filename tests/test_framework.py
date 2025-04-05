import torch
import torch.nn as nn
import pytest
import time
import psutil
import gc
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging
import tracemalloc
from torch.cuda.amp import autocast
import torch.multiprocessing as mp

from model import LogicalReasoningTransformer
from config import (
    ModelConfig,
    TrainingConfig,
    MemoryConfig,
    TrainingEfficiencyConfig,
    EfficientTransformerConfig
)
from utils.memory_profiler import MemoryProfiler
from utils.enhanced_memory_manager import EnhancedMemoryManager

@dataclass
class TestConfig:
    """Configuration for test framework"""
    # Memory testing
    memory_threshold_mb: int = 1000
    leak_detection_epochs: int = 3
    memory_sample_interval: int = 100
    
    # Performance testing
    perf_test_iterations: int = 1000
    warmup_iterations: int = 100
    latency_threshold_ms: float = 100.0
    throughput_threshold: float = 100.0
    
    # Stability testing
    stability_test_epochs: int = 5
    loss_variance_threshold: float = 0.1
    grad_norm_threshold: float = 10.0
    
    # Resource testing
    max_gpu_memory_usage: float = 0.9
    max_cpu_usage: float = 0.8
    
    # Numerical testing
    rtol: float = 1e-5
    atol: float = 1e-8
    
class TestFramework:
    def __init__(
        self,
        model: LogicalReasoningTransformer,
        test_config: TestConfig,
        memory_config: MemoryConfig,
        efficiency_config: TrainingEfficiencyConfig
    ):
        self.model = model
        self.test_config = test_config
        self.memory_profiler = MemoryProfiler(memory_config)
        self.memory_manager = EnhancedMemoryManager(memory_config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracemalloc for memory leak detection
        tracemalloc.start()
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        results = {}
        
        # Memory tests
        self.logger.info("Running memory tests...")
        results['memory'] = self.run_memory_tests()
        
        # Performance tests
        self.logger.info("Running performance tests...")
        results['performance'] = self.run_performance_tests()
        
        # Stability tests
        self.logger.info("Running stability tests...")
        results['stability'] = self.run_stability_tests()
        
        # Resource tests
        self.logger.info("Running resource tests...")
        results['resources'] = self.run_resource_tests()
        
        # Numerical tests
        self.logger.info("Running numerical tests...")
        results['numerical'] = self.run_numerical_tests()
        
        return results
        
    def run_memory_tests(self) -> Dict[str, Any]:
        """Run memory leak detection and tracking"""
        snapshot1 = tracemalloc.take_snapshot()
        
        # Run model multiple times
        for _ in range(self.test_config.leak_detection_epochs):
            with self.memory_manager.track_memory("test_iteration"):
                self._run_test_iteration()
                
        snapshot2 = tracemalloc.take_snapshot()
        
        # Compare memory snapshots
        memory_diff = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Get detailed memory stats
        memory_stats = self.memory_profiler.get_memory_stats()
        
        return {
            'leaks': [str(stat) for stat in memory_diff[:10]],  # Top 10 memory differences
            'peak_memory': memory_stats.peak_memory,
            'current_memory': memory_stats.current_memory,
            'memory_timeline': memory_stats.timeline
        }
        
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and latency tests"""
        latencies = []
        throughputs = []
        
        # Warmup
        for _ in range(self.test_config.warmup_iterations):
            self._run_test_iteration()
            
        # Actual testing
        batch_size = 32
        for _ in range(self.test_config.perf_test_iterations):
            start_time = time.time()
            
            with self.memory_manager.track_memory("perf_test"):
                self._run_test_iteration(batch_size=batch_size)
                
            end_time = time.time()
            
            # Calculate metrics
            latency = (end_time - start_time) * 1000  # ms
            throughput = batch_size / (end_time - start_time)  # samples/sec
            
            latencies.append(latency)
            throughputs.append(throughput)
            
        return {
            'avg_latency': np.mean(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'avg_throughput': np.mean(throughputs),
            'peak_throughput': np.max(throughputs)
        }
        
    def run_stability_tests(self) -> Dict[str, Any]:
        """Run training stability tests"""
        losses = []
        grad_norms = []
        
        for epoch in range(self.test_config.stability_test_epochs):
            epoch_losses = []
            epoch_grads = []
            
            for _ in range(100):  # 100 iterations per epoch
                loss, grad_norm = self._run_training_iteration()
                epoch_losses.append(loss)
                epoch_grads.append(grad_norm)
                
            losses.append(np.mean(epoch_losses))
            grad_norms.append(np.mean(epoch_grads))
            
        return {
            'loss_variance': np.var(losses),
            'loss_trend': np.polyfit(range(len(losses)), losses, 1)[0],
            'grad_norm_mean': np.mean(grad_norms),
            'grad_norm_std': np.std(grad_norms)
        }
        
    def run_resource_tests(self) -> Dict[str, Any]:
        """Run resource utilization tests"""
        gpu_memory = []
        cpu_usage = []
        
        for _ in range(100):  # 100 measurements
            with self.memory_manager.track_memory("resource_test"):
                self._run_test_iteration()
                
            if torch.cuda.is_available():
                gpu_memory.append(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
            
            cpu_usage.append(psutil.cpu_percent() / 100.0)
            
        return {
            'gpu_memory_usage': np.mean(gpu_memory) if gpu_memory else None,
            'cpu_usage': np.mean(cpu_usage),
            'memory_efficiency': self.memory_profiler.get_memory_efficiency()
        }
        
    def run_numerical_tests(self) -> Dict[str, Any]:
        """Run numerical stability tests"""
        # Test with different dtypes
        dtypes = [torch.float32, torch.float16] if torch.cuda.is_available() else [torch.float32]
        results = {}
        
        base_output = None
        for dtype in dtypes:
            self.model = self.model.to(dtype)
            output = self._run_test_iteration()
            
            if base_output is None:
                base_output = output
            else:
                # Compare with base output
                results[str(dtype)] = {
                    'max_diff': torch.max(torch.abs(output - base_output)).item(),
                    'mean_diff': torch.mean(torch.abs(output - base_output)).item(),
                    'is_close': torch.allclose(output, base_output, 
                                             rtol=self.test_config.rtol,
                                             atol=self.test_config.atol)
                }
                
        return results
        
    def _run_test_iteration(self, batch_size: int = 32) -> torch.Tensor:
        """Run a single test iteration"""
        self.model.eval()
        with torch.no_grad():
            # Create dummy input
            input_ids = torch.randint(0, self.model.config.vocab_size, (batch_size, 128)).to(self.device)
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            return outputs['logits']
            
    def _run_training_iteration(self) -> Tuple[float, float]:
        """Run a single training iteration"""
        self.model.train()
        
        # Create dummy input and labels
        input_ids = torch.randint(0, self.model.config.vocab_size, (32, 128)).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        labels = torch.randint(0, self.model.config.num_classes, (32,)).to(self.device)
        
        # Forward and backward pass
        outputs = self.model(input_ids, attention_mask, labels)
        loss = outputs['loss']
        loss.backward()
        
        # Calculate gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        return loss.item(), grad_norm.item()
        
    def cleanup(self):
        """Cleanup resources"""
        tracemalloc.stop()
        self.memory_profiler.cleanup()
        self.memory_manager.cleanup()
        torch.cuda.empty_cache()
        gc.collect() 