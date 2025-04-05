import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import time
import psutil
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import logging
import threading
from contextlib import contextmanager
import gc

@dataclass
class MemoryAllocation:
    """Tracks memory allocation details"""
    size: int  # in bytes
    timestamp: float
    source: str
    lifetime: float
    device: str
    dtype: torch.dtype
    shape: tuple

@dataclass
class MemoryBottleneck:
    """Identifies memory bottlenecks"""
    component: str
    severity: float  # 0 to 1
    impact: float  # MB
    suggestion: str
    priority: int  # 1 to 5

@dataclass
class MemoryStats:
    peak_memory: float
    current_memory: float
    timeline: List[Dict]

class MemoryProfiler:
    """Advanced memory profiling and optimization system"""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking
        self.allocations = defaultdict(list)
        self.peak_memory = defaultdict(float)
        self.component_memory = defaultdict(float)
        self.bottlenecks = []
        self.timeline = []
        
        # Monitoring state
        self.monitoring = False
        self.start_time = time.time()
        self._tracking = False
        
        # Setup hooks
        self.allocation_hooks = []
        self.model_hooks = []
        
    def start_profiling(self, model: Optional[nn.Module] = None):
        """Start memory profiling"""
        self.monitoring = True
        self._tracking = True
        if model:
            self._setup_hooks(model)
        self._start_monitoring_thread()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
    def stop_profiling(self):
        """Stop memory profiling"""
        self.monitoring = False
        self._tracking = False
        self._remove_hooks()
        
    def _setup_hooks(self, model: nn.Module):
        """Setup memory tracking hooks"""
        def forward_hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                self._track_allocation(
                    out,
                    f"forward_{type(module).__name__}",
                    out.device
                )
        
        def backward_hook(module, grad_in, grad_out):
            if isinstance(grad_out, tuple):
                for grad in grad_out:
                    if isinstance(grad, torch.Tensor):
                        self._track_allocation(
                            grad,
                            f"backward_{type(module).__name__}",
                            grad.device
                        )
        
        # Register hooks for all modules
        for name, module in model.named_modules():
            self.model_hooks.extend([
                module.register_forward_hook(forward_hook),
                module.register_backward_hook(backward_hook)
            ])
            
    def _track_allocation(self, tensor: torch.Tensor, source: str, device: str):
        """Track tensor allocation"""
        if not self.monitoring:
            return
            
        allocation = MemoryAllocation(
            size=tensor.numel() * tensor.element_size(),
            timestamp=time.time() - self.start_time,
            source=source,
            lifetime=0.0,  # Will be updated on deallocation
            device=str(device),
            dtype=tensor.dtype,
            shape=tuple(tensor.shape)
        )
        
        self.allocations[source].append(allocation)
        self.component_memory[source] += allocation.size / 1024**2  # Convert to MB
        
        # Update peak memory
        if torch.cuda.is_available() and 'cuda' in str(device):
            current = torch.cuda.memory_allocated(device) / 1024**2
            self.peak_memory[str(device)] = max(
                self.peak_memory[str(device)],
                current
            )
            
    def analyze_bottlenecks(self) -> List[MemoryBottleneck]:
        """Analyze and identify memory bottlenecks"""
        bottlenecks = []
        
        # Analyze component memory usage
        total_memory = sum(self.component_memory.values())
        for component, memory in self.component_memory.items():
            ratio = memory / total_memory
            
            if ratio > 0.3:  # Component uses >30% of memory
                bottleneck = MemoryBottleneck(
                    component=component,
                    severity=ratio,
                    impact=memory,
                    suggestion=self._get_optimization_suggestion(component, ratio),
                    priority=self._get_priority(ratio)
                )
                bottlenecks.append(bottleneck)
                
        # Analyze allocation patterns
        for source, allocs in self.allocations.items():
            if len(allocs) > 1000:  # Many small allocations
                bottleneck = MemoryBottleneck(
                    component=source,
                    severity=0.7,
                    impact=sum(a.size for a in allocs) / 1024**2,
                    suggestion="Consider using tensor pooling or reducing allocation frequency",
                    priority=3
                )
                bottlenecks.append(bottleneck)
                
        # Sort by priority and severity
        bottlenecks.sort(key=lambda x: (-x.priority, -x.severity))
        self.bottlenecks = bottlenecks
        return bottlenecks
        
    def _get_optimization_suggestion(self, component: str, ratio: float) -> str:
        """Get optimization suggestion based on component and usage ratio"""
        if "attention" in component.lower():
            return "Consider using FlashAttention or memory-efficient attention variants"
        elif "embedding" in component.lower():
            return "Consider using quantized or factorized embeddings"
        elif "linear" in component.lower():
            return "Consider using memory-efficient linear layers with activation checkpointing"
        elif ratio > 0.5:
            return "Consider gradient checkpointing or reducing model size for this component"
        else:
            return "Monitor usage patterns and consider optimization if trend continues"
            
    def _get_priority(self, ratio: float) -> int:
        """Get priority level based on memory impact"""
        if ratio > 0.5:
            return 5  # Critical
        elif ratio > 0.3:
            return 4  # High
        elif ratio > 0.2:
            return 3  # Medium
        elif ratio > 0.1:
            return 2  # Low
        else:
            return 1  # Very Low
            
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary"""
        return {
            'peak_memory': dict(self.peak_memory),
            'component_memory': dict(self.component_memory),
            'allocation_counts': {
                k: len(v) for k, v in self.allocations.items()
            },
            'bottlenecks': [
                {
                    'component': b.component,
                    'severity': b.severity,
                    'impact': b.impact,
                    'suggestion': b.suggestion
                }
                for b in self.bottlenecks
            ]
        }
        
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get prioritized optimization recommendations"""
        recommendations = []
        stats = self.get_memory_stats()
        
        if torch.cuda.is_available() and stats.peak_memory > 0.9 * torch.cuda.get_device_properties(0).total_memory:
            recommendations.append({
                'component': 'GPU Memory',
                'recommendation': 'Consider reducing batch size or enabling gradient checkpointing',
                'priority': 'high'
            })
        
        # Analyze memory patterns
        for component, allocs in self.allocations.items():
            avg_size = np.mean([a.size for a in allocs])
            count = len(allocs)
            
            if avg_size > 1024**2 * 100:  # Large allocations
                recommendations.append({
                    'component': component,
                    'issue': 'Large tensor allocations',
                    'recommendation': 'Consider gradient checkpointing or model parallelism',
                    'priority': 'High'
                })
            elif count > 1000:  # Frequent allocations
                recommendations.append({
                    'component': component,
                    'issue': 'Frequent small allocations',
                    'recommendation': 'Use tensor pooling or reduce allocation frequency',
                    'priority': 'Medium'
                })
                
        # Add bottleneck-based recommendations
        for bottleneck in self.analyze_bottlenecks():
            recommendations.append({
                'component': bottleneck.component,
                'issue': f'Memory bottleneck (severity: {bottleneck.severity:.2f})',
                'recommendation': bottleneck.suggestion,
                'priority': 'High' if bottleneck.priority >= 4 else 'Medium'
            })
            
        return recommendations
        
    def print_memory_report(self):
        """Print formatted memory profiling report"""
        self.logger.info("\n=== Memory Profiling Report ===")
        
        # Print peak memory usage
        self.logger.info("\nPeak Memory Usage:")
        for device, peak in self.peak_memory.items():
            self.logger.info(f"{device}: {peak:.2f} MB")
            
        # Print component memory usage
        self.logger.info("\nComponent Memory Usage:")
        sorted_components = sorted(
            self.component_memory.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for component, memory in sorted_components:
            self.logger.info(f"{component}: {memory:.2f} MB")
            
        # Print bottlenecks
        self.logger.info("\nIdentified Bottlenecks:")
        for bottleneck in self.bottlenecks:
            self.logger.info(
                f"\nComponent: {bottleneck.component}"
                f"\n  Severity: {bottleneck.severity:.2f}"
                f"\n  Impact: {bottleneck.impact:.2f} MB"
                f"\n  Suggestion: {bottleneck.suggestion}"
            )
            
        # Print optimization recommendations
        self.logger.info("\nOptimization Recommendations:")
        for rec in self.get_optimization_recommendations():
            self.logger.info(
                f"\n{rec['component']}:"
                f"\n  Issue: {rec['issue']}"
                f"\n  Recommendation: {rec['recommendation']}"
                f"\n  Priority: {rec['priority']}"
            )
            
    def cleanup(self):
        """Cleanup profiler resources"""
        self.stop_profiling()
        self.allocations.clear()
        self.component_memory.clear()
        self.bottlenecks.clear()
        self.timeline = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    def get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        stats = {
            'cpu_used': psutil.Process().memory_info().rss / (1024 * 1024 * 1024),
            'cpu_percent': psutil.cpu_percent()
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_used': torch.cuda.memory_allocated() / (1024 * 1024 * 1024),
                'gpu_cached': torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
            })
            
        return stats

    def get_memory_stats(self) -> MemoryStats:
        """Get memory statistics"""
        current = self.get_current_memory_usage()
        return MemoryStats(
            peak_memory=self.peak_memory,
            current_memory=current['gpu_used'] if torch.cuda.is_available() else current['cpu_used'],
            timeline=self.timeline
        )

    def get_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        if not self.timeline:
            return 0.0
            
        avg_usage = sum(t['memory'] for t in self.timeline) / len(self.timeline)
        return avg_usage / self.peak_memory if self.peak_memory > 0 else 0.0