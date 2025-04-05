import gc
import torch
from typing import Optional, List, Dict
from contextlib import contextmanager

class EnhancedMemoryManager:
    def __init__(self, monitor_tensors: bool = True):
        self.monitor_tensors = monitor_tensors
        self.tensor_snapshots: List[Dict[str, int]] = []
        self.peak_memory: float = 0
        self.baseline_memory: float = 0

    @contextmanager
    def track_memory(self, label: str = ""):
        try:
            self.start_tracking()
            yield
        finally:
            self.end_tracking(label)

    def start_tracking(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.baseline_memory = torch.cuda.memory_allocated()

    def end_tracking(self, label: str):
        current_memory = torch.cuda.memory_allocated()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        if self.monitor_tensors:
            self.tensor_snapshots.append({
                'label': label,
                'active_tensors': len([obj for obj in gc.get_objects() 
                                     if torch.is_tensor(obj)]),
                'memory_allocated': current_memory,
                'memory_delta': current_memory - self.baseline_memory
            })

    def get_memory_stats(self) -> Dict[str, float]:
        return {
            'peak_memory_mb': self.peak_memory / (1024 * 1024),
            'current_memory_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'max_memory_delta_mb': max(
                (snap['memory_delta'] for snap in self.tensor_snapshots),
                default=0
            ) / (1024 * 1024)
        }

    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.tensor_snapshots.clear()
        self.peak_memory = 0
        self.baseline_memory = 0
