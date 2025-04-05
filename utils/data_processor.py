import psutil
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ResourceMetrics:
    cpu_percent: float
    memory_usage: float
    gpu_utilization: Optional[float]
    disk_io: Dict[str, float]

class ResourceMonitor:
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.metrics_history: List[ResourceMetrics] = []

    def sample_metrics(self) -> ResourceMetrics:
        return ResourceMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            gpu_utilization=self._get_gpu_utilization(),
            disk_io=self._get_disk_io_stats()
        )

    def _get_gpu_utilization(self) -> Optional[float]:
        try:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            return info.gpu
        except:
            return None

    def _get_disk_io_stats(self) -> Dict[str, float]:
        disk_io = psutil.disk_io_counters()
        return {
            'read_bytes': disk_io.read_bytes,
            'write_bytes': disk_io.write_bytes
        }

class ProcessingMonitor:
    def __init__(self):
        self.start_time = None
        self.processed_items = 0
        self.error_count = 0
        self.processing_times: List[float] = []

    def start_batch(self):
        self.start_time = time.time()

    def end_batch(self, items_processed: int, errors: int = 0):
        if self.start_time is None:
            return
            
        duration = time.time() - self.start_time
        self.processing_times.append(duration)
        self.processed_items += items_processed
        self.error_count += errors
        self.start_time = None

    def get_statistics(self) -> Dict[str, float]:
        return {
            'total_processed': self.processed_items,
            'error_rate': self.error_count / max(1, self.processed_items),
            'avg_processing_time': sum(self.processing_times) / max(1, len(self.processing_times)),
            'throughput': self.processed_items / max(1, sum(self.processing_times))
        }
