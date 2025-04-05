from dataclasses import dataclass
import time
import psutil
from typing import Dict

@dataclass
class ResourceMetrics:
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    memory_used_mb: float
    
@dataclass
class ProcessingMetrics:
    samples_processed: int
    total_time: float
    tokenization_time: float
    preprocessing_time: float
    samples_per_second: float
    avg_batch_time: float

class ResourceMonitor:
    """Monitor system resource usage"""
    def __init__(self):
        self.process = psutil.Process()
        
    def get_metrics(self) -> ResourceMetrics:
        return ResourceMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage_percent=psutil.disk_usage('/').percent,
            memory_used_mb=self.process.memory_info().rss / 1024 / 1024
        )

class ProcessingMonitor:
    """Monitor data processing metrics"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.start_time = time.time()
        self.samples_processed = 0
        self.total_tokenization_time = 0.0
        self.total_preprocessing_time = 0.0
        self.total_batch_time = 0.0
        self.num_batches = 0
        
    def update(self, batch_size: int, batch_time: float, 
               tokenization_time: float, preprocessing_time: float):
        self.samples_processed += batch_size
        self.total_tokenization_time += tokenization_time
        self.total_preprocessing_time += preprocessing_time
        self.total_batch_time += batch_time
        self.num_batches += 1
        
    def get_metrics(self) -> ProcessingMetrics:
        total_time = time.time() - self.start_time
        return ProcessingMetrics(
            samples_processed=self.samples_processed,
            total_time=total_time,
            tokenization_time=self.total_tokenization_time,
            preprocessing_time=self.total_preprocessing_time,
            samples_per_second=self.samples_processed / total_time if total_time > 0 else 0,
            avg_batch_time=self.total_batch_time / self.num_batches if self.num_batches > 0 else 0
        )
