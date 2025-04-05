import time
import psutil
import logging
from dataclasses import dataclass
from typing import Dict, Any
import prometheus_client as prom

@dataclass
class ModelMetrics:
    latency: float
    memory_usage: float
    gpu_usage: float
    request_count: int
    error_count: int

class ModelMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Prometheus metrics
        self.latency_gauge = prom.Gauge('model_latency_seconds', 'Model inference latency')
        self.memory_gauge = prom.Gauge('model_memory_bytes', 'Model memory usage')
        self.request_counter = prom.Counter('model_requests_total', 'Total requests processed')
        self.error_counter = prom.Counter('model_errors_total', 'Total errors encountered')

    def record_inference(self, start_time: float, success: bool):
        latency = time.time() - start_time
        self.latency_gauge.set(latency)
        self.request_counter.inc()
        if not success:
            self.error_counter.inc()

    def get_resource_usage(self) -> ModelMetrics:
        return ModelMetrics(
            latency=self.latency_gauge._value.get(),
            memory_usage=psutil.Process().memory_info().rss,
            gpu_usage=self._get_gpu_usage(),
            request_count=self.request_counter._value.get(),
            error_count=self.error_counter._value.get()
        )

    def _get_gpu_usage(self) -> float:
        try:
            import torch
            return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        except:
            return 0
