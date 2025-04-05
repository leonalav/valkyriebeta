"""
Metrics collector for LLM model inference and training.
Collects and reports metrics using the metrics module.
"""

import time
import logging
import psutil
import threading
from typing import Dict, Any, List, Optional, Union
import os
from datetime import datetime, timedelta
import json

from .metrics import (
    Counter, 
    Gauge, 
    Histogram, 
    Summary, 
    MetricRegistry,
    MetricsExporter,
    PROMETHEUS_AVAILABLE
)

logger = logging.getLogger("monitoring.metrics_collector")

class MetricsCollector:
    """Collects and reports metrics for LLM model operation"""
    
    def __init__(self, prefix: str = "llm"):
        """Initialize metrics collector
        
        Args:
            prefix: Prefix for all metric names
        """
        self.prefix = prefix
        
        # Initialize metrics
        self._setup_metrics()
        
        # Start resource monitoring
        self.resource_monitor_thread = None
        self.resource_monitoring_active = False
        
    def _setup_metrics(self):
        """Set up all metrics used by the collector"""
        # Inference metrics
        self.inference_requests = Counter(f"{self.prefix}_inference_requests", 
                                         "Total number of inference requests")
        self.inference_latency = Histogram(f"{self.prefix}_inference_latency_seconds", 
                                         "Inference latency in seconds")
        self.tokens_generated = Counter(f"{self.prefix}_tokens_generated", 
                                      "Total number of tokens generated")
        self.input_tokens_processed = Counter(f"{self.prefix}_input_tokens_processed", 
                                           "Total number of input tokens processed")
        self.error_rate = Counter(f"{self.prefix}_errors_total", 
                                "Total number of errors in inference")
        self.batch_size = Histogram(f"{self.prefix}_batch_size", 
                                  "Batch size distribution")
        self.tokens_per_second = Histogram(f"{self.prefix}_tokens_per_second", 
                                         "Tokens per second throughput")
        
        # Model metrics
        self.model_load_time = Histogram(f"{self.prefix}_model_load_time_seconds", 
                                       "Time taken to load model in seconds")
        self.cache_hits = Counter(f"{self.prefix}_cache_hits", 
                                "Total number of cache hits")
        self.cache_misses = Counter(f"{self.prefix}_cache_misses", 
                                  "Total number of cache misses")
        
        # Resource metrics
        self.cpu_usage = Gauge(f"{self.prefix}_cpu_usage_percent", 
                             "CPU usage percentage")
        self.memory_usage = Gauge(f"{self.prefix}_memory_usage_bytes", 
                                "Memory usage in bytes")
        self.gpu_memory_used = Gauge(f"{self.prefix}_gpu_memory_used_bytes", 
                                   "GPU memory used in bytes")
        self.gpu_utilization = Gauge(f"{self.prefix}_gpu_utilization_percent", 
                                   "GPU utilization percentage")
        
        # Request queue metrics
        self.queue_size = Gauge(f"{self.prefix}_queue_size", 
                              "Number of requests in the queue")
        self.queue_wait_time = Histogram(f"{self.prefix}_queue_wait_time_seconds", 
                                      "Time spent in queue before processing")
                                      
        # Training metrics
        self.training_steps = Counter(f"{self.prefix}_training_steps", 
                                    "Total number of training steps")
        self.training_loss = Gauge(f"{self.prefix}_training_loss", 
                                 "Current training loss value")
        self.training_throughput = Gauge(f"{self.prefix}_training_throughput_samples_per_second", 
                                      "Training throughput in samples per second")
        self.gradient_norm = Histogram(f"{self.prefix}_gradient_norm", 
                                     "Gradient norm distribution")
        
    def start_resource_monitoring(self, interval_seconds: float = 5.0):
        """Start monitoring system and GPU resources in the background
        
        Args:
            interval_seconds: Interval between resource measurements
        """
        if self.resource_monitor_thread and self.resource_monitor_thread.is_alive():
            logger.warning("Resource monitoring already running")
            return
            
        self.resource_monitoring_active = True
        self.resource_monitor_thread = threading.Thread(
            target=self._resource_monitor_loop,
            args=(interval_seconds,),
            daemon=True,
            name="resource-monitor"
        )
        self.resource_monitor_thread.start()
        logger.info(f"Started resource monitoring with interval {interval_seconds}s")
        
    def stop_resource_monitoring(self):
        """Stop the resource monitoring thread"""
        if self.resource_monitor_thread and self.resource_monitor_thread.is_alive():
            self.resource_monitoring_active = False
            self.resource_monitor_thread.join(timeout=5.0)
            logger.info("Stopped resource monitoring")
        
    def _resource_monitor_loop(self, interval_seconds: float):
        """Background thread to monitor system resources
        
        Args:
            interval_seconds: Interval between measurements
        """
        while self.resource_monitoring_active:
            try:
                # CPU and memory monitoring
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                
                # Update CPU and memory metrics
                self.cpu_usage.set(cpu_percent)
                self.memory_usage.set(memory_info.used)
                
                # GPU monitoring if available
                try:
                    self._update_gpu_metrics()
                except Exception as e:
                    logger.warning(f"Failed to collect GPU metrics: {str(e)}")
                    
                # Sleep until next interval
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
                time.sleep(interval_seconds)
        
    def _update_gpu_metrics(self):
        """Update GPU metrics if CUDA is available"""
        try:
            import torch
            if torch.cuda.is_available():
                # For each GPU
                for i in range(torch.cuda.device_count()):
                    # Memory metrics
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    
                    # Update metrics
                    self.gpu_memory_used.set(memory_allocated)
                    
                    # Try to get utilization if nvidia-smi is available
                    try:
                        import subprocess
                        result = subprocess.run(
                            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True,
                            text=True
                        )
                        utilization = float(result.stdout.strip())
                        self.gpu_utilization.set(utilization)
                    except Exception:
                        # Utilization measurement not available
                        pass
        except ImportError:
            # CUDA not available
            pass
            
    def log_inference(self, duration: float, tokens_generated: int, error: bool = False, 
                     input_tokens: int = 0):
        """Log a model inference request
        
        Args:
            duration: Duration of inference in seconds
            tokens_generated: Number of tokens generated
            error: Whether the inference resulted in an error
            input_tokens: Number of input tokens processed
        """
        # Increment request counter
        self.inference_requests.inc()
        
        # Log input tokens
        if input_tokens > 0:
            self.input_tokens_processed.inc(input_tokens)
        
        # If successful, log performance metrics
        if not error:
            self.inference_latency.observe(duration)
            self.tokens_generated.inc(tokens_generated)
            
            # Calculate tokens per second
            if duration > 0 and tokens_generated > 0:
                tokens_per_second = tokens_generated / duration
                self.tokens_per_second.observe(tokens_per_second)
        else:
            # Log error
            self.error_rate.inc()
            
    def log_batch_inference(self, duration: float, batch_size: int, tokens_generated: int = 0, 
                          error: bool = False):
        """Log a batch inference request
        
        Args:
            duration: Duration of batch inference in seconds
            batch_size: Size of the batch
            tokens_generated: Total tokens generated across batch
            error: Whether the inference resulted in an error
        """
        # Log the batch size
        self.batch_size.observe(batch_size)
        
        # Log individual requests in the batch
        if not error and tokens_generated > 0:
            # Log overall metrics
            self.inference_requests.inc(batch_size)
            self.tokens_generated.inc(tokens_generated)
            
            # Calculate average latency per item
            avg_latency = duration / batch_size
            for _ in range(batch_size):
                self.inference_latency.observe(avg_latency)
        elif error:
            # Log batch error
            self.inference_requests.inc(batch_size)
            self.error_rate.inc(batch_size)
            
    def log_model_load_time(self, duration: float):
        """Log time taken to load a model
        
        Args:
            duration: Time in seconds to load the model
        """
        self.model_load_time.observe(duration)
        
    def log_cache_hit(self):
        """Log a cache hit"""
        self.cache_hits.inc()
        
    def log_cache_miss(self):
        """Log a cache miss"""
        self.cache_misses.inc()
        
    def log_queue_metrics(self, queue_size: int, wait_time: Optional[float] = None):
        """Log queue metrics
        
        Args:
            queue_size: Current size of the request queue
            wait_time: Optional wait time in queue in seconds
        """
        self.queue_size.set(queue_size)
        if wait_time is not None:
            self.queue_wait_time.observe(wait_time)
            
    def log_training_step(self, loss: float, gradient_norm: Optional[float] = None, 
                        samples_per_second: Optional[float] = None):
        """Log a training step
        
        Args:
            loss: Current loss value
            gradient_norm: Optional gradient norm value
            samples_per_second: Optional throughput measurement
        """
        self.training_steps.inc()
        self.training_loss.set(loss)
        
        if gradient_norm is not None:
            self.gradient_norm.observe(gradient_norm)
            
        if samples_per_second is not None:
            self.training_throughput.set(samples_per_second)
            
    def get_metrics_as_dict(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary
        
        Returns:
            Dictionary with all metrics and their values
        """
        metrics = {
            "inference": {
                "requests": self.inference_requests.get(),
                "tokens_generated": self.tokens_generated.get(),
                "errors": self.error_rate.get(),
                "latency": self.inference_latency.get_stats(),
                "throughput": self.tokens_per_second.get_stats()
            },
            "model": {
                "load_time": self.model_load_time.get_stats(),
                "cache": {
                    "hits": self.cache_hits.get(),
                    "misses": self.cache_misses.get(),
                    "hit_ratio": self.cache_hits.get() / (self.cache_hits.get() + self.cache_misses.get()) 
                             if (self.cache_hits.get() + self.cache_misses.get()) > 0 else 0
                }
            },
            "resources": {
                "cpu_usage": self.cpu_usage.get(),
                "memory_usage": self.memory_usage.get(),
                "gpu_memory_used": self.gpu_memory_used.get(),
                "gpu_utilization": self.gpu_utilization.get()
            },
            "queue": {
                "size": self.queue_size.get(),
                "wait_time": self.queue_wait_time.get_stats()
            },
            "training": {
                "steps": self.training_steps.get(),
                "loss": self.training_loss.get(),
                "throughput": self.training_throughput.get(),
                "gradient_norm": self.gradient_norm.get_stats()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format
        
        Args:
            format: Format to export ('json' or 'prometheus')
            
        Returns:
            String representation of metrics in requested format
        """
        exporter = MetricsExporter()
        
        if format.lower() == "json":
            return exporter.export_json()
        elif format.lower() == "prometheus" and PROMETHEUS_AVAILABLE:
            return exporter.export_prometheus()
        else:
            return json.dumps(self.get_metrics_as_dict(), indent=2)
            
    def export_to_file(self, filename: str, format: str = "json"):
        """Export metrics to a file
        
        Args:
            filename: Path to output file
            format: Format to export ('json' or 'prometheus')
        """
        exporter = MetricsExporter()
        exporter.export_to_file(filename, format)
        
class ProductionMetricsCollector(MetricsCollector):
    """Extended metrics collector with additional production metrics"""
    
    def __init__(self, prefix: str = "llm"):
        """Initialize production metrics collector
        
        Args:
            prefix: Prefix for all metric names
        """
        super().__init__(prefix)
        
        # Add production-specific metrics
        self._setup_production_metrics()
        
        # Start resource monitoring by default
        self.start_resource_monitoring()
        
    def _setup_production_metrics(self):
        """Set up production-specific metrics"""
        # Reliability metrics
        self.health_checks = Counter(f"{self.prefix}_health_checks", 
                                   "Total number of health checks")
        self.health_check_failures = Counter(f"{self.prefix}_health_check_failures", 
                                          "Total number of failed health checks")
        self.service_uptime = Gauge(f"{self.prefix}_service_uptime_seconds", 
                                  "Service uptime in seconds")
        self.restart_count = Counter(f"{self.prefix}_restart_count", 
                                   "Number of service restarts")
        
        # Rate limiting metrics
        self.rate_limited_requests = Counter(f"{self.prefix}_rate_limited_requests", 
                                          "Number of rate-limited requests")
                                          
        # Concurrency metrics
        self.concurrent_requests = Gauge(f"{self.prefix}_concurrent_requests", 
                                       "Number of concurrent requests being processed")
        self.max_concurrent_requests = Gauge(f"{self.prefix}_max_concurrent_requests", 
                                          "Maximum number of concurrent requests observed")
                                          
        # Business metrics
        self.billable_tokens = Counter(f"{self.prefix}_billable_tokens", 
                                     "Total billable tokens (input + output)")
        self.request_cost = Counter(f"{self.prefix}_request_cost", 
                                  "Accumulated cost of requests in USD")
                                  
        # Service start time for uptime calculation
        self.start_time = time.time()
        
        # Start a thread to update uptime
        self._start_uptime_tracker()
        
    def _start_uptime_tracker(self):
        """Start a background thread to track service uptime"""
        def update_uptime():
            while True:
                try:
                    uptime = time.time() - self.start_time
                    self.service_uptime.set(uptime)
                    # Update every minute
                    time.sleep(60)
                except Exception as e:
                    logger.error(f"Error updating uptime: {str(e)}")
                    time.sleep(60)
                    
        thread = threading.Thread(
            target=update_uptime,
            daemon=True,
            name="uptime-tracker"
        )
        thread.start()
        
    def log_health_check(self, success: bool = True):
        """Log a health check
        
        Args:
            success: Whether the health check was successful
        """
        self.health_checks.inc()
        if not success:
            self.health_check_failures.inc()
            
    def log_rate_limited(self, count: int = 1):
        """Log rate-limited requests
        
        Args:
            count: Number of rate-limited requests
        """
        self.rate_limited_requests.inc(count)
        
    def update_concurrent_requests(self, count: int):
        """Update the number of concurrent requests
        
        Args:
            count: Current number of concurrent requests
        """
        self.concurrent_requests.set(count)
        
        # Update max concurrent if needed
        current_max = self.max_concurrent_requests.get()
        if count > current_max:
            self.max_concurrent_requests.set(count)
            
    def log_billable_request(self, input_tokens: int, output_tokens: int, 
                           input_cost_per_1k: float = 0.0005, 
                           output_cost_per_1k: float = 0.0015):
        """Log billable tokens and cost for a request
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            input_cost_per_1k: Cost per 1K input tokens
            output_cost_per_1k: Cost per 1K output tokens
        """
        total_tokens = input_tokens + output_tokens
        self.billable_tokens.inc(total_tokens)
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost
        
        self.request_cost.inc(total_cost)
        
    def log_service_restart(self):
        """Log a service restart"""
        self.restart_count.inc()
        # Reset start time for uptime calculation
        self.start_time = time.time()
        
    def get_production_metrics(self) -> Dict[str, Any]:
        """Get production-specific metrics
        
        Returns:
            Dictionary with production metrics
        """
        metrics = super().get_metrics_as_dict()
        
        # Add production metrics
        metrics.update({
            "reliability": {
                "uptime": self.service_uptime.get(),
                "health_checks": {
                    "total": self.health_checks.get(),
                    "failures": self.health_check_failures.get(),
                    "success_rate": 1.0 - (self.health_check_failures.get() / self.health_checks.get() 
                                        if self.health_checks.get() > 0 else 0)
                },
                "restarts": self.restart_count.get()
            },
            "concurrency": {
                "current": self.concurrent_requests.get(),
                "max_observed": self.max_concurrent_requests.get()
            },
            "rate_limiting": {
                "limited_requests": self.rate_limited_requests.get()
            },
            "billing": {
                "billable_tokens": self.billable_tokens.get(),
                "total_cost_usd": self.request_cost.get()
            }
        })
        
        return metrics
