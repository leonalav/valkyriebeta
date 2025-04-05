"""
Comprehensive metrics system for monitoring model performance in production environments.
Supports both Prometheus and custom metrics storage with visualization capabilities.
"""

import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Union, Callable
from collections import deque
import statistics
from datetime import datetime, timedelta

try:
    import prometheus_client as prom
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger("monitoring.metrics")

class MetricRegistry:
    """Central registry for metrics to avoid duplication and enable global access"""
    
    _instance = None
    _lock = threading.RLock()
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of the registry"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        """Initialize the registry - should only be called once by get_instance()"""
        self.metrics = {}
        self.prometheus_metrics = {}
        
    def register(self, name: str, metric: Any, metric_type: str, description: str = "") -> None:
        """Register a metric in the registry
        
        Args:
            name: Unique name for the metric
            metric: The metric object
            metric_type: Type of metric (counter, gauge, histogram, etc.)
            description: Description of the metric
        """
        with self._lock:
            self.metrics[name] = {
                "metric": metric,
                "type": metric_type,
                "description": description,
                "created_at": datetime.now()
            }
            
    def register_prometheus(self, name: str, metric: Any) -> None:
        """Register a Prometheus metric
        
        Args:
            name: Unique name for the metric
            metric: Prometheus metric object
        """
        with self._lock:
            self.prometheus_metrics[name] = metric
            
    def get(self, name: str) -> Any:
        """Get a metric by name
        
        Args:
            name: Name of the metric
            
        Returns:
            Metric object or None if not found
        """
        return self.metrics.get(name, {}).get("metric")
        
    def get_prometheus(self, name: str) -> Any:
        """Get a Prometheus metric by name
        
        Args:
            name: Name of the metric
            
        Returns:
            Prometheus metric object or None if not found
        """
        return self.prometheus_metrics.get(name)
        
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered metrics
        
        Returns:
            Dictionary of all metrics
        """
        return self.metrics

class Metric:
    """Base class for all metrics"""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize a metric
        
        Args:
            name: Unique name for the metric
            description: Description of the metric
        """
        self.name = name
        self.description = description
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary representation
        
        Returns:
            Dictionary representation of the metric
        """
        return {
            "name": self.name,
            "description": self.description
        }

class Counter(Metric):
    """Counter metric for tracking counts that only increase
    
    Used for counting events like requests, errors, etc.
    """
    
    def __init__(self, name: str, description: str = ""):
        """Initialize a counter
        
        Args:
            name: Unique name for the counter
            description: Description of the counter
        """
        super().__init__(name, description)
        self.value = 0
        self._prometheus = None
        self._lock = threading.RLock()
        
        # Register with registry
        registry = MetricRegistry.get_instance()
        registry.register(name, self, "counter", description)
        
        # Create Prometheus metric if available
        if PROMETHEUS_AVAILABLE:
            self._prometheus = prom.Counter(name, description)
            registry.register_prometheus(name, self._prometheus)
        
    def inc(self, value: float = 1.0) -> None:
        """Increment the counter
        
        Args:
            value: Value to increment by (default 1.0)
        """
        with self._lock:
            self.value += value
            
        # Update Prometheus if available
        if self._prometheus:
            self._prometheus.inc(value)
            
    def get(self) -> float:
        """Get the current value
        
        Returns:
            Current counter value
        """
        with self._lock:
            return self.value
            
    def reset(self) -> None:
        """Reset the counter to zero (not applicable for Prometheus)"""
        with self._lock:
            # Cannot reset Prometheus counters, only our internal value
            self.value = 0
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert counter to dictionary
        
        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        with self._lock:
            result["value"] = self.value
            result["type"] = "counter"
        return result

class Gauge(Metric):
    """Gauge metric for tracking values that can go up and down
    
    Used for measuring current values like memory usage, queue size, etc.
    """
    
    def __init__(self, name: str, description: str = ""):
        """Initialize a gauge
        
        Args:
            name: Unique name for the gauge
            description: Description of the gauge
        """
        super().__init__(name, description)
        self.value = 0.0
        self._prometheus = None
        self._lock = threading.RLock()
        
        # Register with registry
        registry = MetricRegistry.get_instance()
        registry.register(name, self, "gauge", description)
        
        # Create Prometheus metric if available
        if PROMETHEUS_AVAILABLE:
            self._prometheus = prom.Gauge(name, description)
            registry.register_prometheus(name, self._prometheus)
        
    def set(self, value: float) -> None:
        """Set the gauge value
        
        Args:
            value: New value for the gauge
        """
        with self._lock:
            self.value = value
            
        # Update Prometheus if available
        if self._prometheus:
            self._prometheus.set(value)
            
    def inc(self, value: float = 1.0) -> None:
        """Increment the gauge
        
        Args:
            value: Value to increment by (default 1.0)
        """
        with self._lock:
            self.value += value
            
        # Update Prometheus if available
        if self._prometheus:
            self._prometheus.inc(value)
            
    def dec(self, value: float = 1.0) -> None:
        """Decrement the gauge
        
        Args:
            value: Value to decrement by (default 1.0)
        """
        with self._lock:
            self.value -= value
            
        # Update Prometheus if available
        if self._prometheus:
            self._prometheus.dec(value)
            
    def get(self) -> float:
        """Get the current value
        
        Returns:
            Current gauge value
        """
        with self._lock:
            return self.value
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert gauge to dictionary
        
        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        with self._lock:
            result["value"] = self.value
            result["type"] = "gauge"
        return result

class Histogram(Metric):
    """Histogram metric for tracking distributions of values
    
    Used for measuring distributions like response times, token counts, etc.
    """
    
    def __init__(
        self,
        name: str, 
        description: str = "", 
        buckets: Optional[List[float]] = None
    ):
        """Initialize a histogram
        
        Args:
            name: Unique name for the histogram
            description: Description of the histogram
            buckets: Optional list of bucket boundaries
        """
        super().__init__(name, description)
        self.values = []
        self.count = 0
        self.sum = 0.0
        self._prometheus = None
        self._lock = threading.RLock()
        
        # Keep a window of recent values for percentile calculations
        self.recent_values = deque(maxlen=1000)
        
        # Set default buckets if not provided
        self.buckets = buckets or [
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60
        ]
        
        # Initialize bucket counts
        self.bucket_counts = {bucket: 0 for bucket in self.buckets}
        
        # Register with registry
        registry = MetricRegistry.get_instance()
        registry.register(name, self, "histogram", description)
        
        # Create Prometheus metric if available
        if PROMETHEUS_AVAILABLE:
            self._prometheus = prom.Histogram(name, description, buckets=self.buckets)
            registry.register_prometheus(name, self._prometheus)
        
    def observe(self, value: float) -> None:
        """Record an observation
        
        Args:
            value: Value to record
        """
        with self._lock:
            self.values.append(value)
            self.recent_values.append(value)
            self.count += 1
            self.sum += value
            
            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    self.bucket_counts[bucket] += 1
            
        # Update Prometheus if available
        if self._prometheus:
            self._prometheus.observe(value)
            
    def get_count(self) -> int:
        """Get the observation count
        
        Returns:
            Number of observations
        """
        with self._lock:
            return self.count
            
    def get_sum(self) -> float:
        """Get the sum of observed values
        
        Returns:
            Sum of observations
        """
        with self._lock:
            return self.sum
            
    def get_avg(self) -> float:
        """Get the average of observed values
        
        Returns:
            Average of observations or 0 if no observations
        """
        with self._lock:
            return self.sum / self.count if self.count > 0 else 0
            
    def get_percentile(self, percentile: float) -> Optional[float]:
        """Get a percentile value (from recent observations)
        
        Args:
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Percentile value or None if no observations
        """
        with self._lock:
            values = list(self.recent_values)
            if not values:
                return None
                
            # Sort values and calculate percentile
            sorted_values = sorted(values)
            index = int(len(sorted_values) * (percentile / 100))
            return sorted_values[index]
            
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            values = list(self.recent_values)
            stats = {
                "count": self.count,
                "sum": self.sum,
                "avg": self.get_avg(),
            }
            
            # Calculate percentiles if we have values
            if values:
                # Calculate percentiles
                sorted_values = sorted(values)
                stats.update({
                    "min": sorted_values[0],
                    "max": sorted_values[-1],
                    "p50": sorted_values[int(len(sorted_values) * 0.5)] if len(sorted_values) > 0 else 0,
                    "p90": sorted_values[int(len(sorted_values) * 0.9)] if len(sorted_values) > 0 else 0,
                    "p95": sorted_values[int(len(sorted_values) * 0.95)] if len(sorted_values) > 0 else 0,
                    "p99": sorted_values[int(len(sorted_values) * 0.99)] if len(sorted_values) > 0 else 0,
                })
                
                # Add standard deviation if we have multiple values
                if len(sorted_values) > 1:
                    stats["std_dev"] = statistics.stdev(sorted_values)
                    
            # Add bucket counts
            stats["buckets"] = self.bucket_counts
                
            return stats
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert histogram to dictionary
        
        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result.update({
            "type": "histogram",
            "stats": self.get_stats()
        })
        return result

class Summary(Metric):
    """Summary metric for tracking value distribution with time decay
    
    Similar to histogram but optimized for calculating sliding window statistics
    """
    
    def __init__(
        self, 
        name: str, 
        description: str = "", 
        max_age_seconds: int = 600,
        percentiles: Optional[List[float]] = None
    ):
        """Initialize a summary
        
        Args:
            name: Unique name for the summary
            description: Description of the summary
            max_age_seconds: Maximum age of observations to keep
            percentiles: Percentiles to track (0-1)
        """
        super().__init__(name, description)
        self.max_age_seconds = max_age_seconds
        self.percentiles = percentiles or [0.5, 0.9, 0.95, 0.99]
        self._prometheus = None
        self._lock = threading.RLock()
        
        # Store values with timestamps for time decay
        self.values = []  # List of (value, timestamp) tuples
        
        # Register with registry
        registry = MetricRegistry.get_instance()
        registry.register(name, self, "summary", description)
        
        # Create Prometheus metric if available
        if PROMETHEUS_AVAILABLE:
            self._prometheus = prom.Summary(name, description, percentiles=self.percentiles)
            registry.register_prometheus(name, self._prometheus)
        
    def observe(self, value: float) -> None:
        """Record an observation
        
        Args:
            value: Value to record
        """
        now = time.time()
        
        with self._lock:
            # Clean old values
            self._clean_old_values(now)
            
            # Add new value
            self.values.append((value, now))
            
        # Update Prometheus if available
        if self._prometheus:
            self._prometheus.observe(value)
            
    def _clean_old_values(self, now: Optional[float] = None) -> None:
        """Remove values older than max_age_seconds
        
        Args:
            now: Current timestamp or None to use current time
        """
        if now is None:
            now = time.time()
            
        cutoff = now - self.max_age_seconds
        self.values = [(v, t) for v, t in self.values if t >= cutoff]
        
    def get_count(self) -> int:
        """Get the observation count
        
        Returns:
            Number of observations
        """
        with self._lock:
            self._clean_old_values()
            return len(self.values)
            
    def get_sum(self) -> float:
        """Get the sum of observed values
        
        Returns:
            Sum of observations
        """
        with self._lock:
            self._clean_old_values()
            return sum(v for v, _ in self.values)
            
    def get_avg(self) -> float:
        """Get the average of observed values
        
        Returns:
            Average of observations or 0 if no observations
        """
        with self._lock:
            self._clean_old_values()
            if not self.values:
                return 0.0
            return sum(v for v, _ in self.values) / len(self.values)
            
    def get_percentile(self, percentile: float) -> Optional[float]:
        """Get a percentile value
        
        Args:
            percentile: Percentile to calculate (0-1)
            
        Returns:
            Percentile value or None if no observations
        """
        with self._lock:
            self._clean_old_values()
            if not self.values:
                return None
                
            # Extract just the values (not timestamps)
            values = [v for v, _ in self.values]
            
            # Sort values and calculate percentile
            sorted_values = sorted(values)
            index = int(len(sorted_values) * percentile)
            return sorted_values[index]
            
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            self._clean_old_values()
            
            if not self.values:
        return {
                    "count": 0,
                    "sum": 0,
                    "avg": 0
                }
                
            values = [v for v, _ in self.values]
            sorted_values = sorted(values)
            
            stats = {
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": sorted_values[0],
                "max": sorted_values[-1]
            }
            
            # Add percentiles
            for p in self.percentiles:
                index = min(int(len(sorted_values) * p), len(sorted_values) - 1)
                stats[f"p{int(p * 100)}"] = sorted_values[index]
                
            return stats
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary
        
        Returns:
            Dictionary representation
        """
        result = super().to_dict()
        result.update({
            "type": "summary",
            "stats": self.get_stats(),
            "max_age_seconds": self.max_age_seconds
        })
        return result

class MetricsExporter:
    """Exports metrics to various formats and destinations"""
    
    def __init__(self, registry: Optional[MetricRegistry] = None):
        """Initialize exporter
        
        Args:
            registry: Metrics registry to export from
        """
        self.registry = registry or MetricRegistry.get_instance()
        
    def export_json(self) -> str:
        """Export metrics as JSON
        
        Returns:
            JSON string with all metrics
        """
        metrics = {}
        for name, data in self.registry.get_all().items():
            metric = data["metric"]
            if hasattr(metric, "to_dict"):
                metrics[name] = metric.to_dict()
                
        return json.dumps(metrics, indent=2)
        
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format
        
        Returns:
            Prometheus text format metrics
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("Prometheus client not available")
            
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return generate_latest().decode('utf-8')
        
    def export_to_file(self, filename: str, format: str = "json") -> None:
        """Export metrics to a file
        
        Args:
            filename: Path to output file
            format: Export format ('json' or 'prometheus')
        """
        if format == "json":
            content = self.export_json()
        elif format == "prometheus":
            content = self.export_prometheus()
            else:
            raise ValueError(f"Unsupported format: {format}")
            
        with open(filename, 'w') as f:
            f.write(content)
            
    def push_to_pushgateway(self, gateway: str, job: str, registry=None):
        """Push metrics to Prometheus Pushgateway
        
        Args:
            gateway: Pushgateway URL
            job: Job name
            registry: Optional custom registry
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("Prometheus client not available")
            
        from prometheus_client import push_to_gateway
        push_to_gateway(gateway, job=job, registry=registry)