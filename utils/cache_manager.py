import time
import threading
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
import logging

logger = logging.getLogger("cache_manager")

class ResponseCache:
    """Thread-safe LRU cache with TTL support for model inference responses"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """Initialize the cache
        
        Args:
            max_size: Maximum number of items to store in the cache
            ttl_seconds: Time-to-live for items in seconds (default: 5 minutes)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()  # {key: (value, timestamp)}
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
        
        # Start the cleanup thread if TTL is enabled
        if ttl_seconds > 0:
            self._start_cleanup_thread()
            
    def _start_cleanup_thread(self):
        """Start a background thread to clean up expired items"""
        def cleanup_task():
            while True:
                # Sleep for half the TTL time
                time.sleep(self.ttl_seconds / 2)
                try:
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Error in cache cleanup: {str(e)}")
                    
        # Create and start daemon thread
        cleanup_thread = threading.Thread(
            target=cleanup_task, 
            daemon=True,
            name="cache-cleanup"
        )
        cleanup_thread.start()
        
    def _cleanup_expired(self):
        """Remove expired items from the cache"""
        now = time.time()
        with self._lock:
            # Create a list of expired keys to avoid modifying during iteration
            expired_keys = [
                k for k, (_, timestamp) in self._cache.items() 
                if now - timestamp > self.ttl_seconds
            ]
            
            # Remove expired keys
            for key in expired_keys:
                del self._cache[key]
                
            if expired_keys:
                logger.debug(f"Removed {len(expired_keys)} expired items from cache")
                
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache
        
        Args:
            key: The cache key
            
        Returns:
            The cached value if found and not expired, None otherwise
        """
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                
                # Check if the item has expired
                if time.time() - timestamp > self.ttl_seconds:
                    # Remove expired item
                    del self._cache[key]
                    self._miss_count += 1
                    return None
                    
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hit_count += 1
                return value
            else:
                self._miss_count += 1
                return None
                
    def set(self, key: str, value: Any):
        """Set a value in the cache
        
        Args:
            key: The cache key
            value: The value to cache
        """
        with self._lock:
            # If key exists, update it and move to end
            if key in self._cache:
                self._cache[key] = (value, time.time())
                self._cache.move_to_end(key)
                return
                
            # If cache is full, remove oldest item
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove first item (oldest)
                
            # Add new item
            self._cache[key] = (value, time.time())
            
    def clear(self):
        """Clear all items from the cache"""
        with self._lock:
            self._cache.clear()
            
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl_seconds
            }

class ModelCache:
    """Cache for loaded models to support multiple model versions"""
    
    def __init__(self, max_models: int = 2):
        """Initialize the model cache
        
        Args:
            max_models: Maximum number of models to keep in memory
        """
        self.max_models = max_models
        self._models = OrderedDict()  # {model_id: (model, last_used_timestamp)}
        self._lock = threading.RLock()
        
    def get(self, model_id: str) -> Optional[Any]:
        """Get a model from the cache
        
        Args:
            model_id: The model identifier (e.g., "model_name:version")
            
        Returns:
            The cached model if found, None otherwise
        """
        with self._lock:
            if model_id in self._models:
                model, _ = self._models[model_id]
                # Update last used timestamp and move to end
                self._models[model_id] = (model, time.time())
                self._models.move_to_end(model_id)
                return model
            return None
            
    def set(self, model_id: str, model: Any):
        """Set a model in the cache
        
        Args:
            model_id: The model identifier
            model: The model object to cache
        """
        with self._lock:
            # If we're at capacity and this is a new model, remove oldest
            if len(self._models) >= self.max_models and model_id not in self._models:
                # Get the oldest model
                oldest_id, _ = next(iter(self._models.items()))
                # Remove it
                self._remove(oldest_id)
                
            # Add or update the model
            self._models[model_id] = (model, time.time())
            self._models.move_to_end(model_id)
            
    def _remove(self, model_id: str):
        """Remove a model from the cache and release resources
        
        Args:
            model_id: The model identifier to remove
        """
        if model_id in self._models:
            model, _ = self._models.pop(model_id)
            
            # Try to clean up GPU memory if model has a cleanup method
            if hasattr(model, "unload"):
                try:
                    model.unload()
                except Exception as e:
                    logger.error(f"Error unloading model {model_id}: {str(e)}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Try to clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, AttributeError):
                pass
                
    def clear(self):
        """Clear all models from the cache"""
        with self._lock:
            for model_id in list(self._models.keys()):
                self._remove(model_id)
                
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                "count": len(self._models),
                "max_models": self.max_models,
                "models": list(self._models.keys())
            } 