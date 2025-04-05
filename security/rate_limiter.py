import time
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple
import redis
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("security.rate_limiter")

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    TOKEN_BUCKET = "token_bucket"
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    burst_size: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    redis_url: Optional[str] = None
    sliding_window_size: int = 60  # seconds (for sliding window strategy)
    
class RateLimiter:
    """Rate limiter using token bucket algorithm with redis support for distributed scenarios
    
    Implements multiple rate limiting algorithms with support for both in-memory and Redis backends.
    """
    
    def __init__(
        self, 
        requests_per_minute: int = 60, 
        burst_size: int = 10,
        strategy: str = "token_bucket",
        redis_url: Optional[str] = None
    ):
        """Initialize the rate limiter
        
        Args:
            requests_per_minute: Maximum requests allowed per minute
            burst_size: Maximum burst size (for token bucket)
            strategy: Rate limiting strategy to use
            redis_url: Redis URL for distributed rate limiting
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.strategy = RateLimitStrategy(strategy) if isinstance(strategy, str) else strategy
        self.redis_url = redis_url
        self.redis_client = None
        
        # For token bucket, calculate token refill rate
        self.refill_rate = self.requests_per_minute / 60.0  # tokens per second
        self.refill_interval = 1.0 / self.refill_rate if self.refill_rate > 0 else 0
        
        # In-memory storage for clients
        self._in_memory_storage: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Initialize Redis if URL provided
        if self.redis_url:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                logger.info(f"Connected to Redis for distributed rate limiting: {self.redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis ({self.redis_url}): {str(e)}")
                logger.warning("Falling back to in-memory rate limiting")
                self.redis_url = None
                
        logger.info(f"Rate limiter initialized: {self.requests_per_minute} RPM, "
                   f"burst: {self.burst_size}, strategy: {self.strategy.value}")
                
    def allow_request(self, client_id: str) -> bool:
        """Check if a request from a client is allowed based on rate limits
        
        Args:
            client_id: Identifier for the client (e.g., IP address)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        # Choose strategy based on configuration
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._token_bucket(client_id)
        elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._fixed_window(client_id)
        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._sliding_window(client_id)
        else:
            # Default to token bucket if unknown strategy
            return self._token_bucket(client_id)
            
    def _token_bucket(self, client_id: str) -> bool:
        """Token bucket algorithm for rate limiting
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        
        if self.redis_client:
            # Redis-based token bucket implementation
            return self._redis_token_bucket(client_id, now)
        else:
            # In-memory token bucket implementation
            with self._lock:
                # Initialize client state if needed
                if client_id not in self._in_memory_storage:
                    self._in_memory_storage[client_id] = {
                        "tokens": self.burst_size,
                        "last_update": now
                    }
                    return True
                    
                # Get client state
                client_state = self._in_memory_storage[client_id]
                tokens = client_state["tokens"]
                last_update = client_state["last_update"]
                
                # Calculate token refill since last update
                time_passed = now - last_update
                new_tokens = time_passed * self.refill_rate
                
                # Update tokens (capped at burst size)
                tokens = min(tokens + new_tokens, self.burst_size)
                
                # Check if we have at least one token for this request
                if tokens < 1:
                    logger.warning(f"Rate limit exceeded for client: {client_id}")
                    # Update last_update time but don't consume a token
                    client_state["last_update"] = now
                    return False
                    
                # Consume one token and update state
                tokens -= 1
                client_state["tokens"] = tokens
                client_state["last_update"] = now
                
                return True
                
    def _redis_token_bucket(self, client_id: str, now: float) -> bool:
        """Redis-based token bucket implementation
        
        Args:
            client_id: Client identifier
            now: Current timestamp
            
        Returns:
            True if request is allowed, False otherwise
        """
        # Redis key for this client
        key = f"rate_limit:token_bucket:{client_id}"
        
        # Lua script for atomic token bucket operation
        script = """
        local key = KEYS[1]
        local burst_size = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        -- Get or initialize bucket data
        local data = redis.call('HMGET', key, 'tokens', 'last_update')
        local tokens = tonumber(data[1])
        local last_update = tonumber(data[2])
        
        if not tokens or not last_update then
            -- Initialize new bucket
            tokens = burst_size
            last_update = now
        else
            -- Calculate token refill
            local time_passed = now - last_update
            local new_tokens = time_passed * refill_rate
            tokens = math.min(tokens + new_tokens, burst_size)
        end
        
        -- Check if request is allowed
        local allowed = 0
        if tokens >= 1 then
            tokens = tokens - 1
            allowed = 1
        end
        
        -- Store updated state
        redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
        redis.call('EXPIRE', key, 3600)  -- Auto-expire after 1 hour
        
        return allowed
        """
        
        try:
            result = self.redis_client.eval(
                script,
                1,  # number of keys
                key,  # key
                self.burst_size,  # burst size
                self.refill_rate,  # refill rate
                now  # current time
            )
            return bool(result)
        except Exception as e:
            logger.error(f"Redis token bucket error: {str(e)}")
            # Fall back to in-memory if Redis fails
            return self._token_bucket(client_id)
            
    def _fixed_window(self, client_id: str) -> bool:
        """Fixed window rate limiting algorithm
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        window_start = int(now - (now % 60))  # Get start of current minute
        
        if self.redis_client:
            # Redis-based fixed window
            key = f"rate_limit:fixed_window:{client_id}:{window_start}"
            try:
                # Increment counter and set expiry
                count = self.redis_client.incr(key)
                self.redis_client.expire(key, 120)  # 2 minutes expiry
                
                # Check if under limit
                return count <= self.requests_per_minute
            except Exception as e:
                logger.error(f"Redis fixed window error: {str(e)}")
                # Fall back to in-memory
                return self._in_memory_fixed_window(client_id, window_start)
        else:
            # In-memory fixed window
            return self._in_memory_fixed_window(client_id, window_start)
            
    def _in_memory_fixed_window(self, client_id: str, window_start: int) -> bool:
        """In-memory fixed window implementation
        
        Args:
            client_id: Client identifier
            window_start: Start timestamp of current window
            
        Returns:
            True if request is allowed, False otherwise
        """
        with self._lock:
            # Create composite key for client and window
            key = f"{client_id}:{window_start}"
            
            # Initialize client state if needed
            if key not in self._in_memory_storage:
                self._in_memory_storage[key] = {"count": 0}
                
                # Cleanup old windows (avoid memory leak)
                self._cleanup_old_windows()
                
            # Increment counter
            self._in_memory_storage[key]["count"] += 1
            
            # Check if under limit
            return self._in_memory_storage[key]["count"] <= self.requests_per_minute
            
    def _sliding_window(self, client_id: str) -> bool:
        """Sliding window rate limiting algorithm
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if request is allowed, False otherwise
        """
        now = time.time()
        window_size = 60  # 1 minute window
        
        if self.redis_client:
            # Redis-based sliding window using sorted sets
            key = f"rate_limit:sliding_window:{client_id}"
            try:
                # Remove old entries outside the window
                self.redis_client.zremrangebyscore(key, 0, now - window_size)
                
                # Count requests in current window
                count = self.redis_client.zcard(key)
                
                # Check if under limit
                if count < self.requests_per_minute:
                    # Add current request to window
                    self.redis_client.zadd(key, {str(now): now})
                    self.redis_client.expire(key, window_size * 2)  # 2x window size expiry
                    return True
                else:
                    return False
            except Exception as e:
                logger.error(f"Redis sliding window error: {str(e)}")
                # Fall back to in-memory
                return self._in_memory_sliding_window(client_id, now, window_size)
        else:
            # In-memory sliding window
            return self._in_memory_sliding_window(client_id, now, window_size)
            
    def _in_memory_sliding_window(self, client_id: str, now: float, window_size: int) -> bool:
        """In-memory sliding window implementation
        
        Args:
            client_id: Client identifier
            now: Current timestamp
            window_size: Window size in seconds
            
        Returns:
            True if request is allowed, False otherwise
        """
        with self._lock:
            # Initialize client state if needed
            if client_id not in self._in_memory_storage:
                self._in_memory_storage[client_id] = {"requests": []}
                
            # Get client request history
            requests = self._in_memory_storage[client_id]["requests"]
            
            # Remove requests outside the window
            cutoff = now - window_size
            requests = [t for t in requests if t > cutoff]
            
            # Check if under limit
            if len(requests) < self.requests_per_minute:
                # Add current request
                requests.append(now)
                self._in_memory_storage[client_id]["requests"] = requests
                return True
            else:
                return False
                
    def _cleanup_old_windows(self):
        """Clean up old window data to prevent memory leaks"""
        now = time.time()
        current_window = int(now - (now % 60))
        keys_to_remove = []
        
        # Find keys older than 2 minutes
        for key in self._in_memory_storage:
            if ":" in key:  # Only process fixed window keys
                parts = key.split(":")
                if len(parts) >= 2:
                    try:
                        window_start = int(parts[-1])
                        if window_start < current_window - 120:  # 2 minutes old
                            keys_to_remove.append(key)
                    except (ValueError, IndexError):
                        pass
                        
        # Remove old keys
        for key in keys_to_remove:
            del self._in_memory_storage[key]
            
    def reset(self, client_id: Optional[str] = None):
        """Reset rate limits for a client or all clients
        
        Args:
            client_id: Specific client to reset, or None to reset all
        """
        if client_id:
            # Reset specific client
            if self.redis_client:
                try:
                    pattern = f"rate_limit:*:{client_id}*"
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                except Exception as e:
                    logger.error(f"Redis reset error: {str(e)}")
                    
            # In-memory reset
            with self._lock:
                keys_to_remove = []
                for key in self._in_memory_storage:
                    if client_id in key:
                        keys_to_remove.append(key)
                        
                for key in keys_to_remove:
                    del self._in_memory_storage[key]
        else:
            # Reset all clients
            if self.redis_client:
                try:
                    keys = self.redis_client.keys("rate_limit:*")
                    if keys:
                        self.redis_client.delete(*keys)
                except Exception as e:
                    logger.error(f"Redis reset all error: {str(e)}")
                    
            # In-memory reset all
            with self._lock:
                self._in_memory_storage.clear()
