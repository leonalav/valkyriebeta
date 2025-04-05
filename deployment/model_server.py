import torch
import logging
import time
import os
from typing import Dict, Any, Optional, List, Union
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import json
from contextlib import asynccontextmanager
import asyncio
from ..model.inference import ModelInference
from ..monitoring.metrics_collector import MetricsCollector, ProductionMetricsCollector
from ..config.model_config import ModelConfig
from ..security.rate_limiter import RateLimiter
from ..security.validator import InputValidator
from ..utils.cache_manager import ResponseCache

# Configure structured JSON logging
import json_log_formatter
import socket
import platform

class CustomJSONFormatter(json_log_formatter.JSONFormatter):
    def json_record(self, message, extra, record):
        extra['message'] = message
        extra['level'] = record.levelname
        extra['logger'] = record.name
        extra['timestamp'] = record.created
        extra['host'] = socket.gethostname()
        extra['system'] = platform.system()
        extra['process'] = record.process
        extra['thread'] = record.thread
        extra['module'] = record.module
        extra['funcName'] = record.funcName
        extra['lineNo'] = record.lineno
        
        if record.exc_info:
            extra['exception'] = self.formatException(record.exc_info)
            
        return extra

# Setup logging with JSON format and multiple handlers
formatter = CustomJSONFormatter()

# File handler for JSON logs
file_handler = logging.FileHandler("server.json.log")
file_handler.setFormatter(formatter)

# Console handler for development
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

# Configure root logger
logger = logging.getLogger("model_server")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Optional: Add centralized logging handler (e.g., Loki, ELK)
# Uncomment and configure as needed
# try:
#     from pythonjsonlogger.jsonlogger import JsonFormatter
#     from logging.handlers import HTTPHandler
#     http_handler = HTTPHandler(
#         host='your-logging-service.com',
#         url='/api/logs',
#         method='POST'
#     )
#     http_handler.setFormatter(JsonFormatter())
#     logger.addHandler(http_handler)
# except ImportError:
#     logger.warning("Centralized logging dependencies not installed")

# Request and response models
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=32768)
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stop_sequences: Optional[List[str]] = Field(default=None)
    stream: bool = Field(default=False)
    
class PredictionResponse(BaseModel):
    prediction: str
    tokens_generated: int
    processing_time_ms: float
    model_version: str
    
class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    uptime_seconds: float
    current_requests: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model into memory
    app.state.startup_time = time.time()
    app.state.current_requests = 0
    logger.info("Starting model server...")
    
    # Give control back to FastAPI
    yield
    
    # Shutdown: Clean up resources
    logger.info("Shutting down model server...")
    # Clean any resources that need explicit cleanup
    
class ModelServer:
    def __init__(self, config: ModelConfig, model_path: str = None):
        self.config = config
        self.model_path = model_path or os.environ.get("MODEL_PATH")
        if not self.model_path:
            raise ValueError("Model path must be provided either in constructor or as MODEL_PATH env variable")
            
        self.model = None  # Will be loaded on demand or in lifespan
        self.metrics = ProductionMetricsCollector()
        self.rate_limiter = RateLimiter(
            requests_per_minute=int(os.environ.get("RATE_LIMIT_RPM", "60")),
            burst_size=int(os.environ.get("RATE_LIMIT_BURST", "10"))
        )
        self.validator = InputValidator()
        self.cache = ResponseCache(ttl_seconds=300)  # 5-minute cache by default
        
        # Create FastAPI app with lifespan
        self.app = FastAPI(
            title="LLM Model Server",
            description="Production API for Large Language Model inference",
            version="1.0.0",
            lifespan=lifespan
        )
        
        # Add middleware
        self._setup_middleware()
        
        # Add routes
        self._setup_routes()
        
    def _setup_middleware(self):
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=json.loads(os.environ.get("CORS_ORIGINS", '["*"]')),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add custom middleware for request tracking
        @self.app.middleware("http")
        async def track_requests(request: Request, call_next):
            # Increment request counter
            request.app.state.current_requests += 1
            
            # Process request timing
            start_time = time.time()
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(process_time)
                
                # Log request timing if not a health check
                if not request.url.path.endswith("/health"):
                    logger.info(f"Request processed in {process_time:.4f} seconds")
                    self.metrics.inference_latency.observe(process_time)
                
                return response
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Internal server error"}
                )
            finally:
                # Decrement request counter
                request.app.state.current_requests -= 1
                
    def _lazy_load_model(self):
        """Lazy load the model if it's not already loaded"""
        if self.model is None:
            logger.info("Loading model for the first time...")
            start_time = time.time()
            self.model = ModelInference(self.config, model_path=self.model_path)
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            
        return self.model
        
    def _setup_routes(self):
        @self.app.post("/v1/predict", response_model=PredictionResponse)
        async def predict(
            request: PredictionRequest,
            background_tasks: BackgroundTasks,
            client_ip: str = Depends(self._get_client_ip)
        ):
            # Apply rate limiting
            if not self.rate_limiter.allow_request(client_ip):
                self.metrics.inference_requests.inc()
                self.metrics.error_rate.inc()
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
                
            # Validate input
            validation_result = self.validator.validate_input(request.text)
            if not validation_result.is_valid:
                self.metrics.inference_requests.inc()
                self.metrics.error_rate.inc()
                logger.warning(f"Input validation failed: {validation_result.message}")
                raise HTTPException(status_code=400, detail=validation_result.message)
                
            # Check cache for identical requests with same parameters
            cache_key = f"{request.text}:{request.max_tokens}:{request.temperature}:{request.top_p}"
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.info("Returning cached response")
                self.metrics.inference_requests.inc()
                return cached_response
                
            # Circuit breaker state
            circuit_open = False
            last_failure_time = 0
            failure_count = 0
            max_retries = 3
            base_delay = 0.1  # seconds
            
            for attempt in range(max_retries + 1):  # +1 for initial attempt
                try:
                    # Check circuit breaker
                    current_time = time.time()
                    if circuit_open and (current_time - last_failure_time) < 30:  # 30s cooldown
                        raise HTTPException(
                            status_code=503,
                            detail="Service temporarily unavailable due to high error rate"
                        )
                    
                    # Ensure model is loaded
                    model = self._lazy_load_model()
                    
                    # Start timing
                    start_time = time.time()
                    
                    # Generate prediction
                    generation_config = {
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "stop_sequences": request.stop_sequences,
                        "stream": request.stream
                    }
                    
                    result = model.generate(request.text, **generation_config)
                    
                    # Calculate processing time
                    processing_time = (time.time() - start_time) * 1000  # ms
                    
                    # Create response
                    response = PredictionResponse(
                        prediction=result["text"],
                        tokens_generated=result.get("tokens_generated", 0),
                        processing_time_ms=processing_time,
                        model_version=self.config.model_version
                    )
                    
                    # Cache the response
                    self.cache.set(cache_key, response)
                    
                    # Update metrics in background
                    background_tasks.add_task(
                        self._update_metrics,
                        processing_time_seconds=processing_time/1000,
                        tokens_generated=result.get("tokens_generated", 0)
                    )
                    
                    # Reset circuit breaker on success
                    if circuit_open:
                        circuit_open = False
                        failure_count = 0
                        logger.info("Circuit breaker reset after successful request")
                    
                    return response
                    
                except HTTPException:
                    # Re-raise HTTP exceptions (rate limiting, validation errors)
                    raise
                    
                except Exception as e:
                    failure_count += 1
                    last_failure_time = time.time()
                    
                    # Trip circuit if too many failures
                    if failure_count >= 5:
                        circuit_open = True
                        logger.error("Circuit breaker tripped due to consecutive failures")
                    
                    if attempt < max_retries:
                        # Exponential backoff
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {str(e)}")
                        await asyncio.sleep(delay)
                        continue
                    
                    # Fallback response if all retries fail
                    logger.error(f"All retries failed for prediction request: {str(e)}", exc_info=True)
                    self.metrics.inference_requests.inc()
                    self.metrics.error_rate.inc()
                    
                    # Return fallback response if configured
                    if hasattr(self.config, "fallback_response"):
                        return PredictionResponse(
                            prediction=self.config.fallback_response,
                            tokens_generated=0,
                            processing_time_ms=0,
                            model_version=self.config.model_version
                        )
                    
                    raise HTTPException(
                        status_code=500,
                        detail="Model inference failed after multiple retries"
                    )
                
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check(request: Request):
            """Health check endpoint with detailed status information"""
            gpu_available = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
            uptime = time.time() - request.app.state.startup_time
            
            return HealthResponse(
                status="healthy",
                version=self.config.model_version,
                gpu_available=gpu_available,
                uptime_seconds=uptime,
                current_requests=request.app.state.current_requests
            )
            
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request headers or direct connection"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Return first IP in list for X-Forwarded-For
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
        
    def _update_metrics(self, processing_time_seconds: float, tokens_generated: int):
        """Update metrics after successful inference"""
        self.metrics.inference_requests.inc()
        self.metrics.inference_latency.observe(processing_time_seconds)
        
        # Update throughput metrics based on tokens generated
        self.metrics.throughput.inc(tokens_generated)
        
        # Update resource metrics
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            self.metrics.gpu_memory_used.set(memory_used)
            
    def start(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """Start the FastAPI server with uvicorn"""
        # Explicitly preload model if configured
        if self.config.preload_model and self.model is None:
            self._lazy_load_model()
            
        # Start server
        logger.info(f"Starting model server on {host}:{port} with {workers} workers")
        uvicorn.run(
            self.app,
            host=host, 
            port=port,
            workers=workers,
            log_level="info"
        )
