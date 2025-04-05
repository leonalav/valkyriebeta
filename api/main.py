from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
from ..deployment.model_server import ModelServer
from ..config.model_config import ModelConfig
from ..security.rate_limiter import RateLimiter
from ..monitoring.metrics_collector import MetricsCollector

app = FastAPI()
config = ModelConfig()
model_server = ModelServer(config)
rate_limiter = RateLimiter()
metrics = MetricsCollector()

class InferenceRequest(BaseModel):
    text: str
    parameters: Optional[Dict[str, Any]] = None

@app.post("/predict")
async def predict(request: InferenceRequest, rate_limit: bool = Depends(rate_limiter.check)):
    try:
        result = await model_server.predict(request.text, request.parameters)
        metrics.log_inference(result["duration"])
        return result
    except Exception as e:
        metrics.log_inference(0, error=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_server.is_ready()}
