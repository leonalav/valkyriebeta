from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import logging
from ..model import LogicalReasoningTransformer
from ..config import EfficientTransformerConfig
from ..monitoring.metrics import MetricsCollector

app = FastAPI()
logger = logging.getLogger(__name__)
metrics = MetricsCollector()

class InferenceRequest(BaseModel):
    text: str
    max_length: int = 2048
    temperature: float = 0.7

class InferenceResponse(BaseModel):
    prediction: str
    confidence: float
    metrics: Dict[str, Any]

@app.on_event("startup")
async def load_model():
    global model, config, tokenizer
    
    config = EfficientTransformerConfig()
    config.validate()
    
    model = LogicalReasoningTransformer(config)
    model.eval()
    
    # Load model weights
    model.load_state_dict(torch.load("/app/checkpoints/best_model.pt"))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    try:
        # Tokenize
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            max_length=request.max_length,
            truncation=True,
            padding=True
        ).to(model.device)
        
        # Profile inference
        with metrics.profile_execution(model) as profile:
            with torch.no_grad():
                outputs = model(**inputs)
        
        # Get prediction
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        prediction = tokenizer.decode(torch.argmax(logits, dim=-1)[0])
        confidence = float(torch.max(probs))
        
        # Calculate metrics
        perf_metrics = metrics.collect_inference_metrics(model, inputs)
        
        return InferenceResponse(
            prediction=prediction,
            confidence=confidence,
            metrics={
                "inference_time": perf_metrics.inference_time,
                "memory_used": perf_metrics.memory_used,
                "device": perf_metrics.device
            }
        )
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
