from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from model import load_model, generate_text

app = FastAPI(title="NanoGPT API")

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8

class GenerationResponse(BaseModel):
    generated_text: str

model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    output = generate_text(
        model,
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    
    return GenerationResponse(generated_text=output)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
