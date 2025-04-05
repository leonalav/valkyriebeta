import torch
from pathlib import Path
from typing import Optional
from model import GPT, GPTConfig

def load_model(checkpoint_path: Optional[str] = None) -> GPT:
    """Load the model from a checkpoint."""
    config = GPTConfig()
    model = GPT(config)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    model.eval()
    return model

def generate_text(
    model: GPT,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8
) -> str:
    """Generate text from a prompt."""
    with torch.no_grad():
        encoded = model.encode(prompt)
        output = model.generate(
            encoded,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True
        )
        return model.decode(output[0].tolist())
