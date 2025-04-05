import torch
import torch.nn as nn
from typing import List, Optional, Tuple
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class SpeculativeConfig:
    num_draft_models: int = 3
    acceptance_threshold: float = 0.7
    max_candidates: int = 5
    temperature: float = 0.7
    use_adaptive_threshold: bool = True
    min_acceptance: float = 0.5
    max_acceptance: float = 0.9

class DraftModel(nn.Module):
    """Wrapper for draft models used in speculative decoding"""
    def __init__(self, model: nn.Module, temperature: float = 0.7):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.device = next(model.parameters()).device
        
    def forward(self, input_ids: torch.Tensor, **kwargs):
        with torch.no_grad():
            outputs = self.model(input_ids, **kwargs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            logits = logits[:, -1, :] / self.temperature
            return torch.softmax(logits, dim=-1)

class SpeculativeGenerator:
    """Implements speculative decoding for faster generation"""
    def __init__(
        self,
        target_model: nn.Module,
        num_draft_models: int = 3,
        acceptance_threshold: float = 0.7,
        max_candidates: int = 5,
        temperature: float = 0.7
    ):
        self.target_model = target_model
        self.config = SpeculativeConfig(
            num_draft_models=num_draft_models,
            acceptance_threshold=acceptance_threshold,
            max_candidates=max_candidates,
            temperature=temperature
        )
        self.device = next(target_model.parameters()).device
        self.draft_models = self._init_draft_models()
        
    def _init_draft_models(self) -> List[DraftModel]:
        """Initialize draft models by copying layers from target model"""
        draft_models = []
        for i in range(self.config.num_draft_models):
            # Create a shallow copy of the target model
            draft = type(self.target_model)(self.target_model.config)
            draft.load_state_dict(self.target_model.state_dict(), strict=False)
            draft.eval()
            
            # Freeze parameters
            for param in draft.parameters():
                param.requires_grad = False
                
            draft_models.append(DraftModel(draft, self.config.temperature))
            logger.info(f"Initialized draft model {i+1}/{self.config.num_draft_models}")
            
        return draft_models
    
    def generate_candidates(
        self, 
        input_ids: torch.Tensor,
        max_length: int,
        **kwargs
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Generate speculative candidates from draft models"""
        candidates = []
        acceptance_probs = []
        current_input = input_ids
        
        for _ in range(self.config.max_candidates):
            # Get predictions from all draft models in parallel
            draft_probs = []
            for draft in self.draft_models:
                probs = draft(current_input, **kwargs)
                draft_probs.append(probs)
            
            # Average probabilities across draft models
            avg_probs = torch.mean(torch.stack(draft_probs), dim=0)
            
            # Sample from averaged distribution
            next_tokens = torch.multinomial(avg_probs, num_samples=1)
            candidates.append(next_tokens)
            
            # Get acceptance probability from target model
            with torch.no_grad():
                target_logits = self.target_model(current_input, **kwargs).logits
                target_logits = target_logits[:, -1, :] / self.config.temperature
                target_probs = torch.softmax(target_logits, dim=-1)
                
                # Calculate acceptance probability
                accept_prob = torch.min(
                    torch.ones_like(avg_probs),
                    target_probs / (avg_probs + 1e-10)
                ).gather(1, next_tokens)
                acceptance_probs.append(accept_prob.item())
                
            # Update input for next step
            current_input = torch.cat([current_input, next_tokens], dim=-1)
            
            # Early stop if reached max length
            if current_input.size(1) >= max_length:
                break
                
        return candidates, acceptance_probs
    
    def verify_candidates(
        self,
        input_ids: torch.Tensor,
        candidates: List[torch.Tensor],
        acceptance_probs: List[float]
    ) -> Tuple[torch.Tensor, int]:
        """Verify speculative candidates against target model"""
        accepted_tokens = []
        num_accepted = 0
        
        # Prepare input for verification
        current_input = input_ids
        
        for token, prob in zip(candidates, acceptance_probs):
            # Accept token if probability exceeds threshold
            if prob >= self.config.acceptance_threshold:
                accepted_tokens.append(token)
                current_input = torch.cat([current_input, token], dim=-1)
                num_accepted += 1
            else:
                # Reject token and get corrected token from target model
                with torch.no_grad():
                    logits = self.target_model(current_input).logits
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    accepted_tokens.append(next_token)
                break
                
        if not accepted_tokens:
            # Fallback to target model if no tokens accepted
            with torch.no_grad():
                logits = self.target_model(input_ids).logits
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                accepted_tokens.append(next_token)
                
        return torch.cat(accepted_tokens, dim=-1), num_accepted
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        **kwargs
    ) -> torch.Tensor:
        """Generate tokens using speculative decoding"""
        generated = input_ids
        total_accepted = 0
        total_generated = 0
        
        while generated.size(1) < max_length:
            # Generate speculative candidates
            candidates, acceptance_probs = self.generate_candidates(
                generated, max_length, **kwargs
            )
            
            # Verify candidates against target model
            new_tokens, num_accepted = self.verify_candidates(
                generated, candidates, acceptance_probs
            )
            
            # Update generated sequence
            generated = torch.cat([generated, new_tokens], dim=-1)
            
            # Update stats
            total_accepted += num_accepted
            total_generated += len(candidates)
            
            # Log progress
            if generated.size(1) % 10 == 0:
                accept_rate = total_accepted / total_generated if total_generated > 0 else 0
                logger.info(
                    f"Generated {generated.size(1)}/{max_length} tokens "
                    f"(accept rate: {accept_rate:.2f})"
                )
                
        return generated
    
    def register_draft_model(self, model: nn.Module):
        """Register an additional draft model"""
        if len(self.draft_models) >= self.config.num_draft_models:
            logger.warning(f"Cannot add more than {self.config.num_draft_models} draft models")
            return
            
        draft = DraftModel(model, self.config.temperature)
        self.draft_models.append(draft)
        logger.info(f"Registered new draft model (total: {len(self.draft_models)})")
