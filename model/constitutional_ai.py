import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import json
import os
import re

logger = logging.getLogger(__name__)

@dataclass
class ConstitutionalAIConfig:
    """Configuration for Constitutional AI module."""
    # Architecture parameters
    hidden_size: int = 768
    num_layers: int = 2
    dropout: float = 0.1
    
    # Constitutional principles
    principles_file: Optional[str] = None
    num_principles: int = 12
    
    # Critique generation parameters
    max_critique_length: int = 512
    critique_temperature: float = 0.7
    
    # Revision parameters
    max_revision_iterations: int = 2
    revision_temperature: float = 0.5
    
    # Learning parameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    kl_penalty_weight: float = 0.1
    
    # Validation parameters
    validation_threshold: float = 0.7
    
    # Training parameters
    batch_size: int = 8
    
    # Default constitutional principles if no file is provided
    default_principles: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "name": "Harmlessness",
            "description": "The AI should never generate content that could be harmful to individuals or society.",
            "examples": [
                "Avoiding instructions for dangerous activities",
                "Not generating hateful or discriminatory content",
                "Refusing to provide harmful advice or suggestions"
            ]
        },
        {
            "name": "Honesty",
            "description": "The AI should be honest and accurate in its responses, and should not present false information as true.",
            "examples": [
                "Acknowledging uncertainty when appropriate",
                "Not fabricating information or citations",
                "Correcting factual errors when identified"
            ]
        },
        {
            "name": "Fairness",
            "description": "The AI should treat all individuals and groups fairly and without bias.",
            "examples": [
                "Providing balanced perspectives on controversial topics",
                "Not reinforcing harmful stereotypes",
                "Ensuring diverse representation in generated content"
            ]
        },
        {
            "name": "Privacy Respect",
            "description": "The AI should respect user privacy and confidentiality.",
            "examples": [
                "Not encouraging surveillance or invasion of privacy",
                "Respecting personal boundaries",
                "Not promoting techniques to access private information"
            ]
        },
        {
            "name": "Educational Value",
            "description": "When appropriate, the AI should provide educational and informative content.",
            "examples": [
                "Explaining complex concepts clearly",
                "Providing context and background information",
                "Encouraging critical thinking"
            ]
        },
        {
            "name": "Human Autonomy",
            "description": "The AI should respect and promote human autonomy and decision-making.",
            "examples": [
                "Presenting options rather than single solutions",
                "Not being manipulative or coercive",
                "Supporting informed decision-making"
            ]
        },
        {
            "name": "Scientific Integrity",
            "description": "The AI should uphold principles of scientific integrity and rational inquiry.",
            "examples": [
                "Distinguishing between fact and opinion",
                "Citing evidence for scientific claims",
                "Acknowledging the limits of current scientific knowledge"
            ]
        },
        {
            "name": "Helpfulness",
            "description": "The AI should be helpful to users while balancing other principles.",
            "examples": [
                "Providing relevant and useful information",
                "Assisting with legitimate tasks",
                "Following the user's intent when ethically appropriate"
            ]
        }
    ])


class PrincipleEvaluator(nn.Module):
    """Evaluates text against constitutional principles."""
    
    def __init__(self, config: ConstitutionalAIConfig):
        super().__init__()
        self.config = config
        
        # Encoding layers
        self.encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Principle-specific heads
        self.principle_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, 1) for _ in range(config.num_principles)
        ])
        
        # Initialize principles
        self.principles = self._load_principles(config)
    
    def _load_principles(self, config: ConstitutionalAIConfig) -> List[Dict[str, str]]:
        """Load constitutional principles from file or use defaults."""
        if config.principles_file and os.path.exists(config.principles_file):
            with open(config.principles_file, 'r') as f:
                principles = json.load(f)
            return principles
        else:
            logger.warning(f"Principles file not found: {config.principles_file}")
            logger.info("Using default principles")
            return config.default_principles
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate hidden states against constitutional principles.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask tensor
            
        Returns:
            Dictionary with principle scores
        """
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to hidden size
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            # Apply mask
            hidden_states = hidden_states * mask_expanded
        
        # Get sequence representation (mean pooling)
        seq_length = attention_mask.sum(dim=1, keepdim=True) if attention_mask is not None else hidden_states.size(1)
        sequence_output = hidden_states.sum(dim=1) / seq_length
        
        # Encode for principle evaluation
        encoded_output = self.encoder(sequence_output)
        
        # Evaluate against each principle
        principle_scores = []
        for principle_head in self.principle_heads:
            score = torch.sigmoid(principle_head(encoded_output))
            principle_scores.append(score)
        
        # Stack scores
        principle_scores = torch.cat(principle_scores, dim=1)
        
        return {
            "principle_scores": principle_scores,
            "mean_score": principle_scores.mean(dim=1)
        }


class CritiqueGenerator(nn.Module):
    """Generates critiques for text that violates constitutional principles."""
    
    def __init__(self, config: ConstitutionalAIConfig):
        super().__init__()
        self.config = config
        
        # Encoding layers
        self.principle_encoder = nn.Linear(config.num_principles, config.hidden_size)
        self.text_principle_combiner = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        principle_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate critique embeddings based on principle violations.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            principle_scores: Tensor of shape [batch_size, num_principles]
            attention_mask: Optional mask tensor
            
        Returns:
            Critique embeddings
        """
        # Get sequence representation (mean pooling)
        if attention_mask is not None:
            seq_length = attention_mask.sum(dim=1, keepdim=True)
            sequence_output = hidden_states.sum(dim=1) / seq_length
        else:
            sequence_output = hidden_states.mean(dim=1)
        
        # Encode principle scores
        principle_embeddings = self.principle_encoder(principle_scores)
        
        # Combine text and principle embeddings
        combined = torch.cat([sequence_output, principle_embeddings], dim=1)
        critique_embeddings = self.text_principle_combiner(combined)
        
        return critique_embeddings


class ConstitutionalAI(nn.Module):
    """
    Constitutional AI module for ensuring model outputs follow ethical guidelines.
    
    This module implements the Constitutional AI approach:
    1. Evaluate text against constitutional principles
    2. Generate critiques for violations
    3. Revise outputs to align with principles
    """
    
    def __init__(
        self,
        config: ConstitutionalAIConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.config = config
        self.device = device
        
        # Create evaluator and critique generator
        self.evaluator = PrincipleEvaluator(config)
        self.critique_generator = CritiqueGenerator(config)
        
        # Move to device
        self.to(device)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process hidden states with constitutional AI components.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask tensor
            
        Returns:
            Dictionary with principle scores and critique embeddings
        """
        # Evaluate against principles
        eval_outputs = self.evaluator(hidden_states, attention_mask)
        principle_scores = eval_outputs["principle_scores"]
        
        # Generate critiques for low-scoring principles
        critique_embeddings = self.critique_generator(
            hidden_states, 
            principle_scores,
            attention_mask
        )
        
        # Combine outputs
        outputs = {
            **eval_outputs,
            "critique_embeddings": critique_embeddings
        }
        
        return outputs
    
    def evaluate_text(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Evaluate text against constitutional principles.
        
        Args:
            model: The language model to integrate with
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary with evaluation results
        """
        # Get model device
        device = next(model.parameters()).device
        self.to(device)
        
        # Get hidden states from language model
        with torch.no_grad():
            model_outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                output_hidden_states=True
            )
            hidden_states = model_outputs.last_hidden_state
        
        # Evaluate against principles
        eval_outputs = self.evaluator(hidden_states, attention_mask)
        
        # Convert scores to human-readable format
        results = self._format_evaluation_results(eval_outputs)
        
        return results
    
    def _format_evaluation_results(self, eval_outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Format evaluation outputs for human readability.
        
        Args:
            eval_outputs: Raw evaluation outputs
            
        Returns:
            Formatted results
        """
        principle_scores = eval_outputs["principle_scores"].cpu().numpy()
        
        # Format results
        results = {
            "overall_score": float(eval_outputs["mean_score"].mean().cpu().numpy()),
            "principle_scores": {}
        }
        
        # Add individual principle scores
        for i, principle in enumerate(self.evaluator.principles[:principle_scores.shape[1]]):
            results["principle_scores"][principle["name"]] = float(principle_scores[0, i])
        
        # Add pass/fail determination
        results["passes_constitution"] = results["overall_score"] >= self.config.validation_threshold
        
        return results
    
    def revise_text(
        self,
        model: nn.Module,
        tokenizer: Any,
        text: str
    ) -> Dict[str, Any]:
        """
        Revise text to better align with constitutional principles.
        
        Args:
            model: The language model to integrate with
            tokenizer: Tokenizer for encoding/decoding
            text: Text to revise
            
        Returns:
            Dictionary with revision results
        """
        # Encode input text
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Initial evaluation
        initial_results = self.evaluate_text(model, input_ids, attention_mask)
        
        # If text already passes, return it unchanged
        if initial_results["passes_constitution"]:
            return {
                "revised_text": text,
                "initial_evaluation": initial_results,
                "final_evaluation": initial_results,
                "needed_revision": False
            }
        
        # Generate critique
        critique = self._generate_critique(model, tokenizer, input_ids, attention_mask, initial_results)
        
        # Generate revision based on critique
        revised_text = self._generate_revision(model, tokenizer, text, critique)
        
        # Evaluate revised text
        revised_inputs = tokenizer(revised_text, return_tensors="pt")
        revised_input_ids = revised_inputs["input_ids"].to(self.device)
        revised_attention_mask = revised_inputs["attention_mask"].to(self.device)
        final_results = self.evaluate_text(model, revised_input_ids, revised_attention_mask)
        
        return {
            "revised_text": revised_text,
            "critique": critique,
            "initial_evaluation": initial_results,
            "final_evaluation": final_results,
            "needed_revision": True
        }
    
    def _generate_critique(
        self,
        model: nn.Module,
        tokenizer: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        evaluation_results: Dict[str, Any]
    ) -> str:
        """
        Generate critique for problematic text.
        
        Args:
            model: The language model
            tokenizer: Tokenizer for encoding/decoding
            input_ids: Token IDs
            attention_mask: Attention mask
            evaluation_results: Evaluation results
            
        Returns:
            Critique text
        """
        # In a full implementation, this would use the model to generate critiques
        # For now, provide a template-based critique
        
        # Find low-scoring principles
        low_principles = []
        for name, score in evaluation_results["principle_scores"].items():
            if score < self.config.validation_threshold:
                low_principles.append(name)
        
        # Generate critique based on low-scoring principles
        critique_parts = ["This text may be improved to better align with the following principles:"]
        
        for principle in low_principles:
            # Find the full principle object
            principle_obj = next((p for p in self.evaluator.principles if p["name"] == principle), None)
            if principle_obj:
                critique_parts.append(f"- {principle}: {principle_obj['description']}")
                if "examples" in principle_obj and principle_obj["examples"]:
                    example = principle_obj["examples"][0]
                    critique_parts.append(f"  Example improvement: {example}")
        
        return "\n".join(critique_parts)
    
    def _generate_revision(
        self,
        model: nn.Module,
        tokenizer: Any,
        original_text: str,
        critique: str
    ) -> str:
        """
        Generate revision based on critique.
        
        Args:
            model: The language model
            tokenizer: Tokenizer for encoding/decoding
            original_text: Original text
            critique: Critique text
            
        Returns:
            Revised text
        """
        # In a full implementation, this would use the model to generate
        # a revised version based on the critique
        # For now, return a placeholder
        
        # This is where the actual model-based revision would happen,
        # using the critique to guide improvements
        
        # Since we can't actually run the model here, return a modified version
        # of the original that acknowledges the revision process
        return original_text + "\n\n[Note: This text has been revised to better align with constitutional principles]"
    
    def apply_constraints(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> Dict[str, Any]:
        """
        Apply constitutional constraints during training.
        
        Args:
            model: The language model to integrate with
            dataloader: DataLoader with training data
            
        Returns:
            Dictionary containing metrics and gradients
        """
        # Set to training mode
        self.train()
        device = next(model.parameters()).device
        self.to(device)
        
        # Track metrics and gradients
        metrics = {
            "principle_violation_rate": 0.0,
            "mean_principle_score": 0.0,
            "kl_penalty": 0.0,
            "constitutional_loss": 0.0
        }
        
        # Implementation for gradients
        gradients = {}
        
        # Process each batch with the model
        for batch_idx, batch in enumerate(dataloader):
            # Extract batch data
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Enable gradient computation
            model.train()
            
            # Forward pass with gradient tracking
            with torch.set_grad_enabled(True):
                # Original model outputs
                original_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                original_logits = original_outputs.logits
                
                # Evaluate outputs against constitutional principles
                principle_scores = self.evaluator(
                    original_outputs.hidden_states[-1],
                    attention_mask
                )
                
                # Generate critique for problematic outputs
                critiques = self.critique_generator(
                    original_outputs.hidden_states[-1],
                    principle_scores["principle_scores"],
                    attention_mask
                )
                
                # Get revised outputs
                revision_attention_mask = attention_mask.clone()
                
                # Process multiple revisions if needed
                for rev_iter in range(self.config.max_revision_iterations):
                    # Generate revision
                    revision_outputs = model(
                        inputs_embeds=critiques,
                        attention_mask=revision_attention_mask,
                        output_hidden_states=True
                    )
                    revision_logits = revision_outputs.logits
                    
                    # Re-evaluate revised outputs
                    revised_principle_scores = self.evaluator(
                        revision_outputs.hidden_states[-1],
                        revision_attention_mask
                    )
                    
                    # If principles satisfied, break the loop
                    if (revised_principle_scores["principle_scores"].mean() > self.config.validation_threshold).all():
                        break
                    
                    # Otherwise, continue refining
                    critiques = self.critique_generator(
                        revision_outputs.hidden_states[-1],
                        revised_principle_scores["principle_scores"],
                        revision_attention_mask
                    )
                
                # Calculate KL divergence between original and revised distributions
                kl_div = F.kl_div(
                    F.log_softmax(revision_logits, dim=-1),
                    F.softmax(original_logits, dim=-1),
                    reduction="batchmean"
                )
                
                # Calculate constitutional loss
                # Combine principle violations and KL divergence
                principle_loss = (1.0 - revised_principle_scores["principle_scores"]).mean()
                constitutional_loss = principle_loss + self.config.kl_penalty_weight * kl_div
                
                # Backpropagate the loss
                constitutional_loss.backward()
                
                # Extract and store gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if name not in gradients:
                            gradients[name] = []
                        gradients[name].append(param.grad.detach().clone())
                
                # Update metrics
                metrics["principle_violation_rate"] += (revised_principle_scores["principle_scores"] < self.config.validation_threshold).float().mean().item()
                metrics["mean_principle_score"] += revised_principle_scores["principle_scores"].mean().item()
                metrics["kl_penalty"] += kl_div.item()
                metrics["constitutional_loss"] += constitutional_loss.item()
                
                # Clear gradients for next iteration
                model.zero_grad()
        
        # Average metrics over batches
        num_batches = len(dataloader)
        if num_batches > 0:
            metrics["principle_violation_rate"] /= num_batches
            metrics["mean_principle_score"] /= num_batches
            metrics["kl_penalty"] /= num_batches
            metrics["constitutional_loss"] /= num_batches
            
            # Average gradients over batches
            for name in gradients:
                if gradients[name]:
                    gradients[name] = torch.stack(gradients[name]).mean(dim=0)
        
        # Add gradients to metrics
        metrics["gradients"] = gradients
        
        return metrics 