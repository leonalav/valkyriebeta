import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import math
import logging

from ..numerical_precision import NumericallyStableOperations, NumericalPrecisionConfig

logger = logging.getLogger(__name__)

@dataclass
class FormalVerificationConfig:
    """Configuration for formal verification of mathematical reasoning"""
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    
    # Verification parameters
    verification_threshold: float = 0.8
    uncertainty_threshold: float = 0.2
    verification_samples: int = 5
    
    # Proof system parameters
    num_axioms: int = 64
    max_proof_steps: int = 16
    proof_temperature: float = 0.7
    
    # Proof search parameters
    use_proof_search: bool = True
    max_search_depth: int = 8
    search_width: int = 4
    
    # Automated theorem proving
    use_automated_theorem_proving: bool = True
    use_natural_deduction: bool = True
    use_sequent_calculus: bool = True
    
    # Numerical precision
    numerical_precision: NumericalPrecisionConfig = field(default_factory=NumericalPrecisionConfig)
    
    # Proof checking
    check_proof_validity: bool = True
    check_proof_minimality: bool = True
    
    # Uncertainty quantification
    quantify_uncertainty: bool = True


class ProofStep(nn.Module):
    """Represents a single step in a formal proof"""
    
    def __init__(self, config: FormalVerificationConfig):
        super().__init__()
        self.config = config
        
        # Step validity network
        self.step_validity = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Step generation network
        self.step_generation = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        
    def forward(
        self, 
        previous_step: torch.Tensor, 
        axiom_embedding: torch.Tensor, 
        goal_embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a proof step by applying an axiom to the previous step
        
        Args:
            previous_step: Tensor of shape [batch_size, hidden_size]
            axiom_embedding: Tensor of shape [batch_size, hidden_size]
            goal_embedding: Optional tensor of shape [batch_size, hidden_size]
            
        Returns:
            next_step: Tensor of shape [batch_size, hidden_size]
            validity: Tensor of shape [batch_size, 1]
        """
        batch_size = previous_step.shape[0]
        
        # If goal embedding is not provided, use zeros
        if goal_embedding is None:
            goal_embedding = torch.zeros_like(previous_step)
        
        # Combine previous step, axiom, and goal
        combined_input = torch.stack([previous_step, axiom_embedding, goal_embedding], dim=1)
        
        # Generate next step
        next_step = self.step_generation(combined_input)[:, 0]
        
        # Check validity of the step
        validity_input = torch.cat([previous_step, axiom_embedding, next_step], dim=1)
        validity = self.step_validity(validity_input)
        
        return next_step, validity


class FormalVerifier(nn.Module):
    """Module for formal verification of reasoning"""
    
    def __init__(self, config: FormalVerificationConfig):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        
        # Statement encoder
        self.statement_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        
        # Proof step generator
        self.proof_step = ProofStep(config)
        
        # Axiom embeddings for formal verification
        self.axiom_embeddings = nn.Parameter(
            torch.randn(config.num_axioms, hidden_size)
        )
        
        # Verification head to check logical consistency
        self.verification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def encode_statement(self, statement: torch.Tensor) -> torch.Tensor:
        """Encode a statement for verification"""
        # If statement is already encoded (a vector), just return it
        if statement.dim() == 2 and statement.size(1) == self.config.hidden_size:
            return statement
            
        # Otherwise, apply statement encoder
        encoded = self.statement_encoder(statement)
        
        # Use mean pooling for fixed-size representation
        return encoded.mean(dim=1)
    
    def verify_reasoning(
        self, 
        premises: torch.Tensor, 
        conclusion: torch.Tensor,
        reasoning_steps: Optional[List[torch.Tensor]] = None
    ) -> Tuple[bool, float, Dict]:
        """
        Verify if the reasoning from premises to conclusion is valid
        
        Args:
            premises: Tensor for premises
            conclusion: Tensor for conclusion
            reasoning_steps: Optional list of reasoning steps
            
        Returns:
            is_valid: Boolean indicating if reasoning is valid
            confidence: Confidence score for validity
            details: Dictionary with verification details
        """
        # Encode premises and conclusion if needed
        premise_embedding = self.encode_statement(premises)
        conclusion_embedding = self.encode_statement(conclusion)
        
        # If reasoning steps are provided, verify each step
        step_validities = []
        if reasoning_steps is not None and len(reasoning_steps) > 0:
            prev_step = premise_embedding
            
            for i, step in enumerate(reasoning_steps):
                # Encode step if needed
                step_embedding = self.encode_statement(step)
                
                # Determine most relevant axiom for this step
                axiom_scores = torch.matmul(step_embedding, self.axiom_embeddings.transpose(0, 1))
                best_axiom_idx = axiom_scores.argmax(dim=1)
                best_axiom = self.axiom_embeddings[best_axiom_idx]
                
                # Verify step
                _, validity = self.proof_step(prev_step, best_axiom, 
                                            None if i < len(reasoning_steps) - 1 else conclusion_embedding)
                step_validities.append(validity)
                
                # Update previous step
                prev_step = step_embedding
        
        # Check consistency between premises and conclusion
        combined = torch.cat([premise_embedding, conclusion_embedding], dim=1)
        verification_score = self.verification_head(combined.mean(dim=1))
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator(torch.cat([premise_embedding, conclusion_embedding], dim=1))
        
        # Compute confidence as verification score adjusted by uncertainty
        confidence = verification_score * (1 - uncertainty)
        
        # Determine if verification meets threshold
        is_valid = (confidence >= self.config.verification_threshold).all().item()
        
        # Collect verification details
        details = {
            'verification_score': verification_score.detach(),
            'uncertainty': uncertainty.detach(),
            'step_validities': [v.detach() for v in step_validities] if step_validities else None,
            'confidence': confidence.detach(),
            'threshold': self.config.verification_threshold
        }
        
        return is_valid, confidence.mean().item(), details


class UncertaintyAwareVerifier(nn.Module):
    """Verifier that explicitly models uncertainty for more reliable verification"""
    
    def __init__(self, config: FormalVerificationConfig):
        super().__init__()
        self.config = config
        self.verifier = FormalVerifier(config)
        
        # Enhanced uncertainty estimation
        self.uncertainty_module = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        premises: torch.Tensor, 
        conclusion: torch.Tensor,
        reasoning_steps: Optional[List[torch.Tensor]] = None
    ) -> Tuple[bool, float, Dict]:
        """
        Verify reasoning with uncertainty awareness
        
        Args:
            premises: Tensor for premises
            conclusion: Tensor for conclusion
            reasoning_steps: Optional list of reasoning steps
            
        Returns:
            is_valid: Boolean indicating if reasoning is valid with high confidence
            confidence: Confidence score for validity
            details: Dictionary with verification details including uncertainty
        """
        # Get base verification results
        is_valid, confidence, details = self.verifier.verify_reasoning(
            premises, conclusion, reasoning_steps
        )
        
        # Encode premises and conclusion
        premise_embedding = self.verifier.encode_statement(premises)
        conclusion_embedding = self.verifier.encode_statement(conclusion)
        
        # Enhanced uncertainty estimation
        combined = torch.cat([premise_embedding, conclusion_embedding], dim=1)
        uncertainty = self.uncertainty_module(combined)
        
        # Update confidence based on enhanced uncertainty
        adjusted_confidence = confidence * (1 - uncertainty.mean().item())
        
        # Determine validity with uncertainty-aware threshold
        is_uncertain = uncertainty.mean().item() > self.config.uncertainty_threshold
        final_is_valid = is_valid and not is_uncertain
        
        # Update details
        details.update({
            'enhanced_uncertainty': uncertainty.detach(),
            'is_uncertain': is_uncertain,
            'adjusted_confidence': adjusted_confidence,
            'uncertainty_threshold': self.config.uncertainty_threshold
        })
        
        return final_is_valid, adjusted_confidence, details 