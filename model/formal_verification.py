import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import math
import logging

from .numerical_precision import NumericallyStableOperations, NumericalPrecisionConfig

logger = logging.getLogger(__name__)

@dataclass
class FormalVerificationConfig:
    """Configuration for formal verification of mathematical reasoning"""
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    
    # Proof system parameters
    num_axioms: int = 64
    max_proof_steps: int = 16
    proof_temperature: float = 0.7
    
    # Verification parameters
    verification_threshold: float = 0.8
    verification_samples: int = 5
    
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
    uncertainty_threshold: float = 0.2


class AxiomSystem(nn.Module):
    """Represents a system of mathematical axioms for formal verification"""
    
    def __init__(self, config: FormalVerificationConfig):
        super().__init__()
        self.config = config
        
        # Axiom embeddings
        self.axiom_embeddings = nn.Parameter(
            torch.randn(config.num_axioms, config.hidden_size)
        )
        
        # Axiom applicability network
        self.axiom_applicability = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Axiom consequence network
        self.axiom_consequence = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Initialize numerical stability components
        self.stable_ops = NumericallyStableOperations(config.numerical_precision)
        
    def forward(self, statement_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Determine which axioms are applicable to the given statement
        
        Args:
            statement_embedding: Tensor of shape [batch_size, hidden_size]
            
        Returns:
            applicability_scores: Tensor of shape [batch_size, num_axioms]
            consequence_embeddings: Tensor of shape [batch_size, num_axioms, hidden_size]
            uncertainty: Tensor of shape [batch_size, num_axioms]
        """
        batch_size = statement_embedding.shape[0]
        
        # Expand statement embedding for comparison with all axioms
        expanded_statement = statement_embedding.unsqueeze(1).expand(
            batch_size, self.config.num_axioms, -1
        )
        
        # Expand axiom embeddings
        expanded_axioms = self.axiom_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Concatenate statement with each axiom
        combined = torch.cat([expanded_statement, expanded_axioms], dim=2)
        
        # Compute applicability scores
        applicability_scores = self.axiom_applicability(combined).squeeze(-1)
        
        # Compute consequences of applying each axiom
        weighted_axioms = expanded_axioms * applicability_scores.unsqueeze(-1)
        consequence_embeddings = self.axiom_consequence(weighted_axioms)
        
        # Compute uncertainty in applicability
        if self.config.quantify_uncertainty:
            # Uncertainty is higher when applicability score is close to 0.5
            uncertainty = 1.0 - 2.0 * torch.abs(applicability_scores - 0.5)
        else:
            uncertainty = torch.zeros_like(applicability_scores)
        
        return applicability_scores, consequence_embeddings, uncertainty


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
        
        # Initialize numerical stability components
        self.stable_ops = NumericallyStableOperations(config.numerical_precision)
        
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


class ProofSearch(nn.Module):
    """Performs proof search using a tree-based approach"""
    
    def __init__(self, config: FormalVerificationConfig):
        super().__init__()
        self.config = config
        
        # Proof step generator
        self.proof_step = ProofStep(config)
        
        # Axiom system
        self.axiom_system = AxiomSystem(config)
        
        # Goal proximity estimator
        self.goal_proximity = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Initialize numerical stability components
        self.stable_ops = NumericallyStableOperations(config.numerical_precision)
        
    def forward(
        self, 
        premise_embedding: torch.Tensor, 
        goal_embedding: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Search for a proof from premise to goal
        
        Args:
            premise_embedding: Tensor of shape [batch_size, hidden_size]
            goal_embedding: Tensor of shape [batch_size, hidden_size]
            
        Returns:
            proof_steps: List of tensors, each of shape [batch_size, hidden_size]
            proof_validity: Tensor of shape [batch_size, 1]
            proof_uncertainty: Tensor of shape [batch_size, 1]
        """
        batch_size = premise_embedding.shape[0]
        
        # Initialize proof steps with the premise
        proof_steps = [premise_embedding]
        step_validities = []
        
        # Iteratively build the proof
        current_step = premise_embedding
        
        for _ in range(self.config.max_proof_steps):
            # Check if we're already close to the goal
            proximity = self.goal_proximity(
                torch.cat([current_step, goal_embedding], dim=1)
            )
            
            # If we're close enough to the goal, stop
            if (proximity > 0.9).all():
                break
                
            # Find applicable axioms
            applicability, consequences, uncertainty = self.axiom_system(current_step)
            
            # Select the most applicable axiom for each batch item
            best_axiom_idx = torch.argmax(applicability, dim=1)
            
            # Get the corresponding consequences
            selected_consequences = consequences[
                torch.arange(batch_size, device=consequences.device),
                best_axiom_idx
            ]
            
            # Generate the next step
            next_step, validity = self.proof_step(
                current_step, 
                selected_consequences,
                goal_embedding
            )
            
            # Add to proof steps
            proof_steps.append(next_step)
            step_validities.append(validity)
            
            # Update current step
            current_step = next_step
        
        # Compute overall proof validity
        if step_validities:
            step_validities_tensor = torch.stack(step_validities, dim=1)
            proof_validity = torch.prod(step_validities_tensor, dim=1)
        else:
            proof_validity = torch.ones(batch_size, 1, device=premise_embedding.device)
        
        # Compute proof uncertainty
        if self.config.quantify_uncertainty:
            # Uncertainty increases with proof length and decreases with validity
            proof_length = len(proof_steps) - 1  # Exclude premise
            length_factor = torch.tensor(
                min(1.0, proof_length / self.config.max_proof_steps),
                device=premise_embedding.device
            )
            proof_uncertainty = (1.0 - proof_validity) * (0.5 + 0.5 * length_factor)
        else:
            proof_uncertainty = torch.zeros_like(proof_validity)
        
        return proof_steps, proof_validity, proof_uncertainty


class FormalVerifier(nn.Module):
    """Verifies mathematical reasoning using formal methods"""
    
    def __init__(self, config: FormalVerificationConfig):
        super().__init__()
        self.config = config
        
        # Proof search component
        self.proof_search = ProofSearch(config)
        
        # Statement encoder
        self.statement_encoder = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        
        # Proof checker
        self.proof_checker = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Initialize numerical stability components
        self.stable_ops = NumericallyStableOperations(config.numerical_precision)
        
    def encode_statement(self, statement: torch.Tensor) -> torch.Tensor:
        """
        Encode a mathematical statement
        
        Args:
            statement: Tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            encoded: Tensor of shape [batch_size, hidden_size]
        """
        # Apply transformer encoder
        encoded = self.statement_encoder(statement)
        
        # Pool to get a single vector
        return encoded.mean(dim=1)
    
    def verify_reasoning(
        self, 
        premises: torch.Tensor, 
        conclusion: torch.Tensor,
        reasoning_steps: Optional[List[torch.Tensor]] = None
    ) -> Tuple[bool, float, Dict]:
        """
        Verify if the conclusion follows from the premises
        
        Args:
            premises: Tensor of shape [batch_size, num_premises, seq_len, hidden_size]
            conclusion: Tensor of shape [batch_size, seq_len, hidden_size]
            reasoning_steps: Optional list of tensors representing intermediate steps
            
        Returns:
            is_valid: Boolean indicating if the reasoning is valid
            confidence: Float indicating confidence in the verification
            details: Dictionary with verification details
        """
        batch_size = premises.shape[0]
        
        # Encode premises and conclusion
        encoded_premises = []
        for i in range(premises.shape[1]):
            encoded_premise = self.encode_statement(premises[:, i])
            encoded_premises.append(encoded_premise)
        
        # Combine all premises
        if len(encoded_premises) > 1:
            combined_premise = torch.stack(encoded_premises, dim=1).mean(dim=1)
        else:
            combined_premise = encoded_premises[0]
            
        encoded_conclusion = self.encode_statement(conclusion)
        
        # If reasoning steps are provided, verify them directly
        if reasoning_steps:
            encoded_steps = [self.encode_statement(step) for step in reasoning_steps]
            
            # Check validity of each step
            step_validities = []
            for i in range(1, len(encoded_steps)):
                prev_step = encoded_steps[i-1]
                curr_step = encoded_steps[i]
                
                validity = self.proof_checker(
                    torch.cat([prev_step, curr_step], dim=1)
                )
                step_validities.append(validity)
            
            # Compute overall validity
            if step_validities:
                step_validities_tensor = torch.stack(step_validities, dim=1)
                overall_validity = torch.prod(step_validities_tensor, dim=1)
            else:
                overall_validity = torch.ones(batch_size, 1, device=premises.device)
                
            confidence = overall_validity.mean().item()
            is_valid = confidence > self.config.verification_threshold
            
            return is_valid, confidence, {
                "step_validities": [v.mean().item() for v in step_validities],
                "overall_validity": overall_validity.mean().item()
            }
        
        # Otherwise, search for a proof
        proof_steps, proof_validity, proof_uncertainty = self.proof_search(
            combined_premise, encoded_conclusion
        )
        
        # Check if the proof is valid
        confidence = proof_validity.mean().item()
        uncertainty = proof_uncertainty.mean().item()
        
        is_valid = confidence > self.config.verification_threshold and uncertainty < self.config.uncertainty_threshold
        
        return is_valid, confidence, {
            "proof_length": len(proof_steps) - 1,  # Exclude premise
            "proof_validity": proof_validity.mean().item(),
            "proof_uncertainty": uncertainty
        }


class UncertaintyAwareVerifier(nn.Module):
    """Extends formal verification with uncertainty quantification"""
    
    def __init__(self, config: FormalVerificationConfig):
        super().__init__()
        self.config = config
        self.formal_verifier = FormalVerifier(config)
        
        # Uncertainty estimation network
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 2)  # Mean and variance
        )
        
        # Initialize numerical stability components
        self.stable_ops = NumericallyStableOperations(config.numerical_precision)
        
    def forward(
        self, 
        premises: torch.Tensor, 
        conclusion: torch.Tensor,
        reasoning_steps: Optional[List[torch.Tensor]] = None
    ) -> Tuple[bool, float, Dict]:
        """
        Verify reasoning with uncertainty quantification
        
        Args:
            premises: Tensor of shape [batch_size, num_premises, seq_len, hidden_size]
            conclusion: Tensor of shape [batch_size, seq_len, hidden_size]
            reasoning_steps: Optional list of tensors representing intermediate steps
            
        Returns:
            is_valid: Boolean indicating if the reasoning is valid
            confidence: Float indicating confidence in the verification
            details: Dictionary with verification details including uncertainty
        """
        # Get base verification result
        is_valid, confidence, details = self.formal_verifier.verify_reasoning(
            premises, conclusion, reasoning_steps
        )
        
        # If uncertainty quantification is enabled, estimate uncertainty
        if self.config.quantify_uncertainty:
            # Encode premises and conclusion
            encoded_premises = []
            for i in range(premises.shape[1]):
                encoded_premise = self.formal_verifier.encode_statement(premises[:, i])
                encoded_premises.append(encoded_premise)
            
            # Combine all premises
            if len(encoded_premises) > 1:
                combined_premise = torch.stack(encoded_premises, dim=1).mean(dim=1)
            else:
                combined_premise = encoded_premises[0]
                
            encoded_conclusion = self.formal_verifier.encode_statement(conclusion)
            
            # Estimate uncertainty
            uncertainty_input = torch.cat([combined_premise, encoded_conclusion], dim=1)
            uncertainty_output = self.uncertainty_estimator(uncertainty_input)
            
            # Extract mean and variance
            mean = uncertainty_output[:, 0]
            variance = self.stable_ops.safe_exp(uncertainty_output[:, 1])  # Ensure positive variance
            
            # Compute confidence interval
            lower_bound = mean - 1.96 * torch.sqrt(variance)
            upper_bound = mean + 1.96 * torch.sqrt(variance)
            
            # Update details with uncertainty information
            details.update({
                "uncertainty_mean": mean.mean().item(),
                "uncertainty_variance": variance.mean().item(),
                "confidence_lower_bound": lower_bound.mean().item(),
                "confidence_upper_bound": upper_bound.mean().item()
            })
            
            # Adjust validity based on uncertainty
            if upper_bound.mean().item() < self.config.verification_threshold:
                is_valid = False
            elif lower_bound.mean().item() > self.config.verification_threshold:
                is_valid = True
            else:
                # Uncertain case, use original validity but with lower confidence
                confidence = confidence * (1.0 - variance.mean().item())
        
        return is_valid, confidence, details 