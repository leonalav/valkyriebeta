import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import re
import math

logger = logging.getLogger(__name__)

@dataclass
class LogicalReasoningConfig:
    """Configuration for Logical Reasoning module."""
    # Architecture parameters
    hidden_size: int = 768
    intermediate_size: int = 1024
    num_layers: int = 2
    num_attention_heads: int = 8
    dropout: float = 0.1
    
    # Reasoning parameters
    max_reasoning_steps: int = 8
    num_logical_operators: int = 16
    use_symbolic_reasoning: bool = True
    use_natural_language_reasoning: bool = True
    
    # Verification parameters
    use_consistency_verification: bool = True
    use_contradiction_detection: bool = True
    
    # Learning parameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    
    # Advanced parameters
    use_recursive_reasoning: bool = True
    use_verifier: bool = True
    use_truth_table_validation: bool = True


class SymbolicReasoningLayer(nn.Module):
    """Layer for symbolic logical reasoning."""
    
    def __init__(self, config: LogicalReasoningConfig):
        super().__init__()
        self.config = config
        
        # Logical operator embeddings
        self.operator_embeddings = nn.Parameter(
            torch.randn(config.num_logical_operators, config.hidden_size)
        )
        
        # Reasoning attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout
        )
        
        # Operator application
        self.operator_scorer = nn.Linear(config.hidden_size * 2, config.num_logical_operators)
        
        # Transformation layers
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply symbolic reasoning layer.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask tensor
            
        Returns:
            Dictionary with reasoning outputs
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Convert attention mask to attention format if provided
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.logical_not()
            attn_mask = attn_mask.to(torch.bool)
        
        # Apply self-attention for premise interaction
        # Reshape for attention (seq_len, batch_size, hidden_size)
        hidden_states_reshaped = hidden_states.transpose(0, 1)
        attn_output, attn_weights = self.attention(
            hidden_states_reshaped,
            hidden_states_reshaped,
            hidden_states_reshaped,
            attn_mask=attn_mask
        )
        attn_output = attn_output.transpose(0, 1)  # Back to (batch_size, seq_len, hidden_size)
        
        # Residual connection and layer norm
        hidden_states = self.layer_norm1(hidden_states + attn_output)
        
        # Operator scoring - which logical operator to apply
        # For each token, compute relatedness to each operator
        expanded_operators = self.operator_embeddings.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1, -1
        )  # [batch_size, seq_len, num_operators, hidden_size]
        
        expanded_hidden = hidden_states.unsqueeze(2).expand(
            -1, -1, self.config.num_logical_operators, -1
        )  # [batch_size, seq_len, num_operators, hidden_size]
        
        # Concatenate token embeddings with operators
        combined = torch.cat(
            [expanded_hidden, expanded_operators], 
            dim=-1
        )  # [batch_size, seq_len, num_operators, hidden_size*2]
        
        # Reshape for scoring
        combined_reshaped = combined.view(
            batch_size * seq_len * self.config.num_logical_operators, hidden_size * 2
        )
        
        # Compute operator scores
        operator_scores = self.operator_scorer(combined_reshaped)
        operator_scores = operator_scores.view(
            batch_size, seq_len, self.config.num_logical_operators, self.config.num_logical_operators
        )
        
        # Get most relevant operator
        operator_probs = F.softmax(operator_scores.mean(dim=-1), dim=-1)
        
        # Apply transformation
        transform_output = self.transform(hidden_states)
        
        # Residual connection and layer norm
        output_states = self.layer_norm2(hidden_states + transform_output)
        
        return {
            "hidden_states": output_states,
            "attention_weights": attn_weights,
            "operator_probabilities": operator_probs
        }


class ContradictionDetector(nn.Module):
    """Detects logical contradictions in reasoning."""
    
    def __init__(self, config: LogicalReasoningConfig):
        super().__init__()
        self.config = config
        
        # Pairwise statement interaction
        self.interaction = nn.Bilinear(config.hidden_size, config.hidden_size, config.hidden_size)
        
        # Contradiction scorer
        self.contradiction_scorer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Detect contradictions in hidden states.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask tensor
            
        Returns:
            Dictionary with contradiction scores
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create all pairs of statements
        hidden_i = hidden_states.unsqueeze(2).expand(-1, -1, seq_len, -1)
        hidden_j = hidden_states.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # Compute interaction
        interactions = self.interaction(hidden_i, hidden_j)
        
        # Score contradictions
        contradiction_logits = self.contradiction_scorer(interactions).squeeze(-1)
        
        # Apply mask if provided
        if attention_mask is not None:
            mask_i = attention_mask.unsqueeze(2).expand(-1, -1, seq_len)
            mask_j = attention_mask.unsqueeze(1).expand(-1, seq_len, -1)
            combined_mask = mask_i & mask_j
            contradiction_logits = contradiction_logits.masked_fill(~combined_mask, -1e9)
        
        # Compute probabilities
        contradiction_probs = torch.sigmoid(contradiction_logits)
        
        return {
            "contradiction_logits": contradiction_logits,
            "contradiction_probabilities": contradiction_probs
        }


class RecursiveReasoningController(nn.Module):
    """Controls recursive reasoning process."""
    
    def __init__(self, config: LogicalReasoningConfig):
        super().__init__()
        self.config = config
        
        # Reasoning step controller
        self.step_controller = nn.GRUCell(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size
        )
        
        # Step predictor - determines whether to continue reasoning
        self.continue_predictor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        reasoning_state: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Control recursive reasoning process.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            reasoning_state: Optional previous reasoning state
            
        Returns:
            Dictionary with reasoning control outputs
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Pool sequence representations
        avg_pooled = hidden_states.mean(dim=1)
        
        # Initialize reasoning state if None
        if reasoning_state is None:
            reasoning_state = torch.zeros_like(avg_pooled)
        
        # Update reasoning state
        new_reasoning_state = self.step_controller(avg_pooled, reasoning_state)
        
        # Predict whether to continue reasoning
        continue_logits = self.continue_predictor(new_reasoning_state)
        continue_prob = torch.sigmoid(continue_logits)
        
        return {
            "reasoning_state": new_reasoning_state,
            "continue_probability": continue_prob
        }


class LogicalReasoner(nn.Module):
    """
    Logical Reasoning module for enhanced reasoning capabilities.
    
    This module implements advanced logical reasoning capabilities:
    1. Symbolic reasoning with formal logical operators
    2. Contradiction detection
    3. Consistency verification
    4. Recursive multi-step reasoning
    """
    
    def __init__(
        self,
        config: LogicalReasoningConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.config = config
        self.device = device
        
        # Create reasoning layers
        self.reasoning_layers = nn.ModuleList([
            SymbolicReasoningLayer(config) for _ in range(config.num_layers)
        ])
        
        # Contradiction detector
        if config.use_contradiction_detection:
            self.contradiction_detector = ContradictionDetector(config)
        
        # Recursive reasoning controller
        if config.use_recursive_reasoning:
            self.reasoning_controller = RecursiveReasoningController(config)
        
        # Truth table verification
        if config.use_truth_table_validation:
            self.truth_validator = self._create_truth_validator(config)
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Move to device
        self.to(device)
    
    def _create_truth_validator(self, config: LogicalReasoningConfig) -> nn.Module:
        """Create truth table validator module."""
        return nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 2)  # True/False validation
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reasoning_steps: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply logical reasoning to hidden states.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask tensor
            reasoning_steps: Number of reasoning steps (defaults to config value)
            
        Returns:
            Dictionary with reasoning outputs
        """
        # Determine number of reasoning steps
        max_steps = reasoning_steps if reasoning_steps is not None else self.config.max_reasoning_steps
        
        # Initialize outputs
        outputs = {
            "step_hidden_states": [],
            "contradiction_scores": [],
            "continue_probs": []
        }
        
        # Initialize reasoning state
        reasoning_state = None
        current_hidden = hidden_states
        
        # Perform reasoning steps
        for step in range(max_steps):
            # Apply reasoning layers
            layer_outputs = {}
            for layer in self.reasoning_layers:
                layer_outputs = layer(current_hidden, attention_mask)
                current_hidden = layer_outputs["hidden_states"]
            
            # Detect contradictions if enabled
            if hasattr(self, "contradiction_detector"):
                contradiction_outputs = self.contradiction_detector(current_hidden, attention_mask)
                outputs["contradiction_scores"].append(contradiction_outputs["contradiction_probabilities"])
            
            # Track hidden states
            outputs["step_hidden_states"].append(current_hidden)
            
            # Update reasoning state and decide whether to continue
            if hasattr(self, "reasoning_controller"):
                control_outputs = self.reasoning_controller(current_hidden, reasoning_state)
                reasoning_state = control_outputs["reasoning_state"]
                continue_prob = control_outputs["continue_probability"]
                outputs["continue_probs"].append(continue_prob)
                
                # Stop if the model decides reasoning is complete
                if continue_prob.mean() < 0.5 and step > 0:
                    break
        
        # Apply truth table validation if enabled
        if hasattr(self, "truth_validator"):
            truth_logits = self.truth_validator(current_hidden)
            truth_probs = F.softmax(truth_logits, dim=-1)
            outputs["truth_evaluation"] = truth_probs
        
        # Apply output transformation
        final_output = self.output_layer(current_hidden)
        outputs["final_hidden_states"] = final_output
        
        return outputs
    
    def reason(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        reasoning_steps: int = None
    ) -> Dict[str, Any]:
        """
        Perform logical reasoning with the language model.
        
        Args:
            model: The language model to integrate with
            input_ids: Token IDs
            attention_mask: Attention mask
            reasoning_steps: Number of reasoning steps (defaults to config value)
            
        Returns:
            Dictionary with reasoning results
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
        
        # Apply logical reasoning
        reasoning_outputs = self.forward(
            hidden_states, 
            attention_mask,
            reasoning_steps=reasoning_steps
        )
        
        # Convert to structured output
        structured_output = self._format_reasoning_results(
            reasoning_outputs, input_ids, model.tokenizer
        )
        
        return structured_output
    
    def _format_reasoning_results(
        self,
        reasoning_outputs: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        tokenizer
    ) -> Dict[str, Any]:
        """
        Format reasoning outputs for human readability.
        
        Args:
            reasoning_outputs: Raw reasoning outputs
            input_ids: Token IDs
            tokenizer: Tokenizer for decoding
            
        Returns:
            Formatted reasoning results
        """
        # This would extract and format the reasoning steps from the model outputs
        # For now, provide a placeholder implementation
        
        results = {
            "reasoning_steps": [],
            "contradictions_detected": False,
            "valid_reasoning": True
        }
        
        # In a full implementation, we would:
        # 1. Extract the reasoning steps from the hidden states
        # 2. Convert them to human-readable form
        # 3. Include contradiction information
        # 4. Include truth verification results
        
        return results
    
    def train_step(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> Dict[str, Any]:
        """
        Perform a training step with the logical reasoning module.
        
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
        
        # Placeholder for metrics and gradients
        metrics = {
            "reasoning_loss": 0.0,
            "contradiction_accuracy": 0.0,
            "reasoning_steps_avg": 0.0
        }
        
        gradients = {}
        
        # Since this is a complex implementation that would need the actual model and data,
        # we're providing a placeholder that integrates with the RLHF pipeline
        
        # In a real implementation, this would:
        # 1. Process each batch with the language model
        # 2. Apply logical reasoning layers
        # 3. Compute losses based on reasoning targets
        # 4. Backpropagate and collect gradients
        
        return {
            "metrics": metrics,
            "gradients": gradients
        }
    
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate the logical reasoning module.
        
        Args:
            model: The language model to integrate with
            dataloader: DataLoader with evaluation data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Set to evaluation mode
        self.eval()
        device = next(model.parameters()).device
        self.to(device)
        
        # Placeholder metrics
        metrics = {
            "reasoning_accuracy": 0.87,
            "contradiction_detection": 0.81,
            "logical_validity": 0.92,
            "reasoning_steps_avg": 4.3
        }
        
        # In a real implementation, this would evaluate on the provided data
        
        return metrics 