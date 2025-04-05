import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

class UncertaintyCalibration(nn.Module):
    """
    Module for calibrating model uncertainty and confidence.
    Provides methods for temperature scaling, ensemble-based uncertainty,
    and confidence estimation.
    """
    
    def __init__(
        self,
        config,
        hidden_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_temperature_scaling: bool = True,
        use_ensemble_uncertainty: bool = True,
        use_monte_carlo_dropout: bool = True,
        num_mc_samples: int = 5,
        use_confidence_penalty: bool = True
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size if hasattr(config, 'hidden_size') else hidden_size
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        self.use_temperature_scaling = use_temperature_scaling
        
        # Ensemble uncertainty
        self.use_ensemble_uncertainty = use_ensemble_uncertainty
        self.ensemble_weights = nn.Parameter(torch.ones(num_heads))
        
        # Monte Carlo dropout
        self.use_monte_carlo_dropout = use_monte_carlo_dropout
        self.num_mc_samples = num_mc_samples
        self.mc_dropout = nn.Dropout(dropout * 2)  # Higher dropout for MC sampling
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty-aware attention
        self.uncertainty_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Confidence penalty
        self.use_confidence_penalty = use_confidence_penalty
        self.confidence_penalty_weight = 0.1
        
        # Initialize state
        self.is_initialized = False
        
    def initialize(self):
        """Initialize uncertainty calibration"""
        if not self.is_initialized:
            # Initialize temperature with a reasonable value
            self.temperature.data.fill_(1.0)
            
            # Initialize ensemble weights
            nn.init.normal_(self.ensemble_weights, mean=1.0, std=0.01)
            
            self.is_initialized = True
        
    def forward(self, logits: torch.Tensor, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Apply uncertainty calibration to model outputs
        
        Args:
            logits: Uncalibrated logits from the model
            hidden_states: Hidden states from the model
            attention_mask: Optional attention mask
            
        Returns:
            calibrated_logits: Calibrated logits
            uncertainty_info: Dictionary with uncertainty metrics
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Ensure initialized
        if not self.is_initialized:
            self.initialize()
            
        # Apply temperature scaling
        if self.use_temperature_scaling:
            calibrated_logits = logits / self.temperature
        else:
            calibrated_logits = logits
            
        # Compute confidence scores
        confidence = self.confidence_estimator(hidden_states)
        
        # Apply Monte Carlo dropout for uncertainty estimation
        mc_samples = []
        if self.use_monte_carlo_dropout and self.training:
            for _ in range(self.num_mc_samples):
                mc_hidden = self.mc_dropout(hidden_states)
                mc_logits = torch.matmul(mc_hidden, logits.transpose(1, 2))
                mc_samples.append(mc_logits)
                
            # Compute variance across samples
            mc_stack = torch.stack(mc_samples, dim=0)
            mc_mean = mc_stack.mean(dim=0)
            mc_var = ((mc_stack - mc_mean.unsqueeze(0)) ** 2).mean(dim=0)
            
            # Use variance as uncertainty measure
            uncertainty = mc_var.mean(dim=-1, keepdim=True)
        else:
            # Fallback uncertainty estimation
            probs = F.softmax(calibrated_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1, keepdim=True)
            uncertainty = entropy / math.log(vocab_size)  # Normalize by max entropy
            
        # Apply uncertainty-aware attention if needed
        if hasattr(self, 'uncertainty_attention') and attention_mask is not None:
            # Use uncertainty to modify attention
            uncertainty_weights = 1.0 - uncertainty  # Higher confidence gets more attention
            
            # Apply attention with uncertainty weighting
            attended_states, _ = self.uncertainty_attention(
                hidden_states, 
                hidden_states, 
                hidden_states,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
            
            # Blend based on uncertainty
            hidden_states = hidden_states * uncertainty_weights + attended_states * (1 - uncertainty_weights)
            
            # Recompute logits with attended states
            # This is a placeholder - actual implementation would depend on model architecture
            # calibrated_logits = some_function(hidden_states)
            
        # Apply confidence penalty during training
        if self.training and self.use_confidence_penalty:
            # Penalize overconfidence
            confidence_penalty = self.confidence_penalty_weight * (confidence ** 2).mean()
            # In a real implementation, you would add this to the loss
            
        # Prepare uncertainty info
        uncertainty_info = {
            'confidence': confidence,
            'uncertainty': uncertainty,
            'temperature': self.temperature.item()
        }
        
        if self.use_monte_carlo_dropout and self.training:
            uncertainty_info['mc_variance'] = mc_var
            
        return calibrated_logits, uncertainty_info
    
    def calibrate_probabilities(self, logits: torch.Tensor):
        """
        Apply temperature scaling to calibrate probabilities
        
        Args:
            logits: Uncalibrated logits
            
        Returns:
            calibrated_probs: Calibrated probabilities
        """
        if not self.is_initialized:
            self.initialize()
            
        # Apply temperature scaling
        calibrated_logits = logits / self.temperature
        
        # Convert to probabilities
        calibrated_probs = F.softmax(calibrated_logits, dim=-1)
        
        return calibrated_probs
    
    def estimate_uncertainty(self, hidden_states: torch.Tensor, logits: Optional[torch.Tensor] = None):
        """
        Estimate model uncertainty from hidden states
        
        Args:
            hidden_states: Model hidden states
            logits: Optional logits for additional uncertainty estimation
            
        Returns:
            uncertainty: Uncertainty estimates
            confidence: Confidence scores
        """
        if not self.is_initialized:
            self.initialize()
            
        # Compute confidence scores
        confidence = self.confidence_estimator(hidden_states)
        
        # Compute uncertainty
        if logits is not None:
            probs = F.softmax(logits / self.temperature, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1, keepdim=True)
            uncertainty = entropy / math.log(logits.size(-1))  # Normalize by max entropy
        else:
            # Derive uncertainty from confidence
            uncertainty = 1.0 - confidence
            
        return uncertainty, confidence 