import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

from .numerical_precision import (
    NumericalPrecisionConfig, 
    NumericallyStableOperations,
    HighPrecisionMathOperations,
    PrecisionAdaptiveLinear
)
from .formal_verification import (
    FormalVerificationConfig,
    FormalVerifier,
    UncertaintyAwareVerifier
)
from .math_reasoning import MathReasoningConfig

logger = logging.getLogger(__name__)

class EnhancedMathematicalReasoning(nn.Module):
    """
    Integrates enhanced mathematical reasoning capabilities into the model.
    
    This module combines numerical precision improvements and formal verification
    to enhance the mathematical reasoning capabilities of the model.
    """
    
    def __init__(
        self,
        hidden_size: int,
        math_config: Optional[MathReasoningConfig] = None,
        precision_config: Optional[NumericalPrecisionConfig] = None,
        verification_config: Optional[FormalVerificationConfig] = None
    ):
        super().__init__()
        
        # Initialize configurations
        self.math_config = math_config or MathReasoningConfig(hidden_size=hidden_size)
        self.precision_config = precision_config or NumericalPrecisionConfig()
        
        # Create verification config if not provided
        if verification_config is None:
            verification_config = FormalVerificationConfig(
                hidden_size=hidden_size,
                numerical_precision=self.precision_config
            )
        self.verification_config = verification_config
        
        # Initialize numerical stability components
        self.stable_ops = NumericallyStableOperations(self.precision_config)
        self.high_precision_ops = HighPrecisionMathOperations(self.precision_config)
        
        # Initialize formal verification components
        self.formal_verifier = UncertaintyAwareVerifier(self.verification_config)
        
        # Integration layers
        self.precision_integration = nn.Sequential(
            PrecisionAdaptiveLinear(hidden_size, hidden_size, config=self.precision_config),
            nn.GELU(),
            PrecisionAdaptiveLinear(hidden_size, hidden_size, config=self.precision_config)
        )
        
        # Mathematical operation detection
        self.operation_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 8)  # 8 operation types
        )
        
        # Operation-specific precision handlers
        self.operation_handlers = nn.ModuleDict({
            "addition": PrecisionAdaptiveLinear(hidden_size, hidden_size, config=self.precision_config),
            "subtraction": PrecisionAdaptiveLinear(hidden_size, hidden_size, config=self.precision_config),
            "multiplication": PrecisionAdaptiveLinear(hidden_size, hidden_size, config=self.precision_config),
            "division": PrecisionAdaptiveLinear(hidden_size, hidden_size, config=self.precision_config),
            "exponentiation": PrecisionAdaptiveLinear(hidden_size, hidden_size, config=self.precision_config),
            "logarithm": PrecisionAdaptiveLinear(hidden_size, hidden_size, config=self.precision_config),
            "matrix_ops": PrecisionAdaptiveLinear(hidden_size, hidden_size, config=self.precision_config),
            "trigonometry": PrecisionAdaptiveLinear(hidden_size, hidden_size, config=self.precision_config)
        })
        
    def detect_mathematical_operations(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Detect which mathematical operations are being performed
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            operation_weights: Tensor of shape [batch_size, seq_len, num_operations]
        """
        # Detect operations
        logits = self.operation_detector(hidden_states)
        
        # Convert to probabilities
        return self.stable_ops.safe_softmax(logits, dim=-1)
    
    def apply_precision_enhancements(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply precision enhancements to the hidden states
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            enhanced_states: Tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Detect operations
        operation_weights = self.detect_mathematical_operations(hidden_states)
        
        # Apply operation-specific handlers
        operation_outputs = []
        for i, (op_name, handler) in enumerate(self.operation_handlers.items()):
            # Apply handler
            op_output = handler(hidden_states)
            
            # Weight by operation probability
            weighted_output = op_output * operation_weights[:, :, i:i+1]
            operation_outputs.append(weighted_output)
        
        # Combine outputs
        if self.precision_config.use_kahan_summation:
            stacked_outputs = torch.stack(operation_outputs, dim=-1)
            combined = self.stable_ops.kahan_sum(stacked_outputs, dim=-1)
        else:
            combined = sum(operation_outputs)
        
        # Apply final integration
        enhanced = self.precision_integration(combined)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask to match hidden states dimensions
            expanded_mask = attention_mask.unsqueeze(-1).expand_as(enhanced)
            # Apply mask (0 for padding tokens)
            enhanced = enhanced * expanded_mask
        
        # Verify no NaN/Inf values
        return self.stable_ops.verify_tensor(enhanced, "precision_enhanced")
    
    def verify_mathematical_reasoning(
        self,
        premises: torch.Tensor,
        conclusion: torch.Tensor,
        reasoning_steps: Optional[List[torch.Tensor]] = None
    ) -> Tuple[bool, float, Dict]:
        """
        Verify mathematical reasoning using formal verification
        
        Args:
            premises: Tensor of shape [batch_size, num_premises, seq_len, hidden_size]
            conclusion: Tensor of shape [batch_size, seq_len, hidden_size]
            reasoning_steps: Optional list of tensors representing intermediate steps
            
        Returns:
            is_valid: Boolean indicating if the reasoning is valid
            confidence: Float indicating confidence in the verification
            details: Dictionary with verification details
        """
        return self.formal_verifier(premises, conclusion, reasoning_steps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        verify_reasoning: bool = False,
        premises: Optional[torch.Tensor] = None,
        conclusion: Optional[torch.Tensor] = None,
        reasoning_steps: Optional[List[torch.Tensor]] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Apply mathematical reasoning enhancements
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            verify_reasoning: Whether to verify mathematical reasoning
            premises: Optional tensor for verification
            conclusion: Optional tensor for verification
            reasoning_steps: Optional list of tensors for verification
            
        Returns:
            enhanced_states: Tensor of shape [batch_size, seq_len, hidden_size]
            verification_details: Optional dictionary with verification details
        """
        # Apply precision enhancements
        enhanced_states = self.apply_precision_enhancements(hidden_states, attention_mask)
        
        # Verify reasoning if requested
        if verify_reasoning and premises is not None and conclusion is not None:
            is_valid, confidence, details = self.verify_mathematical_reasoning(
                premises, conclusion, reasoning_steps
            )
            return enhanced_states, {
                "is_valid": is_valid,
                "confidence": confidence,
                "verification_details": details
            }
        
        return enhanced_states


class PrecisionEnhancedTransformerLayer(nn.Module):
    """
    Transformer layer with enhanced precision for mathematical operations
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        precision_config: Optional[NumericalPrecisionConfig] = None
    ):
        super().__init__()
        
        # Initialize precision config
        self.precision_config = precision_config or NumericalPrecisionConfig()
        
        # Initialize numerical stability components
        self.stable_ops = NumericallyStableOperations(self.precision_config)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward network with precision-adaptive linear layers
        self.feed_forward = nn.Sequential(
            PrecisionAdaptiveLinear(hidden_size, intermediate_size, config=self.precision_config),
            nn.GELU(),
            PrecisionAdaptiveLinear(intermediate_size, hidden_size, config=self.precision_config),
            nn.Dropout(0.1)
        )
        
        # Layer norms
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Mathematical operation detection
        self.is_math_operation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process input through the transformer layer with precision enhancements
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            output: Tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Detect if this is a mathematical operation
        math_probs = self.is_math_operation(hidden_states)
        
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        
        # Use compensated dot product for attention computation
        if attention_mask is not None:
            # Convert attention mask to proper format
            attn_mask = attention_mask.to(dtype=torch.float32)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, float(0.0))
        else:
            attn_mask = None
            
        # Apply attention
        attention_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states,
            attn_mask=attn_mask
        )
        
        # Apply residual connection with verification
        hidden_states = residual + self.dropout(attention_output)
        hidden_states = self.stable_ops.verify_tensor(hidden_states, "attention_output")
        
        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        
        # Apply residual connection with verification
        hidden_states = residual + self.dropout(hidden_states)
        hidden_states = self.stable_ops.verify_tensor(hidden_states, "ffn_output")
        
        # Apply higher precision for mathematical operations
        if self.precision_config.use_double_precision:
            # Conditionally use higher precision based on math operation probability
            high_precision_mask = (math_probs > 0.5).float()
            
            # Convert relevant parts to higher precision
            high_precision_states = hidden_states.double() * high_precision_mask.double()
            regular_states = hidden_states * (1 - high_precision_mask)
            
            # Convert back to original precision
            hidden_states = regular_states + high_precision_states.to(hidden_states.dtype)
        
        return hidden_states


def enhance_model_with_precision(model, precision_config=None, verification_config=None):
    """
    Enhance an existing model with precision improvements for mathematical reasoning
    
    Args:
        model: The model to enhance
        precision_config: Optional numerical precision configuration
        verification_config: Optional formal verification configuration
        
    Returns:
        Enhanced model
    """
    # Default configs if not provided
    if precision_config is None:
        precision_config = NumericalPrecisionConfig()
    
    # Get the hidden size from the model's config
    if hasattr(model, 'config'):
        # Try to get hidden_size or n_embd from config
        if hasattr(model.config, 'hidden_size'):
            hidden_size = model.config.hidden_size
        elif hasattr(model.config, 'n_embd'):
            hidden_size = model.config.n_embd
        else:
            raise ValueError("Could not determine hidden size from model config")
    else:
        raise ValueError("Model does not have a config attribute")
        
    if verification_config is None:
        verification_config = FormalVerificationConfig(
            hidden_size=hidden_size,
            numerical_precision=precision_config
        )
    
    # Replace linear layers with precision-adaptive ones
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Create a precision-adaptive linear layer with the same parameters
            precision_layer = PrecisionAdaptiveLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                config=precision_config
            )
            
            # Copy weights and biases
            precision_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                precision_layer.bias.data.copy_(module.bias.data)
                
            # Replace the module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, precision_layer)
            else:
                setattr(model, child_name, precision_layer)
    
    # Add mathematical reasoning enhancement module
    model.math_reasoning_enhancement = EnhancedMathematicalReasoning(
        hidden_size=hidden_size,
        precision_config=precision_config,
        verification_config=verification_config
    )
    
    # Store original forward method
    original_forward = model.forward
    
    # Define new forward method
    def enhanced_forward(self, *args, **kwargs):
        # Call original forward method
        outputs = original_forward(*args, **kwargs)
        
        # Extract attention mask from kwargs if available
        attention_mask = kwargs.get('attention_mask', None)
        
        # Apply mathematical reasoning enhancements
        if isinstance(outputs, dict) and 'hidden_states' in outputs:
            # Handle dictionary output format (from GPT class)
            hidden_states = outputs['hidden_states']
            enhanced_states = model.math_reasoning_enhancement(
                hidden_states, 
                attention_mask=attention_mask
            )
            
            # Replace hidden states with enhanced ones
            outputs['hidden_states'] = enhanced_states
        elif isinstance(outputs, tuple):
            # Handle tuple output format
            hidden_states = outputs[0]
            enhanced_states = model.math_reasoning_enhancement(
                hidden_states,
                attention_mask=attention_mask
            )
            
            # Replace hidden states with enhanced ones
            outputs = (enhanced_states,) + outputs[1:]
        else:
            # Handle tensor output format
            enhanced_states = model.math_reasoning_enhancement(
                outputs,
                attention_mask=attention_mask
            )
            outputs = enhanced_states
            
        return outputs
    
    # Replace forward method
    model.forward = enhanced_forward.__get__(model)
    
    return model 