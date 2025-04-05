from typing import Optional, Dict, Any, Tuple, Union
import torch
import torch.nn as nn
from .exceptions import ModelError, ResourceError
from .validation import ConfigValidator, ModelValidator
import logging
from transformers import PreTrainedTokenizer
import math
import os
import random
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class InitializationType(Enum):
    """Types of weight initialization"""
    STANDARD = "standard"  # Normal distribution
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    KAIMING_NORMAL = "kaiming_normal"
    ORTHOGONAL = "orthogonal"
    SMALL_NORMAL = "small_normal"  # Lower std dev
    SMALL_UNIFORM = "small_uniform"  # Smaller range
    RWKV_SPECIFIC = "rwkv"  # Special initialization for RWKV models
    
    @classmethod
    def from_string(cls, name: str) -> "InitializationType":
        """Get the initialization type from a string"""
        try:
            return cls(name.lower())
        except ValueError:
            logger.warning(f"Unknown initialization type: {name}. Using standard initialization.")
            return cls.STANDARD

class ModelInitializer:
    """
    Initializes model weights according to different strategies.
    """
    
    def __init__(
        self,
        initialization_type: Union[InitializationType, str] = InitializationType.STANDARD,
        seed: int = 42,
        **kwargs
    ):
        # Set random seed for reproducibility
        self.seed = seed
        self._set_seed(seed)
        
        # Determine initialization type
        if isinstance(initialization_type, str):
            self.initialization_type = InitializationType.from_string(initialization_type)
        else:
            self.initialization_type = initialization_type
        
        # Store additional initialization parameters
        self.kwargs = kwargs
        
        # Special handling for certain layer types
        self.special_layer_handlers = {
            nn.LayerNorm: self._init_layer_norm,
            nn.Embedding: self._init_embedding
        }
    
    def _set_seed(self, seed: int):
        """Set all random seeds"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def initialize_model(self, model: nn.Module) -> nn.Module:
        """
        Initialize model weights according to the selected strategy.
        
        Args:
            model: The model to initialize
            
        Returns:
            The initialized model
        """
        logger.info(f"Initializing model with {self.initialization_type.value} initialization")
        
        # Check if model has custom initialization
        if hasattr(model, '_init_weights') and callable(model._init_weights):
            logger.info("Using model's built-in initialization")
            # Apply model's own initialization
            for module in model.modules():
                model._init_weights(module)
            return model
        
        # Apply initialization based on selected type
        self._apply_initialization(model)
        
        # Apply special RWKV initialization if needed
        if self.initialization_type == InitializationType.RWKV_SPECIFIC:
            self._apply_rwkv_initialization(model)
        
        logger.info("Model initialization complete")
        return model
    
    def _apply_initialization(self, model: nn.Module):
        """Apply the selected initialization to all model parameters"""
        for name, module in model.named_modules():
            # Check if module has a special handler
            module_type = type(module)
            if module_type in self.special_layer_handlers:
                self.special_layer_handlers[module_type](module)
                continue
            
            # Initialize weights and biases if present
            if hasattr(module, 'weight') and module.weight is not None:
                self._init_weight(module.weight, name)
            
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _init_weight(self, weight: nn.Parameter, name: str):
        """Initialize a weight parameter"""
        if self.initialization_type == InitializationType.STANDARD:
            # Standard initialization with normal distribution
            nn.init.normal_(weight, mean=0.0, std=0.02)
        
        elif self.initialization_type == InitializationType.XAVIER_UNIFORM:
            # Xavier/Glorot uniform initialization
            nn.init.xavier_uniform_(weight)
        
        elif self.initialization_type == InitializationType.XAVIER_NORMAL:
            # Xavier/Glorot normal initialization
            nn.init.xavier_normal_(weight)
        
        elif self.initialization_type == InitializationType.KAIMING_UNIFORM:
            # Kaiming/He uniform initialization
            nn.init.kaiming_uniform_(weight, nonlinearity='relu')
        
        elif self.initialization_type == InitializationType.KAIMING_NORMAL:
            # Kaiming/He normal initialization
            nn.init.kaiming_normal_(weight, nonlinearity='relu')
        
        elif self.initialization_type == InitializationType.ORTHOGONAL:
            # Orthogonal initialization
            nn.init.orthogonal_(weight, gain=1.0)
        
        elif self.initialization_type == InitializationType.SMALL_NORMAL:
            # Smaller normal distribution
            nn.init.normal_(weight, mean=0.0, std=0.01)
        
        elif self.initialization_type == InitializationType.SMALL_UNIFORM:
            # Smaller uniform distribution
            nn.init.uniform_(weight, a=-0.01, b=0.01)
        
        elif self.initialization_type == InitializationType.RWKV_SPECIFIC:
            # RWKV-specific initialization handled separately
            pass
    
    def _init_layer_norm(self, module: nn.LayerNorm):
        """Initialize LayerNorm layers"""
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    
    def _init_embedding(self, module: nn.Embedding):
        """Initialize embedding layers"""
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _apply_rwkv_initialization(self, model: nn.Module):
        """Apply RWKV-specific initialization"""
        logger.info("Applying RWKV-specific initialization")
        
        for name, module in model.named_modules():
            # Initialize time mixing parameters
            if 'time_mix' in name:
                if hasattr(module, 'weight'):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
            
            # Initialize time decay parameters
            elif 'time_decay' in name:
                if hasattr(module, 'weight'):
                    # Initialize with values that create moderate decay
                    nn.init.uniform_(module.weight, a=-0.5, b=0.5)
                    # Apply softplus for positive values
                    with torch.no_grad():
                        module.weight.copy_(torch.log(1 + torch.exp(module.weight)))
            
            # Initialize attention parameters
            elif 'attention' in name.lower() or 'attn' in name.lower():
                if hasattr(module, 'weight'):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            # Initialize gate parameters
            elif 'gate' in name.lower():
                if hasattr(module, 'weight'):
                    # Initialize close to 1 to start with dominant residual connections
                    nn.init.normal_(module.weight, mean=0.0, std=self.kwargs.get('gate_init', 1e-3))

def initialize_model(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    initialization_type: str = "standard"
) -> nn.Module:
    """
    Initialize model weights.
    
    Args:
        model: The model to initialize
        config: Optional configuration dictionary
        seed: Random seed for initialization
        initialization_type: Type of initialization to use
        
    Returns:
        The initialized model
    """
    # Extract initialization parameters from config if provided
    kwargs = {}
    if config is not None:
        kwargs = {
            'gate_init': config.get('gate_init', 1e-3),
            'rwkv_init_factors': config.get('rwkv_init_factors', {
                'time_mix': 0.1,
                'time_decay': 0.5
            })
        }
    
    # Create initializer and apply initialization
    initializer = ModelInitializer(
        initialization_type=initialization_type,
        seed=seed,
        **kwargs
    )
    
    return initializer.initialize_model(model)

def initialize_optimizers(
    model: nn.Module,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Initialize optimizers and schedulers with error handling"""
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs']
        )

        return {
            'optimizer': optimizer,
            'scheduler': scheduler
        }

    except Exception as e:
        raise ModelError(
            f"Error initializing optimizers: {str(e)}",
            {'original_error': str(e)}
        )

def initialize_model(
    model: nn.Module,
    device: torch.device,
    use_mixed_precision: bool = True,
    gradient_checkpointing: bool = False,
    use_3b_config: bool = False
) -> nn.Module:
    """
    Initialize model for training.
    
    Args:
        model: Model to initialize
        device: Device to move model to
        use_mixed_precision: Whether to use mixed precision
        gradient_checkpointing: Whether to use gradient checkpointing
        use_3b_config: Whether to use 3B parameter configuration
        
    Returns:
        Initialized model
    """
    # Move model to device
    model = model.to(device)
    
    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    # Enable mixed precision if requested
    if use_mixed_precision:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        logger.info("Enabled mixed precision training")
    
    # Apply 3B config if requested
    if use_3b_config:
        apply_3b_config(model)
    
    # Initialize weights
    initialize_weights(model)
    
    return model

def initialize_weights(model: nn.Module) -> None:
    """
    Initialize model weights.
    
    Args:
        model: Model to initialize weights for
    """
    # Initialize linear layers
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Use Xavier initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Initialize layer normalization
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
        # Initialize embeddings
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    logger.info("Initialized model weights")

def apply_3b_config(model: nn.Module) -> None:
    """
    Apply 3B parameter configuration to model.
    
    Args:
        model: Model to apply configuration to
    """
    if hasattr(model, "config"):
        config = model.config
        
        # Update model dimensions
        config.hidden_size = 3072
        config.num_attention_heads = 32
        config.intermediate_size = 12288
        config.num_hidden_layers = 32
        
        # Update other parameters
        config.max_position_embeddings = 2048
        config.attention_probs_dropout_prob = 0.1
        config.hidden_dropout_prob = 0.1
        
        logger.info("Applied 3B parameter configuration")

def setup_tokenizer(
    tokenizer_path: str,
    model_max_length: Optional[int] = None,
    padding_side: str = "right",
    truncation_side: str = "right"
) -> PreTrainedTokenizer:
    """
    Set up tokenizer for training.
    
    Args:
        tokenizer_path: Path to tokenizer
        model_max_length: Maximum sequence length
        padding_side: Side to pad sequences
        truncation_side: Side to truncate sequences
        
    Returns:
        Configured tokenizer
    """
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Configure tokenizer
    if model_max_length is not None:
        tokenizer.model_max_length = model_max_length
    
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side
    
    # Add special tokens if needed
    special_tokens = {
        "pad_token": "[PAD]",
        "eos_token": "[EOS]",
        "bos_token": "[BOS]",
        "unk_token": "[UNK]",
        "mask_token": "[MASK]"
    }
    
    for token_name, token_value in special_tokens.items():
        if getattr(tokenizer, token_name) is None:
            tokenizer.add_special_tokens({token_name: token_value})
    
    logger.info(f"Set up tokenizer with max length {tokenizer.model_max_length}")
    
    return tokenizer

def setup_model_config(
    config_dict: Dict[str, Any],
    use_3b_config: bool = False
) -> Dict[str, Any]:
    """
    Set up model configuration.
    
    Args:
        config_dict: Base configuration dictionary
        use_3b_config: Whether to use 3B parameter configuration
        
    Returns:
        Updated configuration dictionary
    """
    if use_3b_config:
        config_dict.update({
            "hidden_size": 3072,
            "num_attention_heads": 32,
            "intermediate_size": 12288,
            "num_hidden_layers": 32,
            "max_position_embeddings": 2048,
            "attention_probs_dropout_prob": 0.1,
            "hidden_dropout_prob": 0.1
        })
        logger.info("Applied 3B parameter configuration")
    
    return config_dict
