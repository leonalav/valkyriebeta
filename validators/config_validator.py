import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

@dataclass
class ValidationResult:
    """Results of configuration validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class ConfigValidator:
    """Validates configuration settings"""
    logger = logging.getLogger(__name__)
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate training configuration"""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['batch_size', 'learning_rate', 'num_epochs']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
                
        # Validate batch size
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                errors.append(f"batch_size must be a positive integer, got {batch_size}")
                
        # Validate learning rate
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0 or lr >= 1:
                errors.append(f"learning_rate should be between 0 and 1, got {lr}")
                
        # Check memory usage thresholds
        if 'max_memory_usage' in config:
            mem_usage = config['max_memory_usage']
            if torch.cuda.is_available() and mem_usage > 0.95:
                warnings.append(f"High memory usage threshold ({mem_usage})")
                
        # Check for potential data and output paths
        path_keys = ['data_path', 'output_dir', 'checkpoint_dir', 'log_dir']
        for key in path_keys:
            if key in config and not config[key]:
                warnings.append(f"Empty path for {key}")
                
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
        
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate model architecture configuration"""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['hidden_size', 'num_layers', 'num_heads', 'vocab_size']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
                
        # Validate hidden size
        if 'hidden_size' in config:
            if not isinstance(config['hidden_size'], int) or config['hidden_size'] <= 0:
                errors.append(f"hidden_size must be a positive integer, got {config['hidden_size']}")
            elif config['hidden_size'] % 64 != 0:
                warnings.append(f"hidden_size ({config['hidden_size']}) is not a multiple of 64, which may impact performance")
                
        # Validate number of layers
        if 'num_layers' in config:
            if not isinstance(config['num_layers'], int) or config['num_layers'] <= 0:
                errors.append(f"num_layers must be a positive integer, got {config['num_layers']}")
                
        # Validate number of heads
        if 'num_heads' in config and 'hidden_size' in config:
            num_heads = config['num_heads']
            hidden_size = config['hidden_size']
            
            if not isinstance(num_heads, int) or num_heads <= 0:
                errors.append(f"num_heads must be a positive integer, got {num_heads}")
            elif hidden_size % num_heads != 0:
                errors.append(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
                
        # Validate vocab size
        if 'vocab_size' in config:
            if not isinstance(config['vocab_size'], int) or config['vocab_size'] <= 0:
                errors.append(f"vocab_size must be a positive integer, got {config['vocab_size']}")
                
        # Check for memory optimization settings
        memory_opts = ['use_8bit_training', 'use_4bit_quantization', 'mixed_precision']
        for opt in memory_opts:
            if opt in config and config[opt]:
                if opt == 'use_4bit_quantization' and 'use_8bit_training' in config and config['use_8bit_training']:
                    warnings.append("Both 4-bit quantization and 8-bit training are enabled; this might cause compatibility issues")
                    
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
