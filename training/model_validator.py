from typing import Type, Any, Dict, List, Set
import torch.nn as nn
import inspect
from dataclasses import fields
from collections import defaultdict
import torch
import logging
import math
import os

logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Validates model configurations and setup to ensure they are correct
    before training begins.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_model_setup(self, model: nn.Module, device: torch.device) -> bool:
        """
        Validate the model setup and raise warnings or errors if issues are found.
        
        Args:
            model: The model to validate
            device: The device the model will run on
            
        Returns:
            True if no critical errors, False otherwise
        """
        # Perform various validation checks
        self._check_model_parameters(model)
        self._check_device_compatibility(model, device)
        self._check_memory_requirements(model, device)
        self._check_model_configuration(model)
        self._check_initialization(model)
        
        # Log all warnings
        for warning in self.validation_warnings:
            logger.warning(f"Validation Warning: {warning}")
        
        # Log all errors
        for error in self.validation_errors:
            logger.error(f"Validation Error: {error}")
        
        # Return True if no critical errors
        return len(self.validation_errors) == 0
    
    def _check_model_parameters(self, model: nn.Module):
        """Check model parameter count and setup"""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Log parameter counts
        logger.info(f"Model has {total_params:,} parameters ({trainable_params:,} trainable)")
        
        # Check if any parameters are not initialized (contain NaN)
        has_nan = any(torch.isnan(p).any() for p in model.parameters())
        if has_nan:
            self.validation_errors.append("Model contains NaN parameters. Check initialization.")
        
        # Check if any parameters are not trainable when they should be
        if trainable_params == 0:
            self.validation_errors.append("No trainable parameters found in model.")
        
        # Check parameter efficiency
        if trainable_params / total_params < 0.5:
            self.validation_warnings.append(
                f"Only {trainable_params / total_params:.1%} of parameters are trainable. "
                "Consider unfreezing more parameters for better training."
            )
    
    def _check_device_compatibility(self, model: nn.Module, device: torch.device):
        """Check if model is compatible with the device"""
        # Check if CUDA is available when device is cuda
        if device.type == 'cuda' and not torch.cuda.is_available():
            self.validation_errors.append("CUDA device requested but CUDA is not available.")
            return
        
        # Check if model can be moved to device
        try:
            # Try to move one parameter to device
            next(model.parameters()).to(device)
        except Exception as e:
            self.validation_errors.append(f"Failed to move model to {device}: {str(e)}")
    
    def _check_memory_requirements(self, model: nn.Module, device: torch.device):
        """Estimate memory requirements of the model"""
        if device.type != 'cuda':
            return
        
        # Get model size in bytes (parameters + buffers)
        param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_bytes = param_size_bytes + buffer_size_bytes
        
        # Convert to GB
        model_size_gb = model_size_bytes / (1024 ** 3)
        
        # Get available GPU memory
        available_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        
        # Logging
        logger.info(f"Model size: {model_size_gb:.2f} GB")
        logger.info(f"Available GPU memory: {available_memory_gb:.2f} GB")
        
        # Estimate memory for training (rough estimate: 3x model size for forward/backward/optimizer)
        estimated_training_memory_gb = model_size_gb * 3
        
        # Check if model will fit in memory
        if estimated_training_memory_gb > available_memory_gb * 0.8:  # 80% of available memory
            self.validation_warnings.append(
                f"Model may not fit in GPU memory. "
                f"Estimated training memory: {estimated_training_memory_gb:.2f} GB, "
                f"Available memory: {available_memory_gb:.2f} GB. "
                "Consider enabling memory optimizations like gradient checkpointing."
            )
        
        # Check if batch size is reasonable
        batch_size = self.config.get('batch_size', 1)
        seq_len = self.config.get('max_seq_length', 2048)
        
        # Estimate memory per sample
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            hidden_size = model.config.hidden_size
        else:
            hidden_size = 2048  # Default estimate
        
        # Rough estimate per token: 12 bytes per hidden dimension for activations
        bytes_per_token = hidden_size * 12
        memory_per_sample_mb = (bytes_per_token * seq_len) / (1024 ** 2)
        
        # Check if batch size is too large
        if memory_per_sample_mb * batch_size > available_memory_gb * 1024 * 0.4:  # 40% of available memory
            self.validation_warnings.append(
                f"Batch size {batch_size} may be too large for the GPU memory. "
                f"Estimated memory per sample: {memory_per_sample_mb:.2f} MB. "
                "Consider reducing batch size or using gradient accumulation."
            )
    
    def _check_model_configuration(self, model: nn.Module):
        """Check if model configuration is valid"""
        # Check RWKV-specific configurations
        if hasattr(model, 'rwkv_layer_indices') and hasattr(model, 'layers'):
            # Check if RWKV layer indices are valid
            if max(model.rwkv_layer_indices, default=0) >= len(model.layers):
                self.validation_errors.append(
                    f"RWKV layer indices contain invalid values. "
                    f"Max index: {max(model.rwkv_layer_indices)}, "
                    f"Number of layers: {len(model.layers)}"
                )
        
        # Check sequence length limits
        max_seq_length = self.config.get('max_seq_length', 2048)
        if hasattr(model, 'config') and hasattr(model.config, 'max_position_embeddings'):
            model_max_seq_length = model.config.max_position_embeddings
            if max_seq_length > model_max_seq_length:
                self.validation_warnings.append(
                    f"Configured sequence length ({max_seq_length}) exceeds "
                    f"model's maximum position embeddings ({model_max_seq_length}). "
                    "This may cause unexpected behavior."
                )
        
        # Check for hybrid model compatibility
        if 'hybrid' in str(type(model)).lower():
            # Check transformer/RWKV balance
            if hasattr(model, 'rwkv_layer_indices') and hasattr(model, 'num_layers'):
                rwkv_ratio = len(model.rwkv_layer_indices) / model.num_layers
                if rwkv_ratio < 0.2 or rwkv_ratio > 0.8:
                    self.validation_warnings.append(
                        f"Unusual RWKV/Transformer ratio: {rwkv_ratio:.2f}. "
                        "Consider a more balanced configuration for hybrid models."
                    )
    
    def _check_initialization(self, model: nn.Module):
        """Check model initialization"""
        # Check for missing methods
        if not hasattr(model, 'forward'):
            self.validation_errors.append("Model does not have a forward method.")
        
        # Check for missing attributes
        if hasattr(model, 'rwkv_layer_indices') and not hasattr(model, 'state_dimensions'):
            self.validation_warnings.append(
                "RWKV model missing state_dimensions attribute. "
                "This may cause issues with stateful processing."
            )
        
        # Check for gradient checkpointing capability
        if not hasattr(model, 'enable_gradient_checkpointing') and not hasattr(model, 'gradient_checkpointing_enable'):
            self.validation_warnings.append(
                "Model does not support gradient checkpointing. "
                "This may limit training with long sequences."
            )

def validate_model_setup(
    model: nn.Module,
    config: Dict[str, Any],
    device: torch.device
) -> bool:
    """
    Validate model setup before training.
    
    Args:
        model: The model to validate
        config: Training configuration
        device: Device to use for training
        
    Returns:
        True if validation passed with no critical errors, False otherwise
    """
    validator = ModelValidator(config)
    return validator.validate_model_setup(model, device)

def validate_model(model: nn.Module, config: Any) -> Dict[str, List[str]]:
    """Comprehensive model validation returning all issues"""
    validation_results = defaultdict(list)
    
    # 1. Validate model attributes
    required_attributes = ModelValidator._get_required_attributes(model)
    missing_attributes = []
    for attr in required_attributes:
        if not hasattr(model, attr):
            # Clean up attribute names that contain code fragments
            clean_attr = attr.split(',')[0].split('(')[0].strip()
            if clean_attr.isidentifier():  # Only add valid Python identifiers
                missing_attributes.append(clean_attr)
    if missing_attributes:
        validation_results['missing_attributes'] = sorted(set(missing_attributes))
    
    # 2. Validate config fields
    config_fields = {f.name for f in fields(config)}
    referenced_configs = ModelValidator._find_config_references(model)
    missing_configs = referenced_configs - config_fields
    if missing_configs:
        validation_results['missing_configs'] = list(missing_configs)
    
    # 3. Validate model methods
    required_methods = {'forward'}  # Only require forward by default
    if hasattr(model, 'tree_lstm'):
        required_methods.add('cell')
    if hasattr(model, 'attention'):
        required_methods.add('attention_forward')
    missing_methods = []
    for method in required_methods:
        if not hasattr(model, method):
            missing_methods.append(method)
        elif not callable(getattr(model, method)):
            missing_methods.append(f"{method} (not callable)")
    if missing_methods:
        validation_results['missing_methods'] = missing_methods
    
    # 4. Validate submodules
    invalid_submodules = ModelValidator._validate_submodules_with_report(model)
    if invalid_submodules:
        validation_results['invalid_submodules'] = invalid_submodules
    
    # 5. Report all found attributes, methods, and configs for reference
    validation_results['found_attributes'] = list(ModelValidator._get_all_attributes(model))
    validation_results['found_methods'] = list(ModelValidator._get_all_methods(model))
    validation_results['found_configs'] = list(referenced_configs)
    
    # Format the results before returning
    formatted_results = {}
    for category, items in validation_results.items():
        # Sort and deduplicate items
        formatted_results[category] = sorted(set(items))
    
    # Print the formatted report
    print(ModelValidator._format_validation_report(formatted_results))
    
    return formatted_results

def _get_all_attributes(model: nn.Module) -> Set[str]:
    """Get all attributes in model"""
    return {name for name, _ in inspect.getmembers(model) 
            if not name.startswith('_') and not callable(getattr(model, name))}

def _get_all_methods(model: nn.Module) -> Set[str]:
    """Get all methods in model"""
    return {name for name, _ in inspect.getmembers(model) 
            if not name.startswith('_') and callable(getattr(model, name))}

def _validate_submodules_with_report(model: nn.Module) -> List[str]:
    """Validate submodules and return list of issues"""
    issues = []
    for name, module in model.named_children():
        if not isinstance(module, nn.Module):
            issues.append(f"{name}: Invalid type {type(module)}")
        elif hasattr(module, 'forward') and not callable(module.forward):
            issues.append(f"{name}: Invalid forward method")
    return issues

def _get_required_attributes(model: nn.Module) -> set:
    """Extract required attributes from model code more accurately"""
    attributes = set()
    source = inspect.getsource(model.__class__)
    
    for line in source.split('\n'):
        if 'self.' in line and '=' in line:
            # More precise parsing
            parts = line.split('self.')[1].split('=')[0].strip()
            # Remove any trailing parentheses or arguments
            attr = parts.split('(')[0].strip()
            if attr and not attr.startswith('_'):
                attributes.add(attr)
    
    return attributes

def _find_config_references(model: nn.Module) -> set:
    """Find config references more accurately"""
    config_refs = set()
    source = inspect.getsource(model.__class__)
    
    for line in source.split('\n'):
        if 'config.' in line:
            # More precise parsing
            for part in line.split('config.')[1:]:
                # Extract clean attribute name
                attr = part.split()[0].strip('(),.[]{}').split('(')[0]
                if attr.isidentifier():  # Ensure it's a valid Python identifier
                    config_refs.add(attr)
    
    return config_refs

def _validate_submodules(model: nn.Module) -> None:
    """Validate all submodules recursively"""
    for name, module in model.named_children():
        if not isinstance(module, nn.Module):
            raise TypeError(f"Invalid submodule type for {name}: {type(module)}")
        
        # Check if submodule has required methods
        if hasattr(module, 'forward') and not callable(module.forward):
            raise AttributeError(f"Submodule {name} has invalid forward method")

def _format_validation_report(validation_results: Dict[str, List[str]]) -> str:
    """Format validation results more cleanly"""
    report = ["\n=== Model Validation Report ===\n"]
    
    # Sort categories for consistent output
    for category in sorted(validation_results.keys()):
        if not validation_results[category]:  # Skip empty categories
            continue
        
        title = category.replace('_', ' ').title()
        report.append(f"{title}:")
        # Sort items for consistent output
        for item in sorted(validation_results[category]):
            report.append(f"  - {item}")
        report.append("")
    
    report.append("=============================")
    return "\n".join(report) 
    