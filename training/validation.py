from typing import Dict, Any, Optional, List, Tuple
import torch
from dataclasses import dataclass
from .exceptions import ConfigError, ModelError
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm.auto import tqdm
import math

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    is_valid: bool
    errors: list
    warnings: list

class ConfigValidator:
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []

        # Required fields
        required = ['batch_size', 'learning_rate', 'num_epochs', 'model_name']
        for field in required:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Numeric validation
        if 'batch_size' in config and config['batch_size'] < 1:
            errors.append("batch_size must be positive")
        
        if 'learning_rate' in config and not (0 < config['learning_rate'] < 1):
            errors.append("learning_rate must be between 0 and 1")

        # Memory estimation
        if 'batch_size' in config and 'max_length' in config:
            estimated_memory = (
                config['batch_size'] * 
                config['max_length'] * 
                2 * 4  # Assuming float32 and factor of 2 for gradients
            )
            if estimated_memory > torch.cuda.get_device_properties(0).total_memory * 0.9:
                warnings.append("Batch size may be too large for available GPU memory")

        # Add validation for new memory-related configs
        if 'save_format' in config:
            if config['save_format'] not in ['pytorch', 'safetensors']:
                errors.append(f"Invalid save_format: {config['save_format']}")

        if 'memory_config' in config:
            if not (0 < config['memory_config'].max_memory_usage <= 1.0):
                errors.append("max_memory_usage must be between 0 and 1")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

class ModelValidator:
    @staticmethod
    def validate_model(model: torch.nn.Module) -> ValidationResult:
        errors = []
        warnings = []

        # Check model device
        if not next(model.parameters()).is_cuda and torch.cuda.is_available():
            warnings.append("Model is on CPU but CUDA is available")

        # Check parameter initialization
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                errors.append(f"NaN values found in {name}")
            if torch.isinf(param).any():
                errors.append(f"Inf values found in {name}")

        # Check gradient requirements
        for name, param in model.named_parameters():
            if not param.requires_grad:
                warnings.append(f"Parameter {name} has requires_grad=False")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

def validate_model(
    model: nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
    use_mixed_precision: bool = True,
    max_eval_steps: Optional[int] = None,
    compute_perplexity: bool = True,
    compute_metrics: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    Validate the model.
    
    Args:
        model: Model to validate
        val_dataloader: DataLoader for validation data
        device: Device to use
        use_mixed_precision: Whether to use mixed precision
        max_eval_steps: Maximum number of evaluation steps
        compute_perplexity: Whether to compute perplexity
        compute_metrics: Whether to compute additional metrics
        
    Returns:
        Tuple of (average loss, metrics dictionary)
    """
    model.eval()
    total_loss = 0
    total_steps = 0
    metrics = {}
    
    # Set up progress bar
    progress_bar = tqdm(total=len(val_dataloader), desc="Validating")
    
    try:
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                # Move batch to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass with mixed precision if enabled
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(**batch)
                        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                else:
                    outputs = model(**batch)
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                # Update metrics
                total_loss += loss.item()
                total_steps += 1
                
                # Track RAG metrics if available
                if isinstance(outputs, dict) and "retrieval_scores" in outputs:
                    if "rag_metrics" not in metrics:
                        metrics["rag_metrics"] = {
                            "avg_retrieval_score": 0,
                            "avg_verification_score": 0,
                            "count": 0
                        }
                    metrics["rag_metrics"]["avg_retrieval_score"] += outputs["retrieval_scores"].mean().item()
                    if "verification_scores" in outputs:
                        metrics["rag_metrics"]["avg_verification_score"] += outputs["verification_scores"].mean().item()
                    metrics["rag_metrics"]["count"] += 1
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # Check if we've reached max steps
                if max_eval_steps and step >= max_eval_steps:
                    break
        
        # Calculate average loss
        avg_loss = total_loss / total_steps
        metrics["loss"] = avg_loss
        
        # Compute perplexity if requested
        if compute_perplexity:
            metrics["perplexity"] = math.exp(avg_loss)
            
        # Calculate RAG metrics if available
        if "rag_metrics" in metrics and metrics["rag_metrics"]["count"] > 0:
            metrics["avg_retrieval_score"] = (
                metrics["rag_metrics"]["avg_retrieval_score"] / 
                metrics["rag_metrics"]["count"]
            )
            if metrics["rag_metrics"]["avg_verification_score"] > 0:
                metrics["avg_verification_score"] = (
                    metrics["rag_metrics"]["avg_verification_score"] / 
                    metrics["rag_metrics"]["count"]
                )
            del metrics["rag_metrics"]
        
        # Compute additional metrics if requested
        if compute_metrics:
            # Get logits and labels from last batch
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
            labels = batch["labels"]
            
            # Calculate accuracy
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten logits and labels
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Calculate accuracy
            predictions = torch.argmax(shift_logits, dim=-1)
            correct = (predictions == shift_labels).sum().item()
            total = (shift_labels != -100).sum().item()
            
            metrics["accuracy"] = correct / total if total > 0 else 0.0
        
        logger.info(f"Validation completed. Loss: {avg_loss:.4f}")
        if compute_perplexity:
            logger.info(f"Perplexity: {metrics['perplexity']:.2f}")
        if compute_metrics:
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        
        return avg_loss, metrics
        
    finally:
        progress_bar.close()
        model.train()

def validate_model_setup(
    model: nn.Module,
    training_config: Dict[str, Any],
    device: torch.device
) -> Dict[str, List[str]]:
    """
    Validate model setup and configuration.
    
    Args:
        model: Model to validate
        training_config: Training configuration
        device: Device to use
        
    Returns:
        Dictionary containing validation results
    """
    results = {
        "errors": [],
        "warnings": []
    }
    
    # Check model device
    if next(model.parameters()).device != device:
        results["errors"].append(f"Model is on {next(model.parameters()).device} but should be on {device}")
    
    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if trainable_params == 0:
        results["errors"].append("No trainable parameters found in model")
    else:
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Check learning rate
    lr = training_config.get("learning_rate", 1e-4)
    if lr <= 0:
        results["errors"].append(f"Invalid learning rate: {lr}")
    elif lr > 1:
        results["warnings"].append(f"High learning rate: {lr}")
    
    # Check batch size
    batch_size = training_config.get("batch_size", 8)
    if batch_size <= 0:
        results["errors"].append(f"Invalid batch size: {batch_size}")
    
    # Check gradient accumulation steps
    grad_accum_steps = training_config.get("gradient_accumulation_steps", 1)
    if grad_accum_steps <= 0:
        results["errors"].append(f"Invalid gradient accumulation steps: {grad_accum_steps}")
    
    # Check max gradient norm
    max_grad_norm = training_config.get("max_grad_norm", 1.0)
    if max_grad_norm <= 0:
        results["errors"].append(f"Invalid max gradient norm: {max_grad_norm}")
    
    # Check warmup steps
    warmup_steps = training_config.get("warmup_steps", 1000)
    if warmup_steps < 0:
        results["errors"].append(f"Invalid warmup steps: {warmup_steps}")
    
    # Check model configuration
    if hasattr(model, "config"):
        config = model.config
        
        # Check vocabulary size
        if hasattr(config, "vocab_size"):
            if config.vocab_size <= 0:
                results["errors"].append(f"Invalid vocabulary size: {config.vocab_size}")
        
        # Check hidden size
        if hasattr(config, "hidden_size"):
            if config.hidden_size <= 0:
                results["errors"].append(f"Invalid hidden size: {config.hidden_size}")
        
        # Check number of layers
        if hasattr(config, "num_hidden_layers"):
            if config.num_hidden_layers <= 0:
                results["errors"].append(f"Invalid number of layers: {config.num_hidden_layers}")
        
        # Check number of attention heads
        if hasattr(config, "num_attention_heads"):
            if config.num_attention_heads <= 0:
                results["errors"].append(f"Invalid number of attention heads: {config.num_attention_heads}")
    
    return results
