import traceback
import logging
from typing import Dict, Any, List
import torch
import os
import sys

logger = logging.getLogger(__name__)

class TrainingError(Exception):
    """Base class for training-related errors"""
    pass

class DataError(TrainingError):
    """Data loading and preprocessing errors"""
    pass

class ModelError(TrainingError):
    """Model architecture and initialization errors"""
    pass

class ResourceError(TrainingError):
    """Resource (GPU/Memory) related errors"""
    pass

class ConfigError(TrainingError):
    """Configuration validation errors"""
    pass

def handle_training_error(e: Exception) -> Dict[str, Any]:
    """Central error handler that returns recovery instructions"""
    logger.error(f"Training error encountered: {type(e).__name__}: {str(e)}")
    
    error_info = {
        'error_type': type(e).__name__,
        'message': str(e),
        'traceback': traceback.format_exc(),
        'recovery_steps': [],
        'severity': 'high'
    }

    # Handle specific error types with tailored recovery steps
    if isinstance(e, torch.cuda.OutOfMemoryError):
        error_info['recovery_steps'] = [
            'Reduce batch size in configuration',
            'Enable gradient checkpointing in model config',
            'Use mixed precision training',
            'Reduce model size (fewer layers or smaller hidden dimension)',
            'Clear CUDA cache before training',
            'Use memory efficient attention mechanisms',
            'Implement activation checkpointing'
        ]
        error_info['severity'] = 'critical'
        
    elif isinstance(e, DataError):
        error_info['recovery_steps'] = [
            'Verify data paths and permissions',
            'Check data format matches expected schema',
            'Validate tokenizer configuration',
            'Ensure data files exist and are readable',
            'Check for corrupted data samples',
            'Verify dataset splitting logic'
        ]
        
    elif isinstance(e, ModelError):
        error_info['recovery_steps'] = [
            'Check model architecture configuration',
            'Verify parameter initialization',
            'Validate optimizer settings',
            'Ensure all required modules are implemented',
            'Check for dimension mismatches between layers',
            'Verify loss function implementation'
        ]
        
    elif isinstance(e, ResourceError):
        error_info['recovery_steps'] = [
            'Check GPU availability and drivers',
            'Monitor system resources during training',
            'Free up system memory before training',
            'Check for other processes using GPU resources',
            'Consider distributed training across multiple GPUs'
        ]
        
    elif isinstance(e, ConfigError):
        error_info['recovery_steps'] = [
            'Validate configuration against schema',
            'Check for incorrect parameter types or values',
            'Ensure all required config fields are provided',
            'Verify configuration compatibility with hardware'
        ]
        
    elif isinstance(e, RuntimeError):
        # Handle specific PyTorch runtime errors
        if "size mismatch" in str(e).lower():
            error_info['recovery_steps'] = [
                'Check tensor dimensions in model forward pass',
                'Verify input and target shapes match expectations',
                'Ensure consistent dimensions across pipeline stages',
                'Check for incorrect reshaping operations'
            ]
        elif "not implemented for" in str(e).lower():
            error_info['recovery_steps'] = [
                'Check for operations not supported by current device/dtype',
                'Verify tensor types are compatible with operations',
                'Consider switching to CPU for unsupported operations'
            ]
        else:
            error_info['recovery_steps'] = [
                'Check stack trace for error location',
                'Verify model implementation',
                'Check for numerical instability in computations',
                'Add debugging statements around error location'
            ]
            
    elif isinstance(e, ValueError) or isinstance(e, TypeError):
        error_info['recovery_steps'] = [
            'Check function arguments for correct types and values',
            'Verify input data formatting and preprocessing',
            'Ensure configuration values are within valid ranges',
            'Add validation steps before problematic code sections'
        ]
        
    elif isinstance(e, ImportError) or isinstance(e, ModuleNotFoundError):
        error_info['recovery_steps'] = [
            'Install missing dependencies',
            'Check virtual environment activation',
            'Verify package versions match requirements',
            'Check import paths and module structure'
        ]
        
    elif isinstance(e, FileNotFoundError) or isinstance(e, PermissionError):
        error_info['recovery_steps'] = [
            'Verify file paths and permissions',
            'Check that necessary directories exist',
            'Ensure config paths are correct for the environment',
            'Verify data and checkpoint directories are accessible'
        ]
        
    else:
        # Fallback for unknown error types
        error_info['recovery_steps'] = [
            'Review the error message and traceback',
            'Check logs for earlier warnings or errors',
            'Verify system environment and resources',
            'Add additional logging around error location',
            'Consider filing a bug report with the traceback'
        ]

    # Log the recovery steps
    logger.info("Suggested recovery steps:")
    for i, step in enumerate(error_info['recovery_steps'], 1):
        logger.info(f"  {i}. {step}")

    return error_info
