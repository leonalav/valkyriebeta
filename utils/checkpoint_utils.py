"""
Utility module for managing PyTorch checkpoint functionality consistently across the codebase.
"""

import torch
import logging
from typing import Callable, Any, Tuple, List

logger = logging.getLogger(__name__)

# Import the checkpoint functionality
try:
    from torch.utils.checkpoint import checkpoint, checkpoint_sequential
    HAS_CHECKPOINT = True
except ImportError:
    logger.warning("torch.utils.checkpoint not available, checkpointing will be disabled")
    HAS_CHECKPOINT = False
    
    # Define dummy functions for fallback
    def checkpoint(function, *args, **kwargs):
        return function(*args, **kwargs)
        
    def checkpoint_sequential(functions, segments, input, **kwargs):
        # Simple fallback that just applies functions sequentially
        result = input
        for f in functions:
            result = f(result)
        return result

def apply_checkpoint(function: Callable, *args, use_checkpoint: bool = True, **kwargs) -> Any:
    """
    Apply checkpoint to a function if checkpointing is enabled.
    
    Args:
        function: Function to checkpoint
        *args: Arguments to pass to the function
        use_checkpoint: Whether to use checkpointing (can be disabled for debugging)
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function
    """
    if use_checkpoint and HAS_CHECKPOINT:
        return checkpoint(function, *args, **kwargs)
    else:
        return function(*args, **kwargs)
        
def apply_checkpoint_sequential(
    functions: List[Callable],
    segments: int,
    input: torch.Tensor,
    use_checkpoint: bool = True,
    **kwargs
) -> torch.Tensor:
    """
    Apply sequential checkpointing to a list of functions.
    
    Args:
        functions: List of functions to checkpoint
        segments: Number of segments to split the functions into
        input: Input tensor
        use_checkpoint: Whether to use checkpointing (can be disabled for debugging)
        **kwargs: Keyword arguments to pass to checkpoint_sequential
        
    Returns:
        Result after applying all functions
    """
    if use_checkpoint and HAS_CHECKPOINT:
        return checkpoint_sequential(functions, segments, input, **kwargs)
    else:
        # Fallback implementation
        result = input
        for f in functions:
            result = f(result)
        return result 