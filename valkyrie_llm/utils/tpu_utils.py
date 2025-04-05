"""
Utility functions for TPU training support.
"""

import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

def setup_tpu_strategy():
    """
    Set up TPU strategy for Kaggle.
    
    Returns:
        tuple: (strategy, is_tpu_available)
            - strategy: TF distribution strategy
            - is_tpu_available: Bool indicating if TPU is available
    """
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment 
        # variable is set. On Kaggle this is always the case.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  
        logger.info(f'Running on TPU: {tpu.master()}')
        
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        
        # Set PyTorch to use TPU
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            import torch_xla.distributed.xla_multiprocessing as xmp
            logger.info("PyTorch XLA libraries loaded successfully")
        except ImportError as e:
            logger.warning(f"Unable to import PyTorch XLA libraries: {e}")
            logger.warning("TPU is available but PyTorch XLA support is missing")
            return strategy, False
        
        return strategy, True
        
    except (ValueError, ImportError) as e:
        logger.warning(f"No TPU detected or TPU libraries not available: {e}")
        # default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()
        return strategy, False
    
def get_tpu_device():
    """
    Get the TPU device for PyTorch.
    
    Returns:
        torch.device: TPU device if available, else CPU
    """
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        return device
    except ImportError:
        import torch
        return torch.device("cpu")
    
def tpu_data_loader(dataloader, device):
    """
    Create a TPU-compatible data loader.
    
    Args:
        dataloader: PyTorch DataLoader
        device: TPU device
    
    Returns:
        TPU-compatible data loader
    """
    try:
        import torch_xla.distributed.parallel_loader as pl
        return pl.ParallelLoader(dataloader, [device]).per_device_loader(device)
    except ImportError:
        return dataloader
    
def save_model_for_tpu(model, path):
    """
    Save a model using TPU-compatible methods.
    
    Args:
        model: PyTorch model
        path: Path to save the model
    """
    try:
        import torch_xla.core.xla_model as xm
        xm.save(model.state_dict(), path)
        logger.info(f"Model saved to {path} using TPU-compatible method")
    except ImportError:
        import torch
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path} using standard PyTorch")
        
def is_tpu_available():
    """
    Check if TPU is available for PyTorch.
    
    Returns:
        bool: True if TPU is available, False otherwise
    """
    try:
        import torch_xla.core.xla_model as xm
        return True
    except ImportError:
        return False 