"""
Advanced checkpointing functionality for large language models.
Provides differential checkpointing and other optimizations.
"""

import os
import torch
import logging
import time
import json
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import asdict
import math
import hashlib
try:
    import zstandard as zstd
except ImportError:
    zstd = None
try:
    import lz4.frame
except ImportError:
    lz4 = None

logger = logging.getLogger(__name__)

class DifferentialCheckpointer:
    """
    Differential checkpointing that tracks and saves only changed model parts.
    Reduces storage usage and I/O during training.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        interval: int = 500,
        base_path: str = "checkpoints",
        compression: str = "zstd",  # "none", "zstd", "lz4"
        keep_last_n: int = 3,
        include_optimizer: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize differential checkpointer.
        
        Args:
            model: Model to checkpoint
            interval: Steps between checkpoints
            base_path: Base directory for checkpoints
            compression: Compression algorithm to use
            keep_last_n: Number of recent checkpoints to keep
            include_optimizer: Whether to include optimizer state
            metadata: Additional metadata to save with checkpoints
        """
        self.model = model
        self.interval = interval
        self.base_path = base_path
        self.compression = compression
        self.keep_last_n = keep_last_n
        self.include_optimizer = include_optimizer
        self.metadata = metadata or {}
        
        # Track previous state dict for differential comparison
        self.prev_state = None
        self.step = 0
        self.checkpoint_history = []
        
        # Create base directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        # Write metadata file
        metadata_path = os.path.join(base_path, "checkpoint_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump({
                "compression": compression,
                "interval": interval,
                "include_optimizer": include_optimizer,
                "custom_metadata": self.metadata
            }, f, indent=2)
        
        logger.info(f"Initialized differential checkpointer with compression: {compression}")
    
    def _get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get model state dict, handling distributed wrappers."""
        if hasattr(self.model, 'module'):
            # Handle DDP/FSDP wrapped models
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        
        return state_dict
    
    def _get_optimizer_state(self) -> Optional[Dict[str, Any]]:
        """Get optimizer state if available and configured."""
        if not self.include_optimizer:
            return None
        
        if hasattr(self.model, 'optimizer'):
            return self.model.optimizer.state_dict()
        
        return None
    
    def _compute_tensor_hash(self, tensor: torch.Tensor) -> str:
        """Compute hash of tensor data for change detection."""
        # Convert to CPU and contiguous memory layout
        tensor_cpu = tensor.detach().cpu().contiguous()
        
        # Compute hash of tensor data
        tensor_bytes = tensor_cpu.numpy().tobytes()
        return hashlib.md5(tensor_bytes).hexdigest()
    
    def _find_changed_tensors(
        self, 
        current_state: Dict[str, torch.Tensor]
    ) -> Set[str]:
        """Identify tensors that have changed since last checkpoint."""
        if self.prev_state is None:
            # First checkpoint, all tensors are new
            return set(current_state.keys())
        
        changed_keys = set()
        
        # Check each tensor for changes
        for key, tensor in current_state.items():
            if key not in self.prev_state:
                # New tensor
                changed_keys.add(key)
                continue
            
            # Check if tensor has changed
            current_hash = self._compute_tensor_hash(tensor)
            prev_hash = self._compute_tensor_hash(self.prev_state[key])
            
            if current_hash != prev_hash:
                changed_keys.add(key)
        
        # Also track removed tensors
        removed_keys = set(self.prev_state.keys()) - set(current_state.keys())
        
        return changed_keys
    
    def _compress_tensor(self, tensor: torch.Tensor) -> bytes:
        """Compress tensor data using configured compression algorithm."""
        # Convert to CPU and contiguous memory layout
        tensor_cpu = tensor.detach().cpu().contiguous()
        
        # Convert to bytes
        tensor_bytes = tensor_cpu.numpy().tobytes()
        
        # Compress bytes
        if self.compression == "zstd" and zstd is not None:
            return zstd.compress(tensor_bytes, level=3)
        elif self.compression == "lz4" and lz4 is not None:
            return lz4.frame.compress(tensor_bytes)
        else:
            # No compression or compression library not available
            return tensor_bytes
    
    def _decompress_tensor(self, compressed_bytes: bytes, tensor_info: Dict[str, Any]) -> torch.Tensor:
        """Decompress tensor data and reconstruct tensor."""
        # Decompress bytes
        if self.compression == "zstd" and zstd is not None:
            tensor_bytes = zstd.decompress(compressed_bytes)
        elif self.compression == "lz4" and lz4 is not None:
            tensor_bytes = lz4.frame.decompress(compressed_bytes)
        else:
            # No compression or compression library not available
            tensor_bytes = compressed_bytes
        
        # Reconstruct tensor
        import numpy as np
        array = np.frombuffer(tensor_bytes, dtype=np.dtype(tensor_info["dtype"]))
        array = array.reshape(tensor_info["shape"])
        
        # Convert to tensor
        tensor = torch.from_numpy(array.copy())
        
        return tensor
    
    def save_checkpoint(self, step: int, metrics: Optional[Dict[str, Any]] = None):
        """Save a differential checkpoint at the given step."""
        # Skip if not at checkpoint interval
        if step % self.interval != 0:
            return
        
        self.step = step
        
        # Get current state
        current_state = self._get_state_dict()
        optimizer_state = self._get_optimizer_state()
        
        # Find changed tensors
        changed_keys = self._find_changed_tensors(current_state)
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.base_path, f"checkpoint_{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save checkpoint index
        index = {
            "step": step,
            "timestamp": time.time(),
            "tensors": {},
            "metrics": metrics or {},
            "optimizer": optimizer_state is not None,
            "changed_keys": list(changed_keys),
            "removed_keys": list(set(self.prev_state.keys()) - set(current_state.keys())) if self.prev_state else []
        }
        
        # Save changed tensors
        for key in changed_keys:
            tensor = current_state[key]
            
            # Save tensor metadata
            tensor_info = {
                "dtype": str(tensor.dtype).split(".")[-1],
                "shape": tensor.shape,
                "device": str(tensor.device),
                "requires_grad": tensor.requires_grad
            }
            
            index["tensors"][key] = tensor_info
            
            # Save compressed tensor data
            compressed_data = self._compress_tensor(tensor)
            
            # Split large tensors if needed (for filesystem limits)
            max_chunk_size = 512 * 1024 * 1024  # 512 MB
            if len(compressed_data) > max_chunk_size:
                num_chunks = math.ceil(len(compressed_data) / max_chunk_size)
                tensor_info["chunks"] = num_chunks
                
                for i in range(num_chunks):
                    chunk_start = i * max_chunk_size
                    chunk_end = min((i + 1) * max_chunk_size, len(compressed_data))
                    chunk_data = compressed_data[chunk_start:chunk_end]
                    
                    with open(os.path.join(checkpoint_dir, f"{key.replace('.', '_')}_{i}.bin"), "wb") as f:
                        f.write(chunk_data)
            else:
                # Save in single file
                with open(os.path.join(checkpoint_dir, f"{key.replace('.', '_')}.bin"), "wb") as f:
                    f.write(compressed_data)
        
        # Save optimizer state if needed
        if optimizer_state is not None:
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            torch.save(optimizer_state, optimizer_path)
        
        # Save index file
        with open(os.path.join(checkpoint_dir, "index.json"), "w") as f:
            json.dump(index, f, indent=2)
        
        # Update checkpoint history
        self.checkpoint_history.append(step)
        
        # Update previous state for next differential checkpoint
        self.prev_state = current_state
        
        # Prune old checkpoints if needed
        self._prune_old_checkpoints()
        
        logger.info(f"Saved differential checkpoint at step {step} with {len(changed_keys)} changed tensors")
    
    def _prune_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if len(self.checkpoint_history) <= self.keep_last_n:
            return
        
        # Keep the most recent N checkpoints
        checkpoints_to_remove = self.checkpoint_history[:-self.keep_last_n]
        self.checkpoint_history = self.checkpoint_history[-self.keep_last_n:]
        
        # Remove checkpoint directories
        for step in checkpoints_to_remove:
            checkpoint_dir = os.path.join(self.base_path, f"checkpoint_{step}")
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
        
        logger.info(f"Pruned {len(checkpoints_to_remove)} old checkpoints")
    
    def load_checkpoint(self, step: Optional[int] = None):
        """
        Load a checkpoint into the model.
        
        Args:
            step: Specific step to load, or None for latest
        """
        # Find checkpoint to load
        if step is None:
            # Load latest checkpoint
            available_steps = [int(d.split("_")[-1]) for d in os.listdir(self.base_path) 
                              if d.startswith("checkpoint_")]
            if not available_steps:
                logger.warning("No checkpoints found to load")
                return False
            
            step = max(available_steps)
        
        checkpoint_dir = os.path.join(self.base_path, f"checkpoint_{step}")
        if not os.path.exists(checkpoint_dir):
            logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
            return False
        
        # Load index file
        index_path = os.path.join(checkpoint_dir, "index.json")
        if not os.path.exists(index_path):
            logger.error(f"Checkpoint index not found: {index_path}")
            return False
        
        with open(index_path, "r") as f:
            index = json.load(f)
        
        # Reconstruct state dict
        state_dict = {}
        
        # Current model state to start with
        if self.prev_state is not None:
            state_dict.update(self.prev_state)
        else:
            # First load, need to create a clean slate
            state_dict = self._get_state_dict()
        
        # Remove any keys that were removed in checkpoint
        for key in index.get("removed_keys", []):
            if key in state_dict:
                del state_dict[key]
        
        # Load changed tensors
        for key, tensor_info in index["tensors"].items():
            # Handle multi-chunk tensors
            if "chunks" in tensor_info:
                num_chunks = tensor_info["chunks"]
                compressed_data = bytearray()
                
                for i in range(num_chunks):
                    chunk_path = os.path.join(checkpoint_dir, f"{key.replace('.', '_')}_{i}.bin")
                    with open(chunk_path, "rb") as f:
                        compressed_data.extend(f.read())
            else:
                # Single file tensor
                tensor_path = os.path.join(checkpoint_dir, f"{key.replace('.', '_')}.bin")
                with open(tensor_path, "rb") as f:
                    compressed_data = f.read()
            
            # Decompress and reconstruct tensor
            tensor = self._decompress_tensor(compressed_data, tensor_info)
            
            # Add to state dict
            state_dict[key] = tensor
        
        # Load state dict into model
        if hasattr(self.model, 'module'):
            # Handle DDP/FSDP wrapped models
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        
        # Load optimizer state if available
        if index["optimizer"] and self.include_optimizer:
            optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
            if os.path.exists(optimizer_path) and hasattr(self.model, 'optimizer'):
                self.model.optimizer.load_state_dict(torch.load(optimizer_path))
        
        # Update internal state
        self.prev_state = state_dict
        self.step = index["step"]
        
        logger.info(f"Loaded checkpoint from step {step} with {len(index['tensors'])} tensors")
        return True 