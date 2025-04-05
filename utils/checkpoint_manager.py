import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging
from datetime import datetime

class CheckpointManager:
    def __init__(self, 
                 base_dir: str,
                 max_checkpoints: int = 5,
                 save_optimizer: bool = True):
        self.base_dir = Path(base_dir)
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.logger = logging.getLogger(__name__)
        self.checkpoints = []
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       metrics: Optional[Dict[str, float]] = None,
                       step: int = 0) -> str:
        """Save model checkpoint with metadata"""
        checkpoint_dir = self.base_dir / f"checkpoint_{step}_{datetime.now():%Y%m%d_%H%M%S}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = checkpoint_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save optimizer and scheduler states
        training_state = {
            "step": step,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat()
        }
        
        if self.save_optimizer and optimizer is not None:
            training_state["optimizer"] = optimizer.state_dict()
            
        if scheduler is not None:
            training_state["scheduler"] = scheduler.state_dict()
            
        # Save metadata
        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(training_state, f, indent=2)
            
        self.checkpoints.append(checkpoint_dir)
        self._cleanup_old_checkpoints()
        
        self.logger.info(f"Saved checkpoint: {checkpoint_dir}")
        return str(checkpoint_dir)
        
    def load_checkpoint(self,
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
        """Load model checkpoint and metadata"""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model state
        model_path = checkpoint_dir / "model.pt"
        model.load_state_dict(torch.load(model_path))
        
        # Load metadata and training state
        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Restore optimizer and scheduler states
        if self.save_optimizer and optimizer is not None and "optimizer" in metadata:
            optimizer.load_state_dict(metadata["optimizer"])
            
        if scheduler is not None and "scheduler" in metadata:
            scheduler.load_state_dict(metadata["scheduler"])
            
        self.logger.info(f"Loaded checkpoint: {checkpoint_dir}")
        return metadata
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints"""
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoints = sorted(self.checkpoints)[:-self.max_checkpoints]
            for checkpoint_dir in old_checkpoints:
                if checkpoint_dir.exists():
                    for file in checkpoint_dir.glob("*"):
                        file.unlink()
                    checkpoint_dir.rmdir()
                self.checkpoints.remove(checkpoint_dir)
