"""
Training utilities for GNN models.

This module provides training utilities and helper functions for training 
GNN models with proper configurations and learning rate scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Type

import numpy as np
import time
import logging
from tqdm import tqdm

from .integration import ModelRegistry


class TrainingConfig:
    """
    Configuration for GNN model training.
    
    This class encapsulates training parameters and configuration options
    for training GNN models with proper settings for stability and convergence.
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 32,
        epochs: int = 100,
        warmup_epochs: int = 10,
        lr_scheduler: str = "cosine",  # "cosine" or "one_cycle"
        grad_clip: float = 1.0,
        grad_accumulation_steps: int = 1,
        early_stopping_patience: int = 15,
        optimizer_type: str = "adam",  # "adam", "adamw", "sgd"
        mixed_precision: bool = False,
        save_best_model: bool = True,
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
        contrastive_loss_weight: float = 0.1,
        dropout: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize training configuration.
        
        Args:
            learning_rate: Base learning rate
            weight_decay: Weight decay (L2 regularization)
            batch_size: Batch size for training
            epochs: Number of training epochs
            warmup_epochs: Number of warmup epochs for learning rate
            lr_scheduler: Learning rate scheduler type
            grad_clip: Gradient clipping value
            grad_accumulation_steps: Number of gradient accumulation steps
            early_stopping_patience: Patience for early stopping
            optimizer_type: Type of optimizer to use
            mixed_precision: Whether to use mixed precision training
            save_best_model: Whether to save the best model
            checkpoint_dir: Directory for saving checkpoints
            log_interval: Interval for logging training progress
            contrastive_loss_weight: Weight for contrastive loss
            dropout: Dropout rate
            seed: Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.lr_scheduler = lr_scheduler
        self.grad_clip = grad_clip
        self.grad_accumulation_steps = grad_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.optimizer_type = optimizer_type
        self.mixed_precision = mixed_precision
        self.save_best_model = save_best_model
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.contrastive_loss_weight = contrastive_loss_weight
        self.dropout = dropout
        self.seed = seed
    
    def set_seed(self):
        """Set random seed for reproducibility."""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class GNNTrainer:
    """
    Trainer for GNN models.
    
    This class provides utilities for training GNN models with proper
    learning rate scheduling, gradient clipping, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize GNN trainer.
        
        Args:
            model: GNN model to train
            config: Training configuration
            train_loader: Data loader for training data
            val_loader: Data loader for validation data
            test_loader: Data loader for test data
            device: Device to train on
            logger: Logger for training logs
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Setup logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
        else:
            self.logger = logger
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = float('-inf')
        self.patience_counter = 0
        
        # Set random seed
        self.config.set_seed()
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        params = self.model.parameters()
        
        if self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        
        if self.config.lr_scheduler.lower() == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=1e-6
            )
        elif self.config.lr_scheduler.lower() == "one_cycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=warmup_steps / total_steps if warmup_steps > 0 else 0.3
            )
        else:
            self.logger.warning(f"Unknown scheduler type: {self.config.lr_scheduler}")
            return None
    
    def train(
        self,
        criterion: Callable,
        metric_fn: Optional[Callable] = None,
        optimize_metric: bool = False,
        additional_loss_terms: Optional[List[Callable]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            criterion: Loss function
            metric_fn: Function to compute evaluation metric
            optimize_metric: Whether to optimize for metric instead of loss
            additional_loss_terms: List of additional loss terms to add
            
        Returns:
            Dictionary containing training history
        """
        self.logger.info(f"Starting training on {self.device}...")
        self.logger.info(f"Training configuration: {self.config.to_dict()}")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        if metric_fn is not None:
            history['train_metric'] = []
            history['val_metric'] = []
        
        # Training loop
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_metric = self._train_epoch(
                criterion, metric_fn, additional_loss_terms
            )
            
            # Validation phase
            if self.val_loader is not None:
                val_loss, val_metric = self._validate(criterion, metric_fn)
                
                # Update history
                history['val_loss'].append(val_loss)
                if metric_fn is not None:
                    history['val_metric'].append(val_metric)
                
                # Check for early stopping and model saving
                improved = self._check_improvement(
                    val_loss, val_metric, optimize_metric
                )
                
                if improved and self.config.save_best_model:
                    self._save_checkpoint("best_model.pt")
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info("Early stopping triggered!")
                    break
            
            # Update history
            history['train_loss'].append(train_loss)
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            if metric_fn is not None:
                history['train_metric'].append(train_metric)
            
            # Log progress
            if (epoch + 1) % self.config.log_interval == 0:
                self._log_progress(epoch, train_loss, train_metric, val_loss, val_metric)
            
            # Save checkpoint at intervals
            if (epoch + 1) % 10 == 0 and self.config.save_best_model:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        # Save final model
        if self.config.save_best_model:
            self._save_checkpoint("final_model.pt")
        
        self.logger.info("Training completed!")
        
        return history
    
    def _train_epoch(
        self,
        criterion: Callable,
        metric_fn: Optional[Callable] = None,
        additional_loss_terms: Optional[List[Callable]] = None,
    ) -> Tuple[float, Optional[float]]:
        """
        Train for one epoch.
        
        Args:
            criterion: Loss function
            metric_fn: Function to compute evaluation metric
            additional_loss_terms: List of additional loss terms to add
            
        Returns:
            Average loss and metric for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_metric = 0.0 if metric_fn is not None else None
        
        # Apply learning rate warmup manually if needed
        if self.config.warmup_epochs > 0 and self.current_epoch < self.config.warmup_epochs:
            warmup_factor = (self.current_epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * warmup_factor
        
        # Training loop
        steps = 0
        self.optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")):
            steps += 1
            
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass with mixed precision if enabled
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = criterion(outputs, batch)
                    
                    # Add additional loss terms if provided
                    if additional_loss_terms is not None:
                        for loss_term in additional_loss_terms:
                            term_loss = loss_term(outputs, batch)
                            loss = loss + term_loss
                    
                    # Add contrastive loss if present in outputs
                    if 'contrastive_loss' in outputs:
                        loss = loss + self.config.contrastive_loss_weight * outputs['contrastive_loss']
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss / self.config.grad_accumulation_steps).backward()
                
                # Gradient accumulation
                if (i + 1) % self.config.grad_accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                    # Unscale gradients for clipping
                    self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    if self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                    
                    # Update weights with scaled gradients
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update learning rate scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()
            else:
                # Standard forward pass without mixed precision
                outputs = self.model(**batch)
                loss = criterion(outputs, batch)
                
                # Add additional loss terms if provided
                if additional_loss_terms is not None:
                    for loss_term in additional_loss_terms:
                        term_loss = loss_term(outputs, batch)
                        loss = loss + term_loss
                
                # Add contrastive loss if present in outputs
                if 'contrastive_loss' in outputs:
                    loss = loss + self.config.contrastive_loss_weight * outputs['contrastive_loss']
                
                # Backward pass
                (loss / self.config.grad_accumulation_steps).backward()
                
                # Gradient accumulation
                if (i + 1) % self.config.grad_accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                    # Gradient clipping
                    if self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )
                    
                    # Update weights
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Update learning rate scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()
            
            # Compute metrics if provided
            if metric_fn is not None:
                metric = metric_fn(outputs, batch)
                total_metric += metric.item() if isinstance(metric, torch.Tensor) else metric
            
            # Accumulate loss
            total_loss += loss.item()
        
        # Compute average loss and metric
        avg_loss = total_loss / len(self.train_loader)
        avg_metric = total_metric / len(self.train_loader) if metric_fn is not None else None
        
        return avg_loss, avg_metric
    
    def _validate(
        self,
        criterion: Callable,
        metric_fn: Optional[Callable] = None,
    ) -> Tuple[float, Optional[float]]:
        """
        Validate the model.
        
        Args:
            criterion: Loss function
            metric_fn: Function to compute evaluation metric
            
        Returns:
            Average validation loss and metric
        """
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0 if metric_fn is not None else None
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = self._batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(**batch)
                loss = criterion(outputs, batch)
                
                # Compute metrics if provided
                if metric_fn is not None:
                    metric = metric_fn(outputs, batch)
                    total_metric += metric.item() if isinstance(metric, torch.Tensor) else metric
                
                # Accumulate loss
                total_loss += loss.item()
        
        # Compute average loss and metric
        avg_loss = total_loss / len(self.val_loader)
        avg_metric = total_metric / len(self.val_loader) if metric_fn is not None else None
        
        return avg_loss, avg_metric
    
    def test(
        self,
        criterion: Callable,
        metric_fn: Optional[Callable] = None,
    ) -> Tuple[float, Optional[float]]:
        """
        Test the model.
        
        Args:
            criterion: Loss function
            metric_fn: Function to compute evaluation metric
            
        Returns:
            Average test loss and metric
        """
        if self.test_loader is None:
            self.logger.warning("No test loader provided. Skipping testing.")
            return None, None
        
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0 if metric_fn is not None else None
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Move batch to device
                batch = self._batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(**batch)
                loss = criterion(outputs, batch)
                
                # Compute metrics if provided
                if metric_fn is not None:
                    metric = metric_fn(outputs, batch)
                    total_metric += metric.item() if isinstance(metric, torch.Tensor) else metric
                
                # Accumulate loss
                total_loss += loss.item()
        
        # Compute average loss and metric
        avg_loss = total_loss / len(self.test_loader)
        avg_metric = total_metric / len(self.test_loader) if metric_fn is not None else None
        
        self.logger.info(f"Test loss: {avg_loss:.4f}, Test metric: {avg_metric:.4f}")
        
        return avg_loss, avg_metric
    
    def _check_improvement(
        self,
        val_loss: float,
        val_metric: Optional[float],
        optimize_metric: bool,
    ) -> bool:
        """
        Check if the model improved and update patience counter.
        
        Args:
            val_loss: Validation loss
            val_metric: Validation metric
            optimize_metric: Whether to optimize for metric instead of loss
            
        Returns:
            Whether the model improved
        """
        improved = False
        
        if optimize_metric and val_metric is not None:
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.patience_counter = 0
                improved = True
            else:
                self.patience_counter += 1
        else:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                improved = True
            else:
                self.patience_counter += 1
        
        return improved
    
    def _log_progress(
        self,
        epoch: int,
        train_loss: float,
        train_metric: Optional[float],
        val_loss: Optional[float],
        val_metric: Optional[float],
    ):
        """Log training progress."""
        log_message = f"Epoch {epoch+1}/{self.config.epochs} - "
        log_message += f"Train loss: {train_loss:.4f}"
        
        if train_metric is not None:
            log_message += f", Train metric: {train_metric:.4f}"
        
        if val_loss is not None:
            log_message += f", Val loss: {val_loss:.4f}"
        
        if val_metric is not None:
            log_message += f", Val metric: {val_metric:.4f}"
        
        log_message += f", LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        
        self.logger.info(log_message)
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        import os
        
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metric = checkpoint.get('best_val_metric', float('-inf'))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        return batch


class IntegratedGNNTrainer(GNNTrainer):
    """
    Trainer for models that integrate GNNs with transformers.
    
    This extended trainer handles the complexities of training integrated 
    GNN-transformer models, including handling different learning rates
    for different components.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        logger: Optional[logging.Logger] = None,
        transformer_lr_factor: float = 0.1,  # Lower learning rate for transformer
    ):
        """
        Initialize integrated GNN-transformer trainer.
        
        Args:
            model: Integrated GNN-transformer model
            config: Training configuration
            train_loader: Data loader for training data
            val_loader: Data loader for validation data
            test_loader: Data loader for test data
            device: Device to train on
            logger: Logger for training logs
            transformer_lr_factor: Factor to scale transformer learning rate
        """
        self.transformer_lr_factor = transformer_lr_factor
        
        # Call parent constructor
        super().__init__(
            model, config, train_loader, val_loader, test_loader, device, logger
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with different learning rates for GNN and transformer components.
        
        This method separates parameters into transformer and GNN groups and applies
        different learning rates to each group.
        """
        # Identify transformer and GNN parameters
        transformer_params = []
        gnn_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'transformer' in name.lower():
                transformer_params.append(param)
            elif 'gnn' in name.lower() or 'graph' in name.lower():
                gnn_params.append(param)
            else:
                other_params.append(param)
        
        # Define parameter groups with different learning rates
        param_groups = [
            {'params': transformer_params, 'lr': self.config.learning_rate * self.transformer_lr_factor},
            {'params': gnn_params, 'lr': self.config.learning_rate},
            {'params': other_params, 'lr': self.config.learning_rate}
        ]
        
        # Create optimizer based on configuration
        if self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                param_groups,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                param_groups,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")


def create_gnn_model(
    model_name: str,
    model_config: Dict[str, Any],
    training_config: Optional[TrainingConfig] = None,
) -> nn.Module:
    """
    Create a GNN model with the given configuration.
    
    Args:
        model_name: Name of the model to create
        model_config: Configuration for the model
        training_config: Training configuration (for dropout and other training params)
        
    Returns:
        Instantiated GNN model
    """
    # Apply training config dropout if provided
    if training_config is not None:
        model_config['dropout'] = training_config.dropout
    
    # Create model using registry
    return ModelRegistry.create_model(model_name, **model_config) 