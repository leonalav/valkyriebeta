import logging
import os
import math
import time
from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class TrainingEngine:
    """
    Engine for training ValkyrieLLM models.
    
    Handles the training loop, gradient updates, scheduling, logging,
    validation, checkpointing, and other training-related tasks.
    """
    
    def __init__(self, model, optimizer=None, lr_scheduler=None, training_config=None, tokenizer=None):
        """
        Initialize the training engine.
        
        Args:
            model: The model to train
            optimizer: Optional optimizer (will be created if None)
            lr_scheduler: Optional learning rate scheduler
            training_config: Training configuration
            tokenizer: Tokenizer for processing text
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_config = training_config
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Set up metrics tracking
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rates': [],
            'grad_norms': []
        }
        
        # Set up training steps tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Initialize optimizer if not provided
        if self.optimizer is None and self.training_config is not None:
            self._setup_optimizer()
        
        # Initialize scheduler if not provided
        if self.lr_scheduler is None and self.optimizer is not None:
            self._setup_scheduler()
        
        # Set up gradient accumulation
        if self.training_config is not None:
            self.gradient_accumulation_steps = getattr(
                self.training_config, 'gradient_accumulation_steps', 1)
        else:
            self.gradient_accumulation_steps = 1
        
        # Set up mixed precision training
        self.use_mixed_precision = False
        if self.training_config is not None and getattr(self.training_config, 'use_mixed_precision', False):
            self.use_mixed_precision = True
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Set up gradient clipping
        self.max_grad_norm = 1.0
        if self.training_config is not None:
            self.max_grad_norm = getattr(self.training_config, 'max_grad_norm', 1.0)
    
    def _setup_optimizer(self):
        """Set up optimizer based on training configuration"""
        from torch.optim import AdamW
        
        # Extract parameters for optimizer
        lr = getattr(self.training_config, 'learning_rate', 5e-5)
        weight_decay = getattr(self.training_config, 'weight_decay', 0.01)
        
        # Group parameters by decay/no-decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                         if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                         if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Create optimizer
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            eps=getattr(self.training_config, 'adam_epsilon', 1e-8)
        )
        
        logger.info(f"Created optimizer with lr={lr}, weight_decay={weight_decay}")
    
    def _setup_scheduler(self):
        """Set up learning rate scheduler based on training configuration"""
        # Get total steps
        if not hasattr(self, 'total_steps'):
            self.total_steps = self._estimate_training_steps()
        
        warmup_steps = getattr(self.training_config, 'warmup_steps', 0)
        
        # Create scheduler with linear warmup and linear decay
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(self.total_steps - current_step) / float(max(1, self.total_steps - warmup_steps))
            )
        
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        logger.info(f"Created LR scheduler with warmup_steps={warmup_steps}, total_steps={self.total_steps}")
    
    def _estimate_training_steps(self):
        """Estimate total training steps based on configuration"""
        if not hasattr(self.training_config, 'train_batch_size') or not hasattr(self.training_config, 'num_epochs'):
            return 1000  # Default fallback value
        
        if hasattr(self.training_config, 'total_training_steps'):
            return self.training_config.total_training_steps
        
        train_batch_size = self.training_config.train_batch_size * self.gradient_accumulation_steps
        num_examples = getattr(self.training_config, 'num_train_examples', 100000)
        num_epochs = self.training_config.num_epochs
        
        return (num_examples // train_batch_size) * num_epochs
    
    def train_epoch(self, train_dataloader, epoch=None):
        """
        Train the model for one epoch.
        
        Args:
            train_dataloader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        if epoch is not None:
            self.current_epoch = epoch
        
        # Set model to training mode
        self.model.train()
        
        # Set up progress bar
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        # Initialize metrics for this epoch
        running_loss = 0.0
        step = 0
        
        # Train for one epoch
        for batch_idx, batch in enumerate(epoch_iterator):
            # Process batch
            loss = self._training_step(batch)
            
            # Update metrics
            running_loss += loss.item()
            step += 1
            
            # Update progress bar
            epoch_iterator.set_postfix(loss=running_loss / step)
        
        # Calculate epoch metrics
        avg_loss = running_loss / step
        self.metrics['train_loss'].append(avg_loss)
        
        logger.info(f"Epoch {self.current_epoch} - Avg Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _training_step(self, batch):
        """
        Perform a single training step (forward + backward + optimize).
        
        Args:
            batch: The current batch of data
            
        Returns:
            loss: The training loss for this step
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision if enabled
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Update weights if gradient accumulation steps reached
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # LR scheduler step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Track learning rate
                self.metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        else:
            # Standard forward pass
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if gradient accumulation steps reached
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                if self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm)
                    self.metrics['grad_norms'].append(grad_norm.item())
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # LR scheduler step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # Track learning rate
                self.metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        # Update global step
        self.global_step += 1
        
        return loss * self.gradient_accumulation_steps  # Return unscaled loss for metrics
    
    def evaluate(self, eval_dataloader):
        """
        Evaluate the model on the evaluation dataset.
        
        Args:
            eval_dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        eval_loss = 0.0
        eval_steps = 0
        
        # Evaluate model
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass without gradients
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            # Update metrics
            eval_loss += loss.item()
            eval_steps += 1
        
        # Calculate average loss
        avg_eval_loss = eval_loss / eval_steps
        self.metrics['eval_loss'].append(avg_eval_loss)
        
        # Check if this is the best model so far
        if avg_eval_loss < self.best_loss:
            self.best_loss = avg_eval_loss
        
        logger.info(f"Evaluation - Loss: {avg_eval_loss:.4f}")
        
        # Return metrics
        return {
            "loss": avg_eval_loss,
            "perplexity": math.exp(avg_eval_loss)
        }
    
    def save_checkpoint(self, output_dir, save_optimizer=True, save_lr_scheduler=True):
        """
        Save a model checkpoint.
        
        Args:
            output_dir: Directory to save the checkpoint
            save_optimizer: Whether to save optimizer state
            save_lr_scheduler: Whether to save scheduler state
            
        Returns:
            Path to the saved checkpoint
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "metrics": self.metrics,
            "best_loss": self.best_loss
        }
        
        # Add optimizer state if requested
        if save_optimizer and self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        
        # Add scheduler state if requested
        if save_lr_scheduler and self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save configuration
        if hasattr(self.model, 'config'):
            model_config_path = os.path.join(output_dir, "config.json")
            with open(model_config_path, 'w') as f:
                if hasattr(self.model.config, 'to_json_string'):
                    f.write(self.model.config.to_json_string())
                elif hasattr(self.model.config, '__dict__'):
                    import json
                    json.dump(self.model.config.__dict__, f, indent=2)
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            tokenizer_path = os.path.join(output_dir, "tokenizer")
            if hasattr(self.tokenizer, 'save_pretrained'):
                self.tokenizer.save_pretrained(tokenizer_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path, load_optimizer=True, load_lr_scheduler=True):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
            load_optimizer: Whether to load optimizer state
            load_lr_scheduler: Whether to load scheduler state
            
        Returns:
            Loaded checkpoint data
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state if requested
        if load_optimizer and "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if requested
        if load_lr_scheduler and "scheduler_state_dict" in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)
        self.metrics = checkpoint.get("metrics", self.metrics)
        self.best_loss = checkpoint.get("best_loss", float('inf'))
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.current_epoch}, step {self.global_step})")
        
        return checkpoint
    
    def prepare_for_inference(self, optimize_for_rwkv=False):
        """
        Prepare the model for inference.
        
        Args:
            optimize_for_rwkv: Whether to apply RWKV-specific optimizations
            
        Returns:
            Model prepared for inference
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Apply RWKV-specific optimizations if requested
        if optimize_for_rwkv:
            if hasattr(self.model, 'rwkv_layer_indices'):
                logger.info("Applying RWKV-specific inference optimizations")
                
                # Enable state reuse for efficient inference
                if hasattr(self.model, 'enable_state_reuse'):
                    self.model.enable_state_reuse()
                
                # Set chunk size
                chunk_size = getattr(self.training_config, 'rwkv_chunk_size', 1024)
                if hasattr(self.model, 'chunk_size'):
                    self.model.chunk_size = chunk_size
        
        return self.model
    
    # RWKV-specific methods
    def setup_rwkv_optimizer(self, rwkv_lr_multiplier=1.0, att_weight_decay=0.0, ffn_weight_decay=0.1):
        """
        Set up RWKV-specific optimizer with different learning rates and weight decays.
        
        Args:
            rwkv_lr_multiplier: Multiplier for RWKV layer learning rate
            att_weight_decay: Weight decay for attention weights
            ffn_weight_decay: Weight decay for feed-forward weights
            
        Returns:
            RWKV-specific optimizer
        """
        from torch.optim import AdamW
        
        # Extract base parameters
        lr = getattr(self.training_config, 'learning_rate', 5e-5)
        weight_decay = getattr(self.training_config, 'weight_decay', 0.01)
        
        # Group parameters by type
        no_decay = ["bias", "LayerNorm.weight"]
        rwkv_att_params = []
        rwkv_ffn_params = []
        other_params = []
        
        # Categorize parameters
        for name, param in self.model.named_parameters():
            # RWKV attention parameters
            if "rwkv" in name.lower() and any(att_type in name.lower() for att_type in ["att", "time", "key", "value"]):
                if not any(nd in name for nd in no_decay):
                    rwkv_att_params.append((name, param, att_weight_decay))
                else:
                    rwkv_att_params.append((name, param, 0.0))
            # RWKV feed-forward parameters
            elif "rwkv" in name.lower() and any(ffn_type in name.lower() for ffn_type in ["ffn", "feed", "channel"]):
                if not any(nd in name for nd in no_decay):
                    rwkv_ffn_params.append((name, param, ffn_weight_decay))
                else:
                    rwkv_ffn_params.append((name, param, 0.0))
            # Other parameters
            else:
                if not any(nd in name for nd in no_decay):
                    other_params.append((name, param, weight_decay))
                else:
                    other_params.append((name, param, 0.0))
        
        # Create parameter groups
        optimizer_grouped_parameters = [
            {
                "params": [p for _, p, _ in other_params],
                "weight_decay": weight_decay,
                "lr": lr
            },
            {
                "params": [p for _, p, _ in rwkv_att_params],
                "weight_decay": att_weight_decay,
                "lr": lr * rwkv_lr_multiplier
            },
            {
                "params": [p for _, p, _ in rwkv_ffn_params],
                "weight_decay": ffn_weight_decay,
                "lr": lr * rwkv_lr_multiplier
            }
        ]
        
        # Create optimizer
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            eps=getattr(self.training_config, 'adam_epsilon', 1e-8)
        )
        
        logger.info(f"Created RWKV-specific optimizer with rwkv_lr_multiplier={rwkv_lr_multiplier}")
        
        return self.optimizer
    
    def optimize_rwkv_training_setup(self):
        """
        Apply RWKV-specific training optimizations.
        
        Returns:
            The optimized model
        """
        # Check if the model has RWKV layers
        if not hasattr(self.model, 'rwkv_layer_indices'):
            logger.warning("Model does not have RWKV layers, skipping RWKV optimizations")
            return self.model
        
        logger.info("Applying RWKV-specific training optimizations")
        
        # Set chunk size for chunked processing
        chunk_size = getattr(self.training_config, 'rwkv_chunk_size', 1024)
        if hasattr(self.model, 'chunk_size'):
            self.model.chunk_size = chunk_size
            logger.info(f"Set RWKV chunk size to {chunk_size}")
        
        # Apply memory optimizations
        if hasattr(self.model, 'optimize_memory_usage'):
            self.model.optimize_memory_usage()
            logger.info("Applied RWKV memory optimizations")
        
        # Apply computational optimizations if available
        try:
            from valkyrie_llm.training.components import RWKVIntegrator
            rwkv_integrator = RWKVIntegrator(self.model)
            self.model = rwkv_integrator.apply_rwkv_optimizations()
            logger.info("Applied RWKV computational optimizations")
        except (ImportError, AttributeError):
            logger.warning("RWKV computational optimizations not available")
        
        return self.model
    
    def evaluate_sequence_modeling(self, test_dataloader, max_seq_len=None, chunk_size=None):
        """
        Evaluate sequence modeling performance (particularly important for RWKV models).
        
        Args:
            test_dataloader: DataLoader for test data
            max_seq_len: Maximum sequence length for evaluation
            chunk_size: Chunk size for processing sequences
            
        Returns:
            Dictionary of sequence modeling metrics
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Set up sequence modeling
        if max_seq_len is None:
            max_seq_len = getattr(self.training_config, 'max_seq_len', 1024)
        
        if chunk_size is None and hasattr(self.model, 'chunk_size'):
            chunk_size = self.model.chunk_size
        elif chunk_size is None:
            chunk_size = 1024
        
        # Initialize metrics
        total_tokens = 0
        total_loss = 0.0
        correct_next_tokens = 0
        
        # Process test data
        for batch in tqdm(test_dataloader, desc="Evaluating sequence modeling"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            
            # Process in chunks for efficiency
            batch_size, seq_len = input_ids.shape
            
            # Forward pass without gradients
            with torch.no_grad():
                for i in range(0, seq_len - 1, chunk_size):
                    # Get chunk bounds
                    chunk_start = i
                    chunk_end = min(i + chunk_size, seq_len - 1)
                    
                    # Get input and target for this chunk
                    chunk_input = input_ids[:, chunk_start:chunk_end]
                    chunk_target = input_ids[:, chunk_start + 1:chunk_end + 1]
                    
                    # Get attention mask for this chunk if available
                    chunk_mask = None
                    if attention_mask is not None:
                        chunk_mask = attention_mask[:, chunk_start:chunk_end]
                    
                    # Forward pass for this chunk
                    outputs = self.model(
                        input_ids=chunk_input,
                        attention_mask=chunk_mask,
                        use_cache=True
                    )
                    
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs[1]
                    
                    # Calculate loss
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = chunk_target
                    
                    loss = F.cross_entropy(
                        shift_logits.view(-1, self.model.config.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                    
                    # Update metrics
                    total_loss += loss.item() * shift_labels.numel()
                    total_tokens += shift_labels.numel()
                    
                    # Calculate next token accuracy
                    predictions = torch.argmax(shift_logits, dim=-1)
                    correct = (predictions == shift_labels).sum().item()
                    correct_next_tokens += correct
        
        # Calculate overall metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss)
        accuracy = correct_next_tokens / total_tokens if total_tokens > 0 else 0.0
        
        logger.info(f"Sequence modeling - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.4f}")
        
        # Return metrics
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "next_token_accuracy": accuracy
        } 