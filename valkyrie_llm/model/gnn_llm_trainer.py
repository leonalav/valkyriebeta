"""
GNN-LLM integration training module.

This module provides tools for joint training of Large Language Models (LLMs) with
Graph Neural Networks (GNNs) to enhance language models with graph-structured reasoning.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tqdm import tqdm

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    Trainer,
    TrainingArguments,
)

from valkyrie_llm.model.gnn.integration import TransformerGNNIntegration
from valkyrie_llm.model.gnn.training import TrainingConfig


class GNNLLMTrainer:
    """
    Trainer for integrated GNN-LLM models.
    
    This trainer extends the Hugging Face Trainer to handle the joint training of
    LLMs with GNN components, ensuring proper gradient flow and optimization.
    """
    
    def __init__(
        self,
        llm_model: PreTrainedModel,
        gnn_model: nn.Module,
        integration_model: TransformerGNNIntegration,
        training_args: TrainingArguments,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        gnn_config: Optional[TrainingConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize GNN-LLM trainer.
        
        Args:
            llm_model: Pretrained language model
            gnn_model: Graph neural network model
            integration_model: Model integrating LLM and GNN
            training_args: Hugging Face training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            gnn_config: GNN training configuration
            tokenizer: Tokenizer for the language model
            device: Device to train on
        """
        self.llm_model = llm_model
        self.gnn_model = gnn_model
        self.integration_model = integration_model
        self.training_args = training_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.device = device
        
        # Create GNN config if not provided
        if gnn_config is None:
            self.gnn_config = TrainingConfig(
                learning_rate=training_args.learning_rate / 10,  # Lower LR for GNN
                batch_size=training_args.per_device_train_batch_size,
                epochs=int(training_args.num_train_epochs),
                warmup_epochs=int(0.1 * training_args.num_train_epochs),
                lr_scheduler="cosine",
                grad_clip=training_args.max_grad_norm,
                grad_accumulation_steps=training_args.gradient_accumulation_steps,
                optimizer_type="adamw",
                mixed_precision=training_args.fp16,
                save_best_model=True,
                checkpoint_dir=os.path.join(training_args.output_dir, "gnn_checkpoints"),
                log_interval=training_args.logging_steps,
                dropout=training_args.dropout if hasattr(training_args, "dropout") else 0.1,
                seed=training_args.seed,
            )
        else:
            self.gnn_config = gnn_config
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        
        # Prepare models
        self._prepare_models()
        
        # Set up optimizers and schedulers
        self._setup_training()
    
    def _prepare_models(self):
        """Prepare models for training."""
        # Move models to device
        self.llm_model.to(self.device)
        self.gnn_model.to(self.device)
        self.integration_model.to(self.device)
        
        # Verify that the integration model is properly constructed
        # and connected to both LLM and GNN
        self.logger.info("Verifying model integration...")
        self.integration_model.verify_integration(self.llm_model, self.gnn_model)
        
        # Set models to training mode
        self.llm_model.train()
        self.gnn_model.train()
        self.integration_model.train()
    
    def _setup_training(self):
        """Set up optimizers and schedulers."""
        # Separate parameters for different learning rates
        llm_params = list(self.llm_model.parameters())
        gnn_params = list(self.gnn_model.parameters())
        integration_params = list(self.integration_model.parameters())
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': llm_params, 'lr': self.training_args.learning_rate},
            {'params': gnn_params, 'lr': self.gnn_config.learning_rate},
            {'params': integration_params, 'lr': self.training_args.learning_rate * 0.5},
        ]
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.training_args.weight_decay,
        )
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.training_args.dataloader_num_workers,
        )
        
        if self.eval_dataset is not None:
            self.eval_loader = torch.utils.data.DataLoader(
                self.eval_dataset,
                batch_size=self.training_args.per_device_eval_batch_size,
                shuffle=False,
                num_workers=self.training_args.dataloader_num_workers,
            )
        else:
            self.eval_loader = None
        
        # Calculate number of training steps
        num_update_steps_per_epoch = len(self.train_loader) // self.training_args.gradient_accumulation_steps
        num_training_steps = num_update_steps_per_epoch * self.training_args.num_train_epochs
        
        # Create learning rate scheduler
        warmup_steps = int(self.training_args.warmup_ratio * num_training_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        # Set up mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.training_args.fp16 else None
    
    def train(self):
        """Train the integrated GNN-LLM model."""
        self.logger.info("Starting GNN-LLM joint training...")
        
        # Training loop
        global_step = 0
        tr_loss = 0.0
        best_eval_loss = float('inf')
        early_stopping_counter = 0
        
        for epoch in range(int(self.training_args.num_train_epochs)):
            self.logger.info(f"Starting epoch {epoch+1}/{int(self.training_args.num_train_epochs)}")
            
            epoch_loss = 0.0
            epoch_steps = 0
            
            # Training phase
            self.llm_model.train()
            self.gnn_model.train()
            self.integration_model.train()
            
            for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
                # Prepare batch
                batch = self._prepare_batch(batch)
                
                # Forward pass with mixed precision if enabled
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_pass(batch)
                        loss = outputs['loss'] / self.training_args.gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Update weights if gradient accumulation steps reached
                    if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or step == len(self.train_loader) - 1:
                        # Unscale gradients for clipping
                        self.scaler.unscale_(self.optimizer)
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.llm_model.parameters(), self.training_args.max_grad_norm
                        )
                        torch.nn.utils.clip_grad_norm_(
                            self.gnn_model.parameters(), self.gnn_config.grad_clip
                        )
                        
                        # Update weights
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                else:
                    # Standard forward and backward pass
                    outputs = self._forward_pass(batch)
                    loss = outputs['loss'] / self.training_args.gradient_accumulation_steps
                    
                    loss.backward()
                    
                    # Update weights if gradient accumulation steps reached
                    if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or step == len(self.train_loader) - 1:
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(
                            self.llm_model.parameters(), self.training_args.max_grad_norm
                        )
                        torch.nn.utils.clip_grad_norm_(
                            self.gnn_model.parameters(), self.gnn_config.grad_clip
                        )
                        
                        # Update weights
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                
                # Track loss
                tr_loss += loss.item() * self.training_args.gradient_accumulation_steps
                epoch_loss += loss.item() * self.training_args.gradient_accumulation_steps
                epoch_steps += 1
                
                # Log progress at logging steps
                if global_step % self.training_args.logging_steps == 0:
                    self.logger.info(f"Step {global_step}: loss = {tr_loss / global_step:.6f}, lr = {self.scheduler.get_last_lr()[0]:.8f}")
            
            # Compute average epoch loss
            avg_epoch_loss = epoch_loss / epoch_steps
            self.logger.info(f"Epoch {epoch+1} completed with average loss: {avg_epoch_loss:.6f}")
            
            # Evaluation phase
            if self.eval_loader is not None:
                eval_loss, eval_results = self.evaluate()
                
                # Log evaluation results
                self.logger.info(f"Evaluation results: loss = {eval_loss:.6f}, perplexity = {eval_results.get('perplexity', 'N/A')}")
                
                # Check for improvement and early stopping
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stopping_counter = 0
                    
                    # Save best model
                    if self.training_args.save_strategy == "epoch":
                        self._save_checkpoint("best_model")
                else:
                    early_stopping_counter += 1
                
                # Early stopping
                if self.training_args.early_stopping_patience > 0 and early_stopping_counter >= self.training_args.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Save checkpoint at save_steps or at the end of epoch based on strategy
            if self.training_args.save_strategy == "epoch" or (self.training_args.save_strategy == "steps" and global_step % self.training_args.save_steps == 0):
                self._save_checkpoint(f"checkpoint-{global_step}")
        
        # Save final model
        self._save_checkpoint("final_model")
        
        return TrainOutput(
            global_step=global_step,
            training_loss=tr_loss / global_step,
            best_eval_loss=best_eval_loss,
        )
    
    def evaluate(self):
        """Evaluate the integrated GNN-LLM model."""
        if self.eval_loader is None:
            self.logger.warning("No evaluation dataset provided. Skipping evaluation.")
            return None, {}
        
        self.logger.info("Starting evaluation...")
        
        # Set models to evaluation mode
        self.llm_model.eval()
        self.gnn_model.eval()
        self.integration_model.eval()
        
        eval_loss = 0.0
        eval_steps = 0
        
        # Collect outputs for metrics
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluation"):
                # Prepare batch
                batch = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self._forward_pass(batch)
                
                # Track loss
                eval_loss += outputs['loss'].item()
                eval_steps += 1
                
                # Collect predictions and labels if relevant
                if 'predictions' in outputs and 'labels' in batch:
                    all_preds.append(outputs['predictions'].cpu())
                    all_labels.append(batch['labels'].cpu())
        
        # Compute average evaluation loss
        avg_eval_loss = eval_loss / eval_steps
        
        # Compute perplexity if relevant
        results = {
            'loss': avg_eval_loss,
        }
        
        if outputs.get('loss_type') == 'language_modeling':
            results['perplexity'] = torch.exp(torch.tensor(avg_eval_loss)).item()
        
        # Compute additional metrics if predictions and labels are available
        if all_preds and all_labels:
            # Concatenate predictions and labels
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Compute accuracy
            if all_preds.dim() > 1 and all_preds.size(1) > 1:
                # Classification with multiple classes
                preds = torch.argmax(all_preds, dim=1)
                correct = (preds == all_labels).sum().item()
                results['accuracy'] = correct / len(all_labels)
        
        return avg_eval_loss, results
    
    def _prepare_batch(self, batch):
        """
        Prepare batch for model input.
        
        This method moves all tensors to the device and organizes inputs
        for the integrated model.
        """
        # Convert batch to dictionary format if it's not already
        if not isinstance(batch, dict):
            if isinstance(batch, torch.utils.data.dataloader._DatasetKind):
                # PyTorch DataLoader batch
                batch = {k: v for k, v in batch.items()}
            else:
                # Handle other batch formats (tuple, list, etc.)
                if hasattr(self.train_dataset, 'column_names'):
                    # Hugging Face dataset
                    batch = {k: v for k, v in zip(self.train_dataset.column_names, batch)}
                else:
                    raise ValueError("Unrecognized batch format")
        
        # Move all tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        
        return batch
    
    def _forward_pass(self, batch):
        """
        Perform forward pass through the integrated model.
        
        This method handles the joint forward pass through LLM and GNN components,
        extracting text inputs for the LLM and graph inputs for the GNN.
        """
        # Extract inputs for LLM
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        
        # Extract inputs for GNN
        node_features = batch.get('node_features')
        edge_index = batch.get('edge_index')
        edge_attr = batch.get('edge_attr')
        batch_indices = batch.get('batch', None)  # Node batch assignments
        
        # Get LLM embeddings
        with torch.no_grad():
            llm_outputs = self.llm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            # Get the hidden states from the last layer
            transformer_features = llm_outputs.last_hidden_state
        
        # Run integrated forward pass
        outputs = self.integration_model(
            transformer_features=transformer_features,
            transformer_mask=attention_mask,
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch_indices,
            return_contrastive_loss=True if self.gnn_config.contrastive_loss_weight > 0 else False
        )
        
        # Compute loss if not already provided by the model
        if 'loss' not in outputs:
            if 'labels' in batch:
                # Get predictions from the integration model output
                logits = outputs.get('transformer_outputs', None)
                
                if logits is None and 'graph_outputs' in outputs:
                    # Use graph outputs if transformer outputs not available
                    logits = outputs['graph_outputs']
                
                if logits is not None:
                    # Compute language modeling loss if logits are available
                    if input_ids is not None and input_ids.dim() > 1:
                        # Shift logits and labels for language modeling
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = input_ids[..., 1:].contiguous()
                        
                        # Compute loss
                        loss_fct = torch.nn.CrossEntropyLoss()
                        loss = loss_fct(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                    else:
                        # Standard classification loss
                        loss_fct = torch.nn.CrossEntropyLoss()
                        loss = loss_fct(
                            logits.view(-1, logits.size(-1)),
                            batch['labels'].view(-1)
                        )
                    
                    outputs['loss'] = loss
                    
            # Add contrastive loss if available
            if 'contrastive_loss' in outputs and self.gnn_config.contrastive_loss_weight > 0:
                if 'loss' in outputs:
                    outputs['loss'] += self.gnn_config.contrastive_loss_weight * outputs['contrastive_loss']
                else:
                    outputs['loss'] = self.gnn_config.contrastive_loss_weight * outputs['contrastive_loss']
        
        return outputs
    
    def _save_checkpoint(self, checkpoint_name):
        """Save model checkpoint."""
        # Create output directory if it doesn't exist
        if not os.path.exists(self.training_args.output_dir):
            os.makedirs(self.training_args.output_dir)
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.training_args.output_dir, checkpoint_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # Save LLM model
        self.llm_model.save_pretrained(checkpoint_dir)
        
        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save GNN model
        torch.save(self.gnn_model.state_dict(), os.path.join(checkpoint_dir, "gnn_model.pt"))
        
        # Save integration model
        torch.save(self.integration_model.state_dict(), os.path.join(checkpoint_dir, "integration_model.pt"))
        
        # Save optimizer and scheduler states
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }, os.path.join(checkpoint_dir, "optimizer.pt"))
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")


class TrainOutput:
    """Output of GNN-LLM training."""
    
    def __init__(self, global_step, training_loss, best_eval_loss=None):
        self.global_step = global_step
        self.training_loss = training_loss
        self.best_eval_loss = best_eval_loss 