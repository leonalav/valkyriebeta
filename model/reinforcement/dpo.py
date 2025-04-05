import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tqdm import tqdm
import time
import os
import json
from dataclasses import dataclass

from .config import RLConfig
from ..transformer import TransformerModel

logger = logging.getLogger(__name__)

class DPOTrainer:
    """
    Direct Preference Optimization (DPO) trainer for language models.
    
    This implementation follows the DPO algorithm as described in:
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    by Rafailov et al.
    
    DPO directly optimizes a policy to align with human preferences without explicitly
    modeling a reward function, making it more efficient than traditional RLHF approaches.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: RLConfig,
        reference_model: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Create reference model (frozen)
        if reference_model is not None:
            self.reference_model = reference_model
        else:
            self.reference_model = self._create_reference_model()
        
        self.reference_model.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Track metrics
        self.metrics = {
            "loss": [],
            "policy_chosen_logps": [],
            "policy_rejected_logps": [],
            "ref_chosen_logps": [],
            "ref_rejected_logps": [],
            "accuracies": [],
            "kl_div": []
        }
    
    def _create_reference_model(self) -> nn.Module:
        """Create a copy of the current model to serve as reference model."""
        reference_model = type(self.model)(**self.model.config.to_dict())
        reference_model.load_state_dict(self.model.state_dict())
        reference_model.eval()  # Set to evaluation mode
        
        # Freeze parameters
        for param in reference_model.parameters():
            param.requires_grad = False
            
        return reference_model
    
    def _get_batch_logps(
        self, 
        model: nn.Module, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for a batch of sequences.
        
        Args:
            model: The model to compute log probs with
            input_ids: Tensor of shape [batch_size, seq_len]
            attention_mask: Tensor of shape [batch_size, seq_len]
            labels: Tensor of shape [batch_size, seq_len] with -100 for non-prediction tokens
            
        Returns:
            logps: Tensor of shape [batch_size] with sequence log probs
        """
        with torch.set_grad_enabled(model.training):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            logits = outputs.logits
            
            # Shift logits and labels for autoregressive loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            
            # Get log probs
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log probs using labels
            token_log_probs = torch.gather(
                log_probs, 
                dim=-1, 
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Zero out log probs for padding tokens
            token_log_probs = token_log_probs * shift_attention_mask
            
            # Sum log probs over sequence
            seq_log_probs = token_log_probs.sum(dim=-1)
            
            # Normalize by sequence length
            seq_lengths = shift_attention_mask.sum(dim=-1)
            normalized_log_probs = seq_log_probs / seq_lengths
            
            return normalized_log_probs
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        beta: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the DPO loss for a batch of policy and reference log probs.
        
        Args:
            policy_chosen_logps: Log probs of chosen responses from policy
            policy_rejected_logps: Log probs of rejected responses from policy
            reference_chosen_logps: Log probs of chosen responses from reference model
            reference_rejected_logps: Log probs of rejected responses from reference model
            beta: Temperature parameter for DPO loss
            
        Returns:
            loss: DPO loss
            metrics: Dictionary of metrics
        """
        # Compute log ratios between policy and reference model
        chosen_rewards = policy_chosen_logps - reference_chosen_logps
        rejected_rewards = policy_rejected_logps - reference_rejected_logps
        
        # Compute the DPO loss
        logits = beta * (chosen_rewards - rejected_rewards)
        losses = -F.logsigmoid(logits)
        loss = losses.mean()
        
        # Compute accuracy (how often the model assigns higher reward to chosen vs rejected)
        accuracies = (chosen_rewards > rejected_rewards).float()
        accuracy = accuracies.mean()
        
        # Compute KL divergence from reference model
        kl_chosen = (policy_chosen_logps - reference_chosen_logps).mean()
        kl_rejected = (policy_rejected_logps - reference_rejected_logps).mean()
        kl_div = 0.5 * (kl_chosen + kl_rejected)
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "kl_div": kl_div.item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item()
        }
        
        return loss, metrics
    
    def train_step(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        rejected_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single training step with DPO.
        
        Args:
            chosen_input_ids: Input IDs for chosen responses
            chosen_attention_mask: Attention mask for chosen responses
            chosen_labels: Labels for chosen responses
            rejected_input_ids: Input IDs for rejected responses
            rejected_attention_mask: Attention mask for rejected responses
            rejected_labels: Labels for rejected responses
            
        Returns:
            metrics: Dictionary of metrics for this step
        """
        # Move inputs to device
        chosen_input_ids = chosen_input_ids.to(self.device)
        chosen_attention_mask = chosen_attention_mask.to(self.device)
        chosen_labels = chosen_labels.to(self.device)
        rejected_input_ids = rejected_input_ids.to(self.device)
        rejected_attention_mask = rejected_attention_mask.to(self.device)
        rejected_labels = rejected_labels.to(self.device)
        
        # Get log probs from policy model
        policy_chosen_logps = self._get_batch_logps(
            self.model, chosen_input_ids, chosen_attention_mask, chosen_labels
        )
        policy_rejected_logps = self._get_batch_logps(
            self.model, rejected_input_ids, rejected_attention_mask, rejected_labels
        )
        
        # Get log probs from reference model (no grad)
        with torch.no_grad():
            ref_chosen_logps = self._get_batch_logps(
                self.reference_model, chosen_input_ids, chosen_attention_mask, chosen_labels
            )
            ref_rejected_logps = self._get_batch_logps(
                self.reference_model, rejected_input_ids, rejected_attention_mask, rejected_labels
            )
        
        # Compute DPO loss
        loss, metrics = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.config.dpo_beta
        )
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
        self.optimizer.step()
        
        # Update metrics
        self.metrics["loss"].append(metrics["loss"])
        self.metrics["policy_chosen_logps"].append(policy_chosen_logps.mean().item())
        self.metrics["policy_rejected_logps"].append(policy_rejected_logps.mean().item())
        self.metrics["ref_chosen_logps"].append(ref_chosen_logps.mean().item())
        self.metrics["ref_rejected_logps"].append(ref_rejected_logps.mean().item())
        self.metrics["accuracies"].append(metrics["accuracy"])
        self.metrics["kl_div"].append(metrics["kl_div"])
        
        return metrics
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model using DPO.
        
        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of epochs to train for
            eval_dataloader: Optional DataLoader for evaluation
            callback: Optional callback function called after each epoch
            
        Returns:
            metrics: Dictionary of training metrics
        """
        logger.info(f"Starting DPO training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            
            # Training loop
            epoch_metrics = {
                "loss": [],
                "accuracy": [],
                "kl_div": []
            }
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                chosen_input_ids = batch["chosen_input_ids"]
                chosen_attention_mask = batch["chosen_attention_mask"]
                chosen_labels = batch["chosen_labels"]
                rejected_input_ids = batch["rejected_input_ids"]
                rejected_attention_mask = batch["rejected_attention_mask"]
                rejected_labels = batch["rejected_labels"]
                
                metrics = self.train_step(
                    chosen_input_ids,
                    chosen_attention_mask,
                    chosen_labels,
                    rejected_input_ids,
                    rejected_attention_mask,
                    rejected_labels
                )
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "acc": f"{metrics['accuracy']:.4f}",
                    "kl": f"{metrics['kl_div']:.4f}"
                })
                
                # Update epoch metrics
                for k, v in metrics.items():
                    if k in epoch_metrics:
                        epoch_metrics[k].append(v)
            
            # Compute epoch average metrics
            avg_metrics = {k: sum(v) / len(v) if len(v) > 0 else 0.0 for k, v in epoch_metrics.items()}
            
            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self.evaluate(eval_dataloader)
                avg_metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            
            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
            logger.info(f"Train metrics: {avg_metrics}")
            
            # Call callback if provided
            if callback is not None:
                callback(epoch, self.model, avg_metrics)
        
        logger.info("DPO training completed")
        return self.metrics
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a validation set.
        
        Args:
            eval_dataloader: DataLoader for evaluation data
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.model.eval()
        
        eval_metrics = {
            "loss": [],
            "accuracy": [],
            "kl_div": []
        }
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                chosen_labels = batch["chosen_labels"].to(self.device)
                rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                rejected_labels = batch["rejected_labels"].to(self.device)
                
                # Get log probs
                policy_chosen_logps = self._get_batch_logps(
                    self.model, chosen_input_ids, chosen_attention_mask, chosen_labels
                )
                policy_rejected_logps = self._get_batch_logps(
                    self.model, rejected_input_ids, rejected_attention_mask, rejected_labels
                )
                ref_chosen_logps = self._get_batch_logps(
                    self.reference_model, chosen_input_ids, chosen_attention_mask, chosen_labels
                )
                ref_rejected_logps = self._get_batch_logps(
                    self.reference_model, rejected_input_ids, rejected_attention_mask, rejected_labels
                )
                
                # Compute metrics
                _, metrics = self.dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=self.config.dpo_beta
                )
                
                # Update metrics
                for k, v in metrics.items():
                    if k in eval_metrics:
                        eval_metrics[k].append(v)
        
        # Compute average metrics
        avg_metrics = {k: sum(v) / len(v) if len(v) > 0 else 0.0 for k, v in eval_metrics.items()}
        return avg_metrics
    
    def save(self, save_dir: str) -> None:
        """
        Save the model, optimizer state, and training metrics.
        
        Args:
            save_dir: Directory to save to
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "model.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Save optimizer
        optimizer_path = os.path.join(save_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # Save metrics
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f)
            
        logger.info(f"Model and training state saved to {save_dir}")
    
    def load(self, load_dir: str) -> None:
        """
        Load the model, optimizer state, and training metrics.
        
        Args:
            load_dir: Directory to load from
        """
        # Load model
        model_path = os.path.join(load_dir, "model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load optimizer if exists
        optimizer_path = os.path.join(load_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        
        # Load metrics if exists
        metrics_path = os.path.join(load_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
                
        logger.info(f"Model and training state loaded from {load_dir}")


class PreferenceDataset(Dataset):
    """
    Dataset for preference data used in DPO training.
    
    Each item contains a chosen response and a rejected response for the same prompt.
    """
    
    def __init__(
        self,
        chosen_data: List[Dict[str, Any]],
        rejected_data: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 512
    ):
        assert len(chosen_data) == len(rejected_data), "Chosen and rejected data must have the same length"
        
        self.chosen_data = chosen_data
        self.rejected_data = rejected_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.chosen_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chosen_item = self.chosen_data[idx]
        rejected_item = self.rejected_data[idx]
        
        # Tokenize chosen response
        chosen_tokens = self.tokenizer(
            chosen_item["prompt"] + chosen_item["response"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize rejected response
        rejected_tokens = self.tokenizer(
            rejected_item["prompt"] + rejected_item["response"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels (shift input_ids right by 1)
        chosen_labels = chosen_tokens["input_ids"].clone()
        rejected_labels = rejected_tokens["input_ids"].clone()
        
        # Prepare labels for language modeling (shift right)
        chosen_labels[:, :-1] = chosen_tokens["input_ids"][:, 1:]
        chosen_labels[:, -1] = -100  # Ignore last token
        rejected_labels[:, :-1] = rejected_tokens["input_ids"][:, 1:]
        rejected_labels[:, -1] = -100  # Ignore last token
        
        # Set prompt tokens to -100 (don't compute loss for prompt)
        prompt_length = len(self.tokenizer.encode(chosen_item["prompt"]))
        chosen_labels[:, :prompt_length] = -100
        rejected_labels[:, :prompt_length] = -100
        
        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "chosen_labels": chosen_labels.squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
            "rejected_labels": rejected_labels.squeeze(0)
        } 