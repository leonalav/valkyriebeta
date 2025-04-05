import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from tqdm import tqdm
import os

from .config import RLConfig
from ..transformer import TransformerModel
from ..attention import MultiHeadAttention

logger = logging.getLogger(__name__)

class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer for language models.
    
    This implementation follows the PPO algorithm as described in:
    "Proximal Policy Optimization Algorithms" by Schulman et al.
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
        
        # Create reference model (for KL penalty)
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
        
        # Initialize KL coefficient for adaptive KL penalty
        self.kl_coef = 0.2
        
        # Track metrics
        self.metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "kl": [],
            "total_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
            "explained_variance": []
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
    
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Tensor of shape [batch_size, seq_len]
            values: Tensor of shape [batch_size, seq_len]
            dones: Tensor of shape [batch_size, seq_len]
            next_values: Tensor of shape [batch_size, seq_len]
            
        Returns:
            advantages: Tensor of shape [batch_size, seq_len]
            returns: Tensor of shape [batch_size, seq_len]
        """
        # Initialize advantages and returns
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Initialize gae
        gae = 0
        
        # Compute advantages and returns in reverse order
        for t in reversed(range(rewards.size(1))):
            # If t is the last step, use next_values, otherwise use values[t+1]
            if t == rewards.size(1) - 1:
                next_value = next_values[:, t]
            else:
                next_value = values[:, t + 1]
                
            # Compute delta
            delta = rewards[:, t] + self.config.discount_factor * next_value * (1 - dones[:, t]) - values[:, t]
            
            # Compute GAE
            gae = delta + self.config.discount_factor * self.config.gae_lambda * (1 - dones[:, t]) * gae
            
            # Store advantage and return
            advantages[:, t] = gae
            returns[:, t] = gae + values[:, t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _compute_policy_gradient_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO policy gradient loss with clipping.
        
        Args:
            logprobs: Log probabilities of actions from current policy
            old_logprobs: Log probabilities of actions from old policy
            advantages: Advantage estimates
            clip_ratio: PPO clip ratio
            
        Returns:
            policy_loss: Policy gradient loss
            metrics: Dictionary of metrics
        """
        # Compute ratio between new and old policy
        ratio = torch.exp(logprobs - old_logprobs)
        
        # Compute surrogate losses
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        
        # Take minimum of surrogate losses
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Compute metrics
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (logprobs - old_logprobs)).mean().item()
            clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean().item()
        
        metrics = {
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction
        }
        
        return policy_loss, metrics
    
    def _compute_value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        clip_ratio: float
    ) -> torch.Tensor:
        """
        Compute value function loss with clipping.
        
        Args:
            values: Value estimates from current value function
            old_values: Value estimates from old value function
            returns: Discounted returns
            clip_ratio: PPO clip ratio
            
        Returns:
            value_loss: Value function loss
        """
        # Compute unclipped value loss
        value_loss_unclipped = (values - returns) ** 2
        
        # Compute clipped value loss
        values_clipped = old_values + torch.clamp(values - old_values, -clip_ratio, clip_ratio)
        value_loss_clipped = (values_clipped - returns) ** 2
        
        # Take maximum of clipped and unclipped value losses
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
        
        return value_loss
    
    def _compute_entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy loss to encourage exploration.
        
        Args:
            logits: Logits from policy network
            
        Returns:
            entropy_loss: Entropy loss
        """
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute log probabilities
        logprobs = F.log_softmax(logits, dim=-1)
        
        # Compute entropy
        entropy = -(probs * logprobs).sum(dim=-1).mean()
        
        # Entropy loss (negative entropy to maximize entropy)
        entropy_loss = -entropy
        
        return entropy_loss
    
    def _compute_kl_penalty(
        self,
        logits: torch.Tensor,
        reference_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty to prevent large policy updates.
        
        Args:
            logits: Logits from current policy
            reference_logits: Logits from reference policy
            
        Returns:
            kl_penalty: KL divergence penalty
        """
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        reference_probs = F.softmax(reference_logits, dim=-1)
        
        # Compute KL divergence
        kl = (reference_probs * (torch.log(reference_probs + 1e-8) - torch.log(probs + 1e-8))).sum(dim=-1).mean()
        
        # KL penalty
        kl_penalty = self.kl_coef * kl
        
        return kl_penalty, kl.item()
    
    def _update_kl_coef(self, kl: float) -> None:
        """
        Update KL coefficient based on current KL divergence.
        
        Args:
            kl: Current KL divergence
        """
        if kl < self.config.target_kl / 1.5:
            self.kl_coef /= 1.5
        elif kl > self.config.target_kl * 1.5:
            self.kl_coef *= 1.5
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        actions: torch.Tensor,
        old_logprobs: torch.Tensor,
        old_values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single PPO training step.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            actions: Actions taken
            old_logprobs: Log probabilities of actions from old policy
            old_values: Value estimates from old value function
            rewards: Rewards received
            dones: Done flags
            next_values: Value estimates for next states
            
        Returns:
            metrics: Dictionary of training metrics
        """
        # Move tensors to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        actions = actions.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        old_values = old_values.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_values = next_values.to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages(rewards, old_values, dones, next_values)
        
        # Initialize metrics
        epoch_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "kl_penalty": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "kl": 0.0
        }
        
        # Perform multiple epochs of PPO updates
        for epoch in range(self.config.ppo_epochs):
            # Forward pass through model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get logits and values
            logits = outputs.logits
            values = outputs.get("value", None)
            
            # If model doesn't output values, use a separate value head
            if values is None:
                # This assumes the model outputs hidden states that can be used for value prediction
                hidden_states = outputs.hidden_states
                values = self.value_head(hidden_states)
            
            # Forward pass through reference model
            with torch.no_grad():
                reference_outputs = self.reference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                reference_logits = reference_outputs.logits
            
            # Compute log probabilities of actions
            logprobs = F.log_softmax(logits, dim=-1)
            logprobs_actions = torch.gather(logprobs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
            
            # Compute policy gradient loss
            policy_loss, pg_metrics = self._compute_policy_gradient_loss(
                logprobs_actions, old_logprobs, advantages, self.config.clip_ratio
            )
            
            # Compute value loss
            value_loss = self._compute_value_loss(
                values, old_values, returns, self.config.clip_ratio
            )
            
            # Compute entropy loss
            entropy_loss = self._compute_entropy_loss(logits)
            
            # Compute KL penalty
            kl_penalty, kl = self._compute_kl_penalty(logits, reference_logits)
            
            # Compute total loss
            total_loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                + self.config.entropy_coef * entropy_loss
                + kl_penalty
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # Update parameters
            self.optimizer.step()
            
            # Update metrics
            epoch_metrics["policy_loss"] += policy_loss.item()
            epoch_metrics["value_loss"] += value_loss.item()
            epoch_metrics["entropy_loss"] += entropy_loss.item()
            epoch_metrics["kl_penalty"] += kl_penalty.item()
            epoch_metrics["total_loss"] += total_loss.item()
            epoch_metrics["approx_kl"] += pg_metrics["approx_kl"]
            epoch_metrics["clip_fraction"] += pg_metrics["clip_fraction"]
            epoch_metrics["kl"] += kl
        
        # Average metrics over epochs
        for key in epoch_metrics:
            epoch_metrics[key] /= self.config.ppo_epochs
        
        # Update KL coefficient
        self._update_kl_coef(epoch_metrics["kl"])
        
        # Update metrics history
        for key in self.metrics:
            if key in epoch_metrics:
                self.metrics[key].append(epoch_metrics[key])
        
        return epoch_metrics
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        num_epochs: int,
        callback: Optional[Callable] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model using PPO.
        
        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of training epochs
            callback: Optional callback function called after each epoch
            
        Returns:
            metrics: Dictionary of training metrics
        """
        # Training loop
        for epoch in range(num_epochs):
            epoch_metrics = {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
                "kl_penalty": 0.0,
                "total_loss": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
                "kl": 0.0
            }
            
            # Process batches
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Unpack batch
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                actions = batch["actions"]
                old_logprobs = batch["old_logprobs"]
                old_values = batch["old_values"]
                rewards = batch["rewards"]
                dones = batch["dones"]
                next_values = batch["next_values"]
                
                # Perform training step
                step_metrics = self.train_step(
                    input_ids, attention_mask, actions, old_logprobs,
                    old_values, rewards, dones, next_values
                )
                
                # Update epoch metrics
                for key in epoch_metrics:
                    epoch_metrics[key] += step_metrics[key]
            
            # Average metrics over batches
            for key in epoch_metrics:
                epoch_metrics[key] /= len(train_dataloader)
            
            # Log metrics
            logger.info(f"Epoch {epoch+1}/{num_epochs} metrics:")
            for key, value in epoch_metrics.items():
                logger.info(f"  {key}: {value:.6f}")
            
            # Call callback if provided
            if callback is not None:
                callback(epoch, epoch_metrics, self.model)
        
        return self.metrics
    
    def save(self, save_dir: str) -> None:
        """
        Save the model, optimizer, and training state.
        
        Args:
            save_dir: Directory to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))
        
        # Save optimizer
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
        
        # Save metrics
        torch.save(self.metrics, os.path.join(save_dir, "metrics.pt"))
        
        # Save config
        self.config.save_pretrained(save_dir)
        
        logger.info(f"Saved model, optimizer, and training state to {save_dir}")
    
    def load(self, load_dir: str) -> None:
        """
        Load the model, optimizer, and training state.
        
        Args:
            load_dir: Directory to load from
        """
        # Load model
        self.model.load_state_dict(torch.load(os.path.join(load_dir, "model.pt")))
        
        # Load optimizer
        self.optimizer.load_state_dict(torch.load(os.path.join(load_dir, "optimizer.pt")))
        
        # Load metrics
        self.metrics = torch.load(os.path.join(load_dir, "metrics.pt"))
        
        # Load config
        self.config = RLConfig.from_pretrained(load_dir)
        
        logger.info(f"Loaded model, optimizer, and training state from {load_dir}")


class ValueHead(nn.Module):
    """Value head for estimating state values."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.out_proj = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value head.
        
        Args:
            hidden_states: Hidden states from transformer model
            
        Returns:
            values: Value estimates
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        values = self.out_proj(x)
        return values 