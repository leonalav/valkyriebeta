import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tqdm import tqdm
import time
import random
import json
from dataclasses import dataclass

from .config import RLConfig
from ..transformer import TransformerModel
from ..tree_reasoning_mcts import MonteCarloTreeSearch, MCTSConfig

logger = logging.getLogger(__name__)

class ExpertIterationTrainer:
    """
    Expert Iteration (ExIt) trainer for language models.
    
    Expert Iteration combines Monte Carlo Tree Search (MCTS) with policy improvement
    to create a powerful reinforcement learning algorithm for complex reasoning tasks.
    
    The algorithm alternates between:
    1. Improvement phase: Use MCTS with the current policy to generate improved outputs
    2. Learning phase: Train the policy to imitate the MCTS-improved outputs
    
    References:
    - "Expert Iteration" (Anthony et al., 2017)
    - "AlphaZero" (Silver et al., 2018)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: RLConfig,
        value_model: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize or use provided value model
        if value_model is not None:
            self.value_model = value_model
        else:
            self.value_model = self._create_value_model()
        
        self.value_model.to(self.device)
        
        # Initialize MCTS
        mcts_config = MCTSConfig(
            max_iterations=config.mcts_iterations_per_expert_step,
            exploration_weight=1.0,
            temperature=config.expert_temperature
        )
        self.mcts = MonteCarloTreeSearch(mcts_config)
        
        # Set up optimizers
        self.policy_optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.value_optimizer = optim.AdamW(
            self.value_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Track metrics
        self.metrics = {
            "policy_loss": [],
            "value_loss": [],
            "mcts_reward": [],
            "expert_improvement": []
        }
    
    def _create_value_model(self) -> nn.Module:
        """Create a value model based on the policy model architecture."""
        # This is a simplified version - in practice, you might want
        # to use a different architecture or duplicate the base model
        # with a value head
        
        # For simplicity, we'll use a small MLP on top of the model's embeddings
        hidden_size = self.model.config.hidden_size
        
        class ValueHead(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.GELU(),
                    nn.Linear(hidden_size // 2, 1)
                )
            
            def forward(self, hidden_states):
                # Average pool the hidden states
                pooled = hidden_states.mean(dim=1)
                return self.net(pooled)
        
        return ValueHead(hidden_size)
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 1,
        eval_dataloader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model with Expert Iteration.
        
        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of epochs to train for
            eval_dataloader: Optional DataLoader for evaluation
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting Expert Iteration training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train for one epoch
            self._train_epoch(train_dataloader)
            
            # Evaluate if dataloader provided
            if eval_dataloader is not None:
                eval_metrics = self.evaluate(eval_dataloader)
                logger.info(f"Evaluation metrics: {eval_metrics}")
        
        return self.metrics
    
    def _train_epoch(self, dataloader: DataLoader):
        """Train for one epoch with Expert Iteration."""
        self.model.train()
        self.value_model.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_mcts_reward = 0.0
        total_expert_improvement = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Expert Iteration"):
            batch_policy_loss, batch_value_loss, batch_mcts_reward, batch_improvement = self._train_batch(batch)
            
            total_policy_loss += batch_policy_loss
            total_value_loss += batch_value_loss
            total_mcts_reward += batch_mcts_reward
            total_expert_improvement += batch_improvement
            num_batches += 1
        
        # Compute epoch averages
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_mcts_reward = total_mcts_reward / num_batches
        avg_expert_improvement = total_expert_improvement / num_batches
        
        # Update metrics
        self.metrics["policy_loss"].append(avg_policy_loss)
        self.metrics["value_loss"].append(avg_value_loss)
        self.metrics["mcts_reward"].append(avg_mcts_reward)
        self.metrics["expert_improvement"].append(avg_expert_improvement)
        
        logger.info(f"Epoch summary - Policy loss: {avg_policy_loss:.4f}, "
                   f"Value loss: {avg_value_loss:.4f}, "
                   f"MCTS reward: {avg_mcts_reward:.4f}, "
                   f"Expert improvement: {avg_expert_improvement:.4f}")
    
    def _train_batch(self, batch) -> Tuple[float, float, float, float]:
        """
        Train on a single batch with Expert Iteration.
        
        Returns:
            Tuple of (policy_loss, value_loss, mcts_reward, expert_improvement)
        """
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Phase 1: Improvement Phase - Use MCTS to find better solutions
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            policy_logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
            
            # Get value estimates
            value_estimates = self.value_model(hidden_states).squeeze(-1)
            
            # Run MCTS to get improved policy
            mcts_policy_logits, mcts_rewards = self._run_mcts(
                input_ids, 
                attention_mask, 
                policy_logits, 
                value_estimates
            )
            
            # Compute improvement (difference between MCTS and original policy)
            original_policy = F.softmax(policy_logits[:, -1, :], dim=-1)
            mcts_policy = F.softmax(mcts_policy_logits, dim=-1)
            policy_improvement = (mcts_policy - original_policy).abs().sum(dim=-1).mean().item()
        
        # Phase 2: Learning Phase - Train policy and value models
        # Train policy to imitate MCTS outputs
        self.policy_optimizer.zero_grad()
        
        # Forward pass for policy
        policy_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        policy_logits = policy_outputs.logits
        
        # Policy loss: Cross-entropy with MCTS policy
        policy_loss = self._compute_policy_loss(policy_logits, mcts_policy_logits)
        policy_loss.backward()
        
        # Clip policy gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.policy_optimizer.step()
        
        # Train value function to predict MCTS rewards
        self.value_optimizer.zero_grad()
        
        # Forward pass for value
        with torch.no_grad():
            # Recompute hidden states with updated policy
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
        
        value_predictions = self.value_model(hidden_states).squeeze(-1)
        
        # Value loss: MSE with MCTS rewards
        value_loss = F.mse_loss(value_predictions, mcts_rewards)
        value_loss.backward()
        
        # Clip value gradients
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), self.config.max_grad_norm)
        self.value_optimizer.step()
        
        return (
            policy_loss.item(),
            value_loss.item(),
            mcts_rewards.mean().item(),
            policy_improvement
        )
    
    def _run_mcts(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        policy_logits: torch.Tensor,
        value_estimates: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MCTS to get improved policy and rewards.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            policy_logits: Original policy logits
            value_estimates: Value estimates for states
            
        Returns:
            Tuple of (improved_policy_logits, mcts_rewards)
        """
        batch_size = input_ids.shape[0]
        vocab_size = policy_logits.shape[-1]
        
        # Initialize outputs
        improved_policy_logits = torch.zeros(batch_size, vocab_size, device=self.device)
        mcts_rewards = torch.zeros(batch_size, device=self.device)
        
        # In a real implementation, this would be more complex
        # This is a simplified version that simulates MCTS
        for b in range(batch_size):
            # Get current sequence
            curr_input_ids = input_ids[b]
            curr_attention_mask = attention_mask[b]
            
            # Get last token's policy
            curr_policy = F.softmax(policy_logits[b, -1, :], dim=-1)
            
            # Simulate MCTS exploration
            # In practice, this would be a full MCTS search
            mcts_result = self._simulate_mcts(
                curr_input_ids, 
                curr_attention_mask,
                curr_policy
            )
            
            improved_policy_logits[b] = mcts_result["improved_policy"]
            mcts_rewards[b] = mcts_result["reward"]
        
        return improved_policy_logits, mcts_rewards
    
    def _simulate_mcts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        policy: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate MCTS exploration for a single sequence.
        
        This is a simplified simulation - in a real implementation,
        this would use the actual MCTS algorithm.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            policy: Original policy distribution
            
        Returns:
            Dictionary with improved policy and reward
        """
        vocab_size = policy.shape[-1]
        
        # Simulate MCTS exploration by boosting high-probability tokens
        # and reducing low-probability tokens
        # This is just a simplified simulation!
        
        # Apply a softmax temperature to make distribution more peaked
        temperature = 0.5
        sharpened_policy = F.softmax(torch.log(policy + 1e-10) / temperature, dim=-1)
        
        # Boost top-k tokens even more
        top_k = 5
        top_k_values, top_k_indices = torch.topk(sharpened_policy, k=top_k)
        improved_policy = torch.zeros_like(policy)
        improved_policy.scatter_(0, top_k_indices, top_k_values * 1.5)
        
        # Renormalize
        improved_policy = improved_policy / improved_policy.sum()
        
        # Convert to logits
        improved_policy_logits = torch.log(improved_policy + 1e-10)
        
        # Simulate reward: higher for more concentrated policy
        reward = 1.0 - improved_policy.entropy() / np.log(vocab_size)
        
        return {
            "improved_policy": improved_policy_logits,
            "reward": reward
        }
    
    def _compute_policy_loss(
        self, 
        policy_logits: torch.Tensor, 
        target_policy_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute policy loss to learn from MCTS policy.
        
        Args:
            policy_logits: Model policy logits
            target_policy_logits: MCTS policy logits
            
        Returns:
            Policy loss
        """
        batch_size = policy_logits.shape[0]
        
        # Get the last token's logits
        last_token_logits = policy_logits[:, -1, :]
        
        # Convert target logits to probabilities
        target_probs = F.softmax(target_policy_logits, dim=-1)
        
        # Compute cross-entropy loss
        loss = -torch.sum(target_probs * F.log_softmax(last_token_logits, dim=-1), dim=-1).mean()
        
        return loss
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_dataloader: DataLoader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        self.value_model.eval()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_reward = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Get model outputs
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                policy_logits = outputs.logits
                hidden_states = outputs.hidden_states[-1]
                
                # Get value estimates
                value_estimates = self.value_model(hidden_states).squeeze(-1)
                
                # Run MCTS to get improved policy
                mcts_policy_logits, mcts_rewards = self._run_mcts(
                    input_ids, 
                    attention_mask, 
                    policy_logits, 
                    value_estimates
                )
                
                # Compute policy loss
                policy_loss = self._compute_policy_loss(policy_logits, mcts_policy_logits)
                    
                    # Compute value loss
                value_loss = F.mse_loss(value_estimates, mcts_rewards)
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_reward += mcts_rewards.mean().item()
                num_batches += 1
        
        # Compute averages
        metrics = {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "reward": total_reward / num_batches
        }
        
        return metrics
    
    def generate_with_expert_search(
        self,
        input_text: str,
        max_length: int = 100,
        num_return_sequences: int = 1,
        mcts_iterations: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[str]:
        """
        Generate text using the model with MCTS-guided search.
        
        Args:
            input_text: Input text prompt
            max_length: Maximum generation length
            num_return_sequences: Number of sequences to return
            mcts_iterations: Number of MCTS iterations (defaults to config value)
            temperature: Temperature for sampling (defaults to config value)
            
        Returns:
            List of generated text sequences
        """
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Set iterations and temperature
        actual_iterations = mcts_iterations or self.config.mcts_iterations_per_expert_step
        actual_temperature = temperature or self.config.expert_temperature
        
        # Update MCTS config
        mcts_config = MCTSConfig(
            max_iterations=actual_iterations,
            exploration_weight=1.0,
            temperature=actual_temperature
        )
        self.mcts = MonteCarloTreeSearch(mcts_config)
        
        # Generate output sequences
        generated_sequences = []
        
        for _ in range(num_return_sequences):
            # Generate one sequence with MCTS guidance
            curr_input_ids = input_ids.clone()
            curr_attention_mask = attention_mask.clone()
            
            for _ in range(max_length):
                with torch.no_grad():
                    # Get model outputs
                    outputs = self.model(
                        input_ids=curr_input_ids,
                        attention_mask=curr_attention_mask,
                        output_hidden_states=True
                    )
                    policy_logits = outputs.logits
                    hidden_states = outputs.hidden_states[-1]
                    
                    # Get value estimates
                    value_estimates = self.value_model(hidden_states).squeeze(-1)
                    
                    # Use MCTS to get improved policy for the next token
                    mcts_policy_logits, _ = self._run_mcts(
                        curr_input_ids,
                        curr_attention_mask,
                        policy_logits,
                        value_estimates
                    )
                    
                    # Sample from the improved policy
                    mcts_policy = F.softmax(mcts_policy_logits / actual_temperature, dim=-1)
                    next_token_id = torch.multinomial(mcts_policy, num_samples=1).item()
                    
                    # Append the new token
                    curr_input_ids = torch.cat([curr_input_ids, torch.tensor([[next_token_id]], device=self.device)], dim=1)
                    curr_attention_mask = torch.cat([curr_attention_mask, torch.ones(1, 1, device=self.device)], dim=1)
                    
                    # Check if we've generated an EOS token
                    if next_token_id == self.tokenizer.eos_token_id:
                        break
            
            # Decode and add to results
            generated_text = self.tokenizer.decode(curr_input_ids[0], skip_special_tokens=True)
            generated_sequences.append(generated_text)
        
        return generated_sequences


class ExpertIterationDataset(torch.utils.data.Dataset):
    """Dataset for Expert Iteration training examples."""
    
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        return {
            "input_ids": example["input_ids"],
            "actions": torch.tensor(example["action"], dtype=torch.long),
            "values": torch.tensor(example["value"], dtype=torch.float),
            "visit_counts": torch.tensor(example["visit_count"], dtype=torch.float)
        } 