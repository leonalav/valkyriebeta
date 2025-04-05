import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tqdm import tqdm
import time
from dataclasses import dataclass, field

from .config import RLConfig, RLHFConfig
from .dpo import DPOTrainer
from .ppo import PPOTrainer
from ..reasoning.causal_inference import CausalInferenceEngine, CausalInferenceConfig

logger = logging.getLogger(__name__)

@dataclass
class CausalRLHFConfig:
    """Configuration for Causal RLHF integration."""
    # Base configurations
    rl_config: RLConfig = field(default_factory=RLConfig)
    rlhf_config: RLHFConfig = field(default_factory=RLHFConfig)
    causal_config: CausalInferenceConfig = field(default_factory=CausalInferenceConfig)
    
    # Integration parameters
    use_causal_reward_bonus: bool = True
    causal_reward_weight: float = 0.3
    use_intervention_amplification: bool = True
    use_counterfactual_reward: bool = True
    
    # Specific causal parameters
    causal_discovery_weight: float = 0.4
    intervention_weight: float = 0.3
    counterfactual_weight: float = 0.3
    
    # Training parameters
    num_causal_iterations: int = 3
    causal_batch_size: int = 8
    max_causal_variables: int = 12


class CausalRewardModel(nn.Module):
    """Reward model for causal inference abilities."""
    
    def __init__(
        self, 
        hidden_size: int, 
        num_layers: int = 2,
        dropout: float = 0.1,
        use_intervention_scoring: bool = True,
        use_counterfactual_scoring: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_intervention_scoring = use_intervention_scoring
        self.use_counterfactual_scoring = use_counterfactual_scoring
        
        # Encoder for causal reasoning
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Component-specific scoring layers
        self.causal_discovery_layer = nn.Linear(hidden_size, hidden_size)
        
        if use_intervention_scoring:
            self.intervention_layer = nn.Linear(hidden_size, hidden_size)
        
        if use_counterfactual_scoring:
            self.counterfactual_layer = nn.Linear(hidden_size, hidden_size)
        
        # Final reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        causal_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rewards for causal reasoning.
        
        Args:
            hidden_states: Hidden states from the language model
            causal_outputs: Optional outputs from causal inference engine
            
        Returns:
            Dictionary containing rewards and component scores
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Encode hidden states
        encoded = self.encoder(hidden_states)
        
        # Pool sequence representation
        pooled = encoded.mean(dim=1)
        
        # Extract features for different aspects of causal reasoning
        causal_discovery_features = self.causal_discovery_layer(pooled)
        
        # Initialize component scores
        component_scores = {
            "causal_discovery": torch.sigmoid(causal_discovery_features.mean(dim=-1))
        }
        
        # Add intervention features if enabled
        if self.use_intervention_scoring:
            if causal_outputs and "intervened_states" in causal_outputs:
                # Use actual intervention outputs if available
                intervention_states = causal_outputs["intervened_states"]
                intervention_features = self.intervention_layer(intervention_states.mean(dim=1))
            else:
                # Else use base hidden states
                intervention_features = self.intervention_layer(pooled)
            
            component_scores["intervention"] = torch.sigmoid(intervention_features.mean(dim=-1))
        
        # Add counterfactual features if enabled
        if self.use_counterfactual_scoring:
            if causal_outputs and "counterfactual_encodings" in causal_outputs:
                # Use actual counterfactual outputs if available
                counterfactual_encodings = causal_outputs["counterfactual_encodings"]
                counterfactual_features = self.counterfactual_layer(counterfactual_encodings.mean(dim=(1, 2)))
            else:
                # Else use base hidden states
                counterfactual_features = self.counterfactual_layer(pooled)
            
            component_scores["counterfactual"] = torch.sigmoid(counterfactual_features.mean(dim=-1))
        
        # Combine features for final reward
        combined_features = causal_discovery_features
        
        # Compute base reward
        reward = self.reward_head(combined_features).squeeze(-1)
        
        # Compute component-weighted final reward
        final_reward = reward
        for component, score in component_scores.items():
            final_reward = final_reward + score
        
        # Normalize
        final_reward = final_reward / (1 + len(component_scores))
        
        return {
            "reward": final_reward,
            "base_reward": reward,
            **component_scores
        }


class CausalRLHFIntegration:
    """
    Integrates RLHF with causal inference capabilities.
    
    This class combines reinforcement learning from human feedback with
    specialized causal inference components to create a model that
    excels at understanding cause-effect relationships.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: CausalRLHFConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize RLHF components based on configuration
        if config.rl_config.use_ppo:
            self.rl_trainer = PPOTrainer(
                model=model,
                tokenizer=tokenizer,
                config=config.rl_config,
                device=device
            )
        elif config.rl_config.use_dpo:
            self.rl_trainer = DPOTrainer(
                model=model,
                tokenizer=tokenizer,
                config=config.rl_config,
                device=device
            )
        else:
            raise ValueError("No RL algorithm specified in config")
        
        # Initialize causal inference engine
        self.causal_engine = CausalInferenceEngine(
            config=config.causal_config,
            device=device
        )
        
        # Initialize causal reward model if enabled
        if config.use_causal_reward_bonus:
            self.causal_reward_model = CausalRewardModel(
                hidden_size=model.config.hidden_size,
                use_intervention_scoring=config.use_intervention_amplification,
                use_counterfactual_scoring=config.use_counterfactual_reward
            ).to(device)
        else:
            self.causal_reward_model = None
        
        # Optimizer for causal components
        self.causal_optimizer = optim.AdamW(
            [p for p in self.causal_engine.parameters() if p.requires_grad] +
            ([p for p in self.causal_reward_model.parameters() if p.requires_grad] 
             if self.causal_reward_model else []),
            lr=config.rl_config.learning_rate,
            weight_decay=config.rl_config.weight_decay
        )
        
        # Track training progress
        self.global_step = 0
        self.metrics = {
            "rl_metrics": {},
            "causal_metrics": {},
            "integration_metrics": {}
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        causal_eval_dataloader: Optional[DataLoader] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train with integrated RLHF + causal inference.
        
        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of epochs to train for
            eval_dataloader: Optional DataLoader for evaluation
            causal_eval_dataloader: Optional DataLoader for causal evaluation
            callback: Optional callback function
            
        Returns:
            Dictionary containing metrics
        """
        logger.info(f"Starting Causal RLHF training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train with RLHF + causal integration
            rl_metrics, causal_metrics = self._train_epoch(
                train_dataloader=train_dataloader,
                epoch=epoch
            )
            
            # Update metrics
            self.metrics["rl_metrics"][f"epoch_{epoch}"] = rl_metrics
            self.metrics["causal_metrics"][f"epoch_{epoch}"] = causal_metrics
            
            # Evaluate if dataloaders provided
            if eval_dataloader is not None:
                eval_metrics = self.rl_trainer.evaluate(eval_dataloader)
                self.metrics["integration_metrics"][f"epoch_{epoch}_eval"] = eval_metrics
            
            if causal_eval_dataloader is not None:
                causal_eval_metrics = self.evaluate_causal(causal_eval_dataloader)
                self.metrics["causal_metrics"][f"epoch_{epoch}_eval"] = causal_eval_metrics
            
            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
            logger.info(f"RL metrics: {rl_metrics}")
            logger.info(f"Causal metrics: {causal_metrics}")
            
            # Call callback if provided
            if callback is not None:
                callback(epoch, self.model, self.metrics)
        
        logger.info("Causal RLHF training completed")
        return self.metrics
    
    def _train_epoch(
        self,
        train_dataloader: DataLoader,
        epoch: int
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Train for one epoch with integrated causal RLHF.
        
        Args:
            train_dataloader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Tuple of (rl_metrics, causal_metrics)
        """
        self.model.train()
        self.causal_engine.train()
        if self.causal_reward_model:
            self.causal_reward_model.train()
        
        # First, train with standard RLHF
        rl_metrics = self.rl_trainer.train(
            train_dataloader=train_dataloader,
            num_epochs=1  # Just one epoch at a time
        )
        
        # Then, enhance with causal reasoning
        causal_metrics = self._train_causal_components(train_dataloader)
        
        return rl_metrics, causal_metrics
    
    def _train_causal_components(
        self,
        train_dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Train causal inference components.
        
        Args:
            train_dataloader: DataLoader for training data
            
        Returns:
            Dictionary of causal training metrics
        """
        # Initialize metrics
        metrics = {
            "causal_loss": 0.0,
            "causal_discovery_accuracy": 0.0,
            "intervention_accuracy": 0.0,
            "counterfactual_accuracy": 0.0,
            "reward_mean": 0.0
        }
        
        # Get a subset of data for causal training
        causal_iterations = min(self.config.num_causal_iterations, len(train_dataloader))
        
        total_samples = 0
        causal_loss_sum = 0.0
        reward_sum = 0.0
        
        for i, batch in enumerate(train_dataloader):
            if i >= causal_iterations:
                break
            
            # Extract inputs
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Get model hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]
            
            # Apply causal inference
            causal_outputs = self.causal_engine(
                hidden_states=hidden_states,
                variable_mask=attention_mask
            )
            
            # Compute causal rewards if enabled
            if self.causal_reward_model:
                reward_outputs = self.causal_reward_model(
                    hidden_states=hidden_states,
                    causal_outputs=causal_outputs
                )
                
                # Get reward signal
                rewards = reward_outputs["reward"]
                reward_sum += rewards.mean().item()
                
                # Use rewards to guide causal learning
                # This is a simplified approach - in a real implementation,
                # you would use more sophisticated techniques
                causal_loss = -rewards.mean()
            else:
                # Without reward model, use a simple loss based on edge consistency
                edge_probs = causal_outputs["edge_probabilities"]
                
                # Compute loss encouraging sparse but confident edges
                sparsity_target = 0.1  # Target edge density
                edge_density = edge_probs[:, :, :, 0].mean()  # Causal edge density
                sparsity_loss = F.mse_loss(edge_density, torch.tensor(sparsity_target, device=self.device))
                
                # Encourage confident predictions (high or low, not middle)
                confidence_loss = -((edge_probs[:, :, :, 0] - 0.5).abs().mean() - 0.5)
                
                causal_loss = sparsity_loss + confidence_loss
            
            # Update causal components
            self.causal_optimizer.zero_grad()
            causal_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.causal_engine.parameters() if p.requires_grad] +
                ([p for p in self.causal_reward_model.parameters() if p.requires_grad] 
                 if self.causal_reward_model else []),
                self.config.rl_config.max_grad_norm
            )
            
            self.causal_optimizer.step()
            
            # Track metrics
            causal_loss_sum += causal_loss.item()
            total_samples += 1
            
            # Update global step
            self.global_step += 1
        
        # Compute average metrics
        if total_samples > 0:
            metrics["causal_loss"] = causal_loss_sum / total_samples
            if self.causal_reward_model:
                metrics["reward_mean"] = reward_sum / total_samples
        
        return metrics
    
    def evaluate_causal(
        self,
        eval_dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate causal inference capabilities.
        
        Args:
            eval_dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        self.causal_engine.eval()
        if self.causal_reward_model:
            self.causal_reward_model.eval()
        
        # Initialize metrics
        metrics = {
            "causal_discovery_accuracy": 0.0,
            "intervention_accuracy": 0.0,
            "counterfactual_accuracy": 0.0,
            "reward_mean": 0.0
        }
        
        total_samples = 0
        reward_sum = 0.0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Extract inputs
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Get model hidden states
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]
                
                # Apply causal inference
                causal_outputs = self.causal_engine(
                    hidden_states=hidden_states,
                    variable_mask=attention_mask
                )
                
                # If ground truth causal graphs are available, evaluate accuracy
                if "causal_adjacency" in batch:
                    true_adj = batch["causal_adjacency"].to(self.device)
                    pred_adj = (causal_outputs["edge_probabilities"][:, :, :, 0] > 0.5).float()
                    metrics["causal_discovery_accuracy"] += (pred_adj == true_adj).float().mean().item()
                
                # Compute causal rewards if enabled
                if self.causal_reward_model:
                    reward_outputs = self.causal_reward_model(
                        hidden_states=hidden_states,
                        causal_outputs=causal_outputs
                    )
                    
                    # Get reward signal
                    rewards = reward_outputs["reward"]
                    reward_sum += rewards.mean().item()
                
                total_samples += 1
        
        # Compute average metrics
        if total_samples > 0:
            if "causal_adjacency" in batch:
                metrics["causal_discovery_accuracy"] /= total_samples
            if self.causal_reward_model:
                metrics["reward_mean"] = reward_sum / total_samples
        
        return metrics
    
    def apply_causal_inference(
        self,
        text: str,
        extract_variables: bool = True
    ) -> Dict[str, Any]:
        """
        Apply causal inference to text and return structured results.
        
        Args:
            text: Input text to analyze
            extract_variables: Whether to automatically extract variables
            
        Returns:
            Dictionary with causal inference results
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Extract variables if requested
        variable_spans = None
        if extract_variables:
            variable_spans = self._extract_variable_spans(text)
        
        # Apply causal inference
        results = self.causal_engine.infer_causality(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            variable_spans=variable_spans
        )
        
        return results
    
    def _extract_variable_spans(self, text: str) -> List[Tuple[int, int]]:
        """
        Extract variable spans from text using simple heuristics.
        
        In a real implementation, this would use more sophisticated NLP techniques.
        
        Args:
            text: Input text
            
        Returns:
            List of (start, end) token indices for variables
        """
        # This is a simplified placeholder implementation
        # In a real system, you would use NER or other techniques
        
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)
        variable_spans = []
        
        # Simple heuristic: consider capitalized words as potential variables
        for i, token in enumerate(tokens):
            if token[0].isupper() and len(token) > 1:
                variable_spans.append((i, i+1))
        
        # Limit the number of variables
        max_vars = min(len(variable_spans), self.config.max_causal_variables)
        variable_spans = variable_spans[:max_vars]
        
        return variable_spans 