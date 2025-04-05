"""
Adaptive Reasoning module for dynamically adjusting reasoning strategies.

This module provides components that allow the model to adapt its reasoning approach
based on the complexity and type of the problem it's solving.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import random

logger = logging.getLogger(__name__)

class ReasoningStrategy(Enum):
    """Enum for different reasoning strategies the model can employ."""
    DEFAULT = auto()
    STEP_BY_STEP = auto()
    CHAIN_OF_THOUGHT = auto()
    SYMBOLIC = auto()
    NUMERICAL = auto()
    RETRIEVAL_AUGMENTED = auto()
    ANALOGICAL = auto()
    COMPARATIVE = auto()
    CAUSAL = auto()
    COUNTERFACTUAL = auto()
    ABDUCTIVE = auto()

class AdaptiveReasoningConfig:
    """Configuration for adaptive reasoning capabilities."""
    
    def __init__(
        self,
        enabled=True,
        default_strategy=ReasoningStrategy.DEFAULT,
        strategy_selection_threshold=0.7,
        max_reasoning_steps=10,
        use_meta_learning=False,
        use_reinforcement=False,
        temperature=0.8,
        reasoning_prompt_templates=None,
        strategy_embeddings_size=128,
    ):
        """
        Initialize the adaptive reasoning configuration.
        
        Args:
            enabled: Whether adaptive reasoning is enabled
            default_strategy: Default reasoning strategy to use
            strategy_selection_threshold: Confidence threshold for strategy selection
            max_reasoning_steps: Maximum number of reasoning steps to perform
            use_meta_learning: Whether to use meta-learning for strategy selection
            use_reinforcement: Whether to use reinforcement learning for strategy improvement
            temperature: Temperature for reasoning strategy selection
            reasoning_prompt_templates: Dictionary mapping strategies to prompt templates
            strategy_embeddings_size: Size of the strategy embedding vectors
        """
        self.enabled = enabled
        self.default_strategy = default_strategy
        self.strategy_selection_threshold = strategy_selection_threshold
        self.max_reasoning_steps = max_reasoning_steps
        self.use_meta_learning = use_meta_learning
        self.use_reinforcement = use_reinforcement
        self.temperature = temperature
        self.strategy_embeddings_size = strategy_embeddings_size
        
        # Default reasoning prompt templates
        self._default_templates = {
            ReasoningStrategy.DEFAULT: "",
            ReasoningStrategy.STEP_BY_STEP: "Let's solve this step-by-step:",
            ReasoningStrategy.CHAIN_OF_THOUGHT: "Let's think through this:",
            ReasoningStrategy.SYMBOLIC: "Let's use symbolic reasoning:",
            ReasoningStrategy.NUMERICAL: "Let's solve this numerically:",
            ReasoningStrategy.RETRIEVAL_AUGMENTED: "Let me recall some relevant information:",
            ReasoningStrategy.ANALOGICAL: "This reminds me of a similar problem:",
            ReasoningStrategy.COMPARATIVE: "Let's compare different approaches:",
            ReasoningStrategy.CAUSAL: "Let's analyze the causal relationships:",
            ReasoningStrategy.COUNTERFACTUAL: "Let's consider what would happen if:",
            ReasoningStrategy.ABDUCTIVE: "Let's find the most likely explanation:"
        }
        
        self.reasoning_prompt_templates = reasoning_prompt_templates or self._default_templates

    def __repr__(self):
        return (f"AdaptiveReasoningConfig(enabled={self.enabled}, "
                f"default_strategy={self.default_strategy}, "
                f"max_reasoning_steps={self.max_reasoning_steps})")

class ComplexityEstimator(nn.Module):
    """Estimates the complexity of input text for reasoning"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        
        # Complexity estimation network
        self.complexity_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Features that indicate complexity
        self.complexity_features = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 5)  # 5 complexity features
        )
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate the complexity of input text
        
        Args:
            hidden_states: Hidden states from the transformer [batch_size, seq_len, hidden_size]
            
        Returns:
            complexity: Complexity score [batch_size, 1]
            features: Complexity features [batch_size, 5]
        """
        # Pool sequence dimension
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Estimate complexity
        complexity = self.complexity_net(pooled)  # [batch_size, 1]
        
        # Extract complexity features
        features = self.complexity_features(pooled)  # [batch_size, 5]
        
        return complexity, features

class ComponentSelector(nn.Module):
    """Selects which reasoning components to use based on input complexity and available compute"""
    
    def __init__(self, config: AdaptiveReasoningConfig, hidden_size: int):
        super().__init__()
        self.config = config
        
        # Complexity estimator
        self.complexity_estimator = ComplexityEstimator(hidden_size)
        
        # Component selection network
        self.selection_net = nn.Sequential(
            nn.Linear(5 + 1, 64),  # 5 complexity features + 1 complexity score
            nn.GELU(),
            nn.Linear(64, len(config.component_costs))
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        available_compute: float = 1.0
    ) -> Dict[str, bool]:
        """
        Select which reasoning components to use
        
        Args:
            hidden_states: Hidden states from the transformer [batch_size, seq_len, hidden_size]
            available_compute: Available computation budget (0.0-1.0)
            
        Returns:
            Dictionary mapping component names to whether they should be used
        """
        # Estimate complexity
        complexity, features = self.complexity_estimator(hidden_states)
        complexity = complexity.mean().item()  # Average over batch
        
        # Adjust available compute based on complexity
        adjusted_compute = min(
            self.config.max_computation_budget,
            max(
                self.config.min_computation_budget,
                available_compute * (0.5 + 0.5 * complexity)  # Scale with complexity
            )
        )
        
        # Determine which components to use based on complexity
        if complexity < self.config.low_complexity_threshold:
            # Simple reasoning - use minimal components
            return self._select_components_for_simple_reasoning(adjusted_compute)
        elif complexity < self.config.medium_complexity_threshold:
            # Moderate reasoning - use medium components
            return self._select_components_for_moderate_reasoning(adjusted_compute)
        else:
            # Complex reasoning - use all components if possible
            return self._select_components_for_complex_reasoning(adjusted_compute)
    
    def _select_components_for_simple_reasoning(self, available_compute: float) -> Dict[str, bool]:
        """Select components for simple reasoning tasks"""
        components = {name: False for name in self.config.component_costs}
        
        # Always enable MoE for simple reasoning
        components["moe"] = True
        available_compute -= self.config.component_costs.get("moe", 0)
        
        # Enable memory layer if affordable
        if available_compute >= self.config.component_costs.get("memory_layer", 0):
            components["memory_layer"] = True
            available_compute -= self.config.component_costs.get("memory_layer", 0)
        
        # Enable knowledge reasoning if affordable (useful for factual queries)
        if available_compute >= self.config.component_costs.get("knowledge_reasoning", 0):
            components["knowledge_reasoning"] = True
            available_compute -= self.config.component_costs.get("knowledge_reasoning", 0)
        
        return components
    
    def _select_components_for_moderate_reasoning(self, available_compute: float) -> Dict[str, bool]:
        """Select components for moderate reasoning tasks"""
        components = {name: False for name in self.config.component_costs}
        
        # Sort components by importance for moderate reasoning
        sorted_components = sorted(
            self.config.component_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Enable components in order of importance until we run out of compute
        for name, _ in sorted_components:
            if available_compute >= self.config.component_costs.get(name, 0):
                components[name] = True
                available_compute -= self.config.component_costs.get(name, 0)
            else:
                components[name] = False
        
        return components
    
    def _select_components_for_complex_reasoning(self, available_compute: float) -> Dict[str, bool]:
        """Select components for complex reasoning tasks"""
        components = {name: False for name in self.config.component_costs}
        
        # For complex reasoning, try to enable all components
        # If not enough compute, prioritize tree reasoning and neural symbolic
        priority_components = [
            "tree_reasoning",
            "neural_symbolic",
            "recursive_reasoning",
            "moe",
            "knowledge_reasoning",
            "verifiable_computation",
            "memory_layer"
        ]
        
        for name in priority_components:
            if available_compute >= self.config.component_costs.get(name, 0):
                components[name] = True
                available_compute -= self.config.component_costs.get(name, 0)
            else:
                components[name] = False
        
        return components

class MetaReasoningOptimizer(nn.Module):
    """
    Meta-learning module for optimizing reasoning strategy selection based on feedback.
    
    This module:
    1. Tracks strategy performance across different tasks
    2. Uses reinforcement learning to update strategy selection
    3. Maintains a memory of successful reasoning paths
    4. Adapts selection criteria based on historical performance
    """
    
    def __init__(self, config: AdaptiveReasoningConfig):
        super().__init__()
        self.config = config
        
        # Strategy embeddings (learnable representations of each strategy)
        self.num_strategies = len(ReasoningStrategy)
        self.strategy_embeddings = nn.Parameter(
            torch.randn(self.num_strategies, config.strategy_embeddings_size)
        )
        
        # Task encoder (encodes task features for matching with strategies)
        self.task_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.strategy_embeddings_size)
        )
        
        # Strategy scorer (computes match between task and strategy)
        self.strategy_scorer = nn.Sequential(
            nn.Linear(config.strategy_embeddings_size * 2, config.strategy_embeddings_size),
            nn.GELU(),
            nn.Linear(config.strategy_embeddings_size, 1)
        )
        
        # Performance predictor (predicts success likelihood for a strategy on a task)
        self.performance_predictor = nn.Sequential(
            nn.Linear(config.strategy_embeddings_size * 2, config.strategy_embeddings_size),
            nn.GELU(),
            nn.Linear(config.strategy_embeddings_size, 1),
            nn.Sigmoid()
        )
        
        # Reinforcement learning parameters
        self.gamma = 0.95  # Discount factor
        self.rl_learning_rate = 0.01
        
        # Strategy statistics tracking
        self.strategy_usage_count = {strategy: 0 for strategy in ReasoningStrategy}
        self.strategy_success_count = {strategy: 0 for strategy in ReasoningStrategy}
        self.task_strategy_history = {}  # Maps task signatures to successful strategies
        
        # Reasoning path memory (stores successful sequences of strategies)
        self.reasoning_paths = {}  # Maps task types to successful reasoning paths
        self.min_path_success_rate = 0.7  # Minimum success rate to store a path
        
        # Recent strategy selection history for online learning
        self.recent_selections = []  # List of (task_embedding, strategy, reward) tuples
        self.recent_history_size = 100
        
        # Task clustering for efficient retrieval
        self.task_clusters = {}
        self.num_clusters = 20
        self.min_samples_for_clustering = 50
        self.task_embeddings_history = []
        
        # Exploration vs exploitation
        self.exploration_rate = 0.1  # Initial exploration rate
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.995
        
    def reset_statistics(self):
        """Reset usage statistics"""
        self.strategy_usage_count = {strategy: 0 for strategy in ReasoningStrategy}
        self.strategy_success_count = {strategy: 0 for strategy in ReasoningStrategy}
        
    def compute_task_embedding(self, hidden_states):
        """Compute embedding for the current task"""
        # Take the first token's representation as the task representation
        # This is typically the [CLS] or similar token that captures the task
        task_state = hidden_states[:, 0]
        
        # Encode the task
        task_embedding = self.task_encoder(task_state)
        
        return task_embedding
    
    def select_strategy(self, hidden_states, available_strategies=None, explore=True):
        """
        Select the best reasoning strategy for the given task.
        
        Args:
            hidden_states: Hidden states from the transformer
            available_strategies: Optional list of available strategies (defaults to all)
            explore: Whether to use exploration (for training) or exploitation (for inference)
            
        Returns:
            selected_strategy: The selected reasoning strategy
            confidence: Confidence score for the selection
        """
        batch_size = hidden_states.size(0)
        task_embedding = self.compute_task_embedding(hidden_states)
        
        # Store task embedding for clustering
        if self.training:
            self.task_embeddings_history.append(task_embedding.detach().cpu())
            if len(self.task_embeddings_history) > self.min_samples_for_clustering:
                self._update_task_clusters()
        
        # Check if we have a similar task in our history
        if not self.training and self.task_clusters:
            similar_task_embedding = self._find_similar_task(task_embedding)
            if similar_task_embedding is not None:
                task_signature = self._get_task_signature(similar_task_embedding)
                if task_signature in self.task_strategy_history:
                    # Use historically successful strategy
                    return self.task_strategy_history[task_signature], 1.0
        
        # Get strategy scores
        strategy_scores = []
        
        # Default to all strategies if none specified
        if available_strategies is None:
            available_strategies = list(ReasoningStrategy)
        
        # Compute scores for each strategy
        for strategy in available_strategies:
            strategy_idx = strategy.value - 1  # Enum values start at 1
            strategy_embedding = self.strategy_embeddings[strategy_idx].unsqueeze(0).expand(batch_size, -1)
            
            # Concatenate task and strategy embeddings
            combined = torch.cat([task_embedding, strategy_embedding], dim=-1)
            
            # Compute match score
            score = self.strategy_scorer(combined).squeeze(-1)
            
            # Add to scores
            strategy_scores.append((strategy, score))
        
        # Sort strategies by score
        strategy_scores.sort(key=lambda x: x[1].item(), reverse=True)
        
        # Use epsilon-greedy for exploration during training
        if self.training and explore and random.random() < self.exploration_rate:
            # Randomly select a strategy
            selected_idx = random.randint(0, len(strategy_scores) - 1)
            selected_strategy = strategy_scores[selected_idx][0]
            confidence = strategy_scores[selected_idx][1].item()
            # Update exploration rate
            self.exploration_rate = max(
                self.min_exploration_rate, 
                self.exploration_rate * self.exploration_decay
            )
        else:
            # Use highest scoring strategy
            selected_strategy = strategy_scores[0][0]
            confidence = strategy_scores[0][1].item()
        
        # Update usage statistics
        if self.training:
            self.strategy_usage_count[selected_strategy] += 1
            
            # Store selection for learning
            self.recent_selections.append({
                "task_embedding": task_embedding.detach(),
                "strategy": selected_strategy,
                "reward": None  # Will be updated when feedback is received
            })
            
            # Keep history bounded
            if len(self.recent_selections) > self.recent_history_size:
                self.recent_selections.pop(0)
        
        return selected_strategy, confidence
    
    def update_with_feedback(self, success: bool, reward: float):
        """
        Update strategy selection model with feedback from the most recent selection.
        
        Args:
            success: Whether the strategy was successful
            reward: Reward signal for the strategy (-1 to 1)
        """
        if not self.recent_selections:
            return
        
        # Get most recent selection
        selection = self.recent_selections[-1]
        strategy = selection["strategy"]
        
        # Update statistics
        if success:
            self.strategy_success_count[strategy] += 1
            
            # Store successful strategy for this task
            task_embedding = selection["task_embedding"]
            task_signature = self._get_task_signature(task_embedding)
            self.task_strategy_history[task_signature] = strategy
        
        # Update reward for recent selection
        selection["reward"] = reward
        
        # Online learning update if we have enough samples
        if len(self.recent_selections) >= 10:
            self._update_strategy_embeddings()
    
    def _update_strategy_embeddings(self):
        """Update strategy embeddings using reinforcement learning"""
        # Skip if no rewards yet
        if all(selection["reward"] is None for selection in self.recent_selections):
            return
            
        # Prepare batched data for update
        task_embeddings = []
        strategy_indices = []
        rewards = []
        
        for selection in self.recent_selections:
            if selection["reward"] is not None:
                task_embeddings.append(selection["task_embedding"])
                strategy_indices.append(selection["strategy"].value - 1)  # Convert to 0-indexed
                rewards.append(selection["reward"])
        
        if not rewards:
            return
            
        # Convert to tensors
        task_embeddings = torch.stack(task_embeddings)
        strategy_indices = torch.tensor(strategy_indices, device=task_embeddings.device)
        rewards = torch.tensor(rewards, device=task_embeddings.device)
        
        # Get strategy embeddings for selected strategies
        selected_strategy_embeddings = self.strategy_embeddings[strategy_indices]
        
        # Predict performance
        combined = torch.cat([task_embeddings, selected_strategy_embeddings], dim=-1)
        predicted_performance = self.performance_predictor(combined).squeeze(-1)
        
        # Compute loss (mean squared error between predicted and actual rewards)
        normalized_rewards = (rewards + 1) / 2  # Scale from [-1,1] to [0,1]
        loss = F.mse_loss(predicted_performance, normalized_rewards)
        
        # Compute gradients and update
        loss.backward()
        
        # Manual update with learning rate
        with torch.no_grad():
            # Update only strategy embeddings
            grad_scale = self.rl_learning_rate / (1.0 + len(rewards))
            self.strategy_embeddings.grad *= grad_scale
            self.strategy_embeddings -= self.strategy_embeddings.grad
            self.strategy_embeddings.grad.zero_()
            
            # Update encoder and predictor parameters
            for param in self.task_encoder.parameters():
                if param.grad is not None:
                    param.grad *= grad_scale
                    param -= param.grad
                    param.grad.zero_()
                    
            for param in self.performance_predictor.parameters():
                if param.grad is not None:
                    param.grad *= grad_scale
                    param -= param.grad
                    param.grad.zero_()
    
    def store_reasoning_path(self, task_embedding, strategy_sequence, success_rate):
        """
        Store a successful sequence of reasoning strategies for a task type.
        
        Args:
            task_embedding: Embedding of the task
            strategy_sequence: Sequence of strategies that were used
            success_rate: Success rate of this strategy sequence
        """
        if success_rate < self.min_path_success_rate:
            return
            
        # Get task signature
        task_signature = self._get_task_signature(task_embedding)
        
        # Find appropriate cluster
        cluster_id = self._get_cluster_id(task_embedding)
        
        # Store the reasoning path for this cluster
        if cluster_id not in self.reasoning_paths:
            self.reasoning_paths[cluster_id] = []
            
        # Add path with success rate
        self.reasoning_paths[cluster_id].append({
            "strategy_sequence": strategy_sequence,
            "success_rate": success_rate,
            "task_signature": task_signature,
            "usage_count": 1
        })
        
        # Sort paths by success rate
        self.reasoning_paths[cluster_id].sort(key=lambda x: x["success_rate"], reverse=True)
    
    def get_reasoning_path_for_task(self, hidden_states):
        """
        Retrieve a previously successful reasoning path for a similar task.
        
        Args:
            hidden_states: Hidden states from the transformer
            
        Returns:
            strategy_sequence: Sequence of strategies to apply, or None if no match
        """
        task_embedding = self.compute_task_embedding(hidden_states)
        
        # Find appropriate cluster
        cluster_id = self._get_cluster_id(task_embedding)
        
        # Check if we have reasoning paths for this cluster
        if cluster_id in self.reasoning_paths and self.reasoning_paths[cluster_id]:
            # Return the most successful path
            best_path = self.reasoning_paths[cluster_id][0]
            
            # Update usage count
            best_path["usage_count"] += 1
            
            return best_path["strategy_sequence"]
        
        return None
    
    def _get_task_signature(self, task_embedding):
        """Generate a signature for a task embedding for efficient lookup"""
        # Quantize embedding values for stable signatures
        quantized = (task_embedding * 100).round() / 100
        
        # Convert to tuple for hashing
        signature = tuple(quantized.flatten().tolist())
        
        # Use hash for more compact representation
        return hash(signature)
    
    def _update_task_clusters(self):
        """Update task clusters based on collected task embeddings"""
        if len(self.task_embeddings_history) < self.min_samples_for_clustering:
            return
            
        # Stack embeddings
        embeddings = torch.stack(self.task_embeddings_history)
        
        # Perform clustering (using k-means)
        try:
            from sklearn.cluster import KMeans
            
            # Limit number of samples for efficiency
            max_samples = 1000
            if len(embeddings) > max_samples:
                indices = torch.randperm(len(embeddings))[:max_samples]
                embeddings_subset = embeddings[indices]
            else:
                embeddings_subset = embeddings
            
            # Perform clustering
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            kmeans.fit(embeddings_subset.numpy())
            
            # Store cluster centers
            self.task_clusters = {
                i: torch.tensor(center, dtype=torch.float32) 
                for i, center in enumerate(kmeans.cluster_centers_)
            }
            
            # Clear history to save memory
            self.task_embeddings_history = []
            
        except ImportError:
            logger.warning("sklearn not available for clustering. Using simple clustering.")
            
            # Simple clustering by selecting random centroids
            indices = torch.randperm(len(embeddings))[:self.num_clusters]
            self.task_clusters = {
                i: embeddings[idx] for i, idx in enumerate(indices)
            }
    
    def _get_cluster_id(self, task_embedding):
        """Find the closest cluster for a task embedding"""
        if not self.task_clusters:
            return None
            
        # Compute distances to all cluster centers
        distances = {}
        for cluster_id, center in self.task_clusters.items():
            center_tensor = center.to(task_embedding.device)
            distance = torch.norm(task_embedding - center_tensor)
            distances[cluster_id] = distance.item()
        
        # Return closest cluster ID
        return min(distances.items(), key=lambda x: x[1])[0]
    
    def _find_similar_task(self, task_embedding):
        """Find a similar task from history based on embedding similarity"""
        cluster_id = self._get_cluster_id(task_embedding)
        if cluster_id is None:
            return None
            
        # Get task signatures in this cluster
        signatures_in_cluster = [
            path["task_signature"] 
            for cluster_paths in self.reasoning_paths.values() 
            for path in cluster_paths
        ]
        
        # Find closest matching task
        closest_task = None
        min_distance = float('inf')
        
        for signature in signatures_in_cluster:
            if signature in self.task_strategy_history:
                # This is a task we've seen before
                # We'd need to store the original embeddings to find the closest one
                # For now, just return the first match
                return task_embedding
                
        return closest_task

class AdaptiveReasoningController(nn.Module):
    """
    Controller module for adaptive reasoning capabilities.
    
    This module integrates with a transformer model and provides
    mechanisms to adapt reasoning strategies based on input characteristics.
    """
    
    def __init__(self, config, hidden_size, vocab_size):
        """
        Initialize the adaptive reasoning controller.
        
        Args:
            config: AdaptiveReasoningConfig object
            hidden_size: Size of hidden states in the transformer model
            vocab_size: Size of vocabulary in the model
        """
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Create strategy embeddings
        num_strategies = len(ReasoningStrategy)
        self.strategy_embeddings = nn.Embedding(
            num_strategies, 
            config.strategy_embeddings_size
        )
        
        # Strategy selection network
        self.strategy_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_strategies)
        )
        
        # Strategy integration layer
        self.strategy_integration = nn.Linear(
            hidden_size + config.strategy_embeddings_size, 
            hidden_size
        )
        
        # Reasoning step counter
        self.register_buffer('reasoning_steps', torch.zeros(1, dtype=torch.long))
        
        # Current strategy tracker
        self.current_strategy = config.default_strategy
        
        # Strategy effectiveness tracking
        self.strategy_scores = {strategy: 0.0 for strategy in ReasoningStrategy}
        self.strategy_uses = {strategy: 0 for strategy in ReasoningStrategy}
        
        logger.info(f"Initialized AdaptiveReasoningController with {num_strategies} strategies")
    
    def reset_state(self):
        """Reset the reasoning state."""
        self.reasoning_steps.zero_()
        self.current_strategy = self.config.default_strategy
    
    def select_strategy(self, hidden_states, problem_type=None):
        """
        Select an appropriate reasoning strategy based on the hidden states.
        
        Args:
            hidden_states: Hidden states from the transformer model
            problem_type: Optional explicit problem type hint
            
        Returns:
            The selected reasoning strategy
        """
        if not self.config.enabled:
            return self.config.default_strategy
            
        # Use the mean pooled hidden state for strategy selection
        pooled_hidden = hidden_states.mean(dim=1)
        
        # Get strategy logits
        strategy_logits = self.strategy_selector(pooled_hidden)
        
        # Apply temperature
        strategy_logits = strategy_logits / self.config.temperature
        
        # Convert to probabilities
        strategy_probs = torch.softmax(strategy_logits, dim=-1)
        
        # Get the most likely strategy
        max_prob, max_idx = strategy_probs.max(dim=-1)
        
        # Use default strategy if confidence is too low
        if max_prob.item() < self.config.strategy_selection_threshold:
            selected_strategy = self.config.default_strategy
        else:
            selected_strategy = ReasoningStrategy(max_idx.item() + 1)  # +1 because Enum starts at 1
        
        # Override with explicit problem type if provided
        if problem_type is not None:
            if problem_type == "math":
                selected_strategy = ReasoningStrategy.NUMERICAL
            elif problem_type == "logic":
                selected_strategy = ReasoningStrategy.SYMBOLIC
            elif problem_type == "causal":
                selected_strategy = ReasoningStrategy.CAUSAL
                
        # Update current strategy
        self.current_strategy = selected_strategy
        self.strategy_uses[selected_strategy] += 1
        
        return selected_strategy
    
    def get_reasoning_prompt(self, strategy=None):
        """
        Get the reasoning prompt template for the given strategy.
        
        Args:
            strategy: The reasoning strategy to get a prompt for
            
        Returns:
            The prompt template string
        """
        strategy = strategy or self.current_strategy
        return self.config.reasoning_prompt_templates.get(
            strategy, 
            self.config.reasoning_prompt_templates[ReasoningStrategy.DEFAULT]
        )
    
    def integrate_strategy(self, hidden_states, strategy=None):
        """
        Integrate the strategy embedding into the hidden states.
        
        Args:
            hidden_states: Hidden states from the transformer model
            strategy: The reasoning strategy to integrate
            
        Returns:
            Modified hidden states with strategy integration
        """
        if not self.config.enabled:
            return hidden_states
            
        strategy = strategy or self.current_strategy
        strategy_idx = torch.tensor([strategy.value - 1], device=hidden_states.device)  # -1 because Enum starts at 1
        strategy_embedding = self.strategy_embeddings(strategy_idx)
        
        # Expand strategy embedding to match hidden states shape
        batch_size, seq_len, _ = hidden_states.shape
        strategy_embedding = strategy_embedding.expand(batch_size, seq_len, -1)
        
        # Concatenate hidden states with strategy embedding
        augmented_hidden = torch.cat([hidden_states, strategy_embedding], dim=-1)
        
        # Integrate strategy into hidden states
        integrated_hidden = self.strategy_integration(augmented_hidden)
        
        # Increment reasoning step counter
        self.reasoning_steps += 1
        
        return integrated_hidden
    
    def update_strategy_score(self, strategy, score):
        """
        Update the effectiveness score for a strategy.
        
        Args:
            strategy: The reasoning strategy to update
            score: Score indicating how effective the strategy was
        """
        if strategy in self.strategy_scores:
            # Exponential moving average
            alpha = 0.1
            self.strategy_scores[strategy] = (1 - alpha) * self.strategy_scores[strategy] + alpha * score
    
    def should_continue_reasoning(self):
        """Check if reasoning should continue or terminate."""
        return self.reasoning_steps.item() < self.config.max_reasoning_steps
    
    def forward(self, hidden_states, problem_type=None):
        """
        Forward pass for the adaptive reasoning controller.
        
        Args:
            hidden_states: Hidden states from the transformer model
            problem_type: Optional explicit problem type hint
            
        Returns:
            Modified hidden states with adaptive reasoning applied
        """
        if not self.config.enabled:
            return hidden_states
            
        # Select strategy based on input
        strategy = self.select_strategy(hidden_states, problem_type)
        
        # Apply strategy to modify hidden states
        modified_hidden = self.integrate_strategy(hidden_states, strategy)
        
        return modified_hidden
    
    def get_strategy_stats(self):
        """Get statistics on strategy usage and effectiveness."""
        return {
            "strategy_scores": {str(s): score for s, score in self.strategy_scores.items()},
            "strategy_uses": {str(s): uses for s, uses in self.strategy_uses.items()},
            "current_strategy": str(self.current_strategy),
            "reasoning_steps": self.reasoning_steps.item()
        }

def create_adaptive_config(model_config) -> AdaptiveReasoningConfig:
    """
    Create an adaptive reasoning configuration based on the model configuration
    
    Args:
        model_config: Model configuration
        
    Returns:
        AdaptiveReasoningConfig
    """
    # Extract relevant parameters from model config
    return AdaptiveReasoningConfig(
        low_complexity_threshold=getattr(model_config, 'low_complexity_threshold', 0.3),
        medium_complexity_threshold=getattr(model_config, 'medium_complexity_threshold', 0.7),
        max_computation_budget=getattr(model_config, 'max_computation_budget', 1.0),
        min_computation_budget=getattr(model_config, 'min_computation_budget', 0.2),
        use_early_exit=getattr(model_config, 'use_early_exit', True),
        early_exit_threshold=getattr(model_config, 'early_exit_threshold', 0.9),
        enabled=getattr(model_config, 'enabled', True),
        default_strategy=getattr(model_config, 'default_strategy', ReasoningStrategy.DEFAULT),
        strategy_selection_threshold=getattr(model_config, 'strategy_selection_threshold', 0.7),
        max_reasoning_steps=getattr(model_config, 'max_reasoning_steps', 10),
        use_meta_learning=getattr(model_config, 'use_meta_learning', False),
        use_reinforcement=getattr(model_config, 'use_reinforcement', False),
        temperature=getattr(model_config, 'temperature', 0.8),
        reasoning_prompt_templates=getattr(model_config, 'reasoning_prompt_templates', None),
        strategy_embeddings_size=getattr(model_config, 'strategy_embeddings_size', 128)
    ) 