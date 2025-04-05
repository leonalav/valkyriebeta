"""
Adaptive reasoning module with confidence-based optimization for MCTS and recursive reasoning.
This implementation improves the efficiency of computational reasoning approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ConfidencePredictor(nn.Module):
    """
    Neural network that predicts confidence in reasoning steps
    to determine when deep reasoning is necessary
    """
    
    def __init__(self, hidden_size: int, confidence_hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Confidence prediction network
        self.confidence_network = nn.Sequential(
            nn.Linear(hidden_size, confidence_hidden_size),
            nn.GELU(),
            nn.Linear(confidence_hidden_size, confidence_hidden_size // 2),
            nn.GELU(),
            nn.Linear(confidence_hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Calibration parameters for reliable confidence scores
        self.calibration_slope = nn.Parameter(torch.ones(1))
        self.calibration_bias = nn.Parameter(torch.zeros(1))
        
        # For tracking calibration metrics
        self.register_buffer('confidence_sum', torch.zeros(1))
        self.register_buffer('accuracy_sum', torch.zeros(1))
        self.register_buffer('samples_seen', torch.zeros(1))
    
    def forward(self, x):
        """
        Predict confidence for reasoning steps
        
        Args:
            x: Input representation [batch_size, hidden_size]
            
        Returns:
            confidence: Calibrated confidence scores [batch_size, 1]
        """
        # Raw confidence prediction
        raw_confidence = self.confidence_network(x)
        
        # Apply calibration for reliable confidence estimates
        calibrated_confidence = torch.sigmoid(
            self.calibration_slope * (raw_confidence - 0.5) + self.calibration_bias
        )
        
        return calibrated_confidence
    
    def update_calibration(self, confidence, correct):
        """
        Update calibration statistics based on feedback
        
        Args:
            confidence: Predicted confidence scores
            correct: Whether predictions were correct
        """
        if self.training:
            # Update tracking metrics
            self.confidence_sum += confidence.sum()
            self.accuracy_sum += correct.float().sum()
            self.samples_seen += confidence.size(0)
            
            # Adjust calibration parameters if enough samples
            if self.samples_seen > 100:
                avg_confidence = self.confidence_sum / self.samples_seen
                avg_accuracy = self.accuracy_sum / self.samples_seen
                
                # Adjust calibration bias to align confidence with accuracy
                target_bias = avg_accuracy - avg_confidence
                self.calibration_bias.data += 0.01 * target_bias
                
                # Reset statistics periodically
                if self.samples_seen > 1000:
                    self.confidence_sum.zero_()
                    self.accuracy_sum.zero_()
                    self.samples_seen.zero_()


class AdaptiveMCTSReasoner(nn.Module):
    """
    Monte Carlo Tree Search based reasoning with adaptive depth control based on confidence.
    Optimizes computation by dynamically deciding how much reasoning is needed.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_simulations: int = 100,
        min_simulations: int = 10,
        exploration_weight: float = 1.0,
        value_scale: float = 2.0,
        confidence_threshold: float = 0.9,
        simulation_batch_size: int = 16,
        use_guided_exploration: bool = True,
        rollout_policy_temperature: float = 1.0,
        enable_heuristic_learning: bool = True
    ):
        """
        Initialize the adaptive MCTS reasoner
        
        Args:
            hidden_size: Dimension of hidden states
            max_simulations: Maximum number of MCTS simulations to perform
            min_simulations: Minimum number of MCTS simulations to perform
            exploration_weight: Weight for UCB exploration term
            value_scale: Scaling factor for value network outputs
            confidence_threshold: Threshold for early stopping
            simulation_batch_size: Number of parallel simulations to run
            use_guided_exploration: Whether to use a learned policy for guiding exploration
            rollout_policy_temperature: Temperature for rollout policy
            enable_heuristic_learning: Whether to use learned heuristics for search guidance
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_simulations = max_simulations
        self.min_simulations = min_simulations
        self.exploration_weight = exploration_weight
        self.value_scale = value_scale
        self.confidence_threshold = confidence_threshold
        self.simulation_batch_size = simulation_batch_size
        self.use_guided_exploration = use_guided_exploration
        self.rollout_policy_temperature = rollout_policy_temperature
        self.enable_heuristic_learning = enable_heuristic_learning
        
        # Confidence predictor for early stopping
        self.confidence_predictor = ConfidencePredictor(hidden_size)
        
        # Policy network for guiding tree search
        self.policy_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2)  # Output logits sized to match possible actions
        )
        
        # Value network for position evaluation
        self.value_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Tanh()  # Values between -1 and 1
        )
        
        # Statistics tracking
        self.avg_simulations_used = 0
        self.total_queries = 0
        self.early_stops = 0
        
        # Register buffers to store best action history for analysis
        self.register_buffer('last_selected_action', torch.zeros(1, dtype=torch.long))
        self.register_buffer('last_action_value', torch.zeros(1))
        self.register_buffer('last_action_visits', torch.zeros(1, dtype=torch.long))
        
    def forward(
        self,
        state_representation: torch.Tensor,
        available_actions: List[Any],
        action_embeddings: torch.Tensor,
        state_transition_fn: Callable,
        reward_fn: Callable,
        max_depth: int = 5
    ):
        """
        Perform MCTS reasoning on the given state
        
        Args:
            state_representation: Tensor representing the current state [batch_size, hidden_size]
            available_actions: List of available actions
            action_embeddings: Tensor of action embeddings [num_actions, hidden_size]
            state_transition_fn: Function that takes (state, action) and returns next state
            reward_fn: Function that takes a state and returns a reward
            max_depth: Maximum depth for MCTS search
            
        Returns:
            Tuple of (selected action, action value, confidence)
        """
        batch_size = state_representation.size(0)
        device = state_representation.device
        num_actions = len(available_actions)
        
        # Initialize statistics for this search
        visit_counts = torch.zeros(batch_size, num_actions, device=device)
        action_values = torch.zeros(batch_size, num_actions, device=device)
        
        # Get prior policy from neural network
        policy_logits = self.policy_network(state_representation)
        policy_logits = policy_logits[:, :num_actions]  # Truncate to actual number of actions
        prior_policy = F.softmax(policy_logits / self.rollout_policy_temperature, dim=-1)
        
        # Get initial value estimate
        initial_value = self.value_network(state_representation).squeeze(-1)
        
        # Get confidence estimate
        confidence = self.confidence_predictor(state_representation).squeeze(-1)
        
        # Determine if we need extensive search
        need_search = confidence < self.confidence_threshold
        
        # If high confidence and not forcing minimum, we can return early
        if not need_search.any() and self.min_simulations == 0:
            # Get action with highest policy score
            best_action_indices = torch.argmax(prior_policy, dim=-1)
            best_action_values = initial_value
            
            # Update statistics
            self.early_stops += batch_size
            self.total_queries += batch_size
            self.last_selected_action.copy_(best_action_indices[0:1])
            self.last_action_value.copy_(best_action_values[0:1])
            self.last_action_visits.copy_(torch.zeros(1, device=device))
            
            return best_action_indices, best_action_values, confidence
        
        # Determine number of simulations based on confidence
        simulations_needed = torch.clamp(
            self.max_simulations * (1 - confidence), 
            min=self.min_simulations, 
            max=self.max_simulations
        ).long()
        max_simulations = simulations_needed.max().item()
        
        # Run MCTS simulations
        for sim_idx in range(max_simulations):
            # For each batch item where we need more simulations
            for batch_idx in range(batch_size):
                if sim_idx < simulations_needed[batch_idx]:
                    # Select action using UCB scores
                    ucb_scores = self._compute_ucb_scores(
                        visit_counts[batch_idx],
                        action_values[batch_idx],
                        prior_policy[batch_idx],
                        self.exploration_weight
                    )
                    action_idx = torch.argmax(ucb_scores).item()
                    
                    # Run simulation for selected action
                    simulation_value = self._run_simulation(
                        state_representation[batch_idx],
                        available_actions,
                        action_embeddings,
                        state_transition_fn,
                        reward_fn,
                        depth=max_depth
                    )
                    
                    # Update statistics
                    visit_counts[batch_idx, action_idx] += 1
                    old_value = action_values[batch_idx, action_idx]
                    action_values[batch_idx, action_idx] = (
                        old_value + (simulation_value - old_value) / visit_counts[batch_idx, action_idx]
                    )
        
        # Calculate action selection policy
        action_policy = visit_counts / (visit_counts.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Select best action
        best_action_indices = torch.argmax(action_policy, dim=-1)
        best_action_values = torch.gather(action_values, 1, best_action_indices.unsqueeze(-1)).squeeze(-1)
        
        # Update statistics
        self.avg_simulations_used = (
            (self.avg_simulations_used * self.total_queries + simulations_needed.sum().item()) / 
            (self.total_queries + batch_size)
        )
        self.total_queries += batch_size
        self.early_stops += (simulations_needed < self.max_simulations).sum().item()
        
        # Store last action for debugging
        self.last_selected_action.copy_(best_action_indices[0:1])
        self.last_action_value.copy_(best_action_values[0:1])
        self.last_action_visits.copy_(
            torch.gather(visit_counts[0], 0, best_action_indices[0:1])
        )
        
        return best_action_indices, best_action_values, confidence
    
    def _compute_ucb_scores(
        self, 
        visit_counts: torch.Tensor,
        action_values: torch.Tensor,
        prior_policy: torch.Tensor,
        exploration_weight: float = 1.0
    ):
        """
        Compute UCB scores for action selection
        
        Args:
            visit_counts: Number of visits for each action
            action_values: Estimated Q-value for each action
            prior_policy: Prior policy from neural network
            exploration_weight: Weight for exploration term
            
        Returns:
            UCB scores for each action
        """
        # Scale values to [0, 1] range for better UCB behavior
        scaled_values = (action_values + 1) / 2.0
        
        # Add small epsilon to avoid division by zero
        total_visits = visit_counts.sum() + 1e-8
        
        # Compute UCB scores using PUCT formula
        # (combination of MCTS-UCB and neural net prior)
        ucb_scores = scaled_values + exploration_weight * prior_policy * math.sqrt(total_visits) / (1 + visit_counts)
        
        # Add noise to exploration term for actions that haven't been tried yet
        untried_actions = (visit_counts == 0).float()
        exploration_noise = torch.rand_like(ucb_scores) * 1e-3
        ucb_scores = ucb_scores + untried_actions * exploration_noise
        
        return ucb_scores
    
    def _run_simulation(
        self,
        state: torch.Tensor,
        available_actions: List[Any],
        action_embeddings: torch.Tensor,
        state_transition_fn: Callable,
        reward_fn: Callable,
        depth: int
    ):
        """
        Run a single MCTS simulation from the given state
        
        Args:
            state: Current state representation
            available_actions: List of available actions
            action_embeddings: Embeddings for each action
            state_transition_fn: Function to get next state
            reward_fn: Function to evaluate states
            depth: Maximum depth for this simulation
            
        Returns:
            Estimated value of the simulation
        """
        # Base case: reached maximum depth or terminal state
        if depth <= 0:
            return self.value_network(state).item()
        
        # Get policy for current state
        policy_logits = self.policy_network(state)
        policy_probs = F.softmax(policy_logits[:len(available_actions)], dim=0)
        
        # Sample action from policy
        if self.use_guided_exploration:
            action_idx = torch.multinomial(policy_probs, 1).item()
        else:
            action_idx = random.randint(0, len(available_actions) - 1)
        
        # Get action and compute next state
        action = available_actions[action_idx]
        action_embedding = action_embeddings[action_idx]
        next_state = state_transition_fn(state, action_embedding)
        
        # Recursively simulate from next state
        sim_value = self._run_simulation(
            next_state,
            available_actions,
            action_embeddings,
            state_transition_fn,
            reward_fn,
            depth - 1
        )
        
        # Return discounted value (could use a configurable discount factor)
        return 0.95 * sim_value
    
    def _compute_entropy(self, distribution):
        """Compute entropy of a probability distribution"""
        log_dist = torch.log(distribution + 1e-8)
        return -torch.sum(distribution * log_dist, dim=-1)
    
    def get_average_simulations(self):
        """Get the average number of simulations used"""
        return self.avg_simulations_used
    
    def get_early_stop_ratio(self):
        """Get the ratio of early stops to total queries"""
        return self.early_stops / max(1, self.total_queries)
    
    def reset_metrics(self):
        """Reset the tracking metrics"""
        self.avg_simulations_used = 0
        self.total_queries = 0
        self.early_stops = 0


class AdaptiveRecursiveReasoner(nn.Module):
    """
    Adaptive recursive reasoning that dynamically adjusts reasoning depth.
    Balances computational efficiency with reasoning effectiveness.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_recursive_depth: int = 5,
        min_recursive_depth: int = 1,
        confidence_threshold: float = 0.85,
        use_memory_augmentation: bool = True,
        memory_size: int = 128
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_recursive_depth = max_recursive_depth
        self.min_recursive_depth = min_recursive_depth
        self.confidence_threshold = confidence_threshold
        self.use_memory_augmentation = use_memory_augmentation
        self.memory_size = memory_size
        
        # Confidence predictor for early stopping
        self.confidence_predictor = ConfidencePredictor(hidden_size)
        
        # Recursive reasoning core
        self.reasoning_core = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Residual connection gate
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Memory for previous reasoning states if enabled
        if use_memory_augmentation:
            self.memory_key = nn.Linear(hidden_size, memory_size)
            self.memory_value = nn.Linear(hidden_size, memory_size)
            self.memory_query = nn.Linear(hidden_size, memory_size)
            self.memory_output = nn.Linear(memory_size, hidden_size)
            
            # Initialize memory buffers
            self.register_buffer('memory_keys', torch.zeros(0, memory_size))
            self.register_buffer('memory_values', torch.zeros(0, memory_size))
            self.memory_capacity = 1000  # Maximum entries in memory
        
        # Statistics tracking
        self.avg_recursion_depth = 0
        self.total_queries = 0
        self.early_stops = 0
    
    def forward(self, x, context=None):
        """
        Execute recursive reasoning with adaptive depth
        
        Args:
            x: Initial state representation [batch_size, hidden_size]
            context: Optional context information
            
        Returns:
            result: Result of recursive reasoning
            depth: Actual recursion depth used
            confidence: Confidence in result
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize state
        current_state = x
        
        # Track recursion depth and states
        recursive_states = [current_state]
        
        # Initial confidence prediction
        initial_confidence = self.confidence_predictor(current_state)
        
        # Determine maximum depth based on initial confidence
        # Lower confidence -> potentially deeper reasoning needed
        confidence_factor = 1.0 - initial_confidence.mean().item()
        adaptive_max_depth = min(
            self.max_recursive_depth,
            self.min_recursive_depth + int(
                confidence_factor * (self.max_recursive_depth - self.min_recursive_depth)
            )
        )
        
        # Update metrics
        if self.training:
            self.total_queries += batch_size
        
        # Execute recursive reasoning
        depth = 0
        confidence = initial_confidence
        
        early_stop = False
        for d in range(adaptive_max_depth):
            depth = d + 1
            
            # Check if minimum depth reached and confidence is sufficient
            if depth >= self.min_recursive_depth and confidence.mean() >= self.confidence_threshold:
                early_stop = True
                break
            
            # Retrieve from memory if available
            if self.use_memory_augmentation and depth > 1:
                # Create memory query
                memory_query = self.memory_query(current_state)
                
                # Compute similarity with stored keys
                memory_similarity = F.cosine_similarity(
                    memory_query.unsqueeze(1),  # [batch_size, 1, memory_size]
                    self.memory_keys.unsqueeze(0),  # [1, num_memories, memory_size]
                    dim=2
                )
                
                # Apply mask for unused memory slots
                memory_mask = (self.memory_usage > 0).float()
                memory_similarity = memory_similarity * memory_mask
                
                # Get most similar memory
                best_match_idx = memory_similarity.argmax(dim=1)
                best_match_sim = memory_similarity.gather(1, best_match_idx.unsqueeze(1))
                
                # Use memory if similarity is high enough
                memory_values = self.memory_values[best_match_idx]
                current_state = torch.where(
                    best_match_sim.unsqueeze(2) > 0.9,
                    memory_values,
                    current_state
                )
            
            # Apply recursion
            next_state = self.reasoning_core(current_state)
            
            # Add to recursive states
            recursive_states.append(next_state)
            
            # Update current state
            current_state = next_state
            
            # Update confidence prediction
            confidence = self.confidence_predictor(current_state)
            
            # Track recursion steps
            if self.training:
                self.total_queries += batch_size
        
            # Track early stopping
            if early_stop and self.training:
                self.early_stops += batch_size
        
            # Store result in memory
            if self.use_memory_augmentation and self.training:
                self._update_memory(x, current_state)
        
        # Track early stopping
        if early_stop and self.training:
            self.early_stops += batch_size
        
        # Return final state, depth used, and confidence
        return current_state, depth, confidence
    
    def _update_memory(self, input_state, output_state):
        """
        Update episodic memory with new reasoning example
        
        Args:
            input_state: Input to reasoning process
            output_state: Output from reasoning process
        """
        batch_size = input_state.size(0)
        
        # Process only first item in batch for simplicity
        x = input_state[0].unsqueeze(0)
        y = output_state[0].unsqueeze(0)
        
        # Create memory key from input
        memory_key = self.memory_key(x)
        
        # Create memory value from output
        memory_value = self.memory_value(y)
        
        # Store in memory (circular buffer)
        idx = self.memory_counter % self.memory_keys.size(0)
        self.memory_keys[idx] = memory_key.squeeze(0)
        self.memory_values[idx] = memory_value.squeeze(0)
        self.memory_usage[idx] = 1.0
        
        # Update counter
        self.memory_counter += 1
    
    def get_average_recursion_depth(self):
        """Get average recursion depth used"""
        if self.total_queries.item() == 0:
            return 0.0
        return self.avg_recursion_depth
    
    def get_early_stop_ratio(self):
        """Get ratio of reasoning calls that stopped early"""
        if self.total_queries.item() == 0:
            return 0.0
        return self.early_stops.item() / self.total_queries.item()
    
    def reset_metrics(self):
        """Reset tracking metrics"""
        self.avg_recursion_depth = 0
        self.total_queries = 0
        self.early_stops = 0


class NeuralSymbolicReasoner(nn.Module):
    """
    Neural-symbolic reasoning module with dynamic theorem proving capabilities
    """
    
    def __init__(
        self,
        hidden_size: int,
        symbolic_embedding_size: int = 256,
        max_theorem_steps: int = 10,
        min_theorem_steps: int = 2,
        use_verification: bool = True,
        theorem_temperature: float = 0.8,
        confidence_threshold: float = 0.9,
        use_guided_search: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.symbolic_embedding_size = symbolic_embedding_size
        self.max_theorem_steps = max_theorem_steps
        self.min_theorem_steps = min_theorem_steps
        self.use_verification = use_verification
        self.theorem_temperature = theorem_temperature
        self.confidence_threshold = confidence_threshold
        self.use_guided_search = use_guided_search
        
        # Neural encoder for symbolic content
        self.symbol_encoder = nn.Sequential(
            nn.Linear(hidden_size, symbolic_embedding_size),
            nn.GELU(),
            nn.Linear(symbolic_embedding_size, symbolic_embedding_size)
        )
        
        # Theorem application network
        self.theorem_network = nn.Sequential(
            nn.Linear(symbolic_embedding_size * 2, symbolic_embedding_size),
            nn.GELU(),
            nn.Linear(symbolic_embedding_size, symbolic_embedding_size)
        )
        
        # Theorem selection policy
        self.theorem_policy = nn.Sequential(
            nn.Linear(symbolic_embedding_size, symbolic_embedding_size // 2),
            nn.GELU(),
            nn.Linear(symbolic_embedding_size // 2, symbolic_embedding_size // 4),
            nn.GELU()
            # Final layer depends on number of theorems
        )
        
        # Verification network
        if use_verification:
            self.verification_network = nn.Sequential(
                nn.Linear(symbolic_embedding_size * 2, symbolic_embedding_size),
                nn.GELU(),
                nn.Linear(symbolic_embedding_size, 1),
                nn.Sigmoid()
            )
        
        # Confidence predictor
        self.confidence_predictor = ConfidencePredictor(symbolic_embedding_size)
        
        # Decoder from symbolic to neural
        self.symbol_decoder = nn.Sequential(
            nn.Linear(symbolic_embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # For tracking metrics
        self.register_buffer('total_theorem_steps', torch.zeros(1))
        self.register_buffer('total_proof_attempts', torch.zeros(1))
        self.register_buffer('successful_proofs', torch.zeros(1))
        self.register_buffer('early_stops', torch.zeros(1))
    
    def forward(
        self,
        x: torch.Tensor,
        theorem_embeddings: torch.Tensor,
        theorem_names: List[str],
        goal_condition: Optional[torch.Tensor] = None
    ):
        """
        Execute neural-symbolic reasoning with dynamic theorem proving
        
        Args:
            x: Initial state representation [batch_size, hidden_size]
            theorem_embeddings: Embeddings for available theorems [num_theorems, symbolic_embedding_size]
            theorem_names: Names of theorems for debugging
            goal_condition: Optional goal condition to prove
            
        Returns:
            result: Result of symbolic reasoning
            proof_steps: List of theorem applications used
            confidence: Confidence in result
            valid: Whether proof is valid
        """
        batch_size = x.size(0)
        device = x.device
        
        # Encode input to symbolic representation
        symbolic_state = self.symbol_encoder(x)
        
        # Track theorem applications
        proof_steps = []
        proof_states = [symbolic_state]
        
        # Initial confidence
        initial_confidence = self.confidence_predictor(symbolic_state)
        
        # Adaptive theorem steps based on confidence
        confidence_factor = 1.0 - initial_confidence.mean().item()
        adaptive_max_steps = min(
            self.max_theorem_steps,
            self.min_theorem_steps + int(
                confidence_factor * (self.max_theorem_steps - self.min_theorem_steps)
            )
        )
        
        # Update metrics
        if self.training:
            self.total_proof_attempts += batch_size
        
        # Initialize current state and confidence
        current_state = symbolic_state
        confidence = initial_confidence
        
        # Execute theorem proving
        early_stop = False
        for step in range(adaptive_max_steps):
            # Check for early stopping if minimum steps reached
            if step >= self.min_theorem_steps and confidence.mean() >= self.confidence_threshold:
                early_stop = True
                break
            
            # Compute theorem selection policy
            policy_logits = self.theorem_policy(current_state).matmul(theorem_embeddings.t())
            
            # Apply temperature to control randomness
            policy = F.softmax(policy_logits / self.theorem_temperature, dim=-1)
            
            # Select theorem
            if self.use_guided_search and self.training:
                # Explore different theorems during training
                theorem_idx = torch.multinomial(policy, 1).squeeze(-1)
            else:
                # Use best theorem during inference
                theorem_idx = policy.argmax(dim=-1)
            
            # Get theorem embedding
            selected_theorem = theorem_embeddings[theorem_idx]
            
            # Apply theorem
            theorem_input = torch.cat([current_state, selected_theorem], dim=-1)
            next_state = self.theorem_network(theorem_input)
            
            # Store proof step
            proof_steps.append(theorem_names[theorem_idx[0].item()])
            proof_states.append(next_state)
            
            # Update current state
            current_state = next_state
            
            # Update confidence
            confidence = self.confidence_predictor(current_state)
            
            # Track theorem steps
            if self.training:
                self.total_theorem_steps += batch_size
        
        # Track early stopping
        if early_stop and self.training:
            self.early_stops += batch_size
        
        # Verify proof if required
        valid = torch.ones(batch_size, device=device, dtype=torch.bool)
        if self.use_verification and goal_condition is not None:
            # Compute verification score
            verification_input = torch.cat([current_state, goal_condition], dim=-1)
            verification_score = self.verification_network(verification_input)
            
            # Check if proof is valid
            valid = verification_score >= 0.5
            
            # Update metrics
            if self.training:
                self.successful_proofs += valid.sum()
        
        # Decode back to neural representation
        result = self.symbol_decoder(current_state)
        
        return result, proof_steps, confidence, valid
    
    def get_average_theorem_steps(self):
        """Get average number of theorem steps used"""
        if self.total_proof_attempts.item() == 0:
            return 0.0
        return self.total_theorem_steps.item() / self.total_proof_attempts.item()
    
    def get_success_rate(self):
        """Get success rate of proofs"""
        if self.total_proof_attempts.item() == 0:
            return 0.0
        return self.successful_proofs.item() / self.total_proof_attempts.item()
    
    def get_early_stop_ratio(self):
        """Get ratio of proofs that stopped early"""
        if self.total_proof_attempts.item() == 0:
            return 0.0
        return self.early_stops.item() / self.total_proof_attempts.item()
    
    def reset_metrics(self):
        """Reset tracking metrics"""
        self.total_theorem_steps.zero_()
        self.total_proof_attempts.zero_()
        self.successful_proofs.zero_()
        self.early_stops.zero_()


class ReasoningManager(nn.Module):
    """
    Manages different reasoning components and selects the appropriate one based on input.
    Serves as a unified interface for all adaptive reasoning approaches.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_recursive_depth: int = 3,
        max_mcts_simulations: int = 100,
        use_mcts: bool = True,
        use_recursive: bool = True,
        use_symbolic: bool = False,
        use_hybrid: bool = True
    ):
        """
        Initialize the reasoning manager
        
        Args:
            hidden_size: Hidden state dimension
            max_recursive_depth: Maximum depth for recursive reasoning
            max_mcts_simulations: Maximum simulations for MCTS reasoning
            use_mcts: Whether to use Monte Carlo Tree Search reasoning
            use_recursive: Whether to use recursive reasoning
            use_symbolic: Whether to use symbolic reasoning
            use_hybrid: Whether to use hybrid reasoning approaches
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_recursive_depth = max_recursive_depth
        self.max_mcts_simulations = max_mcts_simulations
        
        # Initialize reasoners
        self.reasoners = {}
        
        if use_recursive:
            self.reasoners["recursive"] = AdaptiveRecursiveReasoner(
                hidden_size=hidden_size,
                max_recursive_depth=max_recursive_depth
            )
        
        if use_mcts:
            self.reasoners["mcts"] = AdaptiveMCTSReasoner(
                hidden_size=hidden_size,
                max_simulations=max_mcts_simulations
            )
        
        # Reasoner selection network
        self.selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, len(self.reasoners) if self.reasoners else 1)
        )
        
        # Reasoning type embedding
        self.reasoning_type_embedding = nn.Embedding(
            len(self.reasoners) + 1 if self.reasoners else 2, 
            hidden_size
        )
        
        # Reasoning integration
        self.reasoning_integration = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Usage statistics
        self.reasoner_usage = {name: 0 for name in self.reasoners}
        self.total_usage = 0
    
    def forward(self, x, context=None, **kwargs):
        """
        Apply adaptive reasoning based on input
        
        Args:
            x: Input hidden states [batch_size, seq_len, hidden_size]
            context: Optional context information
            **kwargs: Additional arguments for specific reasoners
            
        Returns:
            Refined hidden states after reasoning
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # If no reasoners available, return input
        if not self.reasoners:
            return x
        
        # Get mean representation for reasoning type selection
        mean_repr = x.mean(dim=1)  # [batch_size, hidden_size]
        
        # Select reasoning type
        reasoning_logits = self.selector(mean_repr)  # [batch_size, num_reasoners]
        reasoning_probs = F.softmax(reasoning_logits, dim=-1)
        
        # In inference, use the most likely reasoner
        if not self.training:
            selected_reasoner_idx = reasoning_probs.argmax(dim=-1)  # [batch_size]
            
            # Map indices to reasoner names
            reasoner_names = list(self.reasoners.keys())
            selected_reasoners = [reasoner_names[idx.item()] for idx in selected_reasoner_idx]
            
            # Update usage statistics
            for name in selected_reasoners:
                self.reasoner_usage[name] += 1
            self.total_usage += batch_size
            
            # Apply selected reasoner to each batch item
            outputs = []
            for i, reasoner_name in enumerate(selected_reasoners):
                # Get the selected reasoner
                reasoner = self.reasoners[reasoner_name]
                
                # Apply reasoning
                x_i = x[i].unsqueeze(0)  # [1, seq_len, hidden_size]
                
                # Different handling based on reasoner type
                if reasoner_name == "mcts" and "actions" in kwargs:
                    # For MCTS reasoner, we need actions and transition function
                    actions = kwargs.get("actions", [])
                    action_embeddings = kwargs.get("action_embeddings", None)
                    state_transition_fn = kwargs.get("state_transition_fn", lambda s, a: s)
                    reward_fn = kwargs.get("reward_fn", lambda s: 0)
                    
                    # Execute MCTS reasoning
                    _, _, reasoning_output = reasoner(
                        mean_repr[i].unsqueeze(0),  # [1, hidden_size]
                        actions,
                        action_embeddings,
                        state_transition_fn,
                        reward_fn
                    )
                    
                    # Expand to match input shape
                    reasoning_output = reasoning_output.unsqueeze(1).expand(-1, seq_len, -1)
                else:
                    # For other reasoners
                    reasoning_output, _, _ = reasoner(x_i)
                
                outputs.append(reasoning_output)
            
            # Combine outputs
            output = torch.cat(outputs, dim=0)
            
        else:
            # During training, we can use soft selection for better gradient flow
            # Initialize output with zeros
            output = torch.zeros_like(x)
            
            # Apply all reasoners and combine with weighted average
            for i, (name, reasoner) in enumerate(self.reasoners.items()):
                # Get weight for this reasoner
                weight = reasoning_probs[:, i].unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
                
                # Apply reasoning
                if name == "mcts" and "actions" in kwargs:
                    # For MCTS reasoner
                    actions = kwargs.get("actions", [])
                    action_embeddings = kwargs.get("action_embeddings", None)
                    state_transition_fn = kwargs.get("state_transition_fn", lambda s, a: s)
                    reward_fn = kwargs.get("reward_fn", lambda s: 0)
                    
                    # Execute MCTS reasoning (batch processing)
                    _, _, reasoning_output = reasoner(
                        mean_repr,  # [batch_size, hidden_size]
                        actions,
                        action_embeddings,
                        state_transition_fn,
                        reward_fn
                    )
                    
                    # Expand to match input shape
                    reasoning_output = reasoning_output.unsqueeze(1).expand(-1, seq_len, -1)
                else:
                    # For other reasoners
                    reasoning_output, _, _ = reasoner(x)
                
                # Add weighted output
                output = output + weight * reasoning_output
                
                # Update usage statistics (weighted by probability)
                self.reasoner_usage[name] += reasoning_probs[:, i].sum().item()
            
            self.total_usage += batch_size
        
        # Get reasoning type embedding
        if self.training:
            # During training, use soft mixture of embeddings
            reasoning_type_idx = torch.arange(len(self.reasoners), device=device)
            type_embeddings = self.reasoning_type_embedding(reasoning_type_idx)
            
            # Weighted combination based on selection probabilities
            type_embedding = torch.matmul(reasoning_probs, type_embeddings)
        else:
            # During inference, use selected reasoning type
            type_embedding = self.reasoning_type_embedding(selected_reasoner_idx)
        
        # Expand to match sequence dimension
        type_embedding = type_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Integrate reasoning output with type embedding
        integrated_output = self.reasoning_integration(
            torch.cat([output, type_embedding], dim=-1)
        )
        
        # Apply residual connection and layer norm
        final_output = self.layer_norm(x + integrated_output)
        
        return final_output
    
    def get_reasoner_usage(self):
        """
        Get statistics on reasoner usage
        
        Returns:
            Dictionary with usage statistics
        """
        if self.total_usage == 0:
            return {name: 0.0 for name in self.reasoners}
        
        # Compute usage percentages
        usage_stats = {
            name: count / max(1, self.total_usage) 
            for name, count in self.reasoner_usage.items()
        }
        
        return usage_stats
    
    def reset_metrics(self):
        """Reset usage statistics and metrics of all reasoners"""
        self.reasoner_usage = {name: 0 for name in self.reasoners}
        self.total_usage = 0
        
        # Reset metrics of individual reasoners
        for reasoner in self.reasoners.values():
            if hasattr(reasoner, 'reset_metrics'):
                reasoner.reset_metrics() 