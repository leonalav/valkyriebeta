"""
Monte Carlo Tree Search (MCTS) Reasoner for deep learning models.
This module provides MCTS-based reasoning capabilities for LLMs and transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from dataclasses import dataclass
from collections import defaultdict
import concurrent.futures
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class MCTSConfig:
    """Configuration for MCTS reasoner"""
    hidden_size: int = 768
    max_simulations: int = 50
    min_simulations: int = 10
    exploration_weight: float = 1.0
    value_scale: float = 2.0
    use_policy_network: bool = True
    use_value_network: bool = True
    temperature: float = 1.0
    max_search_depth: int = 5
    use_adaptive_simulation_count: bool = True
    confidence_threshold: float = 0.9
    use_dirichlet_noise: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    prune_suboptimal_actions: bool = True
    use_simulation_optimization: bool = True
    simulation_batch_size: int = 16
    use_state_compression: bool = True  # Enable state compression by default
    enable_progressive_widening: bool = False
    store_reasoning_trace: bool = True
    
    # New optimization parameters
    state_compression_ratio: float = 0.25  # Ratio for state compression
    use_value_cache: bool = True  # Enable value function caching
    value_cache_capacity: int = 10000  # Maximum cache size for values
    use_hybrid_search: bool = True  # Use hybrid MCTS-Beam approach
    beam_size: int = 4  # Beam size for hybrid search
    use_async_simulation: bool = False  # Use async parallel simulation
    num_parallel_workers: int = 4  # Number of parallel workers
    use_dynamic_confidence: bool = True  # Dynamic confidence thresholds
    max_action_space_size: int = 128  # Restricted action space size
    use_action_space_reduction: bool = True  # Enable action space reduction
    essential_token_boost: float = 0.05  # Boost for essential tokens

class CompressedStateNode:
    """
    Efficient state representation using dimensionality reduction.
    Uses SVD/PCA-like compression to reduce memory footprint.
    """
    
    def __init__(self, hidden_state, compression_ratio=0.25):
        """
        Compress a hidden state representation for memory efficiency.
        
        Args:
            hidden_state: Original hidden state tensor
            compression_ratio: Ratio of dimensions to keep (0.0-1.0)
        """
        if hidden_state is None:
            self.is_empty = True
            return
            
        self.is_empty = False
        self.original_shape = hidden_state.shape
        self.device = hidden_state.device
        
        # Flatten if needed
        if hidden_state.dim() > 2:
            hidden_state = hidden_state.view(hidden_state.size(0), -1)
        
        # Determine compression dimensions
        hidden_size = hidden_state.size(-1)
        k = max(1, int(hidden_size * compression_ratio))
        
        try:
            # Use SVD for compression
            u, s, v = torch.svd(hidden_state)
            
            # Keep only top-k components
            self.compressed_u = u[:, :k]
            self.compressed_s = s[:k]
            self.compressed_v = v[:, :k]
        except Exception as e:
            # Fallback to simpler compression if SVD fails
            logger.warning(f"SVD compression failed: {e}, using simpler compression")
            # Simple compression by keeping subset of dimensions
            indices = torch.randperm(hidden_size)[:k]
            self.compressed_state = hidden_state[:, indices]
            self.compression_indices = indices
            self.svd_compression = False
            return
            
        self.svd_compression = True
    
    def reconstruct(self):
        """Reconstruct approximation of original state"""
        if self.is_empty:
            return None
            
        if self.svd_compression:
            # Reconstruct using SVD components
            reconstructed = self.compressed_u @ torch.diag(self.compressed_s) @ self.compressed_v.t()
            
            # Reshape to original dimensions
            return reconstructed.view(self.original_shape)
        else:
            # Reconstruct using subset of dimensions
            # Create empty tensor of original size
            reconstructed = torch.zeros(self.original_shape[0], 
                                       self.compressed_state.size(-1) * (self.original_shape[-1] // self.compressed_state.size(-1)),
                                       device=self.device)
            # Fill in compressed dimensions
            reconstructed[:, self.compression_indices] = self.compressed_state
            return reconstructed.view(self.original_shape)
    
    def size_reduction_ratio(self):
        """Calculate memory reduction ratio"""
        if self.is_empty:
            return 0
            
        if self.svd_compression:
            original_params = self.original_shape[0] * self.original_shape[1]
            compressed_params = (self.compressed_u.numel() + self.compressed_s.numel() + 
                                self.compressed_v.numel())
            return compressed_params / original_params
        else:
            return self.compressed_state.numel() / (self.original_shape[0] * self.original_shape[1])

class MCTSNode:
    """
    Node in the MCTS search tree.
    Tracks statistics for action selection and value estimation.
    """
    
    def __init__(self, state=None, action=None, parent=None, prior=0.0, compression_ratio=0.25):
        self.compressed_state = CompressedStateNode(state, compression_ratio) if state is not None else None
        self.action = action  # Action that led to this state
        self.parent = parent
        self.prior = prior  # Prior probability from policy network
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # Maps actions to child nodes
        self.reward = 0.0  # Immediate reward for reaching this state
        self.expanded = False
        
        # For progressive widening (if enabled)
        self.available_actions = []
        self.unexplored_actions = []
        
        # For reasoning trace
        self.reasoning_steps = []
    
    @property
    def value(self):
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
        
    @property
    def state(self):
        """Get reconstructed state"""
        if self.compressed_state is None:
            return None
        return self.compressed_state.reconstruct()
        
    @state.setter
    def state(self, new_state):
        """Set state with compression"""
        if new_state is None:
            self.compressed_state = None
        else:
            compression_ratio = 0.25  # Default ratio
            if self.parent and hasattr(self.parent, 'compression_ratio'):
                compression_ratio = self.parent.compression_ratio
            self.compressed_state = CompressedStateNode(new_state, compression_ratio)
            
    def set_state_with_compression(self, new_state, compression_ratio=0.25):
        """Explicitly set state with specified compression ratio"""
        self.compression_ratio = compression_ratio
        self.compressed_state = CompressedStateNode(new_state, compression_ratio)
    
    def add_exploration_noise(self, dirichlet_alpha, dirichlet_weight):
        """Add Dirichlet noise to prior probabilities for exploration"""
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        
        for action, n in zip(actions, noise):
            child = self.children[action]
            # Ensure prior is a float (not a list or tuple)
            if isinstance(child.prior, (list, tuple)):
                # Handle potentially nested lists/tuples
                prior_value = child.prior
                while isinstance(prior_value, (list, tuple)) and prior_value:
                    prior_value = prior_value[0]
                child.prior = float(prior_value)
            child.prior = child.prior * (1 - dirichlet_weight) + n * dirichlet_weight
    
    def select_child(self, exploration_weight):
        """Select child node using UCB formula"""
        # Calculate UCB score for each child
        ucb_scores = {
            action: self._ucb_score(child, exploration_weight)
            for action, child in self.children.items()
        }
        
        # Select action with highest UCB score
        action = max(ucb_scores.items(), key=lambda x: x[1])[0]
        return action, self.children[action]
    
    def _ucb_score(self, child, exploration_weight):
        """Calculate UCB score for a child node"""
        # Ensure prior is a float
        if isinstance(child.prior, (list, tuple)):
            # Handle potentially nested lists/tuples
            prior_value = child.prior
            while isinstance(prior_value, (list, tuple)) and prior_value:
                prior_value = prior_value[0]
            prior = float(prior_value)
        else:
            prior = float(child.prior)
        
        # Exploration bonus
        exploration = exploration_weight * prior * math.sqrt(self.visit_count) / (child.visit_count + 1)
        
        # Value component
        if child.visit_count == 0:
            value = 0.0
        else:
            value = child.value
        
        return value + exploration
    
    def expand(self, state, actions, priors, rewards=None):
        """Expand node with given actions and their prior probabilities"""
        self.expanded = True
        self.state = state
        self.available_actions = actions.copy() if actions else []
        self.unexplored_actions = actions.copy() if actions else []
        
        # Make sure priors is the right length
        if priors is not None and len(priors) != len(actions):
            if len(priors) > len(actions):
                priors = priors[:len(actions)]
            else:
                # Pad with uniform probability
                uniform_prob = 1.0 / len(actions)
                priors.extend([uniform_prob] * (len(actions) - len(priors)))
        
        # Initialize children with prior probabilities
        for i, action in enumerate(actions):
            if priors is not None and i < len(priors):
                prior = priors[i]
            else:
                prior = 1.0 / len(actions)
                
            reward = rewards[i] if rewards is not None and i < len(rewards) else 0.0
            self.children[action] = MCTSNode(
                state=None,  # State will be determined when this child is selected
                action=action,
                parent=self,
                prior=prior
            )
            self.children[action].reward = reward
    
    def update(self, value):
        """Update node statistics with backpropagated value"""
        self.visit_count += 1
        self.value_sum += value
        
    def record_reasoning_step(self, message):
        """Record a reasoning step for explainability"""
        self.reasoning_steps.append(message)
    
    def get_visit_count_distribution(self):
        """Get distribution of visit counts for all children"""
        if not self.children:
            return []
        
        # Get visit counts for all actions
        visits = np.array([child.visit_count for child in self.children.values()])
        
        # Normalize to get distribution
        if visits.sum() > 0:
            distribution = visits / visits.sum()
        else:
            distribution = np.ones_like(visits) / len(visits)
            
        return distribution.tolist()

class MCTSCache:
    """
    Cache for MCTS value function results to avoid redundant computation.
    Uses an efficient state hashing mechanism for fast lookups.
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize cache with specified capacity.
        
        Args:
            capacity: Maximum number of entries in cache
        """
        self.cache = {}
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self.collisions = 0
        
    def get_state_hash(self, state):
        """
        Create an efficient hash of state tensor.
        
        For language models, uses a combination of:
        - Downsampled tensor values
        - Mean and std values
        - Selected key positions
        
        Args:
            state: Hidden state tensor
            
        Returns:
            A hashable value representing the state
        """
        if state is None:
            return None
            
        # Convert to CPU for hashing
        if isinstance(state, torch.Tensor):
            # If it's a large tensor, compute a more efficient hash
            if state.numel() > 1000:
                # Downsample by taking sparse samples
                step = max(1, state.size(-1) // 100)
                samples = state[..., ::step].detach().cpu()
                
                # Get summary statistics
                mean_val = torch.mean(state).cpu().item()
                std_val = torch.std(state).cpu().item()
                
                # Get values at key positions (beginning, middle, end)
                if state.dim() > 1:
                    begin = tuple(state[0, :10].detach().cpu().tolist())
                    middle_idx = state.size(0) // 2
                    middle = tuple(state[middle_idx, :10].detach().cpu().tolist())
                    end = tuple(state[-1, :10].detach().cpu().tolist())
                else:
                    begin = tuple(state[:10].detach().cpu().tolist())
                    middle_idx = state.size(0) // 2
                    middle = tuple(state[middle_idx:middle_idx+10].detach().cpu().tolist())
                    end = tuple(state[-10:].detach().cpu().tolist())
                
                # Combine everything into a hashable tuple
                return hash((mean_val, std_val, begin, middle, end))
            else:
                # For small tensors, use the full tensor for hashing
                return hash(tuple(state.detach().cpu().flatten().tolist()))
        else:
            # Fallback for non-tensor states
            return hash(state)
        
    def lookup_value(self, state):
        """
        Look up value for a state in the cache.
        
        Args:
            state: State tensor to look up
            
        Returns:
            Cached value or None if not found
        """
        state_hash = self.get_state_hash(state)
        if state_hash in self.cache:
            self.hits += 1
            return self.cache[state_hash]
        else:
            self.misses += 1
            return None
        
    def store_value(self, state, value):
        """
        Store value for a state in the cache.
        
        Args:
            state: State tensor
            value: Value to cache
        """
        state_hash = self.get_state_hash(state)
        
        # Check for hash collisions (different states with same hash)
        if state_hash in self.cache:
            self.collisions += 1
            
        # Implement cache eviction if at capacity
        if len(self.cache) >= self.capacity and state_hash not in self.cache:
            # Simple random eviction policy
            key_to_remove = random.choice(list(self.cache.keys()))
            self.cache.pop(key_to_remove)
            
        # Store the value
        self.cache[state_hash] = value
        
    def get_stats(self):
        """Get cache performance statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / max(1, total)
        return {
            "size": len(self.cache),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses, 
            "hit_rate": hit_rate,
            "collisions": self.collisions
        }
        
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.collisions = 0

class AdaptiveActionSpace:
    """
    Intelligently restricts action space to manage branching factor.
    
    For language models, this can dramatically reduce search complexity
    by focusing on likely tokens based on context.
    """
    
    def __init__(self, max_actions=128, essential_token_boost=0.05):
        """
        Initialize adaptive action space.
        
        Args:
            max_actions: Maximum number of actions to consider
            essential_token_boost: Probability boost for essential tokens
        """
        self.max_actions = max_actions
        self.essential_token_boost = essential_token_boost
        
        # These should be defined based on the tokenizer
        # Common tokens that are important for structural coherence
        self.essential_tokens = []
        self.token_frequencies = {}
        
    def set_essential_tokens(self, tokens):
        """Set essential tokens that should always be considered"""
        self.essential_tokens = tokens
        
    def set_token_frequencies(self, frequencies):
        """Set token frequency information for better pruning"""
        self.token_frequencies = frequencies
        
    def restrict_action_space(self, state, logits, tokenizer=None):
        """
        Restrict action space based on state and logits.
        
        Args:
            state: Current state representation
            logits: Full logits from model prediction
            tokenizer: Optional tokenizer for token-specific logic
            
        Returns:
            restricted_actions: List of selected action indices
            restricted_priors: Probability distribution over restricted actions
        """
        # Get top-k actions by logits
        topk_values, topk_indices = torch.topk(
            logits, min(self.max_actions, logits.size(-1))
        )
        
        # Convert to softmax probabilities
        topk_probs = F.softmax(topk_values, dim=-1)
        
        # If no tokenizer, just return top-k
        if tokenizer is None:
            return topk_indices.tolist(), topk_probs.tolist()
            
        # Get list of actions to keep
        actions_to_keep = set(topk_indices.tolist())
        
        # Add essential tokens if they exist
        if self.essential_tokens and tokenizer:
            for token in self.essential_tokens:
                if isinstance(token, str):
                    token_id = tokenizer.encode(token, add_special_tokens=False)
                    if token_id:
                        token_id = token_id[0]
                        actions_to_keep.add(token_id)
                else:
                    # Assume it's already a token ID
                    actions_to_keep.add(token)
        
        # Convert to list and limit to max size
        restricted_actions = list(actions_to_keep)[:self.max_actions]
        
        # Get probabilities for restricted actions
        restricted_logits = logits[restricted_actions]
        restricted_probs = F.softmax(restricted_logits, dim=-1)
        
        # Apply boost to essential tokens
        if self.essential_token_boost > 0:
            for i, action in enumerate(restricted_actions):
                if action in self.essential_tokens:
                    restricted_probs[i] += self.essential_token_boost
                    
            # Renormalize
            restricted_probs = restricted_probs / restricted_probs.sum()
            
        return restricted_actions, restricted_probs.tolist()

class DynamicConfidenceThreshold:
    """
    Dynamically adjusts confidence thresholds based on context.
    
    Adapts thresholds based on:
    - Generation position/depth
    - Token uncertainty (entropy)
    - Remaining sequence length
    - Overall model confidence
    """
    
    def __init__(self, base_threshold=0.9, min_threshold=0.6, max_threshold=0.95):
        """
        Initialize dynamic confidence threshold.
        
        Args:
            base_threshold: Base confidence threshold
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
        """
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
    def calculate_threshold(self, depth, remaining_tokens, entropy, recent_values=None):
        """
        Calculate adaptive confidence threshold.
        
        Args:
            depth: Current depth in generation
            remaining_tokens: Estimated remaining tokens to generate
            entropy: Entropy of current token distribution (uncertainty)
            recent_values: Recent evaluation values (optional)
            
        Returns:
            Adjusted confidence threshold
        """
        # Start with base threshold
        threshold = self.base_threshold
        
        # Adjust based on depth in sequence
        # Early in generation: lower threshold to explore more
        # Later in generation: higher threshold for convergence
        depth_factor = min(1.0, depth / 100) * 0.1
        
        # Adjust based on entropy (prediction uncertainty)
        # Higher entropy: lower threshold to explore more
        # Lower entropy: higher threshold to converge faster
        entropy_factor = min(entropy / 4.0, 1.0) * 0.2
        
        # Adjust based on remaining tokens
        # More tokens remaining: lower threshold to explore options
        # Few tokens remaining: higher threshold to finalize
        remaining_factor = min(remaining_tokens / 100, 1.0) * 0.1
        
        # Adjust based on recent values if available
        recent_factor = 0.0
        if recent_values and len(recent_values) > 1:
            # If values are improving, raise threshold
            # If values are declining, lower threshold
            recent_trend = recent_values[-1] - recent_values[0]
            recent_factor = max(-0.1, min(0.1, recent_trend))
        
        # Calculate final threshold
        threshold = threshold + depth_factor - entropy_factor - remaining_factor + recent_factor
        
        # Clamp to allowed range
        return max(self.min_threshold, min(self.max_threshold, threshold))
        
    def get_threshold_for_node(self, node, max_depth, entropy=None):
        """
        Get threshold for a specific node.
        
        Args:
            node: MCTS node
            max_depth: Maximum search depth
            entropy: Entropy of action distribution (optional)
            
        Returns:
            Node-specific confidence threshold
        """
        # Calculate depth
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
            
        # If entropy not provided, estimate from node's children
        if entropy is None and node.children:
            visit_counts = [child.visit_count for child in node.children.values()]
            if sum(visit_counts) > 0:
                probs = [count/sum(visit_counts) for count in visit_counts]
                entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probs)
            else:
                entropy = math.log(len(node.children)) if node.children else 0
        elif entropy is None:
            entropy = 0
            
        # Estimate remaining tokens
        remaining_tokens = max_depth - depth
        
        # Get recent values
        recent_values = []
        if node.parent and hasattr(node.parent, 'children'):
            recent_values = [child.value for child in node.parent.children.values() 
                            if child.visit_count > 0]
            
        return self.calculate_threshold(depth, remaining_tokens, entropy, recent_values)

class HybridMCTSBeamSearch:
    """
    Hybrid MCTS-Beam Search for efficient reasoning.
    
    Uses beam search to quickly identify promising candidates,
    then applies MCTS for in-depth analysis of those candidates.
    """
    
    def __init__(self, beam_size=4, temperature=1.0, mcts_reasoner=None):
        """
        Initialize hybrid search.
        
        Args:
            beam_size: Size of the beam
            temperature: Temperature for sampling
            mcts_reasoner: MCTS reasoner instance
        """
        self.beam_size = beam_size
        self.temperature = temperature
        self.mcts_reasoner = mcts_reasoner
        
    def beam_search(self, state, action_logits, available_actions, top_k=None):
        """
        Perform beam search to find promising candidates.
        
        Args:
            state: Current state
            action_logits: Logits for available actions
            available_actions: List of available actions
            top_k: Number of candidates to return
            
        Returns:
            List of (action, score) pairs
        """
        if top_k is None:
            top_k = self.beam_size
            
        # Get probabilities from logits
        probs = F.softmax(action_logits / self.temperature, dim=-1)
        
        # Get top-k actions and probabilities
        top_probs, top_indices = torch.topk(probs, min(top_k, probs.size(-1)))
        
        # Convert to list of (action, score) pairs
        candidates = []
        for i in range(top_indices.size(0)):
            idx = top_indices[i].item()
            if idx < len(available_actions):
                action = available_actions[idx]
                score = top_probs[i].item()
                candidates.append((action, score))
                
        return candidates
        
    def allocate_simulations(self, candidates, total_simulations):
        """
        Allocate simulation budget across candidates.
        
        Args:
            candidates: List of (action, score) pairs
            total_simulations: Total simulation budget
            
        Returns:
            List of simulation counts for each candidate
        """
        # Extract scores
        scores = [score for _, score in candidates]
        total_score = sum(scores)
        
        # Allocate proportionally to scores
        if total_score > 0:
            allocations = [int(score / total_score * total_simulations) for score in scores]
        else:
            # Equal allocation if all scores are zero
            allocations = [total_simulations // len(candidates)] * len(candidates)
            
        # Ensure minimum simulations per candidate
        min_simulations = max(1, total_simulations // (len(candidates) * 2))
        for i in range(len(allocations)):
            if allocations[i] < min_simulations:
                allocations[i] = min_simulations
                
        # Adjust to match total budget
        while sum(allocations) > total_simulations:
            # Remove from highest allocation
            max_idx = allocations.index(max(allocations))
            allocations[max_idx] -= 1
            
        while sum(allocations) < total_simulations:
            # Add to highest score
            max_idx = scores.index(max(scores))
            allocations[max_idx] += 1
            
        return allocations
        
    def search(self, state, action_logits, available_actions, transition_fn, reward_fn, 
              total_simulations, constraints=None):
        """
        Perform hybrid search.
        
        Args:
            state: Current state
            action_logits: Logits for available actions
            available_actions: List of available actions
            transition_fn: Function to get next state given state and action
            reward_fn: Function to get reward given state and action
            total_simulations: Total simulation budget
            constraints: Optional constraints on search
            
        Returns:
            Best action and its value
        """
        # Perform beam search to get candidates
        candidates = self.beam_search(state, action_logits, available_actions)
        
        # Allocate simulation budget
        allocations = self.allocate_simulations(candidates, total_simulations)
        
        # Run MCTS on each candidate
        results = []
        for i, ((action, score), simulations) in enumerate(zip(candidates, allocations)):
            if simulations <= 0:
                continue
                
            # Get next state for this action
            next_state = transition_fn(state, action)
            
            # Run MCTS from this state
            # Use a portion of the simulation budget
            if self.mcts_reasoner is not None:
                value = self.mcts_reasoner._evaluate_with_simulations(
                    next_state, available_actions, transition_fn, reward_fn, simulations
                )
            else:
                # Fallback if no MCTS reasoner
                value = score
                
            results.append((action, value))
            
        # Return best action
        if results:
            best_action, best_value = max(results, key=lambda x: x[1])
            return best_action, best_value
        else:
            # Fallback to highest beam search score
            return candidates[0][0], candidates[0][1]

class AsyncMCTS:
    """
    Asynchronous parallel MCTS simulation.
    
    Runs multiple simulations in parallel for higher throughput,
    especially beneficial for larger models and batch processing.
    """
    
    def __init__(self, num_workers=4, timeout=10):
        """
        Initialize async MCTS.
        
        Args:
            num_workers: Number of worker threads
            timeout: Timeout for parallel execution
        """
        self.num_workers = num_workers
        self.timeout = timeout
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.node_locks = {}  # Locks for thread-safe node updates
        
    def get_node_lock(self, node):
        """Get a lock for a specific node"""
        node_id = id(node)
        if node_id not in self.node_locks:
            self.node_locks[node_id] = Lock()
        return self.node_locks[node_id]
        
    def run_parallel_simulations(self, root_node, num_simulations, 
                               select_fn, expand_fn, evaluate_fn, backpropagate_fn):
        """
        Run multiple MCTS simulations in parallel.
        
        Args:
            root_node: Root node of search tree
            num_simulations: Number of simulations to run
            select_fn: Function to select path through tree
            expand_fn: Function to expand a leaf node
            evaluate_fn: Function to evaluate a leaf node
            backpropagate_fn: Function to backpropagate values
            
        Returns:
            root_node: Updated root node
        """
        # Create a shared context for functions
        context = {
            "node_locks": self.node_locks,
            "select_fn": select_fn,
            "expand_fn": expand_fn,
            "evaluate_fn": evaluate_fn,
            "backpropagate_fn": backpropagate_fn
        }
        
        # Run simulations in parallel
        futures = []
        for _ in range(num_simulations):
            futures.append(
                self.executor.submit(
                    self._run_single_simulation, root_node, context
                )
            )
            
        # Process results as they complete
        completed = 0
        for future in concurrent.futures.as_completed(futures, timeout=self.timeout):
            try:
                result = future.result()
                completed += 1
            except Exception as e:
                logger.error(f"Error in parallel simulation: {e}")
                
        # Clean up locks for garbage-collected nodes
        self._clean_locks()
                
        return root_node, completed
        
    def _run_single_simulation(self, root_node, context):
        """
        Run a single MCTS simulation.
        
        Args:
            root_node: Root node of search tree
            context: Shared context with functions
            
        Returns:
            True if simulation was successful
        """
        try:
            # Extract functions from context
            select_fn = context["select_fn"]
            expand_fn = context["expand_fn"]
            evaluate_fn = context["evaluate_fn"]
            backpropagate_fn = context["backpropagate_fn"]
            
            # Selection phase - thread safe with locks
            search_path = []
            node = root_node
            
            # Traverse tree with appropriate locking
            while node.expanded and node.children:
                # Get lock for current node
                with self.get_node_lock(node):
                    # Select child
                    action, node = select_fn(node)
                
                search_path.append(node)
                
            # Expansion phase - need lock for selected leaf node
            with self.get_node_lock(node):
                if not node.expanded:
                    expand_fn(node)
                    
            # Evaluation phase
            value = evaluate_fn(node)
            
            # Backpropagation phase - lock each node during update
            for node in reversed(search_path):
                with self.get_node_lock(node):
                    backpropagate_fn(node, value)
                    
            return True
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return False
            
    def _clean_locks(self):
        """Clean up locks for nodes that may have been garbage collected"""
        # Simple implementation: rebuild with existing references
        # A more sophisticated version would use weak references
        current_locks = {}
        for node_id, lock in self.node_locks.items():
            if lock._is_owned():  # Check if lock is currently held
                current_locks[node_id] = lock
                
        self.node_locks = current_locks
        
    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=True)

class MCTSReasoner(nn.Module):
    """
    Monte Carlo Tree Search reasoning module for enhanced decision making.
    
    This module uses MCTS to simulate future states and explore different reasoning paths,
    allowing for more strategic and long-term thinking in language models.
    """
    
    def __init__(self, config: MCTSConfig = None, **kwargs):
        """
        Initialize the MCTS reasoner.
        
        Args:
            config: MCTS configuration
            **kwargs: Additional parameters to override config
        """
        super().__init__()
        self.config = config or MCTSConfig()
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Policy network for action prior probabilities
        if self.config.use_policy_network:
            self.policy_network = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
            )
        else:
            self.policy_network = None
        
        # Value network for state evaluation
        if self.config.use_value_network:
            self.value_network = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.config.hidden_size, 1),
                nn.Tanh()  # Output in [-1, 1] range
            )
        else:
            self.value_network = None
        
        # State compression for memory efficiency (optional)
        if self.config.use_state_compression:
            self.state_encoder = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.config.hidden_size // 2, self.config.hidden_size // 4)
            )
            
            self.state_decoder = nn.Sequential(
                nn.Linear(self.config.hidden_size // 4, self.config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.config.hidden_size // 2, self.config.hidden_size)
            )
        
        # Confidence predictor for adaptive simulation count
        if self.config.use_adaptive_simulation_count:
            self.confidence_predictor = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.config.hidden_size // 2, 1),
                nn.Sigmoid()
            )
        
        # Initialize optimization components
        if self.config.use_value_cache:
            self.value_cache = MCTSCache(capacity=self.config.value_cache_capacity)
        else:
            self.value_cache = None
            
        if self.config.use_action_space_reduction:
            self.action_space_adapter = AdaptiveActionSpace(
                max_actions=self.config.max_action_space_size,
                essential_token_boost=self.config.essential_token_boost
            )
        else:
            self.action_space_adapter = None
            
        if self.config.use_dynamic_confidence:
            self.confidence_threshold_adapter = DynamicConfidenceThreshold(
                base_threshold=self.config.confidence_threshold
            )
        else:
            self.confidence_threshold_adapter = None
            
        if self.config.use_hybrid_search:
            self.hybrid_search = HybridMCTSBeamSearch(
                beam_size=self.config.beam_size,
                temperature=self.config.temperature,
                mcts_reasoner=self
            )
        else:
            self.hybrid_search = None
            
        if self.config.use_async_simulation:
            self.async_mcts = AsyncMCTS(
                num_workers=self.config.num_parallel_workers
            )
        else:
            self.async_mcts = None
            
        # Statistics tracking
        self.register_buffer('total_simulations', torch.tensor(0))
        self.register_buffer('total_searches', torch.tensor(0))
        self.register_buffer('total_nodes_created', torch.tensor(0))
        
        # For analysis
        self.last_reasoning_trace = []
        self.last_search_statistics = {}
    
    def forward(
        self,
        state: torch.Tensor,
        available_actions: List[Any],
        action_embeddings: Optional[torch.Tensor] = None,
        transition_fn: Optional[Callable] = None,
        reward_fn: Optional[Callable] = None,
        constraints: Optional[Dict[str, Any]] = None
    ):
        """
        Perform MCTS reasoning to select the best action.
        
        Args:
            state: Current state representation [batch_size, hidden_size]
            available_actions: List of available actions
            action_embeddings: Tensor of action embeddings [num_actions, hidden_size]
            transition_fn: Function to get next state given state and action
            reward_fn: Function to compute reward for a state
            constraints: Optional constraints on the search (e.g., max_depth)
            
        Returns:
            Tuple of (selected actions, action probabilities, search info)
        """
        batch_size = state.size(0)
        device = state.device
        
        # Default functions if not provided
        if transition_fn is None:
            # Simple default transition function
            transition_fn = lambda s, a: s + 0.1 * a
        
        if reward_fn is None:
            # Default reward function returns 0
            reward_fn = lambda s: torch.zeros(s.size(0), device=s.device)
        
        # Get action embeddings if not provided
        if action_embeddings is None and isinstance(available_actions[0], str):
            # Create simple embeddings based on token IDs or hashes
            action_ids = [hash(str(a)) % 10000 for a in available_actions]
            action_embeddings = torch.randn(len(available_actions), self.config.hidden_size, device=device)
            action_embeddings = F.normalize(action_embeddings, dim=-1)
        
        # Process batch items one by one (can be optimized for parallel execution)
        selected_actions = []
        action_probs = []
        search_info = []
        
        for i in range(batch_size):
            # Get state for this batch item
            state_i = state[i].unsqueeze(0)  # [1, hidden_size]
            
            # Determine number of simulations based on confidence if adaptive
            if self.config.use_adaptive_simulation_count:
                confidence = self.confidence_predictor(state_i).item()
                num_simulations = max(
                    self.config.min_simulations,
                    min(
                        self.config.max_simulations,
                        int(self.config.max_simulations * (1 - confidence))
                    )
                )
            else:
                num_simulations = self.config.max_simulations
            
            # Run MCTS search
            root = self._run_mcts(
                state_i,
                available_actions,
                action_embeddings,
                transition_fn,
                reward_fn,
                num_simulations
            )
            
            # Get action probabilities based on visit counts
            visit_counts = np.array([
                root.children[a].visit_count for a in available_actions if a in root.children
            ])
            
            # Apply temperature
            if self.config.temperature > 0:
                visit_counts = visit_counts ** (1 / self.config.temperature)
            
            # Normalize to get probabilities
            if visit_counts.sum() > 0:
                probs = visit_counts / visit_counts.sum()
            else:
                probs = np.ones_like(visit_counts) / len(visit_counts)
            
            # Select action (most visited during search)
            best_idx = int(np.argmax(visit_counts))
            selected_action = available_actions[best_idx]
            
            # Collect results
            selected_actions.append(selected_action)
            action_probs.append(probs)
            
            # Collect search info
            info = {
                'num_simulations': num_simulations,
                'num_nodes': self.total_nodes_created.item(),
                'visit_counts': visit_counts.tolist(),
                'reasoning_trace': root.reasoning_steps if self.config.store_reasoning_trace else []
            }
            search_info.append(info)
            
            # Store for analysis
            self.last_reasoning_trace = root.reasoning_steps
            self.last_search_statistics = info
            
            # Update statistics
            self.total_simulations += num_simulations
            self.total_searches += 1
        
        return selected_actions, action_probs, search_info
    
    def _run_mcts(
        self,
        root_state: torch.Tensor,
        available_actions: List[Any],
        action_embeddings: torch.Tensor,
        transition_fn: Callable,
        reward_fn: Callable,
        num_simulations: int
    ) -> MCTSNode:
        """
        Run Monte Carlo Tree Search from a root state.
        
        Args:
            root_state: Initial state tensor
            available_actions: List of available actions
            action_embeddings: Embeddings for actions
            transition_fn: Function to get next state given state and action
            reward_fn: Function to get reward given state and action
            num_simulations: Number of simulations to run
            
        Returns:
            Root node of search tree
        """
        # Initialize root node
        if self.config.use_state_compression:
            compression_ratio = self.config.state_compression_ratio
            root = MCTSNode(action=None, parent=None, compression_ratio=compression_ratio)
            root.set_state_with_compression(root_state, compression_ratio)
        else:
            root = MCTSNode(state=root_state, action=None, parent=None)
        
        # Expand root node
        self._expand_node(root, available_actions, action_embeddings)
        
        # Add exploration noise to root node
        if self.config.use_dirichlet_noise:
            root.add_exploration_noise(
                self.config.dirichlet_alpha,
                self.config.dirichlet_weight
            )
            
        # Restrict action space if enabled
        if self.config.use_action_space_reduction and self.action_space_adapter is not None:
            # Get logits for available actions
            with torch.no_grad():
                logits = self.policy_network(root_state)
                
            # Restrict action space
            restricted_actions, restricted_priors = self.action_space_adapter.restrict_action_space(
                root_state, logits
            )
            
            # Update available actions and expand root again
            restricted_action_embeddings = action_embeddings[restricted_actions]
            self._expand_node(root, [available_actions[i] for i in restricted_actions], 
                             restricted_action_embeddings, restricted_priors)
        
        # Use hybrid search if enabled
        if self.config.use_hybrid_search and self.hybrid_search is not None:
            # Get logits for available actions
            with torch.no_grad():
                logits = self.policy_network(root_state)
                
            # Run hybrid search
            best_action, _ = self.hybrid_search.search(
                root_state, logits, available_actions, 
                transition_fn, reward_fn, num_simulations
            )
            
            # Find best action in children
            for action, child in root.children.items():
                if action == best_action:
                    child.visit_count += 1  # Ensure this action is selected
                    break
                    
            return root
        
        # Use asynchronous parallel simulation if enabled
        if self.config.use_async_simulation and self.async_mcts is not None:
            # Define selection function
            def select_fn(node):
                return node.select_child(self.config.exploration_weight)
                
            # Define expansion function
            def expand_fn(node):
                # Get node state
                state = node.state
                
                # Get child state by applying action
                if node.action is not None and node.parent is not None:
                    parent_state = node.parent.state
                    state = transition_fn(parent_state, node.action)
                    node.state = state
                
                # Expand with available actions
                self._expand_node(node, available_actions, action_embeddings)
                
            # Define evaluation function
            def evaluate_fn(node):
                # Get node state
                state = node.state
                
                # Use value cache if enabled
                if self.config.use_value_cache and self.value_cache is not None:
                    cached_value = self.value_cache.lookup_value(state)
                    if cached_value is not None:
                        return cached_value
                
                # Evaluate using rollout or value network
                value = self._evaluate(state)
                
                # Cache value if enabled
                if self.config.use_value_cache and self.value_cache is not None:
                    self.value_cache.store_value(state, value)
                    
                return value
                
            # Define backpropagation function
            def backpropagate_fn(node, value):
                node.update(value)
                
            # Run parallel simulations
            root, completed_simulations = self.async_mcts.run_parallel_simulations(
                root, num_simulations, select_fn, expand_fn, evaluate_fn, backpropagate_fn
            )
            
            # Update statistics
            self.total_simulations += completed_simulations
            
            return root
        
        # Run standard sequential simulations
        for _ in range(num_simulations):
            # Selection phase
            search_path = []
            node = root
            
            while node.expanded and node.children:
                action, node = node.select_child(self.config.exploration_weight)
                search_path.append(node)
            
            # Expansion phase
            if not node.expanded:
                # Get node state
                state = node.state
                
                # Get child state by applying action
                if node.action is not None and node.parent is not None:
                    parent_state = node.parent.state
                    state = transition_fn(parent_state, node.action)
                    node.state = state
                
                # Expand node
                self._expand_node(node, available_actions, action_embeddings)
            
            # Evaluation phase - use value cache if enabled
            if self.config.use_value_cache and self.value_cache is not None:
                cached_value = self.value_cache.lookup_value(node.state)
                if cached_value is not None:
                    value = cached_value
                else:
                    value = self._evaluate(node.state)
                    self.value_cache.store_value(node.state, value)
            else:
                value = self._evaluate(node.state)
            
            # Backpropagation phase
            self._backpropagate(search_path, value)
            
            # Dynamic early stopping if enabled
            if self.config.use_dynamic_confidence and self.confidence_threshold_adapter is not None:
                # Get children visit distribution
                distribution = root.get_visit_count_distribution()
                if distribution:
                    # Calculate entropy of distribution
                    probs = distribution
                    entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probs)
                    
                    # Get dynamic threshold
                    dynamic_threshold = self.confidence_threshold_adapter.calculate_threshold(
                        len(search_path), 
                        self.config.max_search_depth - len(search_path),
                        entropy
                    )
                    
                    # Get most visited child
                    max_visits = max(child.visit_count for child in root.children.values())
                    total_visits = sum(child.visit_count for child in root.children.values())
                    
                    # Check if we've reached the threshold
                    if total_visits > 0 and max_visits / total_visits > dynamic_threshold:
                        if self.config.store_reasoning_trace:
                            root.record_reasoning_step(
                                f"Early stopping after {_ + 1} simulations "
                                f"with confidence {max_visits / total_visits:.3f} > {dynamic_threshold:.3f}"
                            )
                        break
        
        # Update statistics
        self.total_simulations += num_simulations
        
        return root
    
    def _expand_node(self, node: MCTSNode, available_actions: List[Any], action_embeddings: torch.Tensor, priors=None):
        """
        Expand node with available actions.
        
        Args:
            node: Node to expand
            available_actions: List of available actions
            action_embeddings: Embeddings for actions
            priors: Optional prior probabilities for actions
        """
        # Get node state
        state = node.state
        
        # Get policy predictions if no priors provided
        if priors is None and self.policy_network is not None:
            with torch.no_grad():
                policy_logits = self.policy_network(state)
                
                # Ensure policy logits match number of available actions
                if policy_logits.size(-1) != len(available_actions):
                    # Use only first len(available_actions) logits or pad
                    if policy_logits.size(-1) > len(available_actions):
                        policy_logits = policy_logits[..., :len(available_actions)]
                    else:
                        # Pad with small values
                        padding = torch.ones(len(available_actions) - policy_logits.size(-1), device=policy_logits.device) * -1e9
                        policy_logits = torch.cat([policy_logits, padding], dim=-1)
                
                # Create action mask for available actions
                action_mask = torch.ones(len(available_actions), device=policy_logits.device)
                
                # Apply action masking (set unavailable actions to -inf)
                masked_logits = policy_logits + (1 - action_mask) * -1e9
                
                # Get prior probabilities
                priors = F.softmax(masked_logits / self.config.temperature, dim=-1).tolist()
        elif priors is None:
            # Use uniform prior if no policy network
            priors = [1.0 / len(available_actions)] * len(available_actions)
        
        # Get rewards for each action if possible
        rewards = None
        
        # Expand node with actions and priors
        node.expand(state, available_actions, priors, rewards)
    
    def _evaluate(self, state):
        """
        Evaluate state using value network or simple heuristic.
        
        Args:
            state: State to evaluate
            
        Returns:
            Value of state
        """
        # Use value cache if enabled
        if self.config.use_value_cache and self.value_cache is not None:
            cached_value = self.value_cache.lookup_value(state)
            if cached_value is not None:
                return cached_value
                
        # Use value network if available
        if self.value_network is not None:
            with torch.no_grad():
                # Apply state compression if enabled (for value network input)
                if self.config.use_state_compression and hasattr(self, 'state_encoder'):
                    compressed_state = self.state_encoder(state)
                    value = self.value_network(compressed_state)
                else:
                    value = self.value_network(state)
                
                # Handle tensor with multiple elements
                if value.numel() > 1:
                    value = value.mean().item()
                else:
                    value = value.item()
                
                # Scale value to desired range
                value = value * self.config.value_scale
        else:
            # Default to random evaluation
            value = random.uniform(-0.1, 0.1)
            
        # Store in cache if enabled
        if self.config.use_value_cache and self.value_cache is not None:
            self.value_cache.store_value(state, value)
            
        return value
        
    def _evaluate_with_simulations(self, state, available_actions, transition_fn, reward_fn, num_simulations):
        """
        Evaluate a state by running a fixed number of simulations.
        
        Args:
            state: State to evaluate
            available_actions: List of available actions
            transition_fn: Transition function
            reward_fn: Reward function
            num_simulations: Number of simulations to run
            
        Returns:
            Value estimate for the state
        """
        # Create dummy action embeddings if needed
        if not hasattr(self, 'dummy_action_embeddings') or self.dummy_action_embeddings.size(0) < len(available_actions):
            self.dummy_action_embeddings = torch.randn(
                len(available_actions), self.config.hidden_size, 
                device=state.device
            )
            
        # Run MCTS from this state
        node = self._run_mcts(
            state, available_actions, self.dummy_action_embeddings,
            transition_fn, reward_fn, num_simulations
        )
        
        # Return average value of children
        if node.children:
            values = [child.value for child in node.children.values() if child.visit_count > 0]
            if values:
                return sum(values) / len(values)
                
        # Fallback to direct evaluation
        return self._evaluate(state)
    
    def _rollout(
        self,
        state: torch.Tensor,
        available_actions: List[Any],
        action_embeddings: torch.Tensor,
        transition_fn: Callable,
        reward_fn: Callable,
        depth: int = 0
    ) -> float:
        """
        Perform a rollout from the given state to estimate its value.
        
        Args:
            state: Starting state representation
            available_actions: List of available actions
            action_embeddings: Embeddings for actions
            transition_fn: Function to get next state
            reward_fn: Function to compute rewards
            depth: Current depth of rollout
            
        Returns:
            Estimated value from rollout
        """
        # If max depth reached, evaluate state directly
        if depth >= self.config.max_search_depth:
            return self._evaluate(state)
        
        # Select random action for rollout
        action_idx = random.randrange(len(available_actions))
        action_emb = action_embeddings[action_idx]
        
        # Get next state
        next_state = transition_fn(state, action_emb.unsqueeze(0))
        
        # Get reward for this step
        reward = reward_fn(next_state).item()
        
        # Recursive rollout with discount factor
        future_value = self._rollout(
            next_state,
            available_actions,
            action_embeddings,
            transition_fn,
            reward_fn,
            depth + 1
        )
        
        # Combine immediate reward with discounted future value
        gamma = 0.95  # Discount factor
        value = reward + gamma * future_value
        
        return value
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """
        Backpropagate value through the search path.
        
        Args:
            search_path: List of nodes visited during selection
            value: Value to backpropagate
        """
        # Apply backpropagation with optional transformations
        for node in reversed(search_path):
            node.update(value)
            
            # Include immediate reward in value for parent
            if node.parent is not None:
                value = node.reward + 0.95 * value  # Apply discount
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the search process.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_simulations": self.total_simulations.item(),
            "total_searches": self.total_searches.item(),
            "total_nodes_created": self.total_nodes_created.item()
        }
        
        # Add value cache statistics if enabled
        if self.config.use_value_cache and self.value_cache is not None:
            stats["value_cache"] = self.value_cache.get_stats()
        
        # Add memory usage statistics
        stats["memory_usage"] = self._get_memory_usage()
        
        # Add last search info
        stats.update(self.last_search_statistics)
        
        return stats
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """
        Calculate memory usage of the MCTS.
        
        Returns:
            Dictionary of memory usage stats
        """
        import sys
        import gc
        import psutil
        import torch
        
        # Force garbage collection
        gc.collect()
        
        memory_stats = {}
        
        # Get Python memory usage
        process = psutil.Process()
        memory_stats["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)
        
        # Get PyTorch memory usage
        if torch.cuda.is_available():
            memory_stats["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_stats["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            
        return memory_stats
    
    def get_performance_report(self) -> str:
        """
        Generate a human-readable performance report.
        
        Returns:
            String with performance information
        """
        stats = self.get_search_statistics()
        
        # Build report
        report = ["MCTS Performance Report:"]
        report.append(f"- Total simulations: {stats['total_simulations']}")
        report.append(f"- Total searches: {stats['total_searches']}")
        report.append(f"- Total nodes created: {stats['total_nodes_created']}")
        
        # Add value cache stats if available
        if "value_cache" in stats:
            cache_stats = stats["value_cache"]
            hit_rate = cache_stats.get("hit_rate", 0) * 100
            report.append(f"- Value cache: {cache_stats.get('size', 0)}/{cache_stats.get('capacity', 0)} entries")
            report.append(f"- Cache hit rate: {hit_rate:.1f}%")
            report.append(f"- Cache hits/misses: {cache_stats.get('hits', 0)}/{cache_stats.get('misses', 0)}")
            
        # Add memory usage
        if "memory_usage" in stats:
            memory = stats["memory_usage"]
            report.append(f"- Process memory: {memory.get('process_memory_mb', 0):.1f} MB")
            if "cuda_allocated_mb" in memory:
                report.append(f"- CUDA memory: {memory.get('cuda_allocated_mb', 0):.1f} MB allocated, "
                            f"{memory.get('cuda_reserved_mb', 0):.1f} MB reserved")
                
        # Add optimization info
        optimizations = []
        if self.config.use_state_compression:
            optimizations.append("state compression")
        if self.config.use_value_cache:
            optimizations.append("value caching")
        if self.config.use_hybrid_search:
            optimizations.append("hybrid MCTS-beam")
        if self.config.use_action_space_reduction:
            optimizations.append("action space reduction")
        if self.config.use_dynamic_confidence:
            optimizations.append("dynamic confidence")
        if self.config.use_async_simulation:
            optimizations.append("async simulation")
            
        if optimizations:
            report.append(f"- Active optimizations: {', '.join(optimizations)}")
            
        return "\n".join(report)
    
    def get_last_reasoning_trace(self) -> List[str]:
        """
        Get the reasoning trace from the last search.
        
        Returns:
            List of reasoning steps
        """
        return self.last_reasoning_trace
    
    def reset_statistics(self):
        """Reset all search statistics"""
        self.total_simulations.zero_()
        self.total_searches.zero_()
        self.total_nodes_created.zero_()
        self.last_reasoning_trace = []
        self.last_search_statistics = {}
        self.value_cache.clear() 