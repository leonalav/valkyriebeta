import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import random
from collections import defaultdict
from heapq import nlargest

@dataclass
class MCTSConfig:
    """Configuration for MCTS-based reasoning"""
    hidden_size: int = 768
    num_simulations: int = 50
    exploration_constant: float = 1.0
    temperature: float = 1.0
    max_tree_depth: int = 5
    value_scale: float = 0.1
    reward_scale: float = 0.1
    use_value_network: bool = True
    use_policy_network: bool = True
    discount_factor: float = 0.9
    sampling_steps: int = 5
    beam_width: int = 5
    use_transformer_policy: bool = True
    policy_temperature: float = 1.0
    dropout: float = 0.1
    top_k_actions: int = 10
    use_action_embeddings: bool = True
    action_space_size: int = 128
    node_embedding_size: int = 768

class TreeNode:
    """Node in the MCTS tree structure"""
    
    def __init__(
        self,
        state: torch.Tensor, 
        parent=None, 
        action=None, 
        prior: float = 0.0,
        visit_count: int = 0
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = {}
        self.visit_count = visit_count
        self.value_sum = 0.0
        self.depth = 0 if parent is None else parent.depth + 1
        
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def select_child(self, exploration_constant: float) -> 'TreeNode':
        """Select child according to UCB formula"""
        if not self.expanded():
            raise ValueError("Cannot select child of unexpanded node")
        
        # UCB score calculation
        log_visit = math.log(self.visit_count)
        scores = {
            action: (
                child.value() + 
                exploration_constant * child.prior * 
                math.sqrt(log_visit) / (1 + child.visit_count)
            )
            for action, child in self.children.items()
        }
        
        # Find action with highest score
        best_action = max(scores, key=scores.get)
        return self.children[best_action]
    
    def expand(self, actions, action_priors: Dict[Any, float], states: Dict[Any, torch.Tensor]):
        """Expand node with provided actions and their priors and resulting states"""
        for action, prior in action_priors.items():
            if action not in self.children:
                self.children[action] = TreeNode(
                    state=states[action],
                    parent=self,
                    action=action,
                    prior=prior
                )
    
    def backpropagate(self, value: float, discount_factor: float = 1.0):
        """Update value estimates going up the tree"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value *= discount_factor  # Discount for deeper nodes
            node = node.parent


class PolicyNetwork(nn.Module):
    """Policy network for MCTS reasoning"""
    
    def __init__(self, config: MCTSConfig):
        super().__init__()
        self.config = config
        
        if config.use_transformer_policy:
            # Transformer-based policy
            self.policy_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.action_space_size if config.use_action_embeddings else 1)
            )
            
            # Optional action embeddings
            if config.use_action_embeddings:
                self.action_embeddings = nn.Parameter(
                    torch.randn(config.action_space_size, config.node_embedding_size)
                )
        else:
            # Simple MLP policy
            self.policy_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 2),
                nn.LayerNorm(config.hidden_size * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
            nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.action_space_size if config.use_action_embeddings else 1)
            )
            
            # Optional action embeddings
            if config.use_action_embeddings:
                self.action_embeddings = nn.Parameter(
                    torch.randn(config.action_space_size, config.node_embedding_size)
                )
                
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply policy network to get action probabilities
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Dictionary containing:
                - action_logits: Logits over action space
                - action_probs: Probabilities over action space
        """
        outputs = {}
        
        if self.config.use_action_embeddings:
            # Using action embeddings - compute compatibility with all actions
            policy_features = self.policy_head(hidden_states)  # [batch, seq_len, action_space_size]
            
            # Compatibility scores - just use the raw scores
            action_logits = policy_features
            
            # Apply temperature and get probabilities
            action_probs = F.softmax(
                action_logits / self.config.policy_temperature, 
                dim=-1
            )
            
            outputs["action_logits"] = action_logits
            outputs["action_probs"] = action_probs
        else:
            # Project state to a policy embedding
            policy_embedding = self.policy_head(hidden_states)  # [batch, seq_len, hidden_size]
            
            # Compute compatibility with action embeddings
            action_logits = torch.matmul(
                policy_embedding, 
                self.action_embeddings.transpose(0, 1)
            )  # [batch, seq_len, action_space_size]
            
            # Apply temperature and get probabilities
            action_probs = F.softmax(
                action_logits / self.config.policy_temperature, 
                dim=-1
            )
            
            outputs["action_logits"] = action_logits
            outputs["action_probs"] = action_probs
            
        return outputs


class ValueNetwork(nn.Module):
    """Value network for MCTS reasoning"""
    
    def __init__(self, config: MCTSConfig):
        super().__init__()
        self.config = config
        
        # Value estimation network
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply value network to estimate state values
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Dictionary containing:
                - value: Estimated value of the state
        """
        outputs = {}
        
        # Estimate value directly from hidden states
        values = self.value_head(hidden_states).squeeze(-1)  # [batch, seq_len]
        
        # Scale values to a reasonable range
        scaled_values = torch.tanh(values) * self.config.value_scale
        
        outputs["value"] = scaled_values
        
        return outputs


class MCTSEnhancedTreeReasoningModule(nn.Module):
    """Tree reasoning module enhanced with Monte Carlo Tree Search"""
    
    def __init__(self, config: Union[Dict, MCTSConfig]):
        super().__init__()
        
        # Convert dict to config if needed
        if isinstance(config, dict):
            self.config = MCTSConfig(**config)
        else:
            self.config = config
        
        # Policy network for action selection
        self.policy_network = PolicyNetwork(self.config) if self.config.use_policy_network else None
        
        # Value network for state evaluation
        self.value_network = ValueNetwork(self.config) if self.config.use_value_network else None
        
        # State transition function
        self.state_transition = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )
        
        # Action embedding
        if self.config.use_action_embeddings:
            self.action_embeddings = nn.Parameter(
                torch.randn(self.config.action_space_size, self.config.node_embedding_size)
            )
        
        # Layer normalization for inputs
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)
        
        # Final projection back to model dimension
        self.output_projection = nn.Linear(self.config.hidden_size, self.config.hidden_size)
    
    def get_action_embedding(self, action_idx: int) -> torch.Tensor:
        """Get embedding for a particular action"""
        if not self.config.use_action_embeddings:
            # Create a simple one-hot encoding if no action embeddings
            result = torch.zeros(self.config.node_embedding_size, device=self.action_embeddings.device)
            action_idx = min(action_idx, self.config.node_embedding_size - 1)  # Safety check
            result[action_idx] = 1.0
            return result
        
        # Return the learned action embedding
        return self.action_embeddings[action_idx]
    
    def transition_fn(
        self, 
        state: torch.Tensor, 
        action: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """Apply action to state to get next state"""
        # Get action embedding if it's just an index
        if isinstance(action, int):
            action_embedding = self.get_action_embedding(action)
        else:
            action_embedding = action
            
        # Make sure shapes are correct for batch processing
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action_embedding.shape) == 1:
            action_embedding = action_embedding.unsqueeze(0)
            
        # Concatenate state with action embedding
        combined = torch.cat([state, action_embedding], dim=-1)
        
        # Apply transition network
        next_state = self.state_transition(combined)
        
        # Add residual connection
        next_state = next_state + state
        
        return next_state
    
    def get_rewards(self, states: torch.Tensor) -> torch.Tensor:
        """Compute rewards for terminal states"""
        # In a real application, this would be more complex
        # Here we just use the value network estimate
        if self.value_network is not None:
            value_output = self.value_network(states)
            return value_output["value"] * self.config.reward_scale
        
        # Fallback to a simple heuristic
        return torch.sum(states, dim=-1) * 0.01
    
    def mcts_search(
        self, 
        root_state: torch.Tensor, 
        num_simulations: Optional[int] = None
    ) -> Tuple[TreeNode, Dict[str, Any]]:
        """
        Perform MCTS search starting from root_state
        
        Args:
            root_state: Starting state for the search
            num_simulations: Number of MCTS simulations to perform (defaults to config value)
            
        Returns:
            Tuple of (root TreeNode, search_stats)
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations
        
        # Create root node
        root = TreeNode(state=root_state)
        search_stats = {"max_depth": 0, "total_visits": 0, "leaf_values": []}
        
        # Perform simulations
        for _ in range(num_simulations):
            # Selection phase - find leaf node to expand
            leaf = root
            search_path = [leaf]
            
            # Traverse tree to find leaf
            while leaf.expanded() and leaf.depth < self.config.max_tree_depth:
                leaf = leaf.select_child(self.config.exploration_constant)
                search_path.append(leaf)
            
            # Update max depth stat
            search_stats["max_depth"] = max(search_stats["max_depth"], leaf.depth)
            
            # Expansion phase - if not terminal and not at max depth
            if leaf.depth < self.config.max_tree_depth:
                # Get actions and their probabilities
                if self.policy_network is not None:
                    # Use policy network to predict action probabilities
                    policy_output = self.policy_network(leaf.state.unsqueeze(0).unsqueeze(0))
                    action_probs = policy_output["action_probs"].squeeze().cpu().detach().numpy()
                else:
                    # Uniform random policy as fallback
                    action_probs = np.ones(self.config.action_space_size) / self.config.action_space_size
                
                # Get top-k actions
                top_k = min(self.config.top_k_actions, self.config.action_space_size)
                actions = np.argsort(action_probs)[-top_k:]
                
                # Normalize probabilities for selected actions
                selected_probs = action_probs[actions]
                selected_probs = selected_probs / np.sum(selected_probs)
                
                # Create mappings for expansion
                action_priors = {int(action): float(prob) for action, prob in zip(actions, selected_probs)}
                
                # Compute next states for selected actions
                states = {}
                for action in actions:
                    action_embedding = self.get_action_embedding(int(action))
                    next_state = self.transition_fn(leaf.state, action_embedding.to(leaf.state.device))
                    states[int(action)] = next_state.squeeze(0)
                
                # Expand leaf with these actions
                leaf.expand(actions, action_priors, states)
            
            # Simulation/Evaluation phase
            if self.value_network is not None:
                # Evaluate leaf using value network
                value_output = self.value_network(leaf.state.unsqueeze(0).unsqueeze(0))
                value = float(value_output["value"].item())
            else:
                # Simple heuristic if no value network
                value = float(torch.sum(leaf.state).item() * 0.01)
            
            # Track leaf values
            search_stats["leaf_values"].append(value)
            
            # Backpropagation - update node values up the tree
            for node in reversed(search_path):
                search_stats["total_visits"] += 1
                node.backpropagate(value, self.config.discount_factor)
                
        # Return tree root and search stats
        return root, search_stats
    
    def sample_action_from_tree(
        self, 
        root: TreeNode, 
        temperature: Optional[float] = None
    ) -> int:
        """Sample action from the tree search results"""
        if temperature is None:
            temperature = self.config.temperature
            
        # Get visit counts for each action
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        actions = np.array(list(root.children.keys()))
        
        # Apply temperature
        if temperature == 0:
            # Deterministic selection
            best_idx = np.argmax(visit_counts)
            return int(actions[best_idx])
        else:
            # Apply temperature and sample
            visit_counts = visit_counts ** (1 / temperature)
            probs = visit_counts / np.sum(visit_counts)
            action_idx = np.random.choice(len(actions), p=probs)
            return int(actions[action_idx])
    
    def _beam_search_reasoning(
        self, 
        hidden_states: torch.Tensor, 
        beam_width: int, 
        max_steps: int
    ) -> torch.Tensor:
        """Apply beam search reasoning to find best outputs"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # Output states for each batch item
        output_states = []
        
        # Process each batch item separately
        for b in range(batch_size):
            # Initialize beam with starting state
            init_state = hidden_states[b, 0].unsqueeze(0)  # Single starting node
            beam = [(0.0, init_state, [])]  # (score, state, action_history)
            
            # Beam search for specified steps
            for step in range(max_steps):
                # Expand all current candidates
                candidates = []
                
                for score, state, actions in beam:
                    # Get action probabilities for this state
                    if self.policy_network is not None:
                        policy_output = self.policy_network(state.unsqueeze(0))
                        action_probs = policy_output["action_probs"].squeeze(0)
                    else:
                        # Uniform distribution as fallback
                        action_probs = torch.ones(self.config.action_space_size, device=device)
                        action_probs = action_probs / action_probs.sum()
                    
                    # Get value estimate
                    if self.value_network is not None:
                        value_output = self.value_network(state.unsqueeze(0))
                        value = value_output["value"].item()
                    else:
                        value = 0.0
                    
                    # Get top-k actions
                    top_k = min(beam_width, self.config.action_space_size)
                    top_actions = torch.topk(action_probs, top_k).indices
                    
                    # Create new candidates for each action
                    for action_idx in top_actions:
                        action_prob = action_probs[action_idx].item()
                        action_embedding = self.get_action_embedding(action_idx.item())
                        
                        # Apply transition
                        next_state = self.transition_fn(state, action_embedding.to(device))
                        
                        # Get value of next state if available
                        if self.value_network is not None:
                            next_value_output = self.value_network(next_state.unsqueeze(0))
                            next_value = next_value_output["value"].item()
                        else:
                            next_value = 0.0
                        
                        # Compute updated score (higher is better)
                        # This scoring combines:
                        # - Current accumulated score
                        # - Log probability of taking this action
                        # - Value estimate of resulting state
                        new_score = score + math.log(max(action_prob, 1e-10)) + next_value
                        
                        # Add to candidates
                        candidates.append((
                            new_score,
                            next_state.squeeze(0),
                            actions + [action_idx.item()]
                        ))
                
                # Select top candidates for next beam
                beam = nlargest(beam_width, candidates, key=lambda x: x[0])
            
            # Return best final state from beam
            best_score, best_state, _ = max(beam, key=lambda x: x[0])
            output_states.append(best_state)
        
        # Combine output states into a tensor
        return torch.stack(output_states).unsqueeze(1)  # [batch, 1, hidden_dim]
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        reasoning_type: str = "mcts"
    ) -> Dict[str, torch.Tensor]:
        """
        Apply MCTS-enhanced tree reasoning
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            reasoning_type: Type of reasoning to apply ('mcts', 'beam', or None)
            
        Returns:
            Dictionary containing:
                - hidden_states: Updated hidden states
                - search_stats: Optional statistics from the search process
        """
        # Initialize return dict
        outputs = {"hidden_states": hidden_states}
        
        # Skip if not using MCTS reasoning
        if reasoning_type not in ["mcts", "beam"]:
            return outputs
        
        # Normalize inputs
        normalized_states = self.layer_norm(hidden_states)
        
        # Process each example in the batch
        batch_size, seq_len, hidden_dim = normalized_states.shape
        
        # Apply the appropriate reasoning approach
        if reasoning_type == "mcts":
            # MCTS reasoning (monte carlo tree search)
            search_results = []
            search_stats_list = []
            
            for b in range(batch_size):
                # Take first token position as initial state
                # In a full implementation, you might use a more sophisticated state
                root_state = normalized_states[b, 0]
            
        # Perform MCTS search
                root, search_stats = self.mcts_search(root_state)
                
                # Sample action sequence from search tree
                sampled_states = [root.state]
                current_node = root
                
                for _ in range(self.config.sampling_steps):
                    if not current_node.expanded() or len(current_node.children) == 0:
                        break
                    
                    # Sample next action
                    action = self.sample_action_from_tree(current_node)
                    
                    # Get next node
                    if action in current_node.children:
                        current_node = current_node.children[action]
                        sampled_states.append(current_node.state)
                
                # Get final state (either last in sequence or best node by visits)
                if len(sampled_states) > 1:
                    final_state = sampled_states[-1]
                else:
                    # Get the child with the most visits
                    if current_node.expanded() and len(current_node.children) > 0:
                        best_action = max(
                            current_node.children.items(), 
                            key=lambda x: x[1].visit_count
                        )[0]
                        final_state = current_node.children[best_action].state
                    else:
                        final_state = current_node.state
                
                search_results.append(final_state)
                search_stats_list.append(search_stats)
            
            # Combine results
            search_tensor = torch.stack(search_results).unsqueeze(1)  # [batch, 1, hidden_dim]
            outputs["search_stats"] = search_stats_list
            
            # Project back to original hidden dimension
            reasoned_states = self.output_projection(search_tensor)
        
        elif reasoning_type == "beam":
            # Beam search reasoning
            reasoned_states = self._beam_search_reasoning(
                normalized_states, 
                self.config.beam_width, 
                self.config.sampling_steps
            )
            
            # Project back to original hidden dimension
            reasoned_states = self.output_projection(reasoned_states)
        
        # Expand reasoned states to match input sequence length
        expanded_states = reasoned_states.expand(-1, seq_len, -1)
        
        # Apply residual connection
        outputs["hidden_states"] = expanded_states + hidden_states
        
        return outputs 