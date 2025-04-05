"""
Tree-of-Thought Reasoning with Monte Carlo Tree Search

This module enhances the Tree-of-Thought reasoning approach with Monte Carlo Tree Search (MCTS),
enabling more systematic exploration of reasoning paths and improved solution quality.

MCTS follows four main steps:
1. Selection: Select promising nodes using UCB1 formula
2. Expansion: Expand selected node with new children
3. Simulation: Perform rollouts to estimate node value
4. Backpropagation: Update statistics based on simulation results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class MCTSConfig:
    """Configuration for Monte Carlo Tree Search in Tree-of-Thought reasoning"""
    # Core MCTS parameters
    max_iterations: int = 100
    exploration_weight: float = 1.0
    max_depth: int = 10
    rollout_depth: int = 3
    discount_factor: float = 0.95
    
    # Node selection and expansion
    top_k_candidates: int = 5
    min_visits_for_expansion: int = 2
    
    # Value estimation
    use_value_network: bool = True
    value_scale: float = 1.0
    value_shift: float = 0.0
    
    # Simulation parameters
    num_simulations: int = 3
    simulation_temperature: float = 0.7
    
    # Termination criteria
    early_stopping_threshold: float = 0.95
    early_stopping_visits: int = 10
    
    # Computational budget
    max_computation_time: Optional[float] = None  # seconds
    max_nodes: int = 1000
    
    # Beam search integration
    use_beam_search: bool = True
    beam_size: int = 4
    
    # Visualization and logging
    enable_visualization: bool = False
    detailed_logging: bool = False

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.
    
    Each node represents a state in the reasoning process.
    """
    
    def __init__(
        self, 
        state: torch.Tensor,
        state_text: str,
        parent: Optional['MCTSNode'] = None,
        action_taken: Optional[int] = None,
        action_text: Optional[str] = None,
        prior_probability: float = 1.0,
        config: MCTSConfig = None,
    ):
        # Node identification
        self.state = state
        self.state_text = state_text
        self.parent = parent
        self.action_taken = action_taken
        self.action_text = action_text
        self.depth = 0 if parent is None else parent.depth + 1
        
        # Tree structure
        self.children: Dict[int, 'MCTSNode'] = {}
        
        # MCTS statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_probability = prior_probability
        
        # Candidate actions and their probabilities
        self.available_actions: Dict[int, Tuple[float, str]] = {}  # action_id -> (probability, text)
        
        # Configuration
        self.config = config or MCTSConfig()
        
        # Terminal state flag
        self.is_terminal = False
        self.terminal_value = None
        
        # Computation tracking
        self.fully_expanded = False
        self.abandoned = False
    
    @property
    def value(self) -> float:
        """Average value of this node based on rollouts"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def ucb_score(self) -> float:
        """
        Upper Confidence Bound (UCB1) score for node selection.
        
        UCB1 balances exploitation (high value) with exploration (low visit count)
        """
        if self.visit_count == 0:
            return float('inf')
            
        # If parent has no visits, this shouldn't happen but handle it gracefully
        parent_visits = 1 if self.parent is None else max(1, self.parent.visit_count)
        
        # UCB1 formula: value + exploration_weight * sqrt(ln(parent_visits) / visit_count)
        exploitation = self.value
        exploration = self.config.exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visit_count
        )
        
        # Prior term for PUCT (Predictor + UCT) algorithm
        prior_term = self.prior_probability * math.sqrt(parent_visits) / (1 + self.visit_count)
        
        return exploitation + exploration + prior_term
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions from this node have been expanded"""
        # If already marked as fully expanded, return True
        if self.fully_expanded:
            return True
            
        # If there are no available actions, mark as fully expanded
        if not hasattr(self, 'available_actions') or not self.available_actions:
            self.fully_expanded = True
            return True
            
        # If there are available actions not yet expanded, not fully expanded
        return len(self.available_actions) <= len(self.children)
    
    def select_child(self) -> 'MCTSNode':
        """Select the child with the highest UCB score"""
        if not self.children:
            raise ValueError("Cannot select child from a node with no children")
            
        return max(self.children.values(), key=lambda child: child.ucb_score)
    
    def select_action_for_expansion(self) -> Optional[int]:
        """Select an unexpanded action for expansion"""
        # Get actions that have not been expanded yet
        if not hasattr(self, 'available_actions') or not self.available_actions:
            self.fully_expanded = True
            return None
            
        unexpanded_actions = set(self.available_actions.keys()) - set(self.children.keys())
        
        if not unexpanded_actions:
            self.fully_expanded = True
            return None
            
        # Select action with highest prior probability
        return max(
            unexpanded_actions, 
            key=lambda action_id: self.available_actions[action_id][0]
        )
    
    def add_child(self, action: int, state: torch.Tensor, state_text: str, prior_probability: float, action_text: str = None) -> 'MCTSNode':
        """Add a child node for the given action"""
        child = MCTSNode(
            state=state,
            state_text=state_text,
            parent=self,
            action_taken=action,
            action_text=action_text,
            prior_probability=prior_probability,
            config=self.config
        )
        self.children[action] = child
        return child
    
    def update(self, value: float) -> None:
        """Update node statistics after a simulation"""
        self.visit_count += 1
        self.value_sum += value
    
    def get_path_from_root(self) -> List[Tuple[int, str]]:
        """Get the sequence of actions from root to this node"""
        if self.parent is None:
            return []
            
        path = self.parent.get_path_from_root()
        if self.action_taken is not None and self.action_text is not None:
            path.append((self.action_taken, self.action_text))
        return path
    
    def get_best_child(self) -> Optional['MCTSNode']:
        """Get child with highest visit count (most promising)"""
        if not self.children:
            return None
            
        return max(self.children.values(), key=lambda child: child.visit_count)
    
    def __repr__(self) -> str:
        action_str = f" (action={self.action_taken})" if self.action_taken is not None else ""
        return f"MCTSNode(visits={self.visit_count}, value={self.value:.4f}, " + \
               f"children={len(self.children)}{action_str})"

class MonteCarloTreeSearch:
    """
    Monte Carlo Tree Search implementation for Tree-of-Thought reasoning.
    """
    
    def __init__(
        self, 
        config: MCTSConfig,
        state_evaluator: Callable[[torch.Tensor], Tuple[Dict[int, Tuple[float, str]], float]],
        state_expander: Callable[[torch.Tensor, int], Tuple[torch.Tensor, str]],
        state_simulator: Callable[[torch.Tensor, int], float],
        terminal_checker: Callable[[torch.Tensor], Tuple[bool, Optional[float]]],
        state_tokenizer: Optional[Callable[[str], torch.Tensor]] = None,
    ):
        """
        Initialize MCTS with required components.
        
        Args:
            config: MCTS configuration
            state_evaluator: Function that evaluates a state, returning action probabilities and value estimate
            state_expander: Function that expands a state with a given action, returning new state and text
            state_simulator: Function that performs a rollout/simulation from a state, returning estimated value
            terminal_checker: Function that checks if a state is terminal, returning terminal status and value
            state_tokenizer: Optional function to convert state text to tensor representation
        """
        self.config = config
        self.state_evaluator = state_evaluator
        self.state_expander = state_expander
        self.state_simulator = state_simulator
        self.terminal_checker = terminal_checker
        self.state_tokenizer = state_tokenizer
        
        # Tree management
        self.root: Optional[MCTSNode] = None
        self.total_nodes = 0
        self.iteration_count = 0
        
        # Results tracking
        self.execution_trace = []
        self.best_value = -float('inf')
        self.best_sequence = []
        self.best_sequence_text = ""
    
    def initialize_root(self, initial_state: torch.Tensor, initial_text: str) -> None:
        """Initialize the search tree with a root node"""
        self.root = MCTSNode(
            state=initial_state,
            state_text=initial_text,
            config=self.config
        )
        self.total_nodes = 1
        
        # Evaluate the root node
        self._evaluate_node(self.root)
    
    def _evaluate_node(self, node: MCTSNode) -> bool:
        """
        Evaluate a node to get available actions and value estimate.
        Also checks for terminal state.
        
        Returns:
            is_terminal: Whether the node is a terminal state
        """
        # Check if the state is terminal
        is_terminal, terminal_value = self.terminal_checker(node.state)
        
        if is_terminal:
            node.is_terminal = True
            node.terminal_value = terminal_value
            return True
        
        # Evaluate the state to get action probabilities and value estimate
        action_probs, value_estimate = self.state_evaluator(node.state)
        
        # Update node with available actions
        node.available_actions = action_probs
        
        # If first evaluation of this node, record the value
        if node.visit_count == 0:
            node.update(value_estimate)
        
        return False
    
    def _select(self) -> MCTSNode:
        """
        Select a promising leaf node for expansion.
        
        Selection phase of MCTS using UCB1 for node selection.
        """
        node = self.root
        
        # Traverse the tree to a leaf node using UCB scores
        while node.children and not node.is_terminal:
            # If not fully expanded, we can expand from this node
            if not node.is_fully_expanded():
                break
                
            # Select the child with the highest UCB score
            node = node.select_child()
            
            # If we reach a terminal node, return it
            if node.is_terminal:
                return node
        
        return node
    
    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Expand a node by adding a new child.
        
        Expansion phase of MCTS, adding a new node to the tree.
        """
        # Terminal nodes cannot be expanded
        if node.is_terminal:
            return None
        
        # If node hasn't been evaluated yet, evaluate it
        if not node.available_actions and node.visit_count == 0:
            self._evaluate_node(node)
        
        # Select an action for expansion
        action = node.select_action_for_expansion()
        
        # If no action is available for expansion, node is fully expanded
        if action is None:
            node.fully_expanded = True
            return None
        
        # Get the prior probability and text for the selected action
        prior_prob, action_text = node.available_actions[action]
        
        # Expand the node with the selected action
        new_state, new_state_text = self.state_expander(node.state, action)
        
        # Create a new child node
        child = node.add_child(
            action=action,
            state=new_state,
            state_text=new_state_text,
            prior_probability=prior_prob,
            action_text=action_text
        )
        
        # Evaluate the new node
        self._evaluate_node(child)
        
        # Update node count
        self.total_nodes += 1
        
        return child
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Perform rollout from a node to estimate its value.
        
        Simulation phase of MCTS, estimating node value through rollouts.
        """
        # If node is terminal, return its terminal value
        if node.is_terminal:
            return node.terminal_value
        
        # For nodes with existing value estimates from the value network, we can use that
        if self.config.use_value_network and node.visit_count > 0:
            return node.value
        
        # Perform multiple simulations and average the results
        simulation_values = []
        for _ in range(self.config.num_simulations):
            # Use the state simulator to get a value estimate
            value = self.state_simulator(node.state, self.config.rollout_depth)
            simulation_values.append(value)
        
        # Return the average value from simulations
        return sum(simulation_values) / len(simulation_values) if simulation_values else 0.0
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Update statistics for all nodes in the path from the node to the root.
        
        Backpropagation phase of MCTS, updating node statistics.
        """
        # Apply discount factor for depth
        discount = 1.0
        current = node
        
        # Update statistics for all nodes in the path to the root
        while current is not None:
            current.update(value * discount)
            discount *= self.config.discount_factor
            current = current.parent
    
    def search(self, max_iterations: Optional[int] = None) -> Tuple[List[str], float]:
        """
        Perform the MCTS search to find the best reasoning path.
        
        Args:
            max_iterations: Override for the maximum number of iterations
            
        Returns:
            best_sequence: Sequence of reasoning steps in the best path
            best_value: Value of the best path
        """
        if self.root is None:
            raise ValueError("Root node not initialized. Call initialize_root first.")
        
        max_iterations = max_iterations or self.config.max_iterations
        
        # Main MCTS loop
        for iteration in range(max_iterations):
            self.iteration_count = iteration + 1
            
            # Check if we've reached computational budget
            if self.total_nodes >= self.config.max_nodes:
                logger.info(f"Stopping search after reaching max nodes: {self.total_nodes}")
                break
            
            # Check for early stopping based on best value found
            if hasattr(self, 'best_value') and self.best_value > self.config.early_stopping_threshold:
                best_child = self.root.get_best_child()
                if (best_child and 
                    best_child.value > self.config.early_stopping_threshold and
                    best_child.visit_count >= self.config.early_stopping_visits):
                    logger.info(f"Early stopping at iteration {iteration+1}, found high value path: {best_child.value:.4f}")
                    break
            
            try:
                # 1. Selection: Select a promising node
                selected_node = self._select()
                
                # 2. Expansion: Expand the selected node if possible
                if not selected_node.is_terminal and not selected_node.is_fully_expanded():
                    child_node = self._expand(selected_node)
                    # If expansion successful, use the child for simulation
                    if child_node is not None:
                        selected_node = child_node
                
                # 3. Simulation: Estimate the value of the selected node
                simulation_value = self._simulate(selected_node)
                
                # 4. Backpropagation: Update statistics
                self._backpropagate(selected_node, simulation_value)
            
            except Exception as e:
                logger.error(f"Error during MCTS iteration {iteration+1}: {e}")
                # Continue with next iteration
                continue
            
            # Store the best path found so far
            self._update_best_path()
        
        # Return the best reasoning path found
        return self._get_best_path()
    
    def _update_best_path(self) -> None:
        """Update tracking of the best path found so far"""
        # Initialize best value if not exist
        if not hasattr(self, 'best_value'):
            self.best_value = float('-inf')
            self.best_sequence = []
            self.best_sequence_text = ""
            
        # Find the child of the root with the highest visit count
        best_child = self.root.get_best_child()
        if best_child is None:
            return
        
        # Traverse the tree following the most visited path
        current = best_child
        path = []
        path_text = []
        path_value = current.value
        
        # Collect nodes in the path
        while current is not None:
            if current.action_text is not None:
                path.append(current.action_taken)
                path_text.append(current.action_text)
            
            # If reached a terminal node, update with terminal value
            if current.is_terminal:
                path_value = current.terminal_value
                break
            
            # Move to the most visited child
            best_child = current.get_best_child()
            if best_child is None:
                break
                
            current = best_child
        
        # Update the best path if this one is better
        if path_value > self.best_value:
            self.best_value = path_value
            self.best_sequence = path
            self.best_sequence_text = "\n".join(path_text)
    
    def _get_best_path(self) -> Tuple[List[str], float]:
        """Get the best reasoning path found during search"""
        # Initialize best value if not exist
        if not hasattr(self, 'best_value'):
            self.best_value = 0.0
            self.best_sequence = []
            self.best_sequence_text = ""
            
        # If no path found, return empty path
        if not self.best_sequence_text:
            return [], self.best_value
        
        # Split the path text into steps
        steps = self.best_sequence_text.split("\n")
        return steps, self.best_value
    
    def visualize_tree(self, max_depth: int = 3) -> str:
        """Generate a text visualization of the search tree"""
        if self.root is None:
            return "Empty tree"
        
        lines = []
        lines.append(f"MCTS Tree (iterations: {self.iteration_count}, nodes: {self.total_nodes})")
        
        def format_node(node, depth, prefix=""):
            indent = "  " * depth
            action_str = f"[{node.action_text}]" if node.action_text else ""
            return f"{indent}{prefix}Node(v={node.visit_count}, val={node.value:.3f}) {action_str}"
        
        def traverse(node, depth=0):
            if depth > max_depth:
                return
                
            lines.append(format_node(node, depth))
            
            # Sort children by visit count in descending order
            sorted_children = sorted(
                node.children.items(),
                key=lambda x: x[1].visit_count,
                reverse=True
            )
            
            for i, (action, child) in enumerate(sorted_children):
                is_last = i == len(sorted_children) - 1
                child_prefix = "└─ " if is_last else "├─ "
                
                traverse(child, depth + 1)
                
                # Only show top 3 children at each level to avoid clutter
                if i >= 2 and len(sorted_children) > 3:
                    indent = "  " * depth
                    lines.append(f"{indent}  ... ({len(sorted_children) - 3} more children)")
                    break
        
        traverse(self.root)
        return "\n".join(lines)

class MCTSEnhancedTreeReasoningModule(nn.Module):
    """
    Enhanced Tree-of-Thought reasoning module that incorporates Monte Carlo Tree Search
    for more systematic exploration of reasoning paths.
    """
    
    def __init__(self, config, base_tree_reasoning=None):
        """
        Initialize the MCTS-enhanced tree reasoning module.
        
        Args:
            config: Configuration for tree reasoning and MCTS
            base_tree_reasoning: Optional base tree reasoning module to build upon
        """
        super().__init__()
        self.config = config
        self.base_module = base_tree_reasoning
        
        # MCTS configuration
        self.mcts_config = MCTSConfig(
            max_iterations=getattr(config, 'mcts_max_iterations', 100),
            exploration_weight=getattr(config, 'mcts_exploration_weight', 1.0),
            max_depth=getattr(config, 'max_tree_depth', 5),
            rollout_depth=getattr(config, 'mcts_rollout_depth', 3),
            discount_factor=getattr(config, 'mcts_discount_factor', 0.95),
            top_k_candidates=getattr(config, 'branching_factor', 4),
            use_value_network=getattr(config, 'use_value_function', True),
            num_simulations=getattr(config, 'monte_carlo_samples', 8),
            early_stopping_threshold=getattr(config, 'early_stopping_threshold', 0.95),
            use_beam_search=getattr(config, 'use_beam_search', True),
            beam_size=getattr(config, 'beam_size', 4),
        )
        
        # Define hidden size
        self.hidden_size = getattr(config, 'hidden_size', 768)
        
        # Components for value estimation
        self.value_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Components for candidate generation
        self.thought_generator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            ) for _ in range(self.mcts_config.top_k_candidates)
        ])
        
        # Components for thought evaluation
        self.thought_evaluator = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
    
    def _generate_candidates(self, hidden_states):
        """Generate candidate next thoughts from current hidden states"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Extract the last token representation as the current state
        current_state = hidden_states[:, -1, :]
        
        # Generate candidate thoughts
        candidates = []
        logits = []
        
        for generator in self.thought_generator:
            # Generate a candidate thought
            candidate = generator(current_state).unsqueeze(1)  # [batch_size, 1, hidden_size]
            candidates.append(candidate)
            
            # Compute logit/score for this candidate
            combined = torch.cat([current_state, candidate.squeeze(1)], dim=-1)
            logit = self.thought_evaluator(combined)
            logits.append(logit)
        
        # Stack candidates and compute probabilities
        candidates = torch.cat(candidates, dim=1)  # [batch_size, num_candidates, hidden_size]
        logits = torch.cat(logits, dim=1)  # [batch_size, num_candidates]
        probs = F.softmax(logits, dim=-1)
        
        return candidates, probs
    
    def _evaluate_state(self, state):
        """
        Evaluate a state to get available actions and value estimate.
        Used by MCTS as the state_evaluator function.
        """
        # Generate candidate thoughts/actions
        candidates, probs = self._generate_candidates(state.unsqueeze(0).unsqueeze(0))
        
        # Estimate value of the state
        value = self.value_network(state).item()
        
        # Convert to the format expected by MCTS
        action_probs = {}
        for i in range(candidates.shape[1]):
            # Generate a text representation of the candidate
            candidate_text = f"Thought {i+1}: {self._tensor_to_text(candidates[0, i])}"
            action_probs[i] = (probs[0, i].item(), candidate_text)
        
        return action_probs, value
    
    def _tensor_to_text(self, tensor):
        """
        Convert tensor representation to text for visualization.
        In a real implementation, this would use a language model to decode the representation.
        """
        # Simplified representation - just use first few values of tensor
        values = tensor[:5].tolist()
        return f"[{', '.join([f'{v:.2f}' for v in values])}...]"
    
    def _expand_state(self, state, action):
        """
        Expand a state with a given action to get a new state.
        Used by MCTS as the state_expander function.
        """
        # Generate candidate thoughts/actions
        candidates, _ = self._generate_candidates(state.unsqueeze(0).unsqueeze(0))
        
        # Select the candidate corresponding to the chosen action
        new_state = candidates[0, action, :]
        
        # Generate text representation
        new_state_text = f"State after thought {action+1}: {self._tensor_to_text(new_state)}"
        
        return new_state, new_state_text
    
    def _simulate_from_state(self, state, depth):
        """
        Perform a simulation/rollout from a state to estimate its value.
        Used by MCTS as the state_simulator function.
        """
        current_state = state
        cumulative_value = 0.0
        discount = 1.0
        
        # Simple rollout: follow a random policy for specified depth
        for _ in range(depth):
            # Estimate current state value
            state_value = self.value_network(current_state).item()
            cumulative_value += discount * state_value
            
            # Generate next candidates
            candidates, probs = self._generate_candidates(current_state.unsqueeze(0).unsqueeze(0))
            
            # Sample an action based on probabilities
            action_idx = torch.multinomial(probs[0], 1).item()
            
            # Update state
            current_state = candidates[0, action_idx, :]
            
            # Apply discount
            discount *= self.mcts_config.discount_factor
        
        # Final state evaluation
        final_value = self.value_network(current_state).item()
        cumulative_value += discount * final_value
        
        # Normalize by effective rollout length
        effective_length = sum(self.mcts_config.discount_factor ** i for i in range(depth + 1))
        return cumulative_value / effective_length
    
    def _check_terminal(self, state):
        """
        Check if a state is terminal.
        Used by MCTS as the terminal_checker function.
        """
        # For now, use a simple heuristic: high value states are considered terminal
        value = self.value_network(state).item()
        is_terminal = value > 0.9
        
        return is_terminal, value if is_terminal else None
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Perform tree-of-thought reasoning with Monte Carlo Tree Search.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            reasoning_output: Output hidden states after reasoning
            reasoning_trace: Trace of the reasoning process
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Process one example at a time since MCTS is computationally intensive
        all_outputs = []
        all_traces = []
        
        for b in range(batch_size):
            # Extract the sequence for this batch item
            sequence = hidden_states[b]
            
            # Extract the last token representation as the initial state
            initial_state = sequence[-1, :]
            
            # Initialize MCTS
            mcts = MonteCarloTreeSearch(
                config=self.mcts_config,
                state_evaluator=self._evaluate_state,
                state_expander=self._expand_state,
                state_simulator=self._simulate_from_state,
                terminal_checker=self._check_terminal
            )
            
            # Initialize the root node
            mcts.initialize_root(
                initial_state=initial_state,
                initial_text=f"Initial state: {self._tensor_to_text(initial_state)}"
            )
            
            # Perform the search
            reasoning_steps, final_value = mcts.search()
            
            # Create a trace of the reasoning process
            trace = {
                "steps": reasoning_steps,
                "value": final_value,
                "tree_visualization": mcts.visualize_tree() if self.mcts_config.enable_visualization else "",
                "iterations": mcts.iteration_count,
                "nodes_explored": mcts.total_nodes
            }
            
            # For now, we simply extend the hidden states with transformed values
            # In a real implementation, we would generate actual reasoning steps and transform them
            if reasoning_steps:
                extension_length = min(len(reasoning_steps), 10)  # Limit extension length
                extension = torch.zeros(extension_length, hidden_size, device=hidden_states.device)
                
                # Fill with simple transformed values of the original state for demo purposes
                for i in range(extension_length):
                    extension[i] = initial_state * (1.0 + 0.1 * (i + 1))
                
                # Concatenate the original sequence with the extension
                output = torch.cat([sequence, extension], dim=0)
            else:
                output = sequence
            
            all_outputs.append(output)
            all_traces.append(trace)
        
        # Create a padded tensor for outputs (variable length due to reasoning steps)
        max_length = max(output.shape[0] for output in all_outputs)
        padded_outputs = torch.zeros(batch_size, max_length, hidden_size, device=hidden_states.device)
        
        for b, output in enumerate(all_outputs):
            padded_outputs[b, :output.shape[0]] = output
        
        return padded_outputs, all_traces

def create_mcts_reasoning_module(model_config, base_model=None):
    """
    Factory function to create an MCTS-enhanced tree reasoning module.
    
    Args:
        model_config: Configuration for the model
        base_model: Optional base model to build upon
        
    Returns:
        An instance of MCTSEnhancedTreeReasoningModule
    """
    return MCTSEnhancedTreeReasoningModule(model_config, base_model)

class LanguageModelMCTSIntegration:
    """
    Integration of MCTS with language models for tree-of-thought reasoning.
    
    This class provides methods to use language models for generating and evaluating
    reasoning paths within the MCTS framework.
    """
    
    def __init__(
        self, 
        language_model,
        tokenizer,
        mcts_config=None,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=40
    ):
        """
        Initialize the integration.
        
        Args:
            language_model: The language model to use
            tokenizer: Tokenizer for the language model
            mcts_config: Optional MCTS configuration
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        """
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.mcts_config = mcts_config or MCTSConfig()
        
        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Initialize MCTS components
        self.mcts = None
        
        # Set device based on model's device
        try:
            self.device = next(language_model.parameters()).device
        except (StopIteration, AttributeError):
            logger.warning("Could not determine model device, using 'cpu'")
            self.device = torch.device('cpu')
        
    def _generate_candidates(self, state_text, num_candidates=None):
        """
        Generate candidate next thoughts from current state text.
        
        Args:
            state_text: Current reasoning state as text
            num_candidates: Number of candidates to generate
            
        Returns:
            List of (candidate_text, probability) tuples
        """
        num_candidates = num_candidates or self.mcts_config.top_k_candidates
        
        # Prepare prompt for generating next steps
        prompt = f"{state_text}\nNext step in reasoning:"
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # Generate multiple candidates
        candidates = []
        
        for _ in range(num_candidates):
            # Generate with sampling for diversity
            outputs = self.language_model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            # Decode the generated text
            generated_ids = outputs[0][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Estimate probability (simplified)
            # In a real implementation, we would compute the actual probability
            prob = 1.0 / num_candidates
            
            candidates.append((generated_text, prob))
        
        # Normalize probabilities
        total_prob = sum(prob for _, prob in candidates)
        if total_prob > 0:
            normalized_candidates = [(text, prob/total_prob) for text, prob in candidates]
        else:
            normalized_candidates = [(text, 1.0/len(candidates)) for text, prob in candidates]
        
        return normalized_candidates
    
    def _evaluate_state(self, state):
        """
        Evaluate a state to get available actions and value estimate.
        Used by MCTS as the state_evaluator function.
        """
        # Convert state tensor to text
        state_text = state.state_text if hasattr(state, 'state_text') else "Current reasoning state"
        
        # Generate candidate thoughts/actions
        candidates = self._generate_candidates(state_text)
        
        # Estimate value of the state
        value = self._estimate_value(state_text)
        
        # Convert to the format expected by MCTS
        action_probs = {}
        for i, (candidate_text, prob) in enumerate(candidates):
            action_probs[i] = (prob, candidate_text)
        
        return action_probs, value
    
    def _estimate_value(self, state_text):
        """
        Estimate the value of a state using the language model.
        
        Args:
            state_text: State text to evaluate
            
        Returns:
            Estimated value between 0 and 1
        """
        # Prepare prompt for value estimation
        prompt = f"{state_text}\nOn a scale of 0 to 100, how promising is this reasoning path?"
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # Generate a response
        outputs = self.language_model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the generated text
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract a numeric value (simplified)
        # In a real implementation, we would use more robust parsing
        try:
            # Try to extract a number from the text
            import re
            numbers = re.findall(r'\d+', generated_text)
            if numbers:
                value = float(numbers[0]) / 100.0  # Normalize to [0, 1]
                return min(max(value, 0.0), 1.0)  # Clamp to [0, 1]
        except Exception as e:
            print(f"Warning: Failed to extract value from '{generated_text}': {e}")
        
        # Default value if extraction fails
        return 0.5
    
    def _expand_state(self, state, action):
        """
        Expand a state with a given action to get a new state.
        Used by MCTS as the state_expander function.
        """
        # Get the action text
        if not hasattr(state, 'available_actions'):
            raise ValueError(f"State has no available_actions attribute: {state}")
            
        if action not in state.available_actions:
            # If action is invalid, log warning and use a default 
            logger.warning(f"Invalid action {action} for state. Available actions: {list(state.available_actions.keys())}")
            # Try to find a valid action as fallback
            if state.available_actions:
                action = list(state.available_actions.keys())[0]
            else:
                raise ValueError(f"No available actions in state: {state}")
                
        action_text = state.available_actions[action][1]
        
        # Combine current state text with action text
        new_state_text = f"{state.state_text}\n{action_text}"
        
        # Create a tensor representation (simplified)
        # In a real implementation, we would use the language model's embeddings
        # Here we just use a random tensor of appropriate size
        new_state_tensor = torch.randn(768, device=self.device)
        
        return new_state_tensor, new_state_text
    
    def _simulate_from_state(self, state, depth):
        """
        Perform a simulation/rollout from a state to estimate its value.
        Used by MCTS as the state_simulator function.
        """
        # Start with the current state text
        current_text = state.state_text if hasattr(state, 'state_text') else "Current reasoning state"
        
        # Perform a rollout by generating steps up to the specified depth
        cumulative_value = 0.0
        discount = 1.0
        
        try:
            for d in range(depth):
                # Estimate current state value
                state_value = self._estimate_value(current_text)
                cumulative_value += discount * state_value
                
                # Generate next candidates
                candidates = self._generate_candidates(current_text, num_candidates=1)
                
                if not candidates:
                    # No more candidates, end simulation
                    break
                    
                # Update state with the first candidate
                next_text = candidates[0][0]
                if not next_text:  # Skip empty responses
                    break
                    
                current_text = f"{current_text}\n{next_text}"
                
                # Apply discount
                discount *= self.mcts_config.discount_factor
            
            # Final state evaluation
            final_value = self._estimate_value(current_text)
            cumulative_value += discount * final_value
            
            # Normalize by effective rollout length
            effective_length = sum(self.mcts_config.discount_factor ** i for i in range(depth + 1))
            if effective_length > 0:
                return cumulative_value / effective_length
            else:
                return final_value
                
        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            # Return a reasonable default value
            return 0.5
    
    def _check_terminal(self, state):
        """
        Check if a state is terminal.
        Used by MCTS as the terminal_checker function.
        """
        # Get state text
        state_text = state.state_text if hasattr(state, 'state_text') else "Current reasoning state"
        
        # Prepare prompt for checking if reasoning is complete
        prompt = f"{state_text}\nIs this reasoning complete and sufficient to solve the problem? (yes/no)"
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # Generate a response
        outputs = self.language_model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the generated text
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).lower()
        
        # Check if the response indicates completion
        is_terminal = "yes" in generated_text or "complete" in generated_text
        
        # If terminal, estimate final value
        terminal_value = self._estimate_value(state_text) if is_terminal else None
        
        return is_terminal, terminal_value
    
    def solve_problem(self, problem_text, max_iterations=None):
        """
        Solve a problem using tree-of-thought reasoning with MCTS.
        
        Args:
            problem_text: Text describing the problem to solve
            max_iterations: Maximum number of MCTS iterations
            
        Returns:
            solution_text: Text of the solution
            reasoning_trace: Trace of the reasoning process
        """
        if not problem_text:
            logger.warning("Empty problem text provided")
            return "No problem provided", {"steps": [], "value": 0.0, "iterations": 0, "nodes_explored": 0}
            
        # Initialize MCTS
        self.mcts = MonteCarloTreeSearch(
            config=self.mcts_config,
            state_evaluator=self._evaluate_state,
            state_expander=self._expand_state,
            state_simulator=self._simulate_from_state,
            terminal_checker=self._check_terminal
        )
        
        # Create initial state tensor (simplified)
        # In a real implementation, we would use the language model's embeddings
        initial_state = torch.randn(768, device=self.device)
        
        try:
            # Initialize the root node
            self.mcts.initialize_root(
                initial_state=initial_state,
                initial_text=problem_text
            )
            
            # Perform the search
            reasoning_steps, final_value = self.mcts.search(max_iterations)
            
            # Combine the reasoning steps into a solution
            solution_text = problem_text
            for step in reasoning_steps:
                solution_text += f"\n{step}"
            
            # Create a trace of the reasoning process
            trace = {
                "steps": reasoning_steps,
                "value": final_value,
                "tree_visualization": self.mcts.visualize_tree() if self.mcts_config.enable_visualization else "",
                "iterations": self.mcts.iteration_count,
                "nodes_explored": self.mcts.total_nodes
            }
            
            return solution_text, trace
            
        except Exception as e:
            logger.error(f"Error during problem solving: {e}")
            # Return a minimal result in case of error
            error_message = f"Error while solving the problem: {str(e)}"
            return problem_text + "\n" + error_message, {
                "steps": [error_message], 
                "value": 0.0, 
                "tree_visualization": "", 
                "iterations": 0, 
                "nodes_explored": 0
            }

# Example usage
def example_usage():
    """Example of how to use the MCTS-enhanced tree reasoning"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load a language model and tokenizer
        model_name = "gpt2"  # Use a larger model in practice
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Configure MCTS
        mcts_config = MCTSConfig(
            max_iterations=50,
            exploration_weight=1.5,
            rollout_depth=2,
            top_k_candidates=3,
            num_simulations=2,
            enable_visualization=True
        )
        
        # Create the integration
        integration = LanguageModelMCTSIntegration(
            language_model=model,
            tokenizer=tokenizer,
            mcts_config=mcts_config,
            temperature=0.8
        )
        
        # Solve a problem
        problem = "Solve the following problem step by step: If a train travels at 60 mph, how long will it take to travel 150 miles?"
        solution, trace = integration.solve_problem(problem)
        
        print("Problem:")
        print(problem)
        print("\nSolution:")
        print(solution)
        print("\nReasoning steps:")
        for i, step in enumerate(trace["steps"]):
            print(f"{i+1}. {step}")
        print(f"\nFinal value: {trace['value']:.4f}")
        print(f"Iterations: {trace['iterations']}")
        print(f"Nodes explored: {trace['nodes_explored']}")
        
        if trace["tree_visualization"]:
            print("\nTree visualization:")
            print(trace["tree_visualization"])
            
    except ImportError as e:
        print(f"This example requires PyTorch and Transformers libraries: {e}")

if __name__ == "__main__":
    example_usage() 