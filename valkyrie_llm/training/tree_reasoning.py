import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """
    A node in the reasoning tree.
    
    Each node represents a single step in the reasoning process
    and maintains connections to parent and child nodes.
    """
    content: str
    hidden_state: torch.Tensor
    logprob: float = 0.0
    parent: Optional["TreeNode"] = None
    children: List["TreeNode"] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class TreeReasoning(nn.Module):
    """
    Tree-based reasoning module that builds and evaluates reasoning trees.
    
    This module enables the model to perform structured reasoning by constructing
    trees of thought, where each path represents a different reasoning strategy.
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 max_tree_depth: int = 5, 
                 max_children: int = 3,
                 beam_width: int = 5, 
                 value_threshold: float = 0.6,
                 use_iterative_deepening: bool = True,
                 max_iterations: int = 10,
                 temperature: float = 0.8):
        """
        Initialize tree reasoning module.
        
        Args:
            hidden_size: Hidden dimension size
            max_tree_depth: Maximum depth of reasoning trees
            max_children: Maximum number of children per tree node
            beam_width: Width of beam search
            value_threshold: Threshold for value to consider a path promising
            use_iterative_deepening: Whether to use iterative deepening
            max_iterations: Maximum number of iterations for tree search
            temperature: Temperature for sampling reasoning steps
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_tree_depth = max_tree_depth
        self.max_children = max_children
        self.beam_width = beam_width
        self.value_threshold = value_threshold
        self.use_iterative_deepening = use_iterative_deepening
        self.max_iterations = max_iterations
        self.temperature = temperature
        
        # Node expansion network
        self.node_expander = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size * max_children)
        )
        
        # Value estimation network
        self.value_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Path integration network
        self.path_integrator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Tree aggregation network
        self.tree_aggregator = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states, attention_mask=None, context_info=None):
        """
        Forward pass with tree-based reasoning.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask
            context_info: Additional context information
            
        Returns:
            Enhanced hidden states
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Process each example in the batch
        outputs = []
        for i in range(batch_size):
            # Extract single example
            example_hidden = hidden_states[i].unsqueeze(0)  # [1, seq_len, hidden_size]
            example_mask = attention_mask[i].unsqueeze(0) if attention_mask is not None else None
            
            # Use the CLS token as the root node state
            root_state = example_hidden[:, 0]  # [1, hidden_size]
            
            # Construct reasoning tree
            root_node = TreeNode(
                content="Root",
                hidden_state=root_state,
                logprob=0.0
            )
            
            # Build and evaluate reasoning tree
            final_state = self._reason_with_tree(
                root_node, 
                example_hidden, 
                example_mask
            )
            
            # Enhance the input with tree reasoning results
            enhanced_hidden = self._enhance_with_tree_reasoning(
                example_hidden, 
                final_state
            )
            
            outputs.append(enhanced_hidden)
        
        # Combine all outputs
        return torch.cat(outputs, dim=0)
    
    def _reason_with_tree(self, root_node, hidden_states, attention_mask=None):
        """
        Perform tree-based reasoning starting from the root node.
        
        Args:
            root_node: Root node of reasoning tree
            hidden_states: Full sequence hidden states
            attention_mask: Attention mask
            
        Returns:
            Final hidden state after tree reasoning
        """
        if self.use_iterative_deepening:
            return self._iterative_deepening_search(root_node, hidden_states, attention_mask)
        else:
            return self._beam_search(root_node, hidden_states, attention_mask)
    
    def _iterative_deepening_search(self, root_node, hidden_states, attention_mask=None):
        """
        Perform iterative deepening search on the reasoning tree.
        
        Args:
            root_node: Root node of reasoning tree
            hidden_states: Full sequence hidden states
            attention_mask: Attention mask
            
        Returns:
            Final hidden state after tree reasoning
        """
        best_value = 0.0
        best_leaf = root_node
        
        # Iteratively increase depth
        for max_depth in range(1, self.max_tree_depth + 1):
            # Reset tree for this iteration
            root_node.children = []
            
            # Perform limited-depth beam search
            frontier = [root_node]
            for depth in range(max_depth):
                if not frontier:
                    break
                
                next_frontier = []
                for node in frontier:
                    # Expand node
                    children = self._expand_node(node, hidden_states)
                    
                    # Add promising children to frontier
                    for child in children[:self.beam_width]:
                        next_frontier.append(child)
                
                # Keep only the best nodes
                next_frontier.sort(key=lambda x: self._evaluate_node(x), reverse=True)
                frontier = next_frontier[:self.beam_width]
            
            # Find best leaf node from this iteration
            if frontier:
                frontier.sort(key=lambda x: self._evaluate_node(x), reverse=True)
                current_best = frontier[0]
                current_value = self._evaluate_node(current_best)
                
                if current_value > best_value:
                    best_value = current_value
                    best_leaf = current_best
                
                # Early stopping if we found a good enough solution
                if best_value > self.value_threshold:
                    break
        
        # Collect nodes along the best path
        path_nodes = self._collect_path(best_leaf)
        
        # Integrate information along the path
        return self._integrate_path(path_nodes)
    
    def _beam_search(self, root_node, hidden_states, attention_mask=None):
        """
        Perform beam search on the reasoning tree.
        
        Args:
            root_node: Root node of reasoning tree
            hidden_states: Full sequence hidden states
            attention_mask: Attention mask
            
        Returns:
            Final hidden state after tree reasoning
        """
        frontier = [root_node]
        
        # Expand tree with beam search
        for depth in range(self.max_tree_depth):
            if not frontier:
                break
            
            next_frontier = []
            for node in frontier:
                # Expand node
                children = self._expand_node(node, hidden_states)
                
                # Add children to frontier
                next_frontier.extend(children)
            
            # Keep only the best nodes
            next_frontier.sort(key=lambda x: self._evaluate_node(x), reverse=True)
            frontier = next_frontier[:self.beam_width]
        
        # Find best leaf node
        if frontier:
            frontier.sort(key=lambda x: self._evaluate_node(x), reverse=True)
            best_leaf = frontier[0]
            
            # Collect nodes along the best path
            path_nodes = self._collect_path(best_leaf)
            
            # Integrate information along the path
            return self._integrate_path(path_nodes)
        else:
            # Fallback to root node if frontier is empty
            return root_node.hidden_state
    
    def _expand_node(self, node, hidden_states):
        """
        Expand a node by generating children nodes.
        
        Args:
            node: Node to expand
            hidden_states: Full sequence hidden states
            
        Returns:
            List of child nodes
        """
        # Generate child states
        child_states = self.node_expander(node.hidden_state)
        
        # Reshape to get individual child states
        child_states = child_states.view(-1, self.max_children, self.hidden_size)
        
        # Create child nodes
        children = []
        for i in range(self.max_children):
            child_state = child_states[0, i]
            
            # Calculate logprob for this child (simplified)
            # In a real implementation, this would use the language model
            logprob = torch.cosine_similarity(
                child_state.unsqueeze(0), 
                hidden_states.mean(dim=1), 
                dim=1
            ).item()
            
            child_node = TreeNode(
                content=f"Step {len(node.children) + 1}",
                hidden_state=child_state,
                logprob=logprob,
                parent=node
            )
            
            node.children.append(child_node)
            children.append(child_node)
        
        # Sort children by logprob
        children.sort(key=lambda x: x.logprob, reverse=True)
        
        return children
    
    def _evaluate_node(self, node):
        """
        Evaluate a node's value.
        
        Args:
            node: Node to evaluate
            
        Returns:
            Value score for the node
        """
        # Get value estimate
        value = self.value_estimator(node.hidden_state).item()
        
        # Combine with path probability
        path_logprob = self._get_path_logprob(node)
        
        # Balance value and probability
        score = 0.8 * value + 0.2 * path_logprob
        
        return score
    
    def _get_path_logprob(self, node):
        """
        Calculate path log probability for a node.
        
        Args:
            node: Target node
            
        Returns:
            Normalized path log probability
        """
        # Collect log probabilities along path
        path_logprobs = []
        current = node
        
        while current.parent is not None:
            path_logprobs.append(current.logprob)
            current = current.parent
        
        # Normalize by path length
        if path_logprobs:
            return sum(path_logprobs) / len(path_logprobs)
        else:
            return 0.0
    
    def _collect_path(self, leaf_node):
        """
        Collect nodes along path from root to leaf.
        
        Args:
            leaf_node: Leaf node
            
        Returns:
            List of nodes from root to leaf
        """
        path = []
        current = leaf_node
        
        # Traverse up from leaf to root
        while current is not None:
            path.append(current)
            current = current.parent
        
        # Reverse to get root-to-leaf order
        return path[::-1]
    
    def _integrate_path(self, path_nodes):
        """
        Integrate information along a reasoning path.
        
        Args:
            path_nodes: List of nodes in the path
            
        Returns:
            Integrated hidden state
        """
        if len(path_nodes) == 1:
            # Only root node
            return path_nodes[0].hidden_state
        
        # Start with root state
        current_state = path_nodes[0].hidden_state
        
        # Integrate each step along the path
        for i in range(1, len(path_nodes)):
            next_state = path_nodes[i].hidden_state
            
            # Integrate with the current state
            combined = torch.cat([current_state, next_state], dim=-1)
            current_state = self.path_integrator(combined)
        
        return current_state
    
    def _enhance_with_tree_reasoning(self, hidden_states, tree_state):
        """
        Enhance the input hidden states with tree reasoning results.
        
        Args:
            hidden_states: Original hidden states
            tree_state: Final state from tree reasoning
            
        Returns:
            Enhanced hidden states
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Expand tree state to sequence length
        expanded_tree_state = tree_state.expand(batch_size, seq_len, -1)
        
        # Use transformer to attend between original and tree-reasoned states
        combined_states = hidden_states + expanded_tree_state
        
        # Process with transformer
        enhanced_states = self.tree_aggregator(combined_states)
        
        # Project to output dimension
        output_states = self.output_projection(enhanced_states)
        
        # Add residual connection
        return hidden_states + output_states


class BestFirstSearch(nn.Module):
    """
    Best-first search implementation for tree reasoning.
    
    This module provides a more efficient tree search algorithm that prioritizes
    the most promising paths.
    """
    
    def __init__(self, hidden_size: int = 768, max_frontier_size: int = 100):
        """
        Initialize best-first search module.
        
        Args:
            hidden_size: Hidden dimension size
            max_frontier_size: Maximum size of the frontier
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_frontier_size = max_frontier_size
        
        # Heuristic function
        self.heuristic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
    
    def search(self, root_node, expand_fn, goal_fn, max_steps: int = 100):
        """
        Perform best-first search from root node.
        
        Args:
            root_node: Root node to start search from
            expand_fn: Function to expand nodes
            goal_fn: Function to check if node is a goal
            max_steps: Maximum search steps
            
        Returns:
            Goal node if found, otherwise best node visited
        """
        # Initialize frontier with root node
        frontier = [(self._evaluate(root_node), 0, root_node)]  # (score, step, node)
        
        # Keep track of best node found
        best_node = root_node
        best_score = self._evaluate(root_node)
        
        # Visited set to avoid cycles
        visited = set()
        
        # Search
        step = 0
        while frontier and step < max_steps:
            # Get highest-scoring node from frontier
            _, _, current = frontier.pop(0)
            
            # Check if this is a goal node
            if goal_fn(current):
                return current
            
            # Check if this is the best node so far
            current_score = self._evaluate(current)
            if current_score > best_score:
                best_score = current_score
                best_node = current
            
            # Expand node
            children = expand_fn(current)
            
            # Add children to frontier
            for child in children:
                # Generate a hash for the node
                node_hash = hash(str(child.hidden_state.detach().cpu().numpy().tobytes()))
                
                if node_hash not in visited:
                    visited.add(node_hash)
                    
                    # Score is a combination of heuristic and path cost
                    score = self._evaluate(child)
                    
                    # Add to frontier
                    frontier.append((score, step, child))
            
            # Sort frontier by score (highest first)
            frontier.sort(reverse=True)
            
            # Limit frontier size
            if len(frontier) > self.max_frontier_size:
                frontier = frontier[:self.max_frontier_size]
            
            step += 1
        
        # Return best node found
        return best_node
    
    def _evaluate(self, node):
        """
        Evaluate a node's priority score.
        
        Args:
            node: Node to evaluate
            
        Returns:
            Priority score
        """
        # Calculate heuristic value
        heuristic_value = self.heuristic(node.hidden_state).item()
        
        # Calculate path cost (using logprob)
        path_cost = 0.0
        current = node
        depth = 0
        
        while current.parent is not None:
            path_cost += current.logprob
            current = current.parent
            depth += 1
        
        # Normalize path cost
        path_cost = path_cost / max(1, depth)
        
        # Combine heuristic and path cost
        # Higher heuristic and higher path probability (logprob) are better
        return 0.7 * heuristic_value + 0.3 * path_cost


class TreeOfThoughts(nn.Module):
    """
    Tree of Thoughts implementation that enables structured reasoning.
    
    This module builds on tree reasoning to provide a more comprehensive
    implementation of the Tree of Thoughts reasoning approach.
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 max_depth: int = 5,
                 max_breadth: int = 3,
                 search_strategy: str = "best_first",
                 pruning_threshold: float = 0.3,
                 use_value_guidance: bool = True):
        """
        Initialize Tree of Thoughts module.
        
        Args:
            hidden_size: Hidden dimension size
            max_depth: Maximum tree depth
            max_breadth: Maximum breadth at each level
            search_strategy: Search strategy (best_first, beam, dfs, or bfs)
            pruning_threshold: Threshold for pruning low-value branches
            use_value_guidance: Whether to use value estimates for guidance
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_depth = max_depth
        self.max_breadth = max_breadth
        self.search_strategy = search_strategy
        self.pruning_threshold = pruning_threshold
        self.use_value_guidance = use_value_guidance
        
        # Thought generator network
        self.thought_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size * max_breadth)
        )
        
        # Value estimator
        self.value_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Best-first search
        self.bfs_search = BestFirstSearch(hidden_size)
        
        # State integration
        self.state_integrator = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states, attention_mask=None, context_info=None):
        """
        Forward pass with Tree of Thoughts reasoning.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            context_info: Additional context information
            
        Returns:
            Enhanced hidden states
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        outputs = []
        for i in range(batch_size):
            # Process single example
            example_hidden = hidden_states[i].unsqueeze(0)
            example_mask = attention_mask[i].unsqueeze(0) if attention_mask is not None else None
            
            # Use CLS token as root state
            root_state = example_hidden[:, 0]
            
            # Create root node
            root_node = TreeNode(
                content="Root",
                hidden_state=root_state,
                logprob=0.0
            )
            
            # Search in the tree
            if self.search_strategy == "best_first":
                final_node = self.bfs_search.search(
                    root_node,
                    expand_fn=lambda node: self._expand_thoughts(node, example_hidden),
                    goal_fn=lambda node: self._is_goal_state(node, example_hidden),
                    max_steps=100
                )
            else:
                # Fallback to simple expansion
                final_node = self._simple_search(root_node, example_hidden)
            
            # Collect path
            path_nodes = self._collect_path(final_node)
            
            # Integrate states along the path
            path_states = torch.stack([node.hidden_state for node in path_nodes], dim=1)
            
            # Process with transformer
            integrated_state = self._integrate_path_states(path_states)
            
            # Enhance original states
            enhanced = self._enhance_with_tot(example_hidden, integrated_state)
            
            outputs.append(enhanced)
        
        return torch.cat(outputs, dim=0)
    
    def _expand_thoughts(self, node, hidden_states):
        """
        Expand a node by generating thought continuations.
        
        Args:
            node: Node to expand
            hidden_states: Full sequence hidden states
            
        Returns:
            List of child nodes
        """
        # Skip expansion if at max depth
        if self._get_depth(node) >= self.max_depth:
            return []
        
        # Generate thought continuations
        thought_states = self.thought_generator(node.hidden_state)
        
        # Reshape to get individual thought states
        thought_states = thought_states.view(-1, self.max_breadth, self.hidden_size)
        
        # Create child nodes
        children = []
        for i in range(self.max_breadth):
            thought_state = thought_states[0, i]
            
            # Calculate logprob (simplified)
            logprob = torch.cosine_similarity(
                thought_state.unsqueeze(0), 
                hidden_states.mean(dim=1), 
                dim=1
            ).item()
            
            # Skip low probability thoughts
            if logprob < self.pruning_threshold and len(children) > 0:
                continue
            
            child_node = TreeNode(
                content=f"Thought {len(node.children) + 1}",
                hidden_state=thought_state,
                logprob=logprob,
                parent=node
            )
            
            # Apply value guidance
            if self.use_value_guidance:
                # Only add if value is above threshold
                value = self.value_estimator(thought_state).item()
                if value >= self.pruning_threshold or len(children) == 0:
                    node.children.append(child_node)
                    children.append(child_node)
            else:
                node.children.append(child_node)
                children.append(child_node)
        
        # Sort children by value
        if self.use_value_guidance:
            children.sort(
                key=lambda x: self.value_estimator(x.hidden_state).item(), 
                reverse=True
            )
        else:
            # Sort by logprob
            children.sort(key=lambda x: x.logprob, reverse=True)
        
        return children
    
    def _is_goal_state(self, node, hidden_states):
        """
        Check if a node represents a goal state.
        
        Args:
            node: Node to check
            hidden_states: Full sequence hidden states
            
        Returns:
            True if node is a goal state, False otherwise
        """
        # Check depth
        if self._get_depth(node) >= self.max_depth:
            return True
        
        # Check value
        value = self.value_estimator(node.hidden_state).item()
        
        # High value nodes are considered goals
        return value > 0.9
    
    def _get_depth(self, node):
        """
        Get depth of a node in the tree.
        
        Args:
            node: Target node
            
        Returns:
            Depth of the node
        """
        depth = 0
        current = node
        
        while current.parent is not None:
            depth += 1
            current = current.parent
        
        return depth
    
    def _simple_search(self, root_node, hidden_states):
        """
        Perform simple search in the tree.
        
        Args:
            root_node: Root node
            hidden_states: Full sequence hidden states
            
        Returns:
            Best leaf node found
        """
        current = root_node
        
        # Expand to max depth, always choosing the best child
        for _ in range(self.max_depth):
            children = self._expand_thoughts(current, hidden_states)
            
            if not children:
                break
            
            # Choose best child
            current = children[0]
        
        return current
    
    def _collect_path(self, leaf_node):
        """
        Collect nodes along path from root to leaf.
        
        Args:
            leaf_node: Leaf node
            
        Returns:
            List of nodes from root to leaf
        """
        path = []
        current = leaf_node
        
        # Traverse up from leaf to root
        while current is not None:
            path.append(current)
            current = current.parent
        
        # Reverse to get root-to-leaf order
        return path[::-1]
    
    def _integrate_path_states(self, path_states):
        """
        Integrate states along a path.
        
        Args:
            path_states: Tensor of states along path [batch, path_len, hidden_size]
            
        Returns:
            Integrated state
        """
        # Process with transformer
        return self.state_integrator(path_states).mean(dim=1, keepdim=True)
    
    def _enhance_with_tot(self, hidden_states, tot_state):
        """
        Enhance hidden states with Tree of Thoughts reasoning.
        
        Args:
            hidden_states: Original hidden states
            tot_state: State from Tree of Thoughts reasoning
            
        Returns:
            Enhanced hidden states
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Expand ToT state to sequence length
        expanded_tot_state = tot_state.expand(batch_size, seq_len, -1)
        
        # Combine with original states
        combined_states = hidden_states + 0.5 * expanded_tot_state
        
        # Apply output projection
        output_states = self.output_projection(combined_states)
        
        # Residual connection
        return hidden_states + output_states 