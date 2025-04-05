import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)

@dataclass
class TreeOfThoughtConfig:
    """Configuration for Tree-of-Thought reasoning"""
    hidden_size: int = 768
    num_heads: int = 8
    dropout: float = 0.1
    max_tree_depth: int = 3
    branching_factor: int = 4
    pruning_threshold: float = 0.3
    use_value_function: bool = True
    use_beam_search: bool = True
    beam_size: int = 4
    use_monte_carlo: bool = True
    monte_carlo_samples: int = 8
    use_self_consistency: bool = True
    consistency_threshold: float = 0.7
    use_uncertainty_estimation: bool = True
    uncertainty_samples: int = 5
    use_adaptive_computation: bool = True
    max_computation_steps: int = 10
    early_stopping_threshold: float = 0.95

@dataclass
class TreeReasoningConfig:
    """Configuration for tree-based reasoning module"""
    hidden_size: int = 2048
    tree_depth: int = 3
    num_branches: int = 4
    use_residual: bool = True
    use_gating: bool = True
    dropout: float = 0.1
    activation: str = "gelu"
    use_attention: bool = True
    attention_heads: int = 8
    attention_dropout: float = 0.1
    tree_routing_type: str = "learned"  # learned, fixed, attention-based
    num_tree_experts: int = 2
    use_hierarchical_gating: bool = True
    pruning_threshold: float = 0.01
    normalize_gate_probs: bool = True
    gate_temperature: float = 1.0
    
    # Integration with model
    rwkv_compatible: bool = True
    transformer_compatible: bool = True
    use_rwkv_time_shift: bool = True
    
    # Memory optimization
    use_checkpoint_for_branches: bool = True
    optimize_memory: bool = True
    
    def __post_init__(self):
        # Validate config
        if self.tree_depth < 1:
            raise ValueError("Tree depth must be at least 1")
        if self.num_branches < 2:
            raise ValueError("Number of branches must be at least 2")

class ThoughtGenerator(nn.Module):
    """Generates candidate thoughts for exploration"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.branching_factor = config.branching_factor
        
        # Thought generation layers
        self.thought_projector = nn.Linear(self.hidden_size, self.hidden_size)
        self.thought_generator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            ) for _ in range(self.branching_factor)
        ])
        
        # Diversity promotion
        self.diversity_head = nn.Linear(self.hidden_size, self.branching_factor)
        
    def forward(self, hidden_states):
        """
        Generate diverse candidate thoughts
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            thoughts: List of [batch_size, seq_len, hidden_size] tensors
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project input
        projected = self.thought_projector(hidden_states)
        
        # Generate candidate thoughts
        thoughts = []
        for i in range(self.branching_factor):
            thought = self.thought_generator[i](projected)
            thoughts.append(thought)
        
        # Promote diversity through orthogonalization
        if len(thoughts) > 1:
            # Stack thoughts for batch processing
            stacked_thoughts = torch.stack(thoughts, dim=1)  # [batch_size, branching_factor, seq_len, hidden_size]
            
            # Reshape for diversity calculation
            reshaped = stacked_thoughts.view(batch_size, self.branching_factor, -1)  # [batch_size, branching_factor, seq_len*hidden_size]
            
            # Normalize
            normalized = F.normalize(reshaped, p=2, dim=2)
            
            # Compute pairwise similarities
            similarities = torch.bmm(normalized, normalized.transpose(1, 2))  # [batch_size, branching_factor, branching_factor]
            
            # Zero out diagonal (self-similarity)
            diag_mask = torch.eye(self.branching_factor, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1, -1)
            similarities = similarities * (1 - diag_mask)
            
            # Compute diversity loss (negative of average similarity)
            diversity_loss = similarities.sum(dim=(1, 2)) / (self.branching_factor * (self.branching_factor - 1))
            
            # Use diversity head to adjust thoughts
            diversity_weights = F.softmax(self.diversity_head(hidden_states.mean(dim=1)), dim=1)  # [batch_size, branching_factor]
            
            # Apply diversity weights
            for i in range(self.branching_factor):
                thoughts[i] = thoughts[i] * diversity_weights[:, i].view(batch_size, 1, 1)
        
        return thoughts

class ValueFunction(nn.Module):
    """Estimates the value of a thought path"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Value estimation layers
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # Uncertainty estimation
        if config.use_uncertainty_estimation:
            self.uncertainty_net = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.hidden_size // 2, 1)
            )
        
    def forward(self, hidden_states):
        """
        Estimate the value of a thought path
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            value: [batch_size, 1]
            uncertainty: [batch_size, 1] (optional)
        """
        # Pool sequence dimension
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Estimate value
        value = self.value_net(pooled)  # [batch_size, 1]
        
        # Estimate uncertainty if enabled
        if self.config.use_uncertainty_estimation:
            uncertainty = torch.sigmoid(self.uncertainty_net(pooled))  # [batch_size, 1]
            return value, uncertainty
        
        return value, None

class ConsistencyChecker(nn.Module):
    """Checks consistency between different thought paths"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Consistency checking layers
        self.consistency_net = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1)
        )
        
    def forward(self, thoughts):
        """
        Check consistency between thoughts
        
        Args:
            thoughts: List of [batch_size, seq_len, hidden_size] tensors
            
        Returns:
            consistency_scores: [batch_size, num_thoughts, num_thoughts]
        """
        batch_size = thoughts[0].size(0)
        num_thoughts = len(thoughts)
        
        # Pool sequence dimension
        pooled_thoughts = [thought.mean(dim=1) for thought in thoughts]  # List of [batch_size, hidden_size]
        
        # Compute pairwise consistency scores
        consistency_scores = torch.zeros(batch_size, num_thoughts, num_thoughts, device=thoughts[0].device)
        
        for i in range(num_thoughts):
            for j in range(i+1, num_thoughts):
                # Concatenate thought pairs
                paired = torch.cat([pooled_thoughts[i], pooled_thoughts[j]], dim=1)  # [batch_size, hidden_size*2]
                
                # Compute consistency score
                score = torch.sigmoid(self.consistency_net(paired))  # [batch_size, 1]
                
                # Fill in symmetric matrix
                consistency_scores[:, i, j] = score.squeeze(1)
                consistency_scores[:, j, i] = score.squeeze(1)
        
        # Fill diagonal with 1.0 (self-consistency)
        for i in range(num_thoughts):
            consistency_scores[:, i, i] = 1.0
        
        return consistency_scores

class TreeNode:
    """Node in the tree of thought"""
    
    def __init__(self, hidden_states, parent=None, depth=0):
        self.hidden_states = hidden_states
        self.parent = parent
        self.children = []
        self.value = None
        self.uncertainty = None
        self.depth = depth
        
    def add_child(self, child_node):
        self.children.append(child_node)
        
    def is_leaf(self):
        return len(self.children) == 0

class TreeOfThoughtReasoner(nn.Module):
    """Tree-of-Thought reasoning module"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Thought generation
        self.thought_generator = ThoughtGenerator(config)
        
        # Value function
        if config.use_value_function:
            self.value_function = ValueFunction(config)
        
        # Consistency checker
        if config.use_self_consistency:
            self.consistency_checker = ConsistencyChecker(config)
        
        # Output integration
        self.output_integration = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, hidden_states):
        """
        Perform tree-of-thought reasoning
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            reasoning_trace: Dict containing reasoning information
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create root node
        root = TreeNode(hidden_states, depth=0)
        
        # Build tree through recursive exploration
        if self.config.use_beam_search:
            final_nodes = self._beam_search(root)
        else:
            self._expand_node(root)
            final_nodes = self._collect_leaves(root)
        
        # Collect outputs from final nodes
        final_outputs = [node.hidden_states for node in final_nodes]
        
        # Check consistency if enabled
        if self.config.use_self_consistency and len(final_outputs) > 1:
            consistency_scores = self.consistency_checker(final_outputs)
            
            # Find most consistent output
            avg_consistency = consistency_scores.mean(dim=2)  # [batch_size, num_outputs]
            best_indices = avg_consistency.argmax(dim=1)  # [batch_size]
            
            # Select best output for each batch item
            selected_outputs = []
            for b in range(batch_size):
                best_idx = best_indices[b].item()
                selected_outputs.append(final_outputs[best_idx][b:b+1])
            
            # Concatenate along batch dimension
            output = torch.cat(selected_outputs, dim=0)
        else:
            # Use highest value output
            values = [node.value for node in final_nodes]
            best_idx = values.index(max(values))
            output = final_outputs[best_idx]
        
        # Final integration
        output = self.output_integration(output)
        
        # Prepare reasoning trace
        reasoning_trace = {
            'num_nodes_explored': self._count_nodes(root),
            'max_depth': max(node.depth for node in final_nodes),
            'final_values': [node.value.item() if node.value is not None else 0.0 for node in final_nodes],
            'tree_structure': self._serialize_tree(root)
        }
        
        return output, reasoning_trace
    
    def _expand_node(self, node):
        """Recursively expand a node in the tree"""
        # Stop if max depth reached
        if node.depth >= self.config.max_tree_depth:
            return
        
        # Generate candidate thoughts
        thoughts = self.thought_generator(node.hidden_states)
        
        # Evaluate thoughts if value function is enabled
        if self.config.use_value_function:
            for thought in thoughts:
                value, uncertainty = self.value_function(thought)
                
                # Create child node
                child = TreeNode(thought, parent=node, depth=node.depth + 1)
                child.value = value.mean()  # Average over batch
                child.uncertainty = uncertainty.mean() if uncertainty is not None else None
                
                # Add child to parent
                node.add_child(child)
                
                # Recursively expand child if value is high enough
                if child.value > self.config.pruning_threshold:
                    self._expand_node(child)
        else:
            # Without value function, expand all thoughts
            for thought in thoughts:
                child = TreeNode(thought, parent=node, depth=node.depth + 1)
                node.add_child(child)
                self._expand_node(child)
    
    def _beam_search(self, root):
        """Perform beam search through the tree"""
        beam = [root]
        
        for depth in range(self.config.max_tree_depth):
            # Stop if beam is empty
            if not beam:
                break
                
            # Generate all children for nodes in beam
            all_children = []
            
            for node in beam:
                # Generate candidate thoughts
                thoughts = self.thought_generator(node.hidden_states)
                
                # Evaluate thoughts
                for thought in thoughts:
                    value, uncertainty = self.value_function(thought)
                    
                    # Create child node
                    child = TreeNode(thought, parent=node, depth=node.depth + 1)
                    child.value = value.mean()  # Average over batch
                    child.uncertainty = uncertainty.mean() if uncertainty is not None else None
                    
                    # Add child to parent and collection
                    node.add_child(child)
                    all_children.append(child)
            
            # Stop if no children were generated
            if not all_children:
                break
                
            # Select top-k children for next beam
            all_children.sort(key=lambda x: x.value, reverse=True)
            beam = all_children[:self.config.beam_size]
        
        return beam
    
    def _collect_leaves(self, node):
        """Collect all leaf nodes in the tree"""
        if node.is_leaf():
            return [node]
            
        leaves = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
            
        return leaves
    
    def _count_nodes(self, node):
        """Count total nodes in the tree"""
        count = 1  # Count this node
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _serialize_tree(self, node, node_id=0):
        """Serialize tree structure for visualization"""
        result = {
            'id': node_id,
            'depth': node.depth,
            'value': node.value.item() if node.value is not None else None,
            'uncertainty': node.uncertainty.item() if node.uncertainty is not None else None,
            'children': []
        }
        
        next_id = node_id + 1
        for child in node.children:
            child_result, next_id = self._serialize_tree(child, next_id)
            result['children'].append(child_result)
            
        return result, next_id

class AdaptiveTreeReasoner(nn.Module):
    """Tree-of-Thought reasoner with adaptive computation time"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tot_reasoner = TreeOfThoughtReasoner(config)
        
        # Halting mechanism
        self.halting_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states):
        """
        Perform adaptive tree-of-thought reasoning
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            reasoning_trace: Dict containing reasoning information
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Initialize
        current_state = hidden_states
        halting_probability = 0
        remainders = torch.ones(batch_size, 1, device=hidden_states.device)
        n_updates = torch.zeros(batch_size, 1, device=hidden_states.device)
        
        # Initialize outputs
        accumulated_output = torch.zeros_like(hidden_states)
        
        # Track reasoning steps
        reasoning_steps = []
        
        # Adaptive computation loop
        for step in range(self.config.max_computation_steps):
            # Compute halting probability
            halt = self.halting_network(current_state.mean(dim=1, keepdim=True))
            
            # Update halting probability
            halting_probability = halting_probability + remainders * halt
            
            # Compute update weights
            update_weights = remainders * halt
            
            # Update counters
            n_updates = n_updates + remainders
            
            # Perform tree reasoning step
            step_output, step_trace = self.tot_reasoner(current_state)
            reasoning_steps.append(step_trace)
            
            # Accumulate weighted output
            accumulated_output = accumulated_output + update_weights.view(batch_size, 1, 1) * step_output
            
            # Update remainders
            remainders = remainders * (1 - halt)
            
            # Update current state
            current_state = step_output
            
            # Check if all examples have halted
            if remainders.max() < self.config.early_stopping_threshold:
                break
        
        # Add remainders to accumulated output
        accumulated_output = accumulated_output + remainders.view(batch_size, 1, 1) * current_state
        
        # Prepare reasoning trace
        reasoning_trace = {
            'num_steps': step + 1,
            'halting_probabilities': halting_probability.squeeze().tolist(),
            'n_updates': n_updates.squeeze().tolist(),
            'step_traces': reasoning_steps
        }
        
        return accumulated_output, reasoning_trace

class TreeReasoningModule(nn.Module):
    """Main tree reasoning module that integrates all components"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create tree reasoner
        if config.use_adaptive_computation:
            self.reasoner = AdaptiveTreeReasoner(config)
        else:
            self.reasoner = TreeOfThoughtReasoner(config)
        
        # Integration with main model
        self.input_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, hidden_states):
        """
        Apply tree reasoning to input
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            reasoning_info: Dict containing reasoning information
        """
        # Project input
        projected_input = self.input_projection(hidden_states)
        
        # Apply tree reasoning
        reasoned_output, reasoning_trace = self.reasoner(projected_input)
        
        # Project output and add residual connection
        output = self.output_projection(reasoned_output)
        output = self.layer_norm(output + hidden_states)
        
        return output, reasoning_trace

class TreeReasoning(nn.Module):
    """
    Tree-based reasoning module for structured reasoning tasks.
    Implements tree exploration and reasoning over hierarchical structures.
    """
    
    def __init__(
        self,
        config,
        hidden_size: int = 768,
        num_heads: int = 8,
        max_tree_depth: int = 5,
        max_branches: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size if hasattr(config, 'hidden_size') else hidden_size
        self.max_tree_depth = max_tree_depth
        self.max_branches = max_branches
        
        # Tree node representation
        self.node_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Branch prediction
        self.branch_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, max_branches)
        )
        
        # Node expansion
        self.node_expander = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size)
            ) for _ in range(max_branches)
        ])
        
        # Tree traversal attention
        self.tree_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Path evaluation
        self.path_evaluator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Tree reasoning module
        self.tree_reasoning_module = TreeReasoningModule(config)
        
        # Initialize state
        self.is_initialized = False
        
    def initialize(self):
        """Initialize tree reasoning components"""
        if not self.is_initialized:
            # Initialize weights
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                        
            self.is_initialized = True
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Apply tree-based reasoning to input hidden states
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            reasoned_states: Hidden states after tree reasoning
            reasoning_info: Dictionary with reasoning information
        """
        # Ensure initialized
        if not self.is_initialized:
            self.initialize()
            
        batch_size, seq_len, _ = hidden_states.shape
        
        # Encode root node (use CLS token or mean pooling)
        if seq_len > 1:
            root_node = hidden_states[:, 0].clone()  # Use first token as root
        else:
            root_node = hidden_states.squeeze(1)
            
        # Encode node
        root_encoding = self.node_encoder(root_node)
        
        # Initialize tree exploration
        current_nodes = [root_encoding]
        all_nodes = [root_encoding]
        node_depths = [0]  # Track depth of each node
        node_parents = [-1]  # Track parent of each node (-1 for root)
        leaf_nodes = []
        leaf_paths = []
        
        # Explore tree up to max depth
        for depth in range(self.max_tree_depth):
            next_nodes = []
            next_parents = []
            
            # Process each node at current depth
            for node_idx, node in enumerate(current_nodes):
                # Predict branches
                branch_logits = self.branch_predictor(node)
                branch_probs = F.softmax(branch_logits, dim=-1)
                
                # Expand node for each branch
                for branch in range(self.max_branches):
                    # Skip unlikely branches
                    if branch_probs[0, branch].item() < 0.1 and len(next_nodes) > 0:
                        continue
                        
                    # Create child node
                    child_node = self.node_expander[branch](node)
                    
                    # Add to nodes
                    all_nodes.append(child_node)
                    next_nodes.append(child_node)
                    
                    # Track parent
                    parent_idx = len(all_nodes) - len(current_nodes) + node_idx
                    next_parents.append(parent_idx)
                    node_parents.append(parent_idx)
                    
                    # Track depth
                    node_depths.append(depth + 1)
                    
                    # If at max depth, mark as leaf
                    if depth == self.max_tree_depth - 1:
                        leaf_nodes.append(child_node)
                        
                        # Construct path to this leaf
                        path = [child_node]
                        current_parent = parent_idx
                        while current_parent >= 0:
                            path.append(all_nodes[current_parent])
                            current_parent = node_parents[current_parent]
                        leaf_paths.append(torch.stack(path[::-1], dim=1))  # Reverse to get root->leaf
            
            # Update current nodes for next depth
            current_nodes = next_nodes
            
            # Stop if no more nodes to expand
            if not current_nodes:
                break
                
        # Stack all nodes and paths
        all_nodes_tensor = torch.stack(all_nodes, dim=1)
        
        # Evaluate paths if leaves exist
        path_scores = None
        best_path = None
        if leaf_paths:
            # Stack leaf paths
            leaf_paths_tensor = torch.cat(leaf_paths, dim=0)
            
            # Evaluate each path
            path_scores = []
            for path in leaf_paths:
                # Apply attention along path
                path_attn_output, _ = self.tree_attention(
                    path, path, path
                )
                
                # Evaluate path
                path_score = self.path_evaluator(path_attn_output[:, -1])  # Use last node
                path_scores.append(path_score)
                
            # Stack scores
            path_scores = torch.cat(path_scores, dim=0)
            
            # Get best path
            best_path_idx = path_scores.argmax().item()
            best_path = leaf_paths[best_path_idx]
        
        # Apply tree reasoning module
        reasoned_states = self.tree_reasoning_module(hidden_states, attention_mask)
        
        # Prepare reasoning info
        reasoning_info = {
            'all_nodes': all_nodes_tensor,
            'node_depths': node_depths,
            'node_parents': node_parents,
            'path_scores': path_scores,
            'best_path': best_path
        }
        
        return reasoned_states, reasoning_info
    
    def reason(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Apply tree reasoning and return enhanced representations
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask
            
        Returns:
            enhanced_states: Enhanced hidden states after reasoning
        """
        reasoned_states, _ = self.forward(hidden_states, attention_mask)
        return reasoned_states
