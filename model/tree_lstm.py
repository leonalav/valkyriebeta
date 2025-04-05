import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union, Any
import math

class TreeLSTMConfig:
    """Configuration for TreeLSTM model"""
    def __init__(
        self,
        input_size: int = 768,
        hidden_size: int = 768,
        cell_type: str = "child_sum",  # "child_sum" or "n_ary"
        n_ary_factor: int = 2,         # For N-ary TreeLSTM
        max_children: int = 10,
        dropout: float = 0.1,
        layer_norm: bool = True,
        residual_connections: bool = True,
        max_depth: int = 8,
        recursive_mode: bool = False,
        share_weights: bool = False
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.n_ary_factor = n_ary_factor
        self.max_children = max_children
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.residual_connections = residual_connections
        self.max_depth = max_depth
        self.recursive_mode = recursive_mode
        self.share_weights = share_weights


class ChildSumTreeLSTMCell(nn.Module):
    """
    Child-Sum TreeLSTM Cell as described in the paper:
    "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
    by Kai Sheng Tai, Richard Socher, and Christopher D. Manning.
    
    This variant can handle an arbitrary number of children for each node.
    """
    
    def __init__(self, config: TreeLSTMConfig):
        super().__init__()
        self.config = config
        
        # Input transformations for input, output, and update gates
        self.ioux = nn.Linear(config.input_size, 3 * config.hidden_size)
        
        # Hidden state transformation for input, output, and update gates
        self.iouh = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        
        # Forget gate transformations (applied to each child separately)
        self.fx = nn.Linear(config.input_size, config.hidden_size)
        self.fh = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Layer normalization if enabled
        if config.layer_norm:
            self.layer_norm_c = nn.LayerNorm(config.hidden_size)
            self.layer_norm_h = nn.LayerNorm(config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Glorot uniform initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                if 'fx.bias' in name:
                    nn.init.ones_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        child_h: Optional[List[torch.Tensor]] = None,
        child_c: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Child-Sum TreeLSTM Cell
        
        Args:
            x: Input tensor [batch_size, input_size]
            child_h: List of hidden states from children [num_children, batch_size, hidden_size]
            child_c: List of cell states from children [num_children, batch_size, hidden_size]
            
        Returns:
            h: Hidden state [batch_size, hidden_size]
            c: Cell state [batch_size, hidden_size]
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize with zeros if no children
        if not child_h or len(child_h) == 0:
            child_h_sum = torch.zeros(batch_size, self.config.hidden_size, device=device)
            child_c = []
        else:
            # Sum the hidden states of all children
            child_h_sum = torch.sum(torch.stack(child_h), dim=0)
        
        # Input, output, and update gates
        iou = self.ioux(x) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, self.config.hidden_size, dim=-1)
        i, o = torch.sigmoid(i), torch.sigmoid(o)
        u = torch.tanh(u)
        
        # Calculate forget gates for each child
        if not child_h or len(child_h) == 0:
            f = torch.zeros(batch_size, 0, self.config.hidden_size, device=device)
            c = i * u
        else:
            f_sum = 0
            for child in child_h:
                f_gate = torch.sigmoid(self.fx(x) + self.fh(child))
                f_sum += f_gate * child
            
            # Update cell state
            c = i * u + f_sum
        
        # Apply layer normalization if enabled
        if self.config.layer_norm:
            c = self.layer_norm_c(c)
        
        # Output gating
        h = o * torch.tanh(c)
        
        # Apply layer normalization if enabled
        if self.config.layer_norm:
            h = self.layer_norm_h(h)
        
        # Apply dropout
        h = self.dropout(h)
        
        return h, c


class NAryTreeLSTMCell(nn.Module):
    """
    N-ary TreeLSTM Cell as described in the paper:
    "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
    
    This variant assumes a fixed number of children for each node (N),
    which allows for more direct modelling of child relationships.
    """
    
    def __init__(self, config: TreeLSTMConfig):
        super().__init__()
        self.config = config
        self.n_ary = config.n_ary_factor
        
        # Input transformations
        self.ioux = nn.Linear(config.input_size, 3 * config.hidden_size)
        
        # Hidden state transformation for each child
        self.iouh_list = nn.ModuleList([
            nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
            for _ in range(self.n_ary)
        ])
        
        # Forget gate transformations for each child
        self.fx = nn.Linear(config.input_size, self.n_ary * config.hidden_size)
        self.fh_list = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(self.n_ary)
        ])
        
        # Layer normalization if enabled
        if config.layer_norm:
            self.layer_norm_c = nn.LayerNorm(config.hidden_size)
            self.layer_norm_h = nn.LayerNorm(config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Glorot uniform initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                if 'fx.bias' in name:
                    param.data.fill_(1.0 / self.n_ary)
    
    def forward(
        self,
        x: torch.Tensor,
        child_h: Optional[List[torch.Tensor]] = None,
        child_c: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for N-ary TreeLSTM Cell
        
        Args:
            x: Input tensor [batch_size, input_size]
            child_h: List of hidden states from children [n_ary, batch_size, hidden_size]
            child_c: List of cell states from children [n_ary, batch_size, hidden_size]
            
        Returns:
            h: Hidden state [batch_size, hidden_size]
            c: Cell state [batch_size, hidden_size]
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize with zeros if no children or fewer than n_ary children
        if not child_h or len(child_h) == 0:
            child_h = [torch.zeros(batch_size, self.config.hidden_size, device=device)] * self.n_ary
            child_c = [torch.zeros(batch_size, self.config.hidden_size, device=device)] * self.n_ary
        elif len(child_h) < self.n_ary:
            # Pad with zeros if fewer than n_ary children
            padding = [torch.zeros(batch_size, self.config.hidden_size, device=device)] * (self.n_ary - len(child_h))
            child_h.extend(padding)
            
            if child_c:
                child_c.extend(padding)
            else:
                child_c = [torch.zeros(batch_size, self.config.hidden_size, device=device)] * self.n_ary
        
        # Calculate input, output, and update gates
        iou = self.ioux(x)
        for idx, (h, iouh) in enumerate(zip(child_h[:self.n_ary], self.iouh_list)):
            if h is not None:
                iou = iou + iouh(h)
        
        i, o, u = torch.split(iou, self.config.hidden_size, dim=-1)
        i, o = torch.sigmoid(i), torch.sigmoid(o)
        u = torch.tanh(u)
        
        # Calculate forget gates for each child
        f_gates = torch.split(self.fx(x), self.config.hidden_size, dim=-1)
        
        # Initialize cell state
        c = i * u
        
        # Update cell state with each child's contribution
        for idx, (h, c_child, f_gate, fh) in enumerate(zip(
            child_h[:self.n_ary], 
            child_c[:self.n_ary], 
            f_gates,
            self.fh_list
        )):
            if h is not None and c_child is not None:
                f = torch.sigmoid(f_gate + fh(h))
                c = c + f * c_child
        
        # Apply layer normalization if enabled
        if self.config.layer_norm:
            c = self.layer_norm_c(c)
        
        # Output gating
        h = o * torch.tanh(c)
        
        # Apply layer normalization if enabled
        if self.config.layer_norm:
            h = self.layer_norm_h(h)
        
        # Apply dropout
        h = self.dropout(h)
        
        return h, c


class TreeLSTM(nn.Module):
    """
    Tree-Structured Long Short-Term Memory Network
    
    This module implements Tree-LSTMs that can process tree-structured data.
    It supports both Child-Sum and N-ary TreeLSTM variants.
    """
    
    def __init__(self, config: Optional[TreeLSTMConfig] = None, **kwargs):
        super().__init__()
        
        # Create config if not provided
        if config is None:
            config = TreeLSTMConfig(**kwargs)
        self.config = config
        
        # Create the appropriate cell type
        if config.cell_type == "child_sum":
            self.cell = ChildSumTreeLSTMCell(config)
        elif config.cell_type == "n_ary":
            self.cell = NAryTreeLSTMCell(config)
        else:
            raise ValueError(f"Unknown cell type: {config.cell_type}")
        
        # Create additional layers for multi-layer TreeLSTM if max_depth > 1
        if config.max_depth > 1 and not config.share_weights:
            self.cells = nn.ModuleList([self.cell] + [
                ChildSumTreeLSTMCell(config) if config.cell_type == "child_sum" else NAryTreeLSTMCell(config)
                for _ in range(config.max_depth - 1)
            ])
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Residual adaptor if input and hidden sizes differ
        if config.input_size != config.hidden_size and config.residual_connections:
            self.residual_adaptor = nn.Linear(config.input_size, config.hidden_size)
        else:
            self.residual_adaptor = None
        
        # Apply initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for non-cell parameters"""
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
        if self.residual_adaptor is not None:
            nn.init.xavier_uniform_(self.residual_adaptor.weight)
            nn.init.zeros_(self.residual_adaptor.bias)
    
    def _get_cell(self, depth):
        """Get the appropriate cell for the given depth"""
        if self.config.share_weights or self.config.max_depth <= 1:
            return self.cell
        return self.cells[min(depth, len(self.cells) - 1)]
    
    def _process_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        tree: Dict[int, List[int]],
        states: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        depth: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single node in the tree recursively
        
        Args:
            node_idx: Index of the current node
            x: Input tensor [batch_size, seq_len, input_size]
            tree: Dictionary mapping node indices to lists of child indices
            states: Dictionary mapping node indices to (h, c) states
            depth: Current depth in the tree
            
        Returns:
            h: Hidden state for this node
            c: Cell state for this node
        """
        # Check if we've already processed this node
        if node_idx in states:
            return states[node_idx]
        
        # Check if we have children
        if node_idx in tree and tree[node_idx]:
            # Process all children first
            child_states = []
            for child_idx in tree[node_idx]:
                child_h, child_c = self._process_node(child_idx, x, tree, states, depth + 1)
                child_states.append((child_h, child_c))
            
            # Extract child h and c states
            child_h_list = [state[0] for state in child_states]
            child_c_list = [state[1] for state in child_states]
            
            # Get the appropriate cell for this depth
            cell = self._get_cell(depth)
            
            # Process the current node
            node_input = x[:, node_idx]
            h, c = cell(node_input, child_h_list, child_c_list)
        else:
            # Leaf node - no children
            cell = self._get_cell(depth)
            node_input = x[:, node_idx]
            h, c = cell(node_input)
        
        # Add residual connection if enabled
        if self.config.residual_connections:
            node_input = x[:, node_idx]
            if self.residual_adaptor is not None:
                h = h + self.residual_adaptor(node_input)
            else:
                h = h + node_input
        
        # Store the state for this node
        states[node_idx] = (h, c)
        
        return h, c
    
    def _process_flat_sequence(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Process a flat sequence (left-to-right) without tree structure
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            outputs: Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # Initialize states
        h = torch.zeros(batch_size, self.config.hidden_size, device=device)
        c = torch.zeros(batch_size, self.config.hidden_size, device=device)
        
        outputs = []
        
        # Process sequence left-to-right
        for t in range(seq_len):
            cell = self._get_cell(0)
            h, c = cell(x[:, t], [h], [c])
            outputs.append(h)
        
        return torch.stack(outputs, dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        tree_structure: Optional[Dict[int, List[int]]] = None,
        root_idx: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[int, torch.Tensor]]]:
        """
        Forward pass for TreeLSTM
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            tree_structure: Dictionary mapping node indices to lists of child indices
            root_idx: Index of the root node (if None, use max index in tree)
            
        Returns:
            If return_all_nodes is False:
                output: Final hidden state at the root [batch_size, hidden_size]
            If return_all_nodes is True:
                output: Final hidden state at the root [batch_size, hidden_size]
                all_states: Dictionary mapping node indices to hidden states
        """
        batch_size, seq_len, _ = x.size()
        
        # Handle the case where no tree structure is provided
        if tree_structure is None:
            # Process as a flat sequence
            outputs = self._process_flat_sequence(x)
            return outputs
        
        # Determine root node if not specified
        if root_idx is None:
            # Use the highest index that has no parents
            all_nodes = set(tree_structure.keys())
            all_children = set()
            for children in tree_structure.values():
                all_children.update(children)
            
            possible_roots = all_nodes - all_children
            if not possible_roots:
                # If no clear root, use the highest index
                root_idx = max(all_nodes) if all_nodes else seq_len - 1
            else:
                root_idx = max(possible_roots)
        
        # Process the tree
        states = {}
        root_h, root_c = self._process_node(root_idx, x, tree_structure, states)
        
        # Apply final output projection
        output = self.output_projection(root_h)
        
        # Convert states to just hidden states if returning all nodes
        all_hidden_states = {idx: state[0] for idx, state in states.items()}
        
        return output, all_hidden_states
    
    def dynamic_tree_process(
        self,
        x: torch.Tensor,
        adjacency_list: List[List[int]]
    ) -> torch.Tensor:
        """
        Process a dynamically constructed tree based on adjacency list
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            adjacency_list: List of lists where adjacency_list[i] contains indices of i's children
            
        Returns:
            outputs: Output hidden states for all nodes [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        # Convert adjacency list to tree structure
        tree_structure = {i: children for i, children in enumerate(adjacency_list) if children}
        
        # Process the tree
        states = {}
        for node_idx in range(seq_len):
            # Check if this node has already been processed
            if node_idx not in states:
                self._process_node(node_idx, x, tree_structure, states)
        
        # Collect hidden states for all nodes
        outputs = torch.zeros(batch_size, seq_len, self.config.hidden_size, device=device)
        for idx, (h, _) in states.items():
            if idx < seq_len:
                outputs[:, idx] = h
        
        return outputs
    
    def batched_tree_process(
        self,
        x: torch.Tensor,
        batch_trees: List[Dict[int, List[int]]],
        batch_roots: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Process multiple trees in a batch
        
        Args:
            x: Input tensor [batch_size, max_seq_len, input_size]
            batch_trees: List of tree structures, one per batch item
            batch_roots: List of root indices, one per batch item
            
        Returns:
            outputs: Output hidden states for each tree root [batch_size, hidden_size]
        """
        batch_size = len(batch_trees)
        device = x.device
        
        # Process each tree separately
        outputs = []
        for i in range(batch_size):
            tree = batch_trees[i]
            root = batch_roots[i] if batch_roots is not None else None
            
            # Process this tree
            output, _ = self.forward(x[i:i+1], tree, root)
            outputs.append(output)
        
        # Stack all outputs
        return torch.cat(outputs, dim=0)
    
    @staticmethod
    def build_tree_from_parse(
        parse_tree: List[Tuple[int, int, int]],
        seq_len: int
    ) -> Dict[int, List[int]]:
        """
        Build tree structure from parse tree
        
        Args:
            parse_tree: List of (parent, left_child, right_child) triples
            seq_len: Length of the sequence
            
        Returns:
            tree_structure: Dictionary mapping node indices to lists of child indices
        """
        tree_structure = {i: [] for i in range(seq_len)}
        
        for parent, left, right in parse_tree:
            if left >= 0:
                tree_structure[parent].append(left)
            if right >= 0:
                tree_structure[parent].append(right)
        
        return tree_structure
    
    @staticmethod
    def build_linear_tree(seq_len: int) -> Dict[int, List[int]]:
        """
        Build a linear (sequential) tree structure
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            tree_structure: Dictionary mapping node indices to lists of child indices
        """
        tree_structure = {i: [i-1] for i in range(1, seq_len)}
        return tree_structure
    
    @staticmethod
    def build_balanced_tree(seq_len: int) -> Dict[int, List[int]]:
        """
        Build a balanced binary tree structure
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            tree_structure: Dictionary mapping node indices to lists of child indices
        """
        tree_structure = {}
        
        def build_subtree(start, end):
            if start > end:
                return -1
            if start == end:
                return start
            
            mid = (start + end) // 2
            left = build_subtree(start, mid - 1)
            right = build_subtree(mid + 1, end)
            
            tree_structure[mid] = []
            if left >= 0:
                tree_structure[mid].append(left)
            if right >= 0:
                tree_structure[mid].append(right)
            
            return mid
        
        build_subtree(0, seq_len - 1)
        return tree_structure


# Helper functions for integration with the wider model

def create_tree_lstm(config, input_size=None, hidden_size=None, **kwargs):
    """Factory function to create a TreeLSTM module"""
    if isinstance(config, dict):
        config = TreeLSTMConfig(**config)
    elif not isinstance(config, TreeLSTMConfig):
        # Override specific parameters if provided
        params = {
            'input_size': input_size or getattr(config, 'hidden_size', 768),
            'hidden_size': hidden_size or getattr(config, 'hidden_size', 768),
        }
        params.update(kwargs)
        config = TreeLSTMConfig(**params)
    
    return TreeLSTM(config)


class TreeLSTMIntegration(nn.Module):
    """
    Integration module for using TreeLSTM in the broader model architecture.
    
    This module handles the integration between transformer outputs and TreeLSTM,
    providing a cleaner interface for the rest of the model.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        cell_type: str = "child_sum",
        max_depth: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        # Create TreeLSTM configuration
        self.config = TreeLSTMConfig(
            input_size=input_size,
            hidden_size=hidden_size,
            cell_type=cell_type,
            max_depth=max_depth,
            dropout=dropout,
            **kwargs
        )
        
        # Create TreeLSTM
        self.tree_lstm = TreeLSTM(self.config)
        
        # Tree structure prediction (optional)
        self.predict_tree_structure = kwargs.get("predict_tree_structure", False)
        if self.predict_tree_structure:
            self.structure_predictor = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.GELU(),
                nn.Linear(input_size, input_size),
                nn.GELU(),
                # Output logits for each node being a child of each other node
                nn.Linear(input_size, input_size)
            )
    
    def infer_tree_structure(self, hidden_states: torch.Tensor) -> List[Dict[int, List[int]]]:
        """
        Infer tree structure from hidden states
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            tree_structures: List of tree structures, one per batch item
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Compute pairwise affinities between positions
        affinities = torch.matmul(
            hidden_states, hidden_states.transpose(1, 2)
        )  # [batch_size, seq_len, seq_len]
        
        # Zero out invalid parent-child relationships (no self-loops, no future->past)
        mask = torch.triu(torch.ones_like(affinities), diagonal=1)
        affinities = affinities * (1 - mask) - 1e9 * mask
        
        # For each position, select the top-k most likely parents
        k = min(2, seq_len - 1)  # Allow up to 2 parents per node
        _, parent_indices = torch.topk(affinities, k, dim=2)
        
        # Convert to tree structures
        tree_structures = []
        for b in range(batch_size):
            tree = {i: [] for i in range(seq_len)}
            for i in range(seq_len):
                for j in range(k):
                    parent = parent_indices[b, i, j].item()
                    if parent < i:  # Only allow past->future edges
                        tree[parent].append(i)
            tree_structures.append(tree)
        
        return tree_structures
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        tree_structures: Optional[List[Dict[int, List[int]]]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for TreeLSTM integration
        
        Args:
            hidden_states: Hidden states from transformer [batch_size, seq_len, hidden_size]
            tree_structures: List of tree structures, one per batch item
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            outputs: Enhanced hidden states [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Infer tree structure if not provided and prediction is enabled
        if tree_structures is None:
            if self.predict_tree_structure:
                tree_structures = self.infer_tree_structure(hidden_states)
            else:
                # Default to linear tree (sequential processing)
                tree_structures = [TreeLSTM.build_linear_tree(seq_len)] * batch_size
        
        # For each batch item, process with TreeLSTM
        outputs = []
        for i in range(batch_size):
            # Get the tree for this batch item
            tree = tree_structures[i] if i < len(tree_structures) else TreeLSTM.build_linear_tree(seq_len)
            
            # Process with TreeLSTM (use the last node as root if not specified)
            root_idx = max(tree.keys()) if tree else seq_len - 1
            tree_output = self.tree_lstm.dynamic_tree_process(
                hidden_states[i:i+1], 
                [tree.get(j, []) for j in range(seq_len)]
            )
            
            outputs.append(tree_output)
        
        # Combine batch outputs
        return torch.cat(outputs, dim=0) 