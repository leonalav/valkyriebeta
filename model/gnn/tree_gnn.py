"""
Tree-Structured GNN for ValkyrieLLM.

This module implements Tree-Structured Graph Neural Networks that are specialized
for processing hierarchical tree-structured data, which is common in code, natural language
parsing trees, and other hierarchical structures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable


class TreeLSTMCell(nn.Module):
    """
    Tree-LSTM Cell as described in 'Improved Semantic Representations From Tree-Structured
    Long Short-Term Memory Networks' (Tai et al., 2015).
    
    This implements the Child-Sum Tree-LSTM variant which can handle trees with varying
    numbers of children per node.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a Tree-LSTM cell.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input (x) transformations
        self.ioux = nn.Linear(input_dim, 3 * hidden_dim)
        self.fx = nn.Linear(input_dim, hidden_dim)
        
        # Hidden state (h) transformations
        self.iouh = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.fh = nn.Linear(hidden_dim, hidden_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.ioux.weight)
        nn.init.xavier_uniform_(self.fx.weight)
        nn.init.xavier_uniform_(self.iouh.weight)
        nn.init.xavier_uniform_(self.fh.weight)
        nn.init.zeros_(self.ioux.bias)
        nn.init.zeros_(self.fx.bias)
        nn.init.zeros_(self.iouh.bias)
        nn.init.zeros_(self.fh.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        child_h: torch.Tensor,
        child_c: torch.Tensor,
        child_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Tree-LSTM cell.
        
        Args:
            x: Input features for current node [batch_size, input_dim]
            child_h: Hidden states of child nodes [batch_size, num_children, hidden_dim]
            child_c: Cell states of child nodes [batch_size, num_children, hidden_dim]
            child_mask: Mask for valid children [batch_size, num_children], 
                        1 for valid child, 0 for padding
            
        Returns:
            h, c: New hidden and cell states for current node
        """
        batch_size = x.size(0)
        
        if child_mask is not None:
            # Apply mask to child states
            child_h = child_h * child_mask.unsqueeze(-1)
            child_c = child_c * child_mask.unsqueeze(-1)
        
        # Calculate sum of child hidden states
        h_sum = torch.sum(child_h, dim=1)  # [batch_size, hidden_dim]
        
        # Calculate i, o, u gates from input x
        iou_x = self.ioux(x)  # [batch_size, 3 * hidden_dim]
        i_x, o_x, u_x = torch.split(iou_x, self.hidden_dim, dim=1)
        
        # Calculate i, o, u gates from child hidden states sum
        iou_h = self.iouh(h_sum)  # [batch_size, 3 * hidden_dim]
        i_h, o_h, u_h = torch.split(iou_h, self.hidden_dim, dim=1)
        
        # Combine input and hidden for i, o, u
        i = torch.sigmoid(i_x + i_h)  # [batch_size, hidden_dim]
        o = torch.sigmoid(o_x + o_h)  # [batch_size, hidden_dim]
        u = torch.tanh(u_x + u_h)  # [batch_size, hidden_dim]
        
        # Calculate forget gates for each child
        f_x = self.fx(x).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        f_h = self.fh(child_h)  # [batch_size, num_children, hidden_dim]
        f = torch.sigmoid(f_x + f_h)  # [batch_size, num_children, hidden_dim]
        
        # Apply forget gates to each child's cell state
        fc = f * child_c  # [batch_size, num_children, hidden_dim]
        
        # Sum up the forget-gated child cell states
        c = i * u + torch.sum(fc, dim=1)  # [batch_size, hidden_dim]
        
        # Calculate hidden state
        h = o * torch.tanh(c)  # [batch_size, hidden_dim]
        
        return h, c


class RecursiveTreeGNN(nn.Module):
    """
    Recursive Tree-Structured GNN that processes trees in a bottom-up fashion.
    
    This model recursively applies a tree cell (like Tree-LSTM) to process
    hierarchical structures from leaves to root.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        tree_cell: str = "tree_lstm",
        aggregation: str = "sum",
    ):
        """
        Initialize a recursive tree GNN.
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            num_layers: Number of tree layers to stack
            dropout: Dropout probability
            tree_cell: Type of tree cell to use ("tree_lstm")
            aggregation: How to aggregate node embeddings for graph-level representation
                        ("sum", "max", "mean")
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.tree_cell = tree_cell
        self.aggregation = aggregation
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Tree cells for each layer
        self.tree_cells = nn.ModuleList()
        for _ in range(num_layers):
            if tree_cell == "tree_lstm":
                cell = TreeLSTMCell(
                    input_dim=hidden_channels,
                    hidden_dim=hidden_channels
                )
            else:
                raise ValueError(f"Unknown tree cell type: {tree_cell}")
            self.tree_cells.append(cell)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        node_depth: Optional[torch.Tensor] = None,
        node_parent: Optional[torch.Tensor] = None,
        is_leaf: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for recursive tree GNN.
        
        Note: This implementation assumes the nodes are already topologically sorted
        from leaves to root, which is often the case for tree-structured data.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges] (child -> parent)
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            batch: Batch indices [num_nodes] for multiple trees
            node_depth: Depth of each node in the tree [num_nodes]
            node_parent: Parent index for each node [num_nodes]
            is_leaf: Boolean mask indicating leaf nodes [num_nodes]
            
        Returns:
            Dictionary containing node embeddings and tree embeddings
        """
        # Number of nodes
        num_nodes = x.size(0)
        
        # Project input features
        h = self.input_proj(x)
        h = F.relu(h)
        h = self.dropout_layer(h)
        
        # If node_depth, node_parent, or is_leaf are not provided, infer them from edge_index
        if node_depth is None or node_parent is None or is_leaf is None:
            node_depth, node_parent, is_leaf = self._infer_tree_structure(edge_index, num_nodes)
        
        # Process trees layer by layer
        for layer_idx, tree_cell in enumerate(self.tree_cells):
            # Initialize hidden and cell states
            hidden_states = torch.zeros(num_nodes, self.hidden_channels, device=x.device)
            cell_states = torch.zeros(num_nodes, self.hidden_channels, device=x.device)
            
            # Process nodes in topological order (from leaves to root)
            max_depth = node_depth.max().item()
            for depth in range(max_depth, -1, -1):
                # Get nodes at current depth
                depth_mask = (node_depth == depth)
                depth_indices = torch.where(depth_mask)[0]
                
                if depth_indices.size(0) == 0:
                    continue
                
                # For leaf nodes, initialize hidden and cell states
                if depth == max_depth:
                    for idx in depth_indices:
                        # Leaf nodes have no children
                        h_i, c_i = self._process_node(
                            tree_cell, 
                            h[idx].unsqueeze(0), 
                            torch.zeros(1, 0, self.hidden_channels, device=x.device), 
                            torch.zeros(1, 0, self.hidden_channels, device=x.device), 
                            None
                        )
                        hidden_states[idx] = h_i.squeeze(0)
                        cell_states[idx] = c_i.squeeze(0)
                else:
                    # For non-leaf nodes, gather child states
                    for idx in depth_indices:
                        # Find children of this node
                        children = torch.where(node_parent == idx)[0]
                        num_children = children.size(0)
                        
                        if num_children > 0:
                            # Gather child hidden and cell states
                            child_h = hidden_states[children].unsqueeze(0)  # [1, num_children, hidden_dim]
                            child_c = cell_states[children].unsqueeze(0)    # [1, num_children, hidden_dim]
                            
                            # Process node with its children
                            h_i, c_i = self._process_node(
                                tree_cell, 
                                h[idx].unsqueeze(0), 
                                child_h, 
                                child_c, 
                                None
                            )
                            hidden_states[idx] = h_i.squeeze(0)
                            cell_states[idx] = c_i.squeeze(0)
                        else:
                            # Handle nodes with no children (shouldn't happen in a tree but for robustness)
                            h_i, c_i = self._process_node(
                                tree_cell, 
                                h[idx].unsqueeze(0), 
                                torch.zeros(1, 0, self.hidden_channels, device=x.device), 
                                torch.zeros(1, 0, self.hidden_channels, device=x.device), 
                                None
                            )
                            hidden_states[idx] = h_i.squeeze(0)
                            cell_states[idx] = c_i.squeeze(0)
            
            # Update h for next layer
            h = hidden_states
            h = self.dropout_layer(h)
        
        # Apply output projection
        node_embeddings = self.output_proj(h)
        
        # Prepare tree embeddings if batch is provided
        tree_embedding = None
        if batch is not None:
            # Find root nodes for each tree in the batch
            if node_depth is not None:
                root_mask = (node_depth == 0)
                root_indices = torch.where(root_mask)[0]
            else:
                # Assume the node with no parent is the root
                root_indices = torch.where(node_parent == -1)[0]
            
            # Use root node embeddings as tree embeddings
            if self.aggregation == "root":
                tree_embedding = node_embeddings[root_indices]
            # Use aggregation function if specified
            elif self.aggregation in ["sum", "mean", "max"]:
                num_trees = batch.max().item() + 1 if batch.numel() > 0 else 1
                if self.aggregation == "sum":
                    tree_embedding = torch.zeros(num_trees, self.out_channels, device=x.device)
                    tree_embedding.index_add_(0, batch, node_embeddings)
                elif self.aggregation == "mean":
                    tree_embedding = torch.zeros(num_trees, self.out_channels, device=x.device)
                    tree_embedding.index_add_(0, batch, node_embeddings)
                    tree_counts = torch.zeros(num_trees, device=x.device)
                    tree_counts.index_add_(0, batch, torch.ones_like(batch, dtype=torch.float))
                    tree_embedding = tree_embedding / tree_counts.unsqueeze(1).clamp(min=1)
                elif self.aggregation == "max":
                    tree_embedding = torch.full((num_trees, self.out_channels), float('-inf'), device=x.device)
                    for i in range(node_embeddings.size(0)):
                        tree_idx = batch[i]
                        tree_embedding[tree_idx] = torch.max(tree_embedding[tree_idx], node_embeddings[i])
                    # Replace -inf with 0
                    tree_embedding = tree_embedding.masked_fill(tree_embedding == float('-inf'), 0)
        
        # Prepare output dictionary
        output = {
            "node_embeddings": node_embeddings,
        }
        
        if tree_embedding is not None:
            output["tree_embedding"] = tree_embedding
        
        return output
    
    def _process_node(
        self,
        tree_cell: nn.Module,
        x: torch.Tensor,
        child_h: torch.Tensor,
        child_c: torch.Tensor,
        child_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a node with its children using the specified tree cell."""
        return tree_cell(x, child_h, child_c, child_mask)
    
    def _infer_tree_structure(
        self, 
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Infer tree structure from edge_index.
        
        Args:
            edge_index: Edge indices [2, num_edges] (assumed to be child -> parent)
            num_nodes: Number of nodes in the graph
            
        Returns:
            node_depth: Depth of each node [num_nodes]
            node_parent: Parent index for each node [num_nodes]
            is_leaf: Boolean mask indicating leaf nodes [num_nodes]
        """
        device = edge_index.device
        
        # Initialize
        node_parent = torch.full((num_nodes,), -1, device=device)
        is_leaf = torch.ones(num_nodes, dtype=torch.bool, device=device)
        
        # Find parent for each node
        for i in range(edge_index.size(1)):
            child, parent = edge_index[:, i]
            node_parent[child] = parent
            is_leaf[parent] = False  # Parent nodes are not leaves
        
        # Calculate depth for each node, starting from root
        node_depth = torch.full((num_nodes,), -1, device=device)
        
        # Find roots (nodes without parents)
        roots = torch.where(node_parent == -1)[0]
        
        # Set depth 0 for roots
        node_depth[roots] = 0
        
        # BFS to calculate depths
        current_depth = 0
        current_nodes = roots
        
        while current_nodes.size(0) > 0:
            # Find children of current nodes
            next_nodes = []
            for node in current_nodes:
                children = torch.where(node_parent == node)[0]
                next_nodes.append(children)
                node_depth[children] = current_depth + 1
            
            if next_nodes:
                current_nodes = torch.cat(next_nodes)
                current_depth += 1
            else:
                break
        
        return node_depth, node_parent, is_leaf


class TreeGNN(nn.Module):
    """
    Tree-Structured GNN that can process hierarchical data like parse trees,
    abstract syntax trees, or XML/HTML documents represented as graphs.
    
    This model combines top-down and bottom-up passes to effectively process
    tree-structured data for tasks like code understanding or parsing.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        message_passing: str = "tree_lstm",
        bidirectional: bool = True,
        aggregation: str = "root",
    ):
        """
        Initialize a Tree-Structured GNN.
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            num_layers: Number of tree layers to stack
            dropout: Dropout probability
            message_passing: Type of message passing to use ("tree_lstm")
            bidirectional: Whether to use bidirectional (top-down + bottom-up) processing
            aggregation: How to aggregate node embeddings for graph-level representation
                        ("root", "sum", "max", "mean")
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.message_passing = message_passing
        self.bidirectional = bidirectional
        self.aggregation = aggregation
        
        # Bottom-up (leaves to root) processor
        self.bottom_up = RecursiveTreeGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels if bidirectional else out_channels,
            num_layers=num_layers,
            dropout=dropout,
            tree_cell=message_passing,
            aggregation=aggregation
        )
        
        # Top-down (root to leaves) processor (optional)
        if bidirectional:
            self.top_down = RecursiveTreeGNN(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                dropout=dropout,
                tree_cell=message_passing,
                aggregation=aggregation
            )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        self.bottom_up.reset_parameters()
        if self.bidirectional:
            self.top_down.reset_parameters()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        node_depth: Optional[torch.Tensor] = None,
        node_parent: Optional[torch.Tensor] = None,
        is_leaf: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for tree GNN.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges] (child -> parent for bottom-up)
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            batch: Batch indices [num_nodes] for multiple trees
            node_depth: Depth of each node in the tree [num_nodes]
            node_parent: Parent index for each node [num_nodes]
            is_leaf: Boolean mask indicating leaf nodes [num_nodes]
            
        Returns:
            Dictionary containing node embeddings and tree embeddings
        """
        # Bottom-up pass
        bottom_up_output = self.bottom_up(
            x, edge_index, edge_attr, batch, node_depth, node_parent, is_leaf
        )
        
        # Extract node embeddings
        node_embeddings = bottom_up_output["node_embeddings"]
        
        if self.bidirectional:
            # For top-down pass, reverse the edge directions
            top_down_edge_index = torch.flip(edge_index, dims=[0])  # parent -> child
            
            # Top-down pass using embeddings from bottom-up
            top_down_output = self.top_down(
                node_embeddings, top_down_edge_index, edge_attr, batch, node_depth, node_parent, is_leaf
            )
            
            # Final node embeddings
            node_embeddings = top_down_output["node_embeddings"]
            
            # Use tree embedding from top-down pass if available
            if "tree_embedding" in top_down_output:
                tree_embedding = top_down_output["tree_embedding"]
            elif "tree_embedding" in bottom_up_output:
                tree_embedding = bottom_up_output["tree_embedding"]
            else:
                tree_embedding = None
        else:
            # Use tree embedding from bottom-up pass
            tree_embedding = bottom_up_output.get("tree_embedding", None)
        
        # Prepare output dictionary
        output = {
            "node_embeddings": node_embeddings,
        }
        
        if tree_embedding is not None:
            output["tree_embedding"] = tree_embedding
        
        return output 