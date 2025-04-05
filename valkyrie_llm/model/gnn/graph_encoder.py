"""
Graph Encoder for ValkyrieLLM.

This module implements a versatile Graph Neural Network encoder
that can be configured with different GNN layers and architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from .layers import (
    GCNLayer,
    GATLayer,
    GraphSAGELayer,
    GINLayer,
    MPNNLayer,
    EGNN,
    EGNNLayer,
    EdgeGAT,
    GraphTransformer,
    DiffPool,
    HGT,
    HGTLayer
)
from .adapters import HGTAdapter


class GraphEncoder(nn.Module):
    """
    Graph Neural Network encoder that supports multiple GNN architectures.
    
    This module can be configured to use different GNN layers (GCN, GAT, GraphSAGE, etc.)
    and combines them into a full encoder with the specified number of layers.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        gnn_type: str = "gcn",
        dropout: float = 0.1,
        activation: Callable = F.relu,
        residual: bool = True,
        layer_norm: bool = True,
        readout: str = "mean",
        **kwargs
    ):
        """
        Initialize the GraphEncoder.
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layer to use ("gcn", "gat", "sage", "gin", "mpnn")
            dropout: Dropout probability
            activation: Activation function
            residual: Whether to use residual connections
            layer_norm: Whether to use layer normalization
            readout: Readout method for graph-level representations ("mean", "sum", "max")
            **kwargs: Additional arguments for specific GNN layers
                - For GAT: heads, attention_dropout
                - For GIN: train_eps, mlp_hidden_channels
                - For MPNN: edge_dim, aggr
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.activation = activation
        self.residual = residual
        self.layer_norm = layer_norm
        self.readout = readout
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Create GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # Create the first layer (input -> hidden)
        self.gnn_layers.append(self._create_gnn_layer(hidden_channels, hidden_channels, **kwargs))
        
        # Create the intermediate layers (hidden -> hidden)
        for _ in range(num_layers - 2):
            self.gnn_layers.append(self._create_gnn_layer(hidden_channels, hidden_channels, **kwargs))
        
        # Create the final layer (hidden -> output)
        if num_layers > 1:
            self.gnn_layers.append(self._create_gnn_layer(hidden_channels, out_channels, **kwargs))
        
        # Layer normalization layers
        if layer_norm:
            self.layer_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.layer_norms.append(nn.LayerNorm(hidden_channels if _ < num_layers - 1 else out_channels))
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output projection (for graph-level outputs)
        self.output_proj = nn.Linear(out_channels, out_channels)
    
    def _create_gnn_layer(self, in_dim: int, out_dim: int, **kwargs) -> nn.Module:
        """
        Create a GNN layer based on the specified type.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            **kwargs: Additional arguments for specific GNN layers
            
        Returns:
            A GNN layer of the specified type
        """
        if self.gnn_type == "gcn":
            return GCNLayer(
                in_channels=in_dim,
                out_channels=out_dim,
                add_self_loops=kwargs.get("add_self_loops", True),
                normalize=kwargs.get("normalize", True),
                bias=kwargs.get("bias", True),
            )
        elif self.gnn_type == "gat":
            return GATLayer(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=kwargs.get("heads", 8),
                concat=kwargs.get("concat", True),
                negative_slope=kwargs.get("negative_slope", 0.2),
                dropout=kwargs.get("attention_dropout", 0.1),
                bias=kwargs.get("bias", True),
            )
        elif self.gnn_type == "sage":
            return GraphSAGELayer(
                in_features=in_dim,
                out_features=out_dim,
                aggregator_type=kwargs.get("aggregator_type", "mean"),
                use_bias=kwargs.get("bias", True),
                dropout=self.dropout,
                activation=getattr(F, kwargs.get("activation", "relu")),
                use_layer_norm=kwargs.get("layer_norm", False),
                residual=kwargs.get("residual", True),
                normalize=kwargs.get("normalize", True),
            )
        elif self.gnn_type == "gin":
            return GINLayer(
                in_channels=in_dim,
                out_channels=out_dim,
                eps=kwargs.get("eps", 0.0),
                train_eps=kwargs.get("train_eps", False),
                mlp_hidden_channels=kwargs.get("mlp_hidden_channels", None),
                activation=self.activation,
            )
        elif self.gnn_type == "mpnn":
            return MPNNLayer(
                in_channels=in_dim,
                out_channels=out_dim,
                edge_dim=kwargs.get("edge_dim", None),
                aggr=kwargs.get("aggr", "sum"),
                normalize=kwargs.get("normalize", False),
                bias=kwargs.get("bias", True),
            )
        elif self.gnn_type == "egnn":
            # Create EGNN layer (use the EGNNLayer component)
            return EGNNLayer(
                in_channels=in_dim,
                edge_channels=kwargs.get("edge_dim", 32),
                out_channels=out_dim,
                aggr=kwargs.get("aggr", "sum"),
                dropout=self.dropout,
                activation=nn.ReLU(),
                residual=self.residual,
                normalize=kwargs.get("normalize", False),
                use_batch_norm=kwargs.get("use_batch_norm", False),
            )
        elif self.gnn_type == "edgegat":
            # Create EdgeGAT layer
            return EdgeGAT(
                in_channels=in_dim,
                out_channels=out_dim,
                edge_dim=kwargs.get("edge_dim", 32),
                heads=kwargs.get("heads", 8),
                concat=kwargs.get("concat", True),
                negative_slope=kwargs.get("negative_slope", 0.2),
                dropout=kwargs.get("attention_dropout", 0.1),
                bias=kwargs.get("bias", True),
            )
        elif self.gnn_type == "graphtransformer":
            # Create GraphTransformer layer
            return GraphTransformer(
                in_channels=in_dim,
                hidden_channels=in_dim,  # Use same dimension for hidden
                out_channels=out_dim,
                num_layers=1,  # Single layer since we're stacking them in GraphEncoder
                num_heads=kwargs.get("heads", 8),
                dropout=self.dropout,
                edge_dim=kwargs.get("edge_dim", None),
                use_edge_features=kwargs.get("use_edge_features", True) if kwargs.get("edge_dim", None) is not None else False,
                use_path_encoding=kwargs.get("use_path_encoding", True),
                use_degree_encoding=kwargs.get("use_degree_encoding", True),
            )
        elif self.gnn_type == "diffpool":
            # Create DiffPool layer
            return DiffPool(
                in_channels=in_dim,
                hidden_channels=in_dim,
                out_channels=out_dim,
                num_layers=kwargs.get("diffpool_internal_layers", 2),
                num_pooling=kwargs.get("num_pooling", 1),
                pool_ratio=kwargs.get("pool_ratio", 0.25),
                linkpred=kwargs.get("linkpred", True),
                dropout=self.dropout,
                gnn_type=kwargs.get("diffpool_gnn", "gcn"),
            )
        elif self.gnn_type == "hgt":
            # Create HGT layer
            # Note: HGT expects dictionaries for node features and edge indices,
            # so the forward method needs to be adapted for heterogeneous graphs.
            # This is a simplified adapter version.
            hgt_layer = HGTLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                node_types=kwargs.get("node_types", ["default"]),
                edge_types=kwargs.get("edge_types", ["default"]),
                num_heads=kwargs.get("heads", 8),
                dropout=self.dropout,
                use_norm=kwargs.get("use_norm", True),
                use_RTE=kwargs.get("use_RTE", True),
            )
            # Wrap with adapter to handle conversion between formats
            return HGTAdapter(
                hgt_layer,
                default_node_type=kwargs.get("default_node_type", "default"),
                default_edge_type=kwargs.get("default_edge_type", "default")
            )
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}. Supported types: gcn, gat, sage, gin, mpnn, egnn, edgegat, graphtransformer, diffpool, hgt")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_node_embeddings: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the graph encoder.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity in COO format [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch indices [num_nodes] indicating the graph each node belongs to
            return_node_embeddings: Whether to return node embeddings in the output dict
            
        Returns:
            Dictionary containing graph embeddings and optionally node embeddings
        """
        # Project input features
        hidden = self.input_proj(x)
        
        # Process through GNN layers
        node_embeddings = []
        
        # Track previous layer's output for residual connections
        prev_hidden = hidden
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            # Apply GNN layer
            hidden = gnn_layer(hidden, edge_index, edge_attr, batch)
            
            # Apply activation function (except for the last layer)
            if i < len(self.gnn_layers) - 1 or i == len(self.gnn_layers) - 1 and self.gnn_type != "gin":
                hidden = self.activation(hidden)
            
            # Apply layer normalization
            if self.layer_norm:
                hidden = self.layer_norms[i](hidden)
            
            # Apply residual connection
            if self.residual and prev_hidden.shape == hidden.shape:
                hidden = prev_hidden + hidden
            
            # Apply dropout
            hidden = self.dropout_layer(hidden)
            
            # Store for next layer
            prev_hidden = hidden
            
            # Keep track of node embeddings
            node_embeddings.append(hidden)
        
        # Readout for graph-level embeddings
        graph_embedding = None
        if batch is not None:
            if self.readout == "mean":
                graph_embedding = self._readout_mean(hidden, batch)
            elif self.readout == "sum":
                graph_embedding = self._readout_sum(hidden, batch)
            elif self.readout == "max":
                graph_embedding = self._readout_max(hidden, batch)
            else:
                raise ValueError(f"Unsupported readout method: {self.readout}")
            
            # Apply output projection
            graph_embedding = self.output_proj(graph_embedding)
        
        # Prepare output dictionary
        output = {}
        
        if graph_embedding is not None:
            output["graph_embedding"] = graph_embedding
        
        if return_node_embeddings:
            output["node_embeddings"] = node_embeddings[-1]  # Final layer embeddings
            output["all_node_embeddings"] = node_embeddings  # All layer embeddings
        
        return output
    
    def _readout_mean(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Mean readout function to obtain graph-level embeddings.
        
        Args:
            x: Node features [num_nodes, channels]
            batch: Batch indices [num_nodes] indicating the graph each node belongs to
            
        Returns:
            Graph-level embeddings [num_graphs, channels]
        """
        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
        
        # Initialize output tensor
        out = torch.zeros(num_graphs, x.size(-1), device=x.device)
        
        # Count nodes per graph
        ones = torch.ones(batch.size(0), device=x.device)
        count = torch.zeros(num_graphs, device=x.device)
        count.scatter_add_(0, batch, ones)
        
        # Sum node features per graph
        out.scatter_add_(0, batch.unsqueeze(-1).expand(-1, x.size(-1)), x)
        
        # Normalize by count
        count = count.clamp(min=1).unsqueeze(-1)
        return out / count
    
    def _readout_sum(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Sum readout function to obtain graph-level embeddings.
        
        Args:
            x: Node features [num_nodes, channels]
            batch: Batch indices [num_nodes] indicating the graph each node belongs to
            
        Returns:
            Graph-level embeddings [num_graphs, channels]
        """
        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
        
        # Initialize output tensor
        out = torch.zeros(num_graphs, x.size(-1), device=x.device)
        
        # Sum node features per graph
        out.scatter_add_(0, batch.unsqueeze(-1).expand(-1, x.size(-1)), x)
        
        return out
    
    def _readout_max(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Max readout function to obtain graph-level embeddings.
        
        Args:
            x: Node features [num_nodes, channels]
            batch: Batch indices [num_nodes] indicating the graph each node belongs to
            
        Returns:
            Graph-level embeddings [num_graphs, channels]
        """
        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 1
        
        # Initialize output tensor with -inf
        out = torch.full((num_graphs, x.size(-1)), float('-inf'), device=x.device)
        
        # Max node features per graph
        out.scatter_reduce_(0, batch.unsqueeze(-1).expand(-1, x.size(-1)), x, reduce="amax")
        
        # Replace -inf with 0
        out = out.masked_fill(out == float('-inf'), 0)
        
        return out 