"""
Core Graph Neural Network layers for ValkyrieLLM.

This module contains implementations of popular GNN architectures:
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE (Graph SAmple and aggreGatE)
- GIN (Graph Isomorphism Network)
- MPNN (Message Passing Neural Network)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import math


class MessagePassingBase(nn.Module):
    """
    Base class for all message passing GNN layers.
    
    This implements the general message passing framework where the update
    consists of three steps:
    1. Message computation
    2. Message aggregation
    3. Node update
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "sum",
        normalize: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.normalize = normalize
        
        # Transformation for messages
        self.message_lin = nn.Linear(in_channels, out_channels, bias=bias)
        
        # Transformation for node update
        self.update_lin = nn.Linear(in_channels + out_channels, out_channels, bias=bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset parameters using glorot initialization."""
        nn.init.xavier_uniform_(self.message_lin.weight)
        nn.init.xavier_uniform_(self.update_lin.weight)
        if self.message_lin.bias is not None:
            nn.init.zeros_(self.message_lin.bias)
        if self.update_lin.bias is not None:
            nn.init.zeros_(self.update_lin.bias)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute messages from source nodes (j) to target nodes (i).
        
        Args:
            x_i: Target node features [num_edges, in_channels]
            x_j: Source node features [num_edges, in_channels]
            edge_attr: Edge features [num_edges, edge_attr_dim]
            
        Returns:
            Messages from source to target [num_edges, out_channels]
        """
        return self.message_lin(x_j)
    
    def aggregate(self, messages: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Aggregate messages from neighboring nodes.
        
        Args:
            messages: Messages from source to target [num_edges, out_channels]
            edge_index: Graph connectivity in COO format [2, num_edges]
            num_nodes: Number of nodes in the graph
            
        Returns:
            Aggregated messages [num_nodes, out_channels]
        """
        # Extract target node indices
        target_index = edge_index[1]
        
        # Aggregate messages based on the specified aggregation function
        if self.aggr == "sum":
            aggr_messages = torch.zeros(num_nodes, self.out_channels, device=messages.device)
            aggr_messages.scatter_add_(0, target_index.unsqueeze(-1).expand(-1, self.out_channels), messages)
        elif self.aggr == "mean":
            aggr_messages = torch.zeros(num_nodes, self.out_channels, device=messages.device)
            aggr_messages.scatter_add_(0, target_index.unsqueeze(-1).expand(-1, self.out_channels), messages)
            
            # Compute number of neighbors for each node
            ones = torch.ones(target_index.size(0), device=messages.device)
            count = torch.zeros(num_nodes, device=messages.device)
            count.scatter_add_(0, target_index, ones)
            count = count.clamp(min=1).unsqueeze(-1)
            
            # Normalize by number of neighbors
            aggr_messages = aggr_messages / count
        elif self.aggr == "max":
            # Initialize with -inf
            aggr_messages = torch.full((num_nodes, self.out_channels), float('-inf'), device=messages.device)
            
            # Use scatter_reduce with "amax" reduction
            aggr_messages.scatter_reduce_(0, target_index.unsqueeze(-1).expand(-1, self.out_channels), 
                                        messages, reduce="amax")
            
            # Replace -inf with 0
            aggr_messages = aggr_messages.masked_fill(aggr_messages == float('-inf'), 0)
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggr}")
        
        return aggr_messages
    
    def update(self, aggr_messages: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Update node embeddings using the aggregated messages.
        
        Args:
            aggr_messages: Aggregated messages [num_nodes, out_channels]
            x: Node features [num_nodes, in_channels]
            
        Returns:
            Updated node embeddings [num_nodes, out_channels]
        """
        # Concatenate node features with aggregated messages
        updates = torch.cat([x, aggr_messages], dim=-1)
        return self.update_lin(updates)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the message passing layer.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity in COO format [2, num_edges]
            edge_attr: Edge features [num_edges, edge_attr_dim]
            batch: Batch indices [num_nodes] indicating the graph each node belongs to
            
        Returns:
            Updated node embeddings [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        
        # Get source and target node features for each edge
        source, target = edge_index
        x_j = x[source]  # Source node features
        x_i = x[target]  # Target node features
        
        # Compute messages
        messages = self.message(x_i, x_j, edge_attr)
        
        # Aggregate messages
        aggr_messages = self.aggregate(messages, edge_index, num_nodes)
        
        # Update node embeddings
        updated_nodes = self.update(aggr_messages, x)
        
        if self.normalize:
            updated_nodes = F.normalize(updated_nodes, p=2, dim=-1)
        
        return updated_nodes


class GCNLayer(MessagePassingBase):
    """
    Graph Convolutional Network (GCN) layer.
    
    Implementation based on Kipf & Welling (ICLR 2017)
    "Semi-Supervised Classification with Graph Convolutional Networks"
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, aggr="sum", normalize=normalize, bias=bias)
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.cached_result = None
        self.cached_edge_index = None
        self.cached_num_nodes = None
        
        # Override the linear transformations from the base class
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset parameters using glorot initialization."""
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)
        self.cached_result = None
        self.cached_edge_index = None
    
    def norm(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute normalization coefficients for GCN.
        
        Args:
            edge_index: Graph connectivity in COO format [2, num_edges]
            num_nodes: Number of nodes in the graph
            
        Returns:
            Normalization coefficients [num_edges]
        """
        # Check if we can use cached normalization
        if self.cached and self.cached_edge_index is not None:
            if torch.equal(edge_index, self.cached_edge_index) and self.cached_num_nodes == num_nodes:
                return self.cached_result
        
        # Calculate node degree
        row, col = edge_index
        edge_weight = torch.ones(row.size(0), device=row.device)
        
        # Add self loops if requested
        if self.add_self_loops:
            # Add self-loops to edge_index
            loop_indices = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
            loop_indices = loop_indices.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_indices], dim=1)
            
            # Add corresponding edge weights
            loop_weights = torch.ones(num_nodes, device=edge_weight.device)
            edge_weight = torch.cat([edge_weight, loop_weights], dim=0)
        
        row, col = edge_index
        
        # Compute degrees
        degree = torch.zeros(num_nodes, device=edge_index.device)
        degree.scatter_add_(0, row, edge_weight)
        
        # Compute normalization
        degree_inv_sqrt = degree.pow_(-0.5)
        degree_inv_sqrt.masked_fill_(degree_inv_sqrt == float('inf'), 0)
        
        # Symmetric normalization (D^(-1/2) * A * D^(-1/2))
        norm = degree_inv_sqrt[row] * edge_weight * degree_inv_sqrt[col]
        
        # Cache the result if requested
        if self.cached:
            self.cached_edge_index = edge_index
            self.cached_num_nodes = num_nodes
            self.cached_result = norm
        
        return norm
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the GCN layer.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity in COO format [2, num_edges]
            edge_weight: Edge weights [num_edges]
            batch: Batch indices [num_nodes] indicating the graph each node belongs to
            
        Returns:
            Updated node embeddings [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        
        # Apply feature transformation
        x = self.lin(x)
        
        # Compute normalization
        norm = self.norm(edge_index, num_nodes)
        
        # Propagate messages
        row, col = edge_index
        out = torch.zeros_like(x)
        
        # Gather source node features
        source_features = x[row] * norm.view(-1, 1)
        
        # Aggregate to target nodes
        out.scatter_add_(0, col.unsqueeze(-1).expand(-1, self.out_channels), source_features)
        
        return out


class GATLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer.
    
    Implementation based on Veličković et al. (ICLR 2018)
    "Graph Attention Networks"
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Output channels per head
        self.out_channels_per_head = out_channels // heads if concat else out_channels
        
        # Linear transformation applied to node features
        self.lin = nn.Linear(in_channels, heads * self.out_channels_per_head, bias=False)
        
        # Attention mechanism: a's in the paper
        self.att_src = nn.Parameter(torch.Tensor(1, heads, self.out_channels_per_head))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, self.out_channels_per_head))
        
        # Bias term if requested
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * self.out_channels_per_head))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters using glorot initialization."""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the GAT layer.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity in COO format [2, num_edges]
            edge_attr: Edge features [num_edges, edge_attr_dim]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node embeddings [num_nodes, out_channels] or
            (node embeddings, attention weights) if return_attention_weights=True
        """
        num_nodes = x.size(0)
        
        # Apply linear transformation to node features
        x = self.lin(x).view(-1, self.heads, self.out_channels_per_head)
        
        # Extract source and target node indices
        source, target = edge_index
        
        # Compute attention scores
        # Source node attention (α⋅Wh_i in the paper)
        alpha_src = (x * self.att_src).sum(dim=-1)
        # Target node attention (α⋅Wh_j in the paper)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        
        # Compute combined attention scores for each edge
        alpha = alpha_src[source] + alpha_dst[target]
        
        # Apply LeakyReLU non-linearity
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention scores using softmax
        alpha = self._softmax(alpha, target, num_nodes)
        
        # Apply dropout to attention scores
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention weights to source node features
        out = torch.zeros(num_nodes, self.heads, self.out_channels_per_head, device=x.device)
        weighted_features = x[source].mul(alpha.unsqueeze(-1))
        
        # Aggregate weighted features
        out.scatter_add_(0, target.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.out_channels_per_head), weighted_features)
        
        # Combine heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels_per_head)
        else:
            out = out.mean(dim=1)
        
        # Add bias if requested
        if self.bias is not None:
            out = out + self.bias
        
        if return_attention_weights:
            return out, (edge_index, alpha)
        else:
            return out
    
    def _softmax(self, alpha: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute softmax normalization per target node.
        
        Args:
            alpha: Attention scores [num_edges, heads]
            index: Target nodes indices [num_edges]
            num_nodes: Number of nodes in the graph
            
        Returns:
            Normalized attention scores [num_edges, heads]
        """
        # Group by target node
        num_nodes = num_nodes
        
        # For each target node, compute softmax over the incoming edges
        alpha = torch.softmax(alpha, dim=0)
        
        return alpha


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE layer implementation.
    
    Implements the GraphSAGE operation as described in:
    "Inductive Representation Learning on Large Graphs" (Hamilton et al., 2017)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator_type: str = "mean",
        use_bias: bool = True,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        use_layer_norm: bool = False,
        residual: bool = False,
        normalize: bool = True
    ):
        """
        Initialize a GraphSAGE layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            aggregator_type: Aggregation function ('mean', 'max', 'sum', 'lstm')
            use_bias: Whether to use bias
            dropout: Dropout probability
            activation: Activation function to use
            use_layer_norm: Whether to use layer normalization
            residual: Whether to use residual connections
            normalize: Whether to normalize the output embeddings
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator_type = aggregator_type.lower()
        self.use_bias = use_bias
        self.dropout = dropout
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.residual = residual
        self.normalize = normalize
        
        # Self transform for the node's own features
        self.self_linear = nn.Linear(in_features, out_features, bias=use_bias)
        
        # Linear transformation for neighbor features
        self.neigh_linear = nn.Linear(in_features, out_features, bias=False)
        
        # Special components for aggregator types
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(
                input_size=in_features,
                hidden_size=in_features,
                batch_first=True
            )
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_features)
        
        # For residual connections
        if residual and in_features != out_features:
            self.res_proj = nn.Linear(in_features, out_features, bias=False)
        else:
            self.res_proj = None
        
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize the parameters of the GraphSAGE layer."""
        # Initialize linear transformations
        nn.init.xavier_uniform_(self.self_linear.weight)
        nn.init.xavier_uniform_(self.neigh_linear.weight)
        
        # Initialize bias
        if self.use_bias:
            nn.init.zeros_(self.self_linear.bias)
        
        # Initialize residual projection
        if self.res_proj is not None:
            nn.init.xavier_uniform_(self.res_proj.weight)
        
        # Initialize LSTM if used
        if self.aggregator_type == "lstm":
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def mean_aggregator(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Mean aggregation function for GraphSAGE."""
        # Normalize adjacency matrix for proper mean calculation
        # Compute degree matrix (number of neighbors for each node)
        degree = adj_matrix.sum(dim=-1, keepdim=True)  # [batch_size, num_nodes, 1]
        # Avoid division by zero for isolated nodes
        degree = torch.clamp(degree, min=1.0)
        
        # Normalize adjacency matrix by degree
        norm_adj = adj_matrix / degree
        
        # Aggregate neighbor features
        return torch.matmul(norm_adj, node_features)
    
    def max_aggregator(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Max aggregation function for GraphSAGE."""
        batch_size, num_nodes, feat_dim = node_features.shape
        
        # Set non-edge values to a large negative value
        # [batch_size, num_nodes, num_nodes, feat_dim]
        expanded_features = node_features.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        mask = adj_matrix.unsqueeze(-1).expand(-1, -1, -1, feat_dim) == 0
        masked_features = expanded_features.masked_fill(mask, -9e15)
        
        # Take max over neighborhood
        max_features, _ = torch.max(masked_features, dim=2)
        
        # Handle isolated nodes (all neighbors are masked)
        # Set their aggregated features to zeros if they have no neighbors
        isolated = (adj_matrix.sum(dim=-1) == 0).unsqueeze(-1).expand(-1, -1, feat_dim)
        max_features = max_features.masked_fill(isolated, 0.0)
        
        return max_features
    
    def sum_aggregator(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Sum aggregation function for GraphSAGE."""
        return torch.matmul(adj_matrix, node_features)
    
    def lstm_aggregator(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """LSTM aggregation function for GraphSAGE."""
        batch_size, num_nodes, feat_dim = node_features.shape
        result = torch.zeros(batch_size, num_nodes, feat_dim, device=node_features.device)
        
        # Process each node in the batch
        for b in range(batch_size):
            for i in range(num_nodes):
                # Get neighbors of node i
                neighbors = adj_matrix[b, i].nonzero().squeeze(-1)
                
                if neighbors.numel() > 0:
                    # Collect neighbor features
                    neighbor_feats = node_features[b, neighbors]
                    
                    # Apply LSTM to neighbor features
                    _, (hidden, _) = self.lstm(neighbor_feats.unsqueeze(0))
                    
                    # Use the final hidden state as the aggregated feature
                    result[b, i] = hidden.squeeze(0).squeeze(0)
        
        return result
    
    def forward(
        self, 
        node_features: torch.Tensor, 
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the GraphSAGE layer.
        
        Args:
            node_features: Node feature matrix [batch_size, num_nodes, in_features]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated node features [batch_size, num_nodes, out_features]
        """
        # Add self-loops to adjacency matrix
        batch_size, num_nodes, _ = adj_matrix.shape
        identity = torch.eye(num_nodes, device=adj_matrix.device)
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
        adj_with_self_loops = adj_matrix.clone()
        
        # Remove existing self-loops to avoid double counting
        adj_no_self_loops = adj_with_self_loops * (1 - identity)
        
        # Aggregate neighbor features based on the specified aggregator
        if self.aggregator_type == "mean":
            neigh_features = self.mean_aggregator(node_features, adj_no_self_loops)
        elif self.aggregator_type == "max":
            neigh_features = self.max_aggregator(node_features, adj_no_self_loops)
        elif self.aggregator_type == "sum":
            neigh_features = self.sum_aggregator(node_features, adj_no_self_loops)
        elif self.aggregator_type == "lstm":
            neigh_features = self.lstm_aggregator(node_features, adj_no_self_loops)
        else:
            raise ValueError(f"Unknown aggregator type: {self.aggregator_type}")
        
        # Transform features
        transformed_self = self.self_linear(node_features)
        transformed_neigh = self.neigh_linear(neigh_features)
        
        # Combine self and neighbor features
        output = transformed_self + transformed_neigh
        
        # Apply residual connection if requested
        if self.residual:
            if self.res_proj is not None:
                residual = self.res_proj(node_features)
            else:
                residual = node_features
            output = output + residual
        
        # Apply layer normalization if requested
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        # Apply activation and dropout
        output = self.activation(output)
        output = self.dropout_layer(output)
        
        # Normalize output embeddings if requested
        if self.normalize:
            output = F.normalize(output, p=2, dim=-1)
        
        return output


class GINLayer(nn.Module):
    """
    Graph Isomorphism Network (GIN) layer.
    
    Implementation based on Xu et al. (ICLR 2019)
    "How Powerful are Graph Neural Networks?"
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 0.0,
        train_eps: bool = False,
        mlp_hidden_channels: Optional[int] = None,
        activation: Callable = F.relu,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        
        # Epsilon parameter from the GIN paper (either trainable or fixed)
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        
        # Define the MLP
        mlp_hidden_channels = mlp_hidden_channels or out_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_channels),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.ReLU(),
            nn.Linear(mlp_hidden_channels, out_channels),
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters."""
        for layer in self.mlp:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if isinstance(self.eps, nn.Parameter):
            nn.init.zeros_(self.eps)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the GIN layer.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity in COO format [2, num_edges]
            batch: Batch indices [num_nodes] indicating the graph each node belongs to
            
        Returns:
            Updated node embeddings [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        
        # Extract source and target nodes
        source, target = edge_index
        
        # Initialize output with self-loop: (1 + eps) * x_i
        out = (1 + self.eps) * x
        
        # Aggregate neighborhood features (simple sum aggregation in standard GIN)
        aggr_out = torch.zeros_like(x)
        aggr_out.scatter_add_(0, target.unsqueeze(-1).expand(-1, self.in_channels), x[source])
        
        # Combine with self features
        out = out + aggr_out
        
        # Apply MLP
        if batch is not None:
            out = self.mlp(out)
        else:
            # Handle batch normalization for a single graph
            batch_size = out.size(0)
            out = self.mlp[0](out)  # Linear
            out = out.view(1, batch_size, -1)
            out = out.transpose(0, 1).contiguous()
            out = self.mlp[1](out)  # BatchNorm
            out = out.transpose(0, 1).contiguous()
            out = out.view(batch_size, -1)
            out = self.mlp[2](out)  # ReLU
            out = self.mlp[3](out)  # Linear
        
        return out


class MPNNLayer(MessagePassingBase):
    """
    Message Passing Neural Network (MPNN) layer.
    
    Implementation based on Gilmer et al. (ICML 2017)
    "Neural Message Passing for Quantum Chemistry"
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        aggr: str = "sum",
        normalize: bool = False,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, aggr=aggr, normalize=normalize, bias=bias)
        self.edge_dim = edge_dim
        
        # If edge features are provided, create a linear transformation for them
        if edge_dim is not None:
            self.edge_lin = nn.Linear(edge_dim, out_channels, bias=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters using glorot initialization."""
        super().reset_parameters()
        if hasattr(self, 'edge_lin'):
            nn.init.xavier_uniform_(self.edge_lin.weight)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute messages incorporating edge features if available.
        
        Args:
            x_i: Target node features [num_edges, in_channels]
            x_j: Source node features [num_edges, in_channels]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Messages from source to target [num_edges, out_channels]
        """
        # Apply node feature transformation
        messages = self.message_lin(x_j)
        
        # Incorporate edge features if available
        if edge_attr is not None and self.edge_dim is not None:
            edge_embedding = self.edge_lin(edge_attr)
            messages = messages + edge_embedding
        
        return messages 

class EdgeGAT(nn.Module):
    """
    Edge-Enhanced Graph Attention Network (EdgeGAT).
    
    Extends the standard GAT to incorporate edge features in the attention mechanism
    and message passing process. This implementation allows for effectively leveraging
    edge information in graph-based reasoning.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Output channels per head
        self.out_channels_per_head = out_channels // heads if concat else out_channels
        
        # Linear transformation applied to node features
        self.lin = nn.Linear(in_channels, heads * self.out_channels_per_head, bias=False)
        
        # Linear transformation for edge features
        self.edge_lin = nn.Linear(edge_dim, heads * self.out_channels_per_head, bias=False)
        
        # Attention mechanism: a's in the paper
        self.att_src = nn.Parameter(torch.Tensor(1, heads, self.out_channels_per_head))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, self.out_channels_per_head))
        self.att_edge = nn.Parameter(torch.Tensor(1, heads, self.out_channels_per_head))
        
        # Bias term if requested
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * self.out_channels_per_head))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters using glorot initialization."""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.edge_lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.att_edge)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the EdgeGAT layer.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity in COO format [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node embeddings [num_nodes, out_channels] or
            (node embeddings, attention weights) if return_attention_weights=True
        """
        num_nodes = x.size(0)
        
        # Apply linear transformation to node features
        x = self.lin(x).view(-1, self.heads, self.out_channels_per_head)
        
        # Transform edge features
        edge_emb = self.edge_lin(edge_attr).view(-1, self.heads, self.out_channels_per_head)
        
        # Extract source and target node indices
        source, target = edge_index
        
        # Compute attention scores
        # Source node attention (α⋅Wh_i in the paper)
        alpha_src = (x * self.att_src).sum(dim=-1)
        # Target node attention (α⋅Wh_j in the paper)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        # Edge feature attention
        alpha_edge = (edge_emb * self.att_edge).sum(dim=-1)
        
        # Compute combined attention scores for each edge
        alpha = alpha_src[source] + alpha_dst[target] + alpha_edge
        
        # Apply LeakyReLU non-linearity
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Normalize attention scores using softmax
        alpha = self._softmax(alpha, target, num_nodes)
        
        # Apply dropout to attention scores
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention weights to source node features and edge features
        out = torch.zeros(num_nodes, self.heads, self.out_channels_per_head, device=x.device)
        
        # Combine node and edge features with attention weights
        weighted_features = x[source] * alpha.unsqueeze(-1)
        # Add edge contribution
        weighted_features = weighted_features + edge_emb * alpha.unsqueeze(-1)
        
        # Aggregate weighted features
        out.scatter_add_(0, target.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.out_channels_per_head), weighted_features)
        
        # Combine heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels_per_head)
        else:
            out = out.mean(dim=1)
        
        # Add bias if requested
        if self.bias is not None:
            out = out + self.bias
        
        if return_attention_weights:
            return out, (edge_index, alpha)
        else:
            return out
    
    def _softmax(self, alpha: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute softmax normalization per target node.
        
        Args:
            alpha: Attention scores [num_edges, heads]
            index: Target nodes indices [num_edges]
            num_nodes: Number of nodes in the graph
            
        Returns:
            Normalized attention scores [num_edges, heads]
        """
        # Group by target node
        num_nodes = num_nodes
        
        # For each target node, compute softmax over the incoming edges
        alpha = torch.softmax(alpha, dim=0)
        
        return alpha


class EGNNLayer(nn.Module):
    """
    Edge-Enhanced Graph Neural Network Layer.
    
    This layer processes both node and edge features, computing updated node
    representations with a focus on edge information.
    """
    
    def __init__(
        self,
        in_channels: int,
        edge_channels: int,
        out_channels: int,
        aggr: str = "sum",
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        residual: bool = True,
        normalize: bool = False,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.dropout = dropout
        self.activation = activation
        self.residual = residual
        self.normalize = normalize
        self.use_batch_norm = use_batch_norm
        
        # Node feature projections
        self.node_proj = nn.Linear(in_channels, out_channels)
        
        # Edge feature projections
        self.edge_proj = nn.Linear(edge_channels, out_channels)
        
        # Message transformation (combining source node and edge features)
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        
        # Attention mechanism for weighting messages
        self.attention = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_channels, out_channels),
            nn.Tanh(),
            nn.Linear(out_channels, 1),
        )
        
        # Update function for target nodes
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        
        # Optional residual connection
        if residual and in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = None
        
        # Optional batch normalization
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(out_channels)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize the layer parameters."""
        # Initialize linear transformations
        nn.init.xavier_uniform_(self.node_proj.weight)
        nn.init.zeros_(self.node_proj.bias)
        
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.zeros_(self.edge_proj.bias)
        
        # Initialize MLPs
        for layer in self.message_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        for layer in self.update_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        # Initialize residual projection
        if self.res_proj is not None:
            nn.init.xavier_uniform_(self.res_proj.weight)
            nn.init.zeros_(self.res_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the EGNN layer.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity in COO format [2, num_edges]
            edge_attr: Edge features [num_edges, edge_channels]
            
        Returns:
            Updated node embeddings [num_nodes, out_channels]
        """
        # Store original features for residual connection
        identity = x
        
        # Get number of nodes
        num_nodes = x.size(0)
        
        # Extract source and target node indices
        source, target = edge_index
        
        # Compute attention weights
        # Gather source and target node features for each edge
        source_features = x[source]
        target_features = x[target]
        
        # Concatenate source, target, and edge features for attention computation
        attention_input = torch.cat([source_features, target_features, edge_attr], dim=-1)
        
        # Compute attention weights
        attention_weights = self.attention(attention_input)
        attention_weights = F.softmax(attention_weights, dim=0)
        
        # Compute messages (combining source node and edge features)
        message_input = torch.cat([source_features, edge_attr], dim=-1)
        messages = self.message_mlp(message_input)
        
        # Weight messages by attention scores
        weighted_messages = messages * attention_weights
        
        # Aggregate messages based on the specified aggregation function
        aggr_messages = torch.zeros(num_nodes, self.out_channels, device=x.device)
        
        if self.aggr == "sum":
            aggr_messages.scatter_add_(0, target.unsqueeze(-1).expand(-1, self.out_channels), weighted_messages)
        elif self.aggr == "mean":
            aggr_messages.scatter_add_(0, target.unsqueeze(-1).expand(-1, self.out_channels), weighted_messages)
            
            # Compute number of neighbors for each node
            ones = torch.ones(target.size(0), device=x.device)
            count = torch.zeros(num_nodes, device=x.device)
            count.scatter_add_(0, target, ones)
            count = count.clamp(min=1).unsqueeze(-1)
            
            # Normalize by number of neighbors
            aggr_messages = aggr_messages / count
        elif self.aggr == "max":
            # Initialize with -inf
            aggr_messages = torch.full((num_nodes, self.out_channels), float('-inf'), device=x.device)
            
            # Use scatter_reduce with "amax" reduction
            aggr_messages.scatter_reduce_(0, target.unsqueeze(-1).expand(-1, self.out_channels), 
                                        weighted_messages, reduce="amax")
            
            # Replace -inf with 0
            aggr_messages = aggr_messages.masked_fill(aggr_messages == float('-inf'), 0)
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggr}")
        
        # Update node representations
        update_input = torch.cat([x, aggr_messages], dim=-1)
        out = self.update_mlp(update_input)
        
        # Apply residual connection if requested
        if self.residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            out = out + identity
        
        # Apply batch normalization if requested
        if self.use_batch_norm:
            out = self.batch_norm(out)
        
        # Apply activation and dropout
        out = self.activation(out)
        out = self.dropout_layer(out)
        
        # Normalize output embeddings if requested
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return out


class EGNN(nn.Module):
    """
    Edge-Enhanced Graph Neural Network.
    
    A model that leverages edge features for better graph-based reasoning.
    This implementation stacks multiple EGNN layers to capture complex patterns
    and relationships in the graph structure while emphasizing edge information.
    """
    
    def __init__(
        self,
        in_channels: int,
        edge_features: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        residual: bool = True,
        normalize: bool = False,
        use_bn: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.edge_features = edge_features
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.normalize = normalize
        self.use_bn = use_bn
        
        # Input projections
        self.node_proj = nn.Linear(in_channels, hidden_channels)
        self.edge_proj = nn.Linear(edge_features, hidden_channels)
        
        # EGNN layers
        self.layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.layers.append(
            EGNNLayer(
                hidden_channels,
                hidden_channels,
                hidden_channels,
                dropout=dropout,
                residual=residual,
                normalize=normalize,
                use_batch_norm=use_bn,
            )
        )
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.layers.append(
                EGNNLayer(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels,
                    dropout=dropout,
                    residual=residual,
                    normalize=normalize,
                    use_batch_norm=use_bn,
                )
            )
        
        # Last layer: hidden_dim -> output_dim
        if num_layers > 1:
            self.layers.append(
                EGNNLayer(
                    hidden_channels,
                    hidden_channels,
                    out_channels,
                    dropout=dropout,
                    residual=residual and hidden_channels == out_channels,
                    normalize=normalize,
                    use_batch_norm=use_bn,
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels if num_layers == 1 else out_channels, out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize the model parameters."""
        nn.init.xavier_uniform_(self.node_proj.weight)
        nn.init.zeros_(self.node_proj.bias)
        
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.zeros_(self.edge_proj.bias)
        
        for layer in self.layers:
            layer.reset_parameters()
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the EGNN.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity in COO format [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            
        Returns:
            Updated node embeddings [num_nodes, out_channels]
        """
        # Project input features
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        # Apply EGNN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Final projection
        x = self.output_proj(x)
        
        return x


class GraphTransformerLayer(nn.Module):
    """
    Individual layer for the Graph Transformer.
    
    This implements a Transformer encoder layer adapted for graphs,
    with self-attention mechanisms that incorporate graph structure.
    """
    
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        norm_first: bool = False,
        activation: Callable = F.gelu,
    ):
        """
        Initialize a Graph Transformer layer.
        
        Args:
            hidden_channels: Dimension of input/output features
            num_heads: Number of attention heads
            dropout: Dropout rate for MLP
            attention_dropout: Dropout rate for attention weights
            norm_first: Whether to normalize before attention/MLP (Pre-LN Transformer)
            activation: Activation function for MLP
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.norm_first = norm_first
        self.activation = activation
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_channels, 
            num_heads, 
            dropout=attention_dropout,
            batch_first=True
        )
        
        # MLP module
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 4 * hidden_channels),
            nn.GELU(),
            nn.Linear(4 * hidden_channels, hidden_channels),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize layer parameters."""
        # MLP initialization
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the Graph Transformer layer.
        
        Args:
            x: Node features [num_nodes, hidden_channels]
            attn_mask: Attention mask based on graph structure [num_nodes, num_nodes]
                
        Returns:
            Updated node features [num_nodes, hidden_channels]
        """
        # Apply normalization before or after attention based on norm_first
        if self.norm_first:
            # Pre-LN Transformer
            normalized_x = self.norm1(x)
            attn_output, _ = self.self_attn(
                normalized_x, normalized_x, normalized_x,
                attn_mask=attn_mask,
                need_weights=False
            )
            x = x + attn_output
            
            normalized_x = self.norm2(x)
            x = x + self.mlp(normalized_x)
        else:
            # Post-LN Transformer
            attn_output, _ = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                need_weights=False
            )
            x = self.norm1(x + attn_output)
            
            mlp_output = self.mlp(x)
            x = self.norm2(x + mlp_output)
        
        return x


class GraphTransformer(nn.Module):
    """
    Graph Transformer (Graphormer) implementation.
    
    This model incorporates Transformer-style self-attention mechanisms into graph neural networks,
    allowing for more expressive modeling of node interactions beyond immediate neighbors.
    
    Based on "Do Transformers Really Perform Bad for Graph Representation?"
    (Ying et al., NeurIPS 2021)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_edge_features: bool = False,
        edge_dim: Optional[int] = None,
        max_path_length: int = 5,
        centrality_encoding: bool = True,
        spatial_encoding: bool = True,
        norm_first: bool = False,
        activation: Callable = F.gelu,
    ):
        """
        Initialize the Graph Transformer model.
        
        Args:
            in_channels: Dimension of input node features
            hidden_channels: Dimension of hidden layers
            out_channels: Dimension of output node features
            num_layers: Number of transformer layers
            num_heads: Number of attention heads in each layer
            dropout: Dropout rate for MLP and final layers
            attention_dropout: Dropout rate for attention weights
            use_edge_features: Whether to use edge features
            edge_dim: Dimension of edge features if used
            max_path_length: Maximum path length for spatial encoding
            centrality_encoding: Whether to use centrality encoding (degree)
            spatial_encoding: Whether to use spatial encoding (SPD)
            norm_first: Whether to normalize before attention/MLP (Pre-LN Transformer)
            activation: Activation function
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_edge_features = use_edge_features
        self.edge_dim = edge_dim
        self.max_path_length = max_path_length
        self.centrality_encoding = centrality_encoding
        self.spatial_encoding = spatial_encoding
        self.norm_first = norm_first
        self.activation = activation
        
        # Input embedding
        embedding_dim = hidden_channels
        
        # Node feature embedding
        self.node_embedding = nn.Linear(in_channels, hidden_channels)
        
        # Positional encodings
        if centrality_encoding:
            # Node degree embedding (centrality)
            self.degree_embedding = nn.Embedding(64, hidden_channels, padding_idx=0)
        
        if spatial_encoding:
            # Shortest path distance embedding (spatial)
            self.path_embedding = nn.Embedding(max_path_length + 2, hidden_channels, padding_idx=0)
        
        # Edge feature embedding (if used)
        if use_edge_features and edge_dim is not None:
            self.edge_embedding = nn.Linear(edge_dim, hidden_channels)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_channels=hidden_channels,
                num_heads=num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                norm_first=norm_first,
                activation=activation
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_channels)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize model parameters."""
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.zeros_(self.node_embedding.bias)
        
        if self.centrality_encoding:
            nn.init.normal_(self.degree_embedding.weight, mean=0, std=0.02)
        
        if self.spatial_encoding:
            nn.init.normal_(self.path_embedding.weight, mean=0, std=0.02)
        
        if self.use_edge_features and self.edge_dim is not None:
            nn.init.xavier_uniform_(self.edge_embedding.weight)
            nn.init.zeros_(self.edge_embedding.bias)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def compute_shortest_path_distance(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        max_distance: int = None
    ) -> torch.Tensor:
        """
        Compute the shortest path distance matrix from edge_index.
        
        Args:
            edge_index: Edge connectivity [2, num_edges]
            num_nodes: Number of nodes in the graph
            max_distance: Maximum distance to compute
            
        Returns:
            Shortest path distance matrix [num_nodes, num_nodes]
        """
        if max_distance is None:
            max_distance = self.max_path_length
        
        # Create adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        
        # Initialize distance matrix with inf
        distances = torch.full((num_nodes, num_nodes), float('inf'), device=edge_index.device)
        
        # Set diagonal to 0 (distance to self)
        distances.fill_diagonal_(0)
        
        # Set direct connections to 1
        distances[edge_index[0], edge_index[1]] = 1
        
        # Floyd-Warshall algorithm for all-pairs shortest paths
        for k in range(num_nodes):
            # Break early if we've reached the maximum distance
            if (distances <= max_distance).all():
                break
                
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if distances[i, k] + distances[k, j] < distances[i, j]:
                        distances[i, j] = distances[i, k] + distances[k, j]
        
        # Clip distances to max_distance + 1
        distances.clamp_(max=max_distance + 1)
        
        return distances.long()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the Graph Transformer.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Get number of nodes
        num_nodes = x.size(0)
        
        # Embed node features
        h = self.node_embedding(x)
        
        # Prepare structural encodings for attention bias
        attention_bias = None
        
        # Compute shortest path distances if using spatial encoding
        if self.spatial_encoding:
            # Compute all-pairs shortest path distances
            spd_matrix = self.compute_shortest_path_distance(edge_index, num_nodes)
            
            # Convert SPD matrix to attention bias
            attention_bias = self.path_embedding(spd_matrix)
        
        # Add centrality encoding if used
        if self.centrality_encoding:
            # Compute node degrees
            node_degrees = torch.bincount(edge_index[0], minlength=num_nodes)
            node_degrees = torch.clamp(node_degrees, max=63)  # Max supported degree
            
            # Embed degrees and add to node features
            degree_emb = self.degree_embedding(node_degrees)
            h = h + degree_emb
        
        # Add edge features if used
        if self.use_edge_features and edge_attr is not None and hasattr(self, 'edge_embedding'):
            # Create an attention mask based on edges
            edge_attention = torch.zeros(num_nodes, num_nodes, self.hidden_channels, device=x.device)
            
            # Embed edge features
            edge_emb = self.edge_embedding(edge_attr)
            
            # Map edge features to the attention mask
            for i in range(edge_index.size(1)):
                src, dst = edge_index[0, i], edge_index[1, i]
                edge_attention[src, dst] = edge_emb[i]
            
            # Add to attention bias if it exists
            if attention_bias is not None:
                attention_bias = attention_bias + edge_attention
            else:
                attention_bias = edge_attention
        
        # Convert attention bias to mask for nn.MultiheadAttention
        if attention_bias is not None:
            # Reshape to [num_nodes, num_nodes]
            attn_mask = attention_bias.sum(dim=-1) != 0
            
            # Convert boolean mask to float mask with -inf for masked positions
            attn_mask = ~attn_mask  # Invert mask (True for positions to mask)
            attn_mask = attn_mask.float().masked_fill(attn_mask, float("-inf"))
        else:
            attn_mask = None
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, attn_mask)
        
        # Apply final normalization
        h = self.norm(h)
        
        # Apply dropout and final projection
        h = self.dropout_layer(h)
        out = self.output_proj(h)
        
        return out 

class DiffPool(nn.Module):
    """
    Differentiable Pooling (DiffPool) for hierarchical graph representation learning.
    
    DiffPool learns a differentiable soft assignment matrix to cluster nodes at each layer,
    enabling hierarchical graph representation learning.
    
    Based on "Hierarchical Graph Representation Learning with Differentiable Pooling"
    (Ying et al., NeurIPS 2018)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_pooling: int = 1,
        pool_ratio: float = 0.25,
        linkpred: bool = True,
        dropout: float = 0.0,
        gnn_type: str = "gcn",
    ):
        """
        Initialize the DiffPool model.
        
        Args:
            in_channels: Dimension of input node features
            hidden_channels: Dimension of hidden layers
            out_channels: Dimension of output features
            num_layers: Number of GNN layers per DiffPool block
            num_pooling: Number of pooling operations
            pool_ratio: Ratio of nodes to keep after pooling (0-1)
            linkpred: Whether to use link prediction auxiliary loss
            dropout: Dropout rate
            gnn_type: Type of GNN to use ('gcn', 'gat', 'sage')
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_pooling = num_pooling
        self.pool_ratio = pool_ratio
        self.linkpred = linkpred
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()
        
        # Input embedding
        self.input_embed = nn.Linear(in_channels, hidden_channels)
        
        # Build hierarchical GNN structure
        self.diffpool_blocks = nn.ModuleList()
        
        # Initial number of clusters for the first pooling
        num_nodes = None  # Will be determined dynamically
        
        # Diffpool blocks (each with embed GNN, pool GNN, and pooling)
        for i in range(num_pooling):
            # Create a DiffPool block
            block = DiffPoolBlock(
                in_channels=hidden_channels if i == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                gnn_type=gnn_type,
                num_layers=num_layers,
                pool_ratio=pool_ratio,
                dropout=dropout,
                linkpred=linkpred
            )
            self.diffpool_blocks.append(block)
        
        # Final GNN layers after pooling
        self.final_gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'gcn':
                gnn_layer = GCNLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    bias=True
                )
            elif gnn_type == 'gat':
                gnn_layer = GATLayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout
                )
            elif gnn_type == 'sage':
                gnn_layer = GraphSAGELayer(
                    in_features=hidden_channels,
                    out_features=hidden_channels,
                    aggregator_type="mean",
                    dropout=dropout,
                    use_layer_norm=False,
                    residual=True
                )
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
                
            self.final_gnn_layers.append(gnn_layer)
        
        # Global pooling (readout)
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
            
        # Final output layer
        self.output_layer = nn.Linear(hidden_channels, out_channels)
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize model parameters."""
        nn.init.xavier_uniform_(self.input_embed.weight)
        nn.init.zeros_(self.input_embed.bias)
        
        for block in self.diffpool_blocks:
            block.reset_parameters()
        
        for layer in self.final_gnn_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        for layer in self.global_pool:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for DiffPool.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            Tuple containing:
                - Output node features [num_nodes_final, out_channels]
                - List of auxiliary losses (link prediction, entropy)
        """
        # Initial feature embedding
        h = self.input_embed(x)
        h = F.relu(h)
        h = self.dropout_layer(h)
        
        # Get adjacency matrix from edge_index
        adj = self._edge_index_to_adj(edge_index, x.size(0), batch)
        
        # List to store auxiliary losses
        aux_losses = []
        
        # Apply DiffPool blocks
        for block in self.diffpool_blocks:
            h, adj, block_losses = block(h, adj, batch)
            aux_losses.extend(block_losses)
        
        # Apply final GNN layers
        for layer in self.final_gnn_layers:
            if self.gnn_type in ['gcn', 'gat']:
                # Re-compute edge_index from adjacency matrix
                edge_index_coarsened = self._adj_to_edge_index(adj)
                h = layer(h, edge_index_coarsened)
            else:
                # For GraphSAGE, use adjacency matrix directly
                h = layer(h, adj)
            
            h = F.relu(h)
            h = self.dropout_layer(h)
        
        # Global pooling (readout)
        # If batch is provided, do per-graph pooling
        if batch is not None:
            # Since h corresponds to coarsened nodes, need to compute new batch assignment
            # This is complex in general, but for simplicity we can use the fact that 
            # each DiffPool block preserves graph boundaries
            # We compute node-to-graph assignment based on pooled adjacency matrix
            
            # For our implementation, we'll do global mean pooling
            num_graphs = torch.max(batch) + 1 if batch.size(0) > 0 else 1
            graph_output = torch.zeros(num_graphs, h.size(1), device=h.device)
            
            # Simple mean pooling for pooled nodes
            graph_output.index_add_(0, batch, h)
            graph_count = torch.bincount(batch, minlength=num_graphs).float().view(-1, 1)
            graph_output = graph_output / graph_count.clamp(min=1)
            
            # Apply final transformation
            output = self.output_layer(graph_output)
        else:
            # Single graph, just do global pooling
            output = self.global_pool(h.mean(dim=0, keepdim=True))
            output = output.squeeze(0)
        
        return output, aux_losses
    
    def _edge_index_to_adj(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert edge_index to dense adjacency matrix.
        
        Args:
            edge_index: Edge connectivity [2, num_edges]
            num_nodes: Number of nodes in the graph
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            Adjacency matrix [num_nodes, num_nodes] or batch-wise block diagonal matrix
        """
        if batch is None:
            # Single graph
            adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
            adj[edge_index[0], edge_index[1]] = 1
            return adj
        else:
            # Batched graphs - handle each graph separately
            unique_batch = torch.unique(batch)
            max_nodes_per_graph = torch.bincount(batch).max().item()
            
            # Create a batched adjacency matrix
            # Either as a block diagonal matrix or as a batch of adjacency matrices
            batch_adj = torch.zeros(
                len(unique_batch),  # Number of graphs
                max_nodes_per_graph,
                max_nodes_per_graph,
                device=edge_index.device
            )
            
            for b in unique_batch:
                # Get nodes and edges for this graph
                b_mask = (batch == b)
                b_nodes = torch.nonzero(b_mask).squeeze()
                
                # Map global node indices to local indices within this graph
                node_mapper = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
                node_mapper[b_nodes] = torch.arange(b_nodes.size(0), device=edge_index.device)
                
                # Filter edges that belong to this graph
                edge_mask = b_mask[edge_index[0]] & b_mask[edge_index[1]]
                b_edges = edge_index[:, edge_mask]
                
                # Map to local indices
                b_edges_local = node_mapper[b_edges]
                
                # Create adjacency matrix for this graph
                b_adj = torch.zeros(
                    b_nodes.size(0),
                    b_nodes.size(0),
                    device=edge_index.device
                )
                b_adj[b_edges_local[0], b_edges_local[1]] = 1
                
                # Add to batch adjacency tensor (with padding)
                batch_adj[b, :b_adj.size(0), :b_adj.size(1)] = b_adj
            
            return batch_adj
    
    def _adj_to_edge_index(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Convert adjacency matrix back to edge_index.
        
        Args:
            adj: Adjacency matrix [num_nodes, num_nodes] or batched [batch_size, num_nodes, num_nodes]
            
        Returns:
            Edge index [2, num_edges]
        """
        if adj.dim() == 2:
            # Single adjacency matrix
            edge_indices = torch.nonzero(adj > 0.5).t()
            return edge_indices
        else:
            # Batched adjacency matrices
            edge_indices = []
            offset = 0
            
            for b in range(adj.size(0)):
                b_adj = adj[b]
                # Find edges
                b_edges = torch.nonzero(b_adj > 0.5).t()
                
                if b_edges.size(1) > 0:  # If there are edges
                    # Add offset for this graph
                    b_edges = b_edges + offset
                    edge_indices.append(b_edges)
                
                # Update offset for next graph
                offset += b_adj.size(0)
            
            # Concatenate all edges
            if edge_indices:
                return torch.cat(edge_indices, dim=1)
            else:
                # Return empty edge index if no edges
                return torch.zeros(2, 0, dtype=torch.long, device=adj.device)


class DiffPoolBlock(nn.Module):
    """
    A single DiffPool block consisting of embedding GNN, pooling GNN, and pooling operation.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        gnn_type: str = "gcn",
        num_layers: int = 2,
        pool_ratio: float = 0.25,
        dropout: float = 0.0,
        linkpred: bool = True,
    ):
        """
        Initialize a DiffPool block.
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden feature dimension
            out_channels: Output feature dimension
            gnn_type: Type of GNN to use ('gcn', 'gat', 'sage')
            num_layers: Number of GNN layers before pooling
            pool_ratio: Ratio of nodes to keep after pooling (0-1)
            dropout: Dropout rate
            linkpred: Whether to use link prediction auxiliary loss
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.gnn_type = gnn_type.lower()
        self.num_layers = num_layers
        self.pool_ratio = pool_ratio
        self.dropout = dropout
        self.linkpred = linkpred
        
        # GNN layers for node embedding (feature transformation)
        self.embed_gnn_layers = nn.ModuleList()
        
        # First layer: in_channels -> hidden_channels
        if gnn_type == 'gcn':
            self.embed_gnn_layers.append(
                GCNLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    bias=True
                )
            )
        elif gnn_type == 'gat':
            self.embed_gnn_layers.append(
                GATLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout
                )
            )
        elif gnn_type == 'sage':
            self.embed_gnn_layers.append(
                GraphSAGELayer(
                    in_features=in_channels,
                    out_features=hidden_channels,
                    aggregator_type="mean",
                    dropout=dropout,
                    use_layer_norm=False,
                    residual=True
                )
            )
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Hidden layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 1):
            if gnn_type == 'gcn':
                self.embed_gnn_layers.append(
                    GCNLayer(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        bias=True
                    )
                )
            elif gnn_type == 'gat':
                self.embed_gnn_layers.append(
                    GATLayer(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        heads=1,
                        concat=False,
                        dropout=dropout
                    )
                )
            elif gnn_type == 'sage':
                self.embed_gnn_layers.append(
                    GraphSAGELayer(
                        in_features=hidden_channels,
                        out_features=hidden_channels,
                        aggregator_type="mean",
                        dropout=dropout,
                        use_layer_norm=False,
                        residual=True
                    )
                )
        
        # GNN layers for pooling (assignment matrix generation)
        self.pool_gnn_layers = nn.ModuleList()
        
        # First layer: in_channels -> hidden_channels
        if gnn_type == 'gcn':
            self.pool_gnn_layers.append(
                GCNLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    bias=True
                )
            )
        elif gnn_type == 'gat':
            self.pool_gnn_layers.append(
                GATLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout
                )
            )
        elif gnn_type == 'sage':
            self.pool_gnn_layers.append(
                GraphSAGELayer(
                    in_features=in_channels,
                    out_features=hidden_channels,
                    aggregator_type="mean",
                    dropout=dropout,
                    use_layer_norm=False,
                    residual=True
                )
            )
        
        # Hidden layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            if gnn_type == 'gcn':
                self.pool_gnn_layers.append(
                    GCNLayer(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        bias=True
                    )
                )
            elif gnn_type == 'gat':
                self.pool_gnn_layers.append(
                    GATLayer(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        heads=1,
                        concat=False,
                        dropout=dropout
                    )
                )
            elif gnn_type == 'sage':
                self.pool_gnn_layers.append(
                    GraphSAGELayer(
                        in_features=hidden_channels,
                        out_features=hidden_channels,
                        aggregator_type="mean",
                        dropout=dropout,
                        use_layer_norm=False,
                        residual=True
                    )
                )
        
        # Final pooling projection layer (will be initialized dynamically)
        # This outputs the assignment matrix from hidden_channels
        self.pool_assignment = None
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
    
    def reset_parameters(self):
        """Initialize layer parameters."""
        for gnn in self.embed_gnn_layers:
            if hasattr(gnn, 'reset_parameters'):
                gnn.reset_parameters()
        
        for gnn in self.pool_gnn_layers:
            if hasattr(gnn, 'reset_parameters'):
                gnn.reset_parameters()
        
        if self.pool_assignment is not None:
            nn.init.xavier_uniform_(self.pool_assignment.weight)
            nn.init.zeros_(self.pool_assignment.bias)
        
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for a DiffPool block.
        
        Args:
            x: Node features [num_nodes, in_channels] or [batch_size, num_nodes, in_channels]
            adj: Adjacency matrix [num_nodes, num_nodes] or [batch_size, num_nodes, num_nodes]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            Tuple containing:
                - Pooled node features [num_clusters, out_channels]
                - Pooled adjacency matrix [num_clusters, num_clusters]
                - List of auxiliary losses
        """
        # Apply embedding GNN to get node embeddings
        embed = x
        for i, gnn in enumerate(self.embed_gnn_layers):
            if self.gnn_type in ['gcn', 'gat']:
                # For GCN/GAT, we need to convert adj to edge_index
                if adj.dim() == 2:
                    # Single graph
                    edge_index = torch.nonzero(adj > 0.5).t()
                    embed = gnn(embed, edge_index)
                else:
                    # Batched graphs (this is a simplification)
                    # In practice, would need careful handling of batches
                    batch_embed = []
                    for b in range(adj.size(0)):
                        b_edge_index = torch.nonzero(adj[b] > 0.5).t()
                        batch_embed.append(gnn(embed[b], b_edge_index))
                    embed = torch.stack(batch_embed)
            else:
                # For GraphSAGE, pass adjacency matrix directly
                embed = gnn(embed, adj)
            
            # Apply non-linearity and dropout (except for last layer)
            if i < len(self.embed_gnn_layers) - 1:
                embed = F.relu(embed)
                embed = self.dropout_layer(embed)
        
        # Apply pooling GNN to get assignment matrix
        assign = x
        for i, gnn in enumerate(self.pool_gnn_layers):
            if self.gnn_type in ['gcn', 'gat']:
                # For GCN/GAT, we need to convert adj to edge_index
                if adj.dim() == 2:
                    # Single graph
                    edge_index = torch.nonzero(adj > 0.5).t()
                    assign = gnn(assign, edge_index)
                else:
                    # Batched graphs (this is a simplification)
                    batch_assign = []
                    for b in range(adj.size(0)):
                        b_edge_index = torch.nonzero(adj[b] > 0.5).t()
                        batch_assign.append(gnn(assign[b], b_edge_index))
                    assign = torch.stack(batch_assign)
            else:
                # For GraphSAGE, pass adjacency matrix directly
                assign = gnn(assign, adj)
            
            # Apply non-linearity and dropout
            assign = F.relu(assign)
            assign = self.dropout_layer(assign)
        
        # Determine number of clusters based on pool_ratio
        if adj.dim() == 2:
            # Single graph
            num_nodes = adj.size(0)
        else:
            # Batched graphs - take max nodes across batch
            num_nodes = adj.size(1)
        
        num_clusters = max(1, int(self.pool_ratio * num_nodes))
        
        # Create the final assignment projection layer dynamically
        if self.pool_assignment is None or self.pool_assignment.out_features != num_clusters:
            self.pool_assignment = nn.Linear(
                self.hidden_channels, num_clusters, bias=True
            ).to(adj.device)
            
            # Initialize weights
            nn.init.xavier_uniform_(self.pool_assignment.weight)
            nn.init.zeros_(self.pool_assignment.bias)
        
        # Generate the assignment matrix
        assign = self.pool_assignment(assign)
        assign = F.softmax(assign, dim=-1)
        
        # Calculate auxiliary losses
        aux_losses = []
        
        # Link prediction auxiliary objective
        if self.linkpred:
            # Link prediction loss: how well we reconstruct adj from assign
            if adj.dim() == 2:
                link_pred_loss = self._link_prediction_loss(adj, assign)
                aux_losses.append(link_pred_loss)
            else:
                # Handle batched graphs
                for b in range(adj.size(0)):
                    b_link_loss = self._link_prediction_loss(adj[b], assign[b])
                    aux_losses.append(b_link_loss)
        
        # Entropy regularization to encourage cluster assignment sparsity
        # Each node should belong to a clear cluster, not spread across many
        entropy_loss = self._entropy_loss(assign)
        aux_losses.append(entropy_loss)
        
        # Perform pooling: X_coarse = S^T * X
        # S is the assignment matrix
        if adj.dim() == 2:
            # Single graph
            x_pooled = torch.matmul(assign.t(), embed)
            
            # Also pool adjacency matrix: A_coarse = S^T * A * S
            adj_pooled = torch.matmul(torch.matmul(assign.t(), adj), assign)
        else:
            # Batched graphs
            x_pooled = torch.matmul(assign.transpose(1, 2), embed)
            
            # Pool adjacency matrices batch-wise
            adj_pooled = torch.matmul(
                torch.matmul(assign.transpose(1, 2), adj),
                assign
            )
        
        # Apply final projection
        x_pooled = self.output_proj(x_pooled)
        
        # Apply non-linearity
        x_pooled = F.relu(x_pooled)
        
        return x_pooled, adj_pooled, aux_losses
    
    def _link_prediction_loss(self, adj: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
        """
        Calculate link prediction auxiliary loss.
        
        Args:
            adj: Adjacency matrix [num_nodes, num_nodes]
            assign: Assignment matrix [num_nodes, num_clusters]
            
        Returns:
            Link prediction loss
        """
        # Reconstruction: S * S^T should approximate A
        adj_recon = torch.matmul(assign, assign.t())
        
        # Use Frobenius norm as the loss
        link_loss = torch.norm(adj - adj_recon, p='fro')
        
        # Normalize by number of entries
        link_loss = link_loss / (adj.size(0) * adj.size(1))
        
        return link_loss
    
    def _entropy_loss(self, assign: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy regularization loss.
        
        Args:
            assign: Assignment matrix [num_nodes, num_clusters] or [batch_size, num_nodes, num_clusters]
            
        Returns:
            Entropy loss
        """
        if assign.dim() == 2:
            # Single graph
            # Calculate entropy: -sum(P * log(P))
            entropy = -torch.sum(assign * torch.log(assign + 1e-8), dim=-1).mean()
        else:
            # Batched graphs
            entropy = -torch.sum(assign * torch.log(assign + 1e-8), dim=-1).mean()
        
        return entropy 


class HGTLayer(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) layer.
    
    HGT can model different types of nodes and edges in heterogeneous graphs
    by using type-specific parameters while maintaining weight sharing.
    
    Based on "Heterogeneous Graph Transformer" (Hu et al., WWW 2020)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int, 
        node_types: List[str],
        edge_types: List[str],
        num_heads: int = 8,
        dropout: float = 0.2,
        use_norm: bool = True,
        use_RTE: bool = True,
    ):
        """
        Initialize an HGT layer.
        
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            node_types: List of node types in the heterogeneous graph
            edge_types: List of edge types in the heterogeneous graph
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_norm: Whether to use layer normalization
            use_RTE: Whether to use relation type encoding
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_types = node_types
        self.edge_types = edge_types
        self.num_types = len(node_types)
        self.num_relations = len(edge_types)
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.use_RTE = use_RTE
        
        # Initialize type-specific parameters
        
        # K, Q, V linear projections for each node type
        self.k_linears = nn.ModuleDict()
        self.q_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        
        # Type-specific relation embeddings
        self.relation_k = nn.ParameterDict()
        self.relation_q = nn.ParameterDict()
        self.relation_v = nn.ParameterDict()
        self.relation_att = nn.ParameterDict()
        
        # Layer norm for each node type
        if use_norm:
            self.norms = nn.ModuleDict()
        
        # Relation type encoding
        if use_RTE:
            self.relation_embeds = nn.ParameterDict()
        
        # Output projection for each node type
        self.out_linears = nn.ModuleDict()
        
        # Initialize parameters for each node type
        for node_type in node_types:
            # K, Q, V projections
            self.k_linears[node_type] = nn.Linear(in_dim, out_dim)
            self.q_linears[node_type] = nn.Linear(in_dim, out_dim)
            self.v_linears[node_type] = nn.Linear(in_dim, out_dim)
            
            # Output projection
            self.out_linears[node_type] = nn.Linear(out_dim, out_dim)
            
            # Layer normalization
            if use_norm:
                self.norms[node_type] = nn.LayerNorm(out_dim)
        
        # Initialize parameters for each edge type
        for edge_type in edge_types:
            # Relation-specific transforms
            self.relation_k[edge_type] = nn.Parameter(torch.Tensor(num_heads, self.d_k, self.d_k))
            self.relation_q[edge_type] = nn.Parameter(torch.Tensor(num_heads, self.d_k, self.d_k))
            self.relation_v[edge_type] = nn.Parameter(torch.Tensor(num_heads, self.d_k, self.d_k))
            self.relation_att[edge_type] = nn.Parameter(torch.Tensor(num_heads, self.d_k, 1))
            
            # Relation type encoding
            if use_RTE:
                self.relation_embeds[edge_type] = nn.Parameter(torch.Tensor(num_heads, self.d_k))
        
        # Dropout
        self.drop = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize model parameters."""
        # Initialize linear layers
        for node_type in self.node_types:
            nn.init.xavier_uniform_(self.k_linears[node_type].weight)
            nn.init.xavier_uniform_(self.q_linears[node_type].weight)
            nn.init.xavier_uniform_(self.v_linears[node_type].weight)
            nn.init.xavier_uniform_(self.out_linears[node_type].weight)
            nn.init.zeros_(self.k_linears[node_type].bias)
            nn.init.zeros_(self.q_linears[node_type].bias)
            nn.init.zeros_(self.v_linears[node_type].bias)
            nn.init.zeros_(self.out_linears[node_type].bias)
        
        # Initialize relation-specific parameters
        for edge_type in self.edge_types:
            nn.init.xavier_uniform_(self.relation_k[edge_type])
            nn.init.xavier_uniform_(self.relation_q[edge_type])
            nn.init.xavier_uniform_(self.relation_v[edge_type])
            nn.init.xavier_uniform_(self.relation_att[edge_type])
            if self.use_RTE:
                nn.init.xavier_uniform_(self.relation_embeds[edge_type])
    
    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_index: Dict[str, torch.Tensor],
        node_type_indices: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the HGT layer.
        
        Args:
            node_features: Dictionary of node features for each node type
                {node_type: tensor of shape [num_nodes, in_dim]}
            edge_index: Dictionary of edge indices for each edge type
                {edge_type: tensor of shape [2, num_edges]}
            node_type_indices: Dictionary mapping node indices to their type
                {node_type: tensor of node indices that have this type}
            
        Returns:
            Dictionary of updated node features for each node type
            {node_type: tensor of shape [num_nodes, out_dim]}
        """
        # Dictionary to store updated features for each node type
        updated_features = {}
        
        # Process each target node type
        for target_type in self.node_types:
            # Skip if no nodes of this type
            if target_type not in node_features or node_features[target_type].shape[0] == 0:
                continue
            
            # Query projection for target nodes
            target_nodes = node_features[target_type]
            q = self.q_linears[target_type](target_nodes).view(-1, self.num_heads, self.d_k)
            
            # Initialize output for target nodes
            out = torch.zeros(q.size(0), self.out_dim, device=q.device)
            
            # Process each relation type (source_type, edge_type, target_type)
            for edge_type in self.edge_types:
                # Get edges of this type
                if edge_type not in edge_index or edge_index[edge_type].size(1) == 0:
                    continue
                
                # Get source and target nodes for this edge type
                src, dst = edge_index[edge_type]
                
                # Get source node type for this edge type
                # In a practical implementation, this would be a predefined mapping
                # For demonstration, we'll assume source_type is encoded in edge_type
                source_type = edge_type.split('_')[0]  # Example: 'author_writes_paper' -> 'author'
                
                # Skip if source type not in node features
                if source_type not in node_features or node_features[source_type].shape[0] == 0:
                    continue
                
                # Get source node features
                source_nodes = node_features[source_type]
                
                # Project source nodes to key and value space
                k = self.k_linears[source_type](source_nodes).view(-1, self.num_heads, self.d_k)
                v = self.v_linears[source_type](source_nodes).view(-1, self.num_heads, self.d_k)
                
                # Apply relation-specific transformations
                # [n_heads, d_k, d_k] x [n_nodes, n_heads, d_k] -> [n_nodes, n_heads, d_k]
                k = torch.einsum('hdk,nhd->nhd', self.relation_k[edge_type], k)
                v = torch.einsum('hdk,nhd->nhd', self.relation_v[edge_type], v)
                
                # Get target node queries for this edge type
                # Map dst indices to indices in the target_type features
                # This mapping is complex and depends on how node indices are organized
                # Here we assume dst indices directly map to target nodes
                q_dst = q[dst]
                
                # Transform queries with relation-specific parameters
                q_dst = torch.einsum('hdk,nhd->nhd', self.relation_q[edge_type], q_dst)
                
                # Add relation type encoding if enabled
                if self.use_RTE:
                    q_dst = q_dst + self.relation_embeds[edge_type].unsqueeze(0)
                
                # Calculate attention scores
                # [n_edges, n_heads, d_k] x [n_heads, d_k, 1] -> [n_edges, n_heads, 1]
                rel_att = torch.einsum('nhl,hla->nha', k[src], self.relation_att[edge_type])
                att = (q_dst * k[src]).sum(dim=-1, keepdim=True) / self.sqrt_dk + rel_att
                
                # Apply softmax to get attention weights
                # Note: In a real implementation, this would be normalized per target node
                # For simplicity, we'll just apply softmax along all edges
                att = F.softmax(att, dim=0)
                att = self.drop(att)
                
                # Apply attention weights to source node values
                # [n_edges, n_heads, 1] x [n_edges, n_heads, d_k] -> [n_edges, n_heads, d_k]
                h = att * v[src]
                
                # Reshape for output projection
                h = h.view(-1, self.out_dim)
                
                # Add to target node features using scatter_add
                # This aggregates messages from all sources to each target
                # For simplicity, we'll use a dense implementation
                out_dst = torch.zeros(
                    target_nodes.size(0), self.out_dim, device=h.device
                )
                for i in range(dst.size(0)):
                    out_dst[dst[i]] += h[i]
                
                # Add to output
                out += out_dst
            
            # Apply output projection
            out = self.out_linears[target_type](out)
            
            # Apply layer norm if enabled
            if self.use_norm:
                out = self.norms[target_type](out)
            
            # Apply non-linearity and dropout
            out = F.gelu(out)
            out = self.drop(out)
            
            # Store updated features
            updated_features[target_type] = out
        
        return updated_features


class HGT(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) model.
    
    HGT can model different types of nodes and edges in heterogeneous graphs
    through type-dependent parameters while preserving weight sharing.
    
    Based on "Heterogeneous Graph Transformer" (Hu et al., WWW 2020)
    """
    
    def __init__(
        self,
        in_dim: Dict[str, int],
        hidden_dim: int,
        out_dim: Dict[str, int],
        node_types: List[str],
        edge_types: List[str],
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
        use_norm: bool = True,
        use_RTE: bool = True,
    ):
        """
        Initialize a Heterogeneous Graph Transformer model.
        
        Args:
            in_dim: Dictionary of input dimensions for each node type
            hidden_dim: Hidden dimension
            out_dim: Dictionary of output dimensions for each node type
            node_types: List of node types in the heterogeneous graph
            edge_types: List of edge types in the heterogeneous graph
            num_layers: Number of HGT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_norm: Whether to use layer normalization
            use_RTE: Whether to use relation type encoding
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.node_types = node_types
        self.edge_types = edge_types
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_norm = use_norm
        self.use_RTE = use_RTE
        
        # Embedding layers for each node type (to project to hidden_dim)
        self.node_embeddings = nn.ModuleDict()
        for node_type in node_types:
            if node_type in in_dim:
                self.node_embeddings[node_type] = nn.Linear(in_dim[node_type], hidden_dim)
        
        # HGT layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                HGTLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    node_types=node_types,
                    edge_types=edge_types,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_norm=use_norm,
                    use_RTE=use_RTE
                )
            )
        
        # Output projections for each node type
        self.out_projs = nn.ModuleDict()
        for node_type in node_types:
            if node_type in out_dim:
                self.out_projs[node_type] = nn.Linear(hidden_dim, out_dim[node_type])
        
        # Dropout
        self.drop = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize model parameters."""
        # Initialize embeddings
        for node_type in self.node_types:
            if node_type in self.node_embeddings:
                nn.init.xavier_uniform_(self.node_embeddings[node_type].weight)
                nn.init.zeros_(self.node_embeddings[node_type].bias)
        
        # Initialize HGT layers
        for layer in self.layers:
            layer.reset_parameters()
        
        # Initialize output projections
        for node_type in self.node_types:
            if node_type in self.out_projs:
                nn.init.xavier_uniform_(self.out_projs[node_type].weight)
                nn.init.zeros_(self.out_projs[node_type].bias)
    
    def forward(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_index: Dict[str, torch.Tensor],
        node_type_indices: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the HGT model.
        
        Args:
            node_features: Dictionary of node features for each node type
                {node_type: tensor of shape [num_nodes, in_dim]}
            edge_index: Dictionary of edge indices for each edge type
                {edge_type: tensor of shape [2, num_edges]}
            node_type_indices: Dictionary mapping node indices to their type
                {node_type: tensor of node indices that have this type}
            
        Returns:
            Dictionary of output node features for each node type
            {node_type: tensor of shape [num_nodes, out_dim]}
        """
        # Project input features to hidden dimension
        h = {}
        for node_type, features in node_features.items():
            if node_type in self.node_embeddings and features.size(0) > 0:
                h[node_type] = self.node_embeddings[node_type](features)
                h[node_type] = F.relu(h[node_type])
                h[node_type] = self.drop(h[node_type])
            else:
                h[node_type] = features
        
        # Apply HGT layers
        for layer in self.layers:
            h = layer(h, edge_index, node_type_indices)
        
        # Apply output projections
        out = {}
        for node_type, features in h.items():
            if node_type in self.out_projs:
                out[node_type] = self.out_projs[node_type](features)
            else:
                out[node_type] = features
        
        return out