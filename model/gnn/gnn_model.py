import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

class GraphConvolutionLayer(nn.Module):
    """
    Graph Convolutional Network Layer
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_edge_features: bool = False,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        activation: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_edge_features = use_edge_features
        
        # Linear transformation for node features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        
        # Edge feature transformation if edge features are used
        if use_edge_features:
            edge_dim = edge_dim or in_features
            self.edge_weight = nn.Parameter(torch.Tensor(edge_dim, out_features))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Activation function
        self.activation = activation
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights and bias"""
        nn.init.xavier_uniform_(self.weight)
        if self.use_edge_features:
            nn.init.xavier_uniform_(self.edge_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] or None
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Get source and target nodes
        src, dst = edge_index
        
        # Transform node features
        x = torch.matmul(x, self.weight)
        
        # Aggregate messages from neighbors
        out = torch.zeros_like(x)
        
        # For each edge, update target node with source node features
        for i, (s, d) in enumerate(zip(src, dst)):
            # Apply edge features if available
            if self.use_edge_features and edge_attr is not None:
                # Weight message by edge features
                edge_weight = torch.matmul(edge_attr[i].unsqueeze(0), self.edge_weight)
                message = x[s] * edge_weight
            else:
                message = x[s]
                
            out[d] += message
            
        # Apply bias
        if self.bias is not None:
            out += self.bias
            
        # Apply activation
        if self.activation is not None:
            out = self.activation(out)
            
        return out


class GATLayer(nn.Module):
    """
    Graph Attention Network Layer
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        concat: bool = True,
        dropout: float = 0.1,
        use_edge_features: bool = False,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        activation: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        
        # Whether to concatenate or average multi-head outputs
        self.head_dim = out_features // num_heads if concat else out_features
        
        # Node feature projections for each attention head
        self.W = nn.Parameter(torch.Tensor(num_heads, in_features, self.head_dim))
        
        # Attention parameters for each head
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * self.head_dim, 1))
        
        # Edge feature projection if used
        if use_edge_features:
            edge_dim = edge_dim or in_features
            self.edge_proj = nn.Parameter(torch.Tensor(num_heads, edge_dim, self.head_dim))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        else:
            self.register_parameter('bias', None)
            
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.feat_dropout = nn.Dropout(dropout)
        
        # Activation
        self.activation = activation or nn.LeakyReLU(0.2)
        self.output_activation = nn.ELU()
        
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights and bias"""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
        if self.use_edge_features:
            nn.init.xavier_uniform_(self.edge_proj)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] or None
            
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Get dimensions
        num_nodes = x.size(0)
        
        # Transform input features for each attention head
        # [num_nodes, in_features] -> [num_heads, num_nodes, head_dim]
        Wh = torch.einsum('hid,nf->hnd', self.W, x)
        
        # Get source and target nodes
        src, dst = edge_index
        
        # Compute attention scores
        # 1. Get node features for each endpoint of each edge
        Wh_src = Wh[:, src]  # [num_heads, num_edges, head_dim]
        Wh_dst = Wh[:, dst]  # [num_heads, num_edges, head_dim]
        
        # 2. Concatenate source and target features
        # [num_heads, num_edges, 2*head_dim]
        cat_features = torch.cat([Wh_src, Wh_dst], dim=2)
        
        # 3. Compute attention scores
        # [num_heads, num_edges, 1]
        e = torch.einsum('hef,hfl->hel', cat_features, self.a)
        
        # Apply edge features if available
        if self.use_edge_features and edge_attr is not None:
            # Project edge features
            # [num_edges, edge_dim] -> [num_heads, num_edges, head_dim]
            edge_features = torch.einsum('ef,hfl->hel', edge_attr, self.edge_proj)
            
            # Add edge features to attention scores
            # Simple approach: just add the sum of edge features
            edge_sum = edge_features.sum(dim=2, keepdim=True)
            e = e + edge_sum
        
        # Apply activation to attention scores
        e = self.activation(e)
        
        # Apply softmax to normalize attention scores
        alpha = F.softmax(e, dim=1)
        
        # Apply dropout to attention scores
        alpha = self.attn_dropout(alpha)
        
        # Apply attention to node features
        out = torch.zeros(self.num_heads, num_nodes, self.head_dim, device=x.device)
        
        # For each edge, update target node with weighted source node features
        for i, (s, d) in enumerate(zip(src, dst)):
            out[:, d] += Wh[:, s] * alpha[:, i]
        
        # Apply bias
        if self.bias is not None:
            out += self.bias.unsqueeze(1)
            
        # Apply dropout to output features
        out = self.feat_dropout(out)
        
        # Apply output activation
        out = self.output_activation(out)
        
        # Combine attention heads
        if self.concat:
            # Concatenate attention heads
            out = out.transpose(0, 1).reshape(num_nodes, -1)
        else:
            # Average attention heads
            out = out.mean(dim=0)
            
        return out


class GNNEncoder(nn.Module):
    """
    Graph Neural Network Encoder
    
    This module implements a configurable GNN encoder that can use different types
    of graph neural network layers (GCN, GAT, etc.)
    """
    
    def __init__(
        self,
        hidden_size: int,
        gnn_hidden_size: Optional[int] = None,
        gnn_type: str = "gcn",
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        residual: bool = True,
        layer_norm: bool = True,
        pooling_type: str = "mean"
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.gnn_hidden_size = gnn_hidden_size or hidden_size
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.use_edge_features = use_edge_features
        self.residual = residual
        self.layer_norm = layer_norm
        self.pooling_type = pooling_type
        
        # Initial projection
        self.input_projection = nn.Linear(hidden_size, self.gnn_hidden_size)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type.lower() == "gcn":
                layer = GraphConvolutionLayer(
                    in_features=self.gnn_hidden_size,
                    out_features=self.gnn_hidden_size,
                    use_edge_features=use_edge_features,
                    bias=True,
                    activation=nn.ReLU()
                )
            elif gnn_type.lower() == "gat":
                layer = GATLayer(
                    in_features=self.gnn_hidden_size,
                    out_features=self.gnn_hidden_size,
                    num_heads=num_heads,
                    concat=False,  # Use averaging to maintain dimensions
                    dropout=dropout,
                    use_edge_features=use_edge_features,
                    bias=True
                )
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
                
            self.gnn_layers.append(layer)
            
        # Layer normalization
        if layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(self.gnn_hidden_size) for _ in range(num_layers)
            ])
            
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Linear(self.gnn_hidden_size, hidden_size)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            node_features: Node features [num_nodes, hidden_size]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] or None
            batch: Batch indices for each node [num_nodes]
            
        Returns:
            Encoded node features [num_nodes, hidden_size]
        """
        # Apply input projection
        x = self.input_projection(node_features)
        
        # Apply GNN layers
        for i in range(self.num_layers):
            # Store original for residual connection
            residual = x
            
            # Apply GNN layer
            x = self.gnn_layers[i](x, edge_index, edge_attr)
            
            # Apply dropout
            x = self.dropout(x)
            
            # Apply residual connection if enabled
            if self.residual and i > 0:
                x = x + residual
                
            # Apply layer normalization if enabled
            if self.layer_norm:
                x = self.layer_norms[i](x)
                
        # Apply output projection
        x = self.output_projection(x)
        
        # Apply global pooling if requested
        if self.pooling_type is not None and batch is not None:
            if self.pooling_type == "mean":
                x = self._global_mean_pool(x, batch)
            elif self.pooling_type == "max":
                x = self._global_max_pool(x, batch)
            elif self.pooling_type == "sum":
                x = self._global_add_pool(x, batch)
                
        return x
    
    def _global_mean_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Mean pooling across nodes in the same graph"""
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device)
        count = torch.zeros(num_graphs, device=x.device)
        
        # Sum node features for each graph
        for i in range(x.size(0)):
            out[batch[i]] += x[i]
            count[batch[i]] += 1
            
        # Avoid division by zero
        count = torch.max(count, torch.ones_like(count))
        
        # Compute mean
        return out / count.unsqueeze(1)
    
    def _global_max_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Max pooling across nodes in the same graph"""
        num_graphs = batch.max().item() + 1
        out = torch.full((num_graphs, x.size(1)), float('-inf'), device=x.device)
        
        # Find max node features for each graph
        for i in range(x.size(0)):
            out[batch[i]] = torch.max(out[batch[i]], x[i])
            
        return out
    
    def _global_add_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Sum pooling across nodes in the same graph"""
        num_graphs = batch.max().item() + 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device)
        
        # Sum node features for each graph
        for i in range(x.size(0)):
            out[batch[i]] += x[i]
            
        return out 