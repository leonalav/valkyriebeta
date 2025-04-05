"""
Graph Neural Network model implementation for ValkyrieLLM.

This module implements a complete GNN model that can be used for graph-based reasoning
in the ValkyrieLLM framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from .layers import GraphConvolution, GraphAttention, GraphSAGELayer


class GNNEncoder(nn.Module):
    """
    Multi-layer GNN encoder model that processes graph-structured data.
    
    This class stacks multiple GNN layers to create a deep graph neural network
    for encoding graph-structured information.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        gnn_type: str = "gat",
        dropout: float = 0.1,
        residual: bool = True,
        layer_norm: bool = True,
        activation: nn.Module = nn.ReLU(),
        num_heads: int = 8,
        aggregator_type: str = "mean",
        readout_type: str = "mean",
        use_edge_features: bool = False,
        edge_dim: Optional[int] = None,
        concat_heads: bool = True,
        normalize: bool = True
    ):
        """
        Initialize a GNN encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for GNN layers
            output_dim: Output dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layers ('gcn', 'gat', 'graphsage')
            dropout: Dropout probability
            residual: Whether to use residual connections
            layer_norm: Whether to use layer normalization
            activation: Activation function for GNN layers
            num_heads: Number of attention heads for GAT
            aggregator_type: Aggregator type for GraphSAGE
            readout_type: Type of readout function ('mean', 'sum', 'max', 'attention')
            use_edge_features: Whether to use edge features
            edge_dim: Edge feature dimension (if used)
            concat_heads: Whether to concatenate attention heads in GAT
            normalize: Whether to normalize node embeddings
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.activation = activation
        self.readout_type = readout_type.lower()
        self.use_edge_features = use_edge_features
        
        # Validate GNN type
        valid_gnn_types = ["gcn", "gat", "graphsage"]
        if gnn_type not in valid_gnn_types:
            raise ValueError(f"Invalid GNN type: {gnn_type}. Must be one of {valid_gnn_types}")
        
        # Validate readout type
        valid_readout_types = ["mean", "sum", "max", "attention"]
        if readout_type not in valid_readout_types:
            raise ValueError(f"Invalid readout type: {readout_type}. Must be one of {valid_readout_types}")
        
        # Create GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # Input projection layer
        if gnn_type == "gcn":
            self.gnn_layers.append(
                GraphConvolution(
                    in_features=input_dim,
                    out_features=hidden_dim,
                    use_bias=True,
                    dropout=dropout,
                    activation=activation,
                    use_layer_norm=layer_norm,
                    residual=False  # No residual for first layer as dimensions don't match
                )
            )
        elif gnn_type == "gat":
            self.gnn_layers.append(
                GraphAttention(
                    in_features=input_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    concat_heads=concat_heads,
                    dropout=dropout,
                    use_bias=True,
                    use_layer_norm=layer_norm,
                    residual=False  # No residual for first layer as dimensions don't match
                )
            )
        elif gnn_type == "graphsage":
            self.gnn_layers.append(
                GraphSAGELayer(
                    in_features=input_dim,
                    out_features=hidden_dim,
                    aggregator_type=aggregator_type,
                    use_bias=True,
                    dropout=dropout,
                    activation=activation,
                    use_layer_norm=layer_norm,
                    residual=False,  # No residual for first layer as dimensions don't match
                    normalize=normalize
                )
            )
        
        # Hidden layers
        for i in range(num_layers - 2):
            if gnn_type == "gcn":
                self.gnn_layers.append(
                    GraphConvolution(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        use_bias=True,
                        dropout=dropout,
                        activation=activation,
                        use_layer_norm=layer_norm,
                        residual=residual
                    )
                )
            elif gnn_type == "gat":
                self.gnn_layers.append(
                    GraphAttention(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        num_heads=num_heads,
                        concat_heads=concat_heads,
                        dropout=dropout,
                        use_bias=True,
                        use_layer_norm=layer_norm,
                        residual=residual
                    )
                )
            elif gnn_type == "graphsage":
                self.gnn_layers.append(
                    GraphSAGELayer(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        aggregator_type=aggregator_type,
                        use_bias=True,
                        dropout=dropout,
                        activation=activation,
                        use_layer_norm=layer_norm,
                        residual=residual,
                        normalize=normalize
                    )
                )
        
        # Output layer
        if gnn_type == "gcn":
            self.gnn_layers.append(
                GraphConvolution(
                    in_features=hidden_dim,
                    out_features=output_dim,
                    use_bias=True,
                    dropout=dropout,
                    activation=nn.Identity(),  # No activation in the final layer
                    use_layer_norm=False,  # No layer norm in the final layer
                    residual=False  # No residual in the final layer
                )
            )
        elif gnn_type == "gat":
            self.gnn_layers.append(
                GraphAttention(
                    in_features=hidden_dim,
                    out_features=output_dim,
                    num_heads=1,  # Single head for output layer
                    concat_heads=False,
                    dropout=dropout,
                    use_bias=True,
                    use_layer_norm=False,  # No layer norm in the final layer
                    residual=False  # No residual in the final layer
                )
            )
        elif gnn_type == "graphsage":
            self.gnn_layers.append(
                GraphSAGELayer(
                    in_features=hidden_dim,
                    out_features=output_dim,
                    aggregator_type=aggregator_type,
                    use_bias=True,
                    dropout=dropout,
                    activation=nn.Identity(),  # No activation in the final layer
                    use_layer_norm=False,  # No layer norm in the final layer
                    residual=False,  # No residual in the final layer
                    normalize=normalize
                )
            )
        
        # Edge feature projection (if used)
        if use_edge_features and edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # Attention readout parameters (if used)
        if readout_type == "attention":
            self.readout_attention = nn.Sequential(
                nn.Linear(output_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False)
            )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        return_node_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the GNN encoder.
        
        Args:
            node_features: Node feature matrix [batch_size, num_nodes, input_dim]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            node_mask: Node mask [batch_size, num_nodes] (1 for valid nodes, 0 for padding)
            edge_features: Optional edge features [batch_size, num_nodes, num_nodes, edge_dim]
            return_node_embeddings: Whether to return node embeddings along with graph embedding
            
        Returns:
            If return_node_embeddings is False:
                Graph embedding [batch_size, output_dim]
            If return_node_embeddings is True:
                (Graph embedding [batch_size, output_dim], Node embeddings [batch_size, num_nodes, output_dim])
        """
        # Process edge features if available
        if self.use_edge_features and edge_features is not None:
            edge_proj = self.edge_proj(edge_features)
        else:
            edge_proj = None
        
        # Apply GNN layers
        x = node_features
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.gnn_type == "gcn" or self.gnn_type == "graphsage":
                x = gnn_layer(x, adj_matrix)
            elif self.gnn_type == "gat":
                # For GAT, if we have edge features, we could use them here
                # This would require a custom GAT implementation that accepts edge features
                x = gnn_layer(x, adj_matrix)
            
            # Apply dropout between layers (but not after the last layer)
            if i < len(self.gnn_layers) - 1:
                x = self.dropout_layer(x)
        
        # Apply node mask if provided
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1)
        
        # Readout to get graph-level embedding
        if self.readout_type == "mean":
            if node_mask is not None:
                # Masked mean
                graph_embedding = x.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                # Simple mean
                graph_embedding = x.mean(dim=1)
        elif self.readout_type == "sum":
            graph_embedding = x.sum(dim=1)
        elif self.readout_type == "max":
            if node_mask is not None:
                # Masked max
                mask = (1 - node_mask).unsqueeze(-1) * -1e9
                graph_embedding, _ = (x + mask).max(dim=1)
            else:
                # Simple max
                graph_embedding, _ = x.max(dim=1)
        elif self.readout_type == "attention":
            # Attention-based readout
            attention_scores = self.readout_attention(x)  # [batch_size, num_nodes, 1]
            
            if node_mask is not None:
                # Mask attention scores for padding nodes
                attention_scores = attention_scores.masked_fill((node_mask == 0).unsqueeze(-1), -1e9)
            
            attention_weights = F.softmax(attention_scores, dim=1)
            graph_embedding = torch.sum(attention_weights * x, dim=1)
        
        if return_node_embeddings:
            return graph_embedding, x
        else:
            return graph_embedding


class GraphTransformer(nn.Module):
    """
    Graph Transformer model that combines GNN with transformer-like attention.
    
    This model processes graph-structured data using a combination of GNN layers
    for local graph structure and transformer-like attention for global interactions.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_gnn_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        gnn_type: str = "gat",
        use_edge_features: bool = False,
        edge_dim: Optional[int] = None,
        aggregator_type: str = "mean",
        readout_type: str = "attention",
        use_global_pool: bool = True,
        layer_norm: bool = True,
        activation: nn.Module = nn.GELU(),
        normalize: bool = True
    ):
        """
        Initialize a Graph Transformer model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for both GNN and transformer
            output_dim: Output dimension
            num_gnn_layers: Number of GNN layers
            num_transformer_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            gnn_type: Type of GNN layers ('gcn', 'gat', 'graphsage')
            use_edge_features: Whether to use edge features
            edge_dim: Edge feature dimension (if used)
            aggregator_type: Aggregator type for GraphSAGE
            readout_type: Type of readout function ('mean', 'sum', 'max', 'attention')
            use_global_pool: Whether to use global pooling for final output
            layer_norm: Whether to use layer normalization
            activation: Activation function
            normalize: Whether to normalize node embeddings
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.use_edge_features = use_edge_features
        self.readout_type = readout_type
        self.use_global_pool = use_global_pool
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers for local graph structure
        self.gnn = GNNEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_gnn_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            residual=True,
            layer_norm=layer_norm,
            activation=activation,
            num_heads=num_heads,
            aggregator_type=aggregator_type,
            readout_type=readout_type,
            use_edge_features=use_edge_features,
            edge_dim=edge_dim,
            normalize=normalize
        )
        
        # Transformer layers for global interactions
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_transformer_layers)
        ])
        
        # Readout layers
        if use_global_pool:
            if readout_type == "attention":
                self.global_pool = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1, bias=False)
                )
            else:
                self.global_pool = None
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        return_node_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the Graph Transformer model.
        
        Args:
            node_features: Node feature matrix [batch_size, num_nodes, input_dim]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            node_mask: Node mask [batch_size, num_nodes] (1 for valid nodes, 0 for padding)
            edge_features: Optional edge features [batch_size, num_nodes, num_nodes, edge_dim]
            return_node_embeddings: Whether to return node embeddings along with graph embedding
            
        Returns:
            If return_node_embeddings is False:
                Graph embedding [batch_size, output_dim]
            If return_node_embeddings is True:
                (Graph embedding [batch_size, output_dim], Node embeddings [batch_size, num_nodes, output_dim])
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Project input features
        x = self.input_proj(node_features)
        
        # Apply GNN layers to capture local graph structure
        if return_node_embeddings:
            # Get both graph and node embeddings
            _, x = self.gnn(
                node_features=x,
                adj_matrix=adj_matrix,
                node_mask=node_mask,
                edge_features=edge_features,
                return_node_embeddings=True
            )
        else:
            # Get only node embeddings
            x = self.gnn(
                node_features=x,
                adj_matrix=adj_matrix,
                node_mask=node_mask,
                edge_features=edge_features,
                return_node_embeddings=True
            )[1]  # Take node embeddings
        
        # Apply layer normalization if used
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        # Create attention mask from node mask
        if node_mask is not None:
            # 1 = no attention, 0 = attention
            attn_mask = (1 - node_mask).bool().unsqueeze(1).unsqueeze(2)
            # Expand to [batch_size, 1, num_nodes, num_nodes]
            attn_mask = attn_mask.expand(-1, self.num_heads, -1, -1)
            # Reshape to [batch_size * num_heads, num_nodes, num_nodes]
            attn_mask = attn_mask.contiguous().view(batch_size * self.num_heads, num_nodes, num_nodes)
        else:
            attn_mask = None
        
        # Apply transformer layers for global interactions
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, src_key_padding_mask=(node_mask == 0) if node_mask is not None else None)
        
        # Global pooling
        if self.use_global_pool:
            if self.readout_type == "mean":
                if node_mask is not None:
                    # Masked mean
                    graph_embedding = x.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
                else:
                    # Simple mean
                    graph_embedding = x.mean(dim=1)
            elif self.readout_type == "sum":
                graph_embedding = x.sum(dim=1)
            elif self.readout_type == "max":
                if node_mask is not None:
                    # Masked max
                    mask = (1 - node_mask).unsqueeze(-1) * -1e9
                    graph_embedding, _ = (x + mask).max(dim=1)
                else:
                    # Simple max
                    graph_embedding, _ = x.max(dim=1)
            elif self.readout_type == "attention":
                # Attention-based readout
                attention_scores = self.global_pool(x)  # [batch_size, num_nodes, 1]
                
                if node_mask is not None:
                    # Mask attention scores for padding nodes
                    attention_scores = attention_scores.masked_fill((node_mask == 0).unsqueeze(-1), -1e9)
                
                attention_weights = F.softmax(attention_scores, dim=1)
                graph_embedding = torch.sum(attention_weights * x, dim=1)
            
            # Apply output projection
            output = self.output_proj(graph_embedding)
        else:
            # No global pooling, project each node embedding
            output = self.output_proj(x)
        
        if return_node_embeddings:
            return output, x
        else:
            return output


class GraphNodeClassifier(nn.Module):
    """
    Graph Neural Network model for node classification tasks.
    
    This model encodes the graph structure and node features to predict
    labels for nodes.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 3,
        gnn_type: str = "gat",
        dropout: float = 0.1,
        layer_norm: bool = True,
        residual: bool = True,
        num_heads: int = 8,
        aggregator_type: str = "mean",
        use_edge_features: bool = False,
        edge_dim: Optional[int] = None,
        normalize: bool = True,
        activation: nn.Module = nn.ReLU()
    ):
        """
        Initialize a node classification GNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for GNN layers
            num_classes: Number of output classes
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layers ('gcn', 'gat', 'graphsage')
            dropout: Dropout probability
            layer_norm: Whether to use layer normalization
            residual: Whether to use residual connections
            num_heads: Number of attention heads for GAT
            aggregator_type: Aggregator type for GraphSAGE
            use_edge_features: Whether to use edge features
            edge_dim: Edge feature dimension (if used)
            normalize: Whether to normalize node embeddings
            activation: Activation function
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # GNN layers
        self.gnn = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # No need for classification head yet
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            residual=residual,
            layer_norm=layer_norm,
            activation=activation,
            num_heads=num_heads,
            aggregator_type=aggregator_type,
            readout_type="none",  # No readout needed for node classification
            use_edge_features=use_edge_features,
            edge_dim=edge_dim,
            normalize=normalize
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the node classification model.
        
        Args:
            node_features: Node feature matrix [batch_size, num_nodes, input_dim]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            node_mask: Node mask [batch_size, num_nodes] (1 for valid nodes, 0 for padding)
            edge_features: Optional edge features [batch_size, num_nodes, num_nodes, edge_dim]
            
        Returns:
            Node classification logits [batch_size, num_nodes, num_classes]
        """
        # Get node embeddings from GNN
        _, node_embeddings = self.gnn(
            node_features=node_features,
            adj_matrix=adj_matrix,
            node_mask=node_mask,
            edge_features=edge_features,
            return_node_embeddings=True
        )
        
        # Apply classification head
        logits = self.classifier(node_embeddings)
        
        # Apply node mask if provided
        if node_mask is not None:
            logits = logits * node_mask.unsqueeze(-1)
        
        return logits


class GraphLinkPredictor(nn.Module):
    """
    Graph Neural Network model for link prediction tasks.
    
    This model encodes the graph structure and node features to predict
    the likelihood of edges between nodes.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        edge_dim: Optional[int] = None,
        num_layers: int = 3,
        gnn_type: str = "gat",
        dropout: float = 0.1,
        layer_norm: bool = True,
        residual: bool = True,
        num_heads: int = 8,
        aggregator_type: str = "mean",
        scoring_type: str = "dot",
        normalize: bool = True,
        activation: nn.Module = nn.ReLU()
    ):
        """
        Initialize a link prediction GNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for GNN layers
            edge_dim: Edge feature dimension (if used)
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layers ('gcn', 'gat', 'graphsage')
            dropout: Dropout probability
            layer_norm: Whether to use layer normalization
            residual: Whether to use residual connections
            num_heads: Number of attention heads for GAT
            aggregator_type: Aggregator type for GraphSAGE
            scoring_type: Type of scoring function ('dot', 'bilinear', 'mlp')
            normalize: Whether to normalize node embeddings
            activation: Activation function
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.scoring_type = scoring_type.lower()
        
        # Validate scoring type
        valid_scoring_types = ["dot", "bilinear", "mlp"]
        if scoring_type not in valid_scoring_types:
            raise ValueError(f"Invalid scoring type: {scoring_type}. Must be one of {valid_scoring_types}")
        
        # GNN layers
        self.gnn = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            residual=residual,
            layer_norm=layer_norm,
            activation=activation,
            num_heads=num_heads,
            aggregator_type=aggregator_type,
            readout_type="none",  # No readout needed for link prediction
            use_edge_features=edge_dim is not None,
            edge_dim=edge_dim,
            normalize=normalize
        )
        
        # Scoring function
        if scoring_type == "bilinear":
            self.scorer = nn.Bilinear(hidden_dim, hidden_dim, 1)
        elif scoring_type == "mlp":
            self.scorer = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )
    
    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the link prediction model.
        
        Args:
            node_features: Node feature matrix [batch_size, num_nodes, input_dim]
            adj_matrix: Adjacency matrix [batch_size, num_nodes, num_nodes]
            edge_index: Edge indices to predict [batch_size, num_edges, 2]
            node_mask: Node mask [batch_size, num_nodes] (1 for valid nodes, 0 for padding)
            edge_features: Optional edge features [batch_size, num_nodes, num_nodes, edge_dim]
            
        Returns:
            Edge prediction scores [batch_size, num_edges]
        """
        batch_size, num_nodes, _ = node_features.shape
        num_edges = edge_index.size(1)
        
        # Get node embeddings from GNN
        _, node_embeddings = self.gnn(
            node_features=node_features,
            adj_matrix=adj_matrix,
            node_mask=node_mask,
            edge_features=edge_features,
            return_node_embeddings=True
        )
        
        # Extract source and target node embeddings
        s_idx = edge_index[:, :, 0].unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        t_idx = edge_index[:, :, 1].unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        
        # [batch_size, num_edges, hidden_dim]
        source_embeddings = torch.gather(node_embeddings, 1, s_idx)
        target_embeddings = torch.gather(node_embeddings, 1, t_idx)
        
        # Compute edge scores based on scoring type
        if self.scoring_type == "dot":
            # [batch_size, num_edges]
            scores = torch.sum(source_embeddings * target_embeddings, dim=-1)
        elif self.scoring_type == "bilinear":
            # [batch_size, num_edges, 1]
            scores = self.scorer(source_embeddings, target_embeddings)
            scores = scores.squeeze(-1)  # [batch_size, num_edges]
        elif self.scoring_type == "mlp":
            # [batch_size, num_edges, 2*hidden_dim]
            concat_embeddings = torch.cat([source_embeddings, target_embeddings], dim=-1)
            # [batch_size, num_edges, 1]
            scores = self.scorer(concat_embeddings)
            scores = scores.squeeze(-1)  # [batch_size, num_edges]
        
        return scores 