import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

class GraphEncoder(nn.Module):
    """
    Graph encoder that converts text representations into a graph structure
    and encodes the graph with GNN layers.
    """
    
    def __init__(
        self,
        hidden_size: int,
        gnn_hidden_size: Optional[int] = None,
        num_gnn_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        max_nodes: int = 512,
        graph_construction: str = "dynamic"
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.gnn_hidden_size = gnn_hidden_size or hidden_size
        self.num_heads = num_heads
        self.num_gnn_layers = num_gnn_layers
        self.use_edge_features = use_edge_features
        self.max_nodes = max_nodes
        self.graph_construction = graph_construction
        
        # Initial projection
        self.input_projection = nn.Linear(hidden_size, self.gnn_hidden_size)
        
        # Graph construction layers
        if graph_construction == "dynamic":
            self.attention = nn.MultiheadAttention(
                embed_dim=self.gnn_hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Edge feature projection if using edge features
            if use_edge_features:
                self.edge_projection = nn.Linear(self.gnn_hidden_size, self.gnn_hidden_size)
        
        # Node and edge feature dropout
        self.node_dropout = nn.Dropout(dropout)
        self.edge_dropout = nn.Dropout(dropout) if use_edge_features else None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.gnn_hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode text representations into graph features
        
        Args:
            hidden_states: Transformer hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            edge_index: Optional predefined edge indices [2, num_edges]
            edge_attr: Optional predefined edge attributes [num_edges, edge_dim]
            
        Returns:
            Tuple of (node_features, edge_index, edge_attr)
            - node_features: Node features [batch_size, num_nodes, gnn_hidden_size]
            - edge_index: Edge indices [2, num_edges]
            - edge_attr: Edge attributes [num_edges, edge_dim] or None
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Limit the number of nodes if needed
        if seq_len > self.max_nodes:
            hidden_states = hidden_states[:, :self.max_nodes, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_nodes]
            seq_len = self.max_nodes
        
        # Project input to GNN dimension
        node_features = self.input_projection(hidden_states)
        
        # Apply node dropout
        node_features = self.node_dropout(node_features)
        
        # Construct graph if edge_index not provided
        if edge_index is None and self.graph_construction == "dynamic":
            # Use attention to compute edge weights
            attn_output, attn_weights = self.attention(
                query=node_features,
                key=node_features,
                value=node_features,
                key_padding_mask=attention_mask.logical_not() if attention_mask is not None else None
            )
            
            # Convert attention weights to edge index and attributes
            edge_index, edge_attr = self._attention_to_graph(attn_weights, attention_mask)
            
        elif edge_index is None:
            # Default to fully connected graph
            edge_index = self._create_fully_connected_graph(seq_len, batch_size, hidden_states.device)
            edge_attr = None
        
        # Apply edge feature dropout if using edge features
        if self.use_edge_features and edge_attr is not None and self.edge_dropout is not None:
            edge_attr = self.edge_dropout(edge_attr)
        
        # Apply layer normalization to node features
        node_features = self.layer_norm(node_features)
        
        return node_features, edge_index, edge_attr
    
    def _attention_to_graph(
        self,
        attention_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert attention weights to graph structure
        
        Args:
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        device = attention_weights.device
        
        # Average across attention heads
        avg_attention = attention_weights.mean(dim=1)  # [batch_size, seq_len, seq_len]
        
        # Apply threshold to create sparse graph (optional)
        # Here we just keep edges with attention scores in the top 50%
        k = seq_len // 2
        topk_values, topk_indices = torch.topk(avg_attention, k=k, dim=-1)
        
        # Create sparse adjacency matrix
        sparse_adj = torch.zeros_like(avg_attention)
        
        # For each node, set its top-k connections
        for b in range(batch_size):
            for i in range(seq_len):
                if attention_mask is not None and attention_mask[b, i] == 0:
                    continue
                sparse_adj[b, i, topk_indices[b, i]] = 1
        
        # Convert sparse adjacency to edge index and attributes
        edge_indices = []
        edge_attrs = []
        
        for b in range(batch_size):
            # Get indices where adjacency is 1
            src, dst = torch.where(sparse_adj[b] > 0)
            
            # Add batch offset to indices for batched graph
            batch_offset = b * seq_len
            src = src + batch_offset
            dst = dst + batch_offset
            
            # Stack source and destination nodes
            edge_index = torch.stack([src, dst], dim=0)
            
            # Get edge attributes from attention weights if using edge features
            if self.use_edge_features:
                # Get attention scores for these edges
                edge_attr = torch.zeros(len(src), self.gnn_hidden_size, device=device)
                
                # Fill with attention scores
                for i, (s, d) in enumerate(zip(src - batch_offset, dst - batch_offset)):
                    # Use attention score as a feature (converted to one-hot or embedding)
                    attn_score = avg_attention[b, s - batch_offset, d - batch_offset]
                    edge_attr[i, 0] = attn_score
                    
                # Project to edge feature dimension
                edge_attr = self.edge_projection(edge_attr)
                
                edge_attrs.append(edge_attr)
            
            edge_indices.append(edge_index)
        
        # Concatenate across batches
        edge_index = torch.cat(edge_indices, dim=1)
        
        if self.use_edge_features:
            edge_attr = torch.cat(edge_attrs, dim=0)
        else:
            edge_attr = None
            
        return edge_index, edge_attr
    
    def _create_fully_connected_graph(
        self,
        num_nodes: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create a fully connected graph for each batch
        
        Args:
            num_nodes: Number of nodes per batch
            batch_size: Batch size
            device: Device to create tensor on
            
        Returns:
            edge_index: Edge indices [2, num_edges]
        """
        edge_indices = []
        
        for b in range(batch_size):
            # Create all possible pairs
            src = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
            dst = torch.arange(num_nodes, device=device).repeat(num_nodes)
            
            # Remove self-loops
            mask = src != dst
            src, dst = src[mask], dst[mask]
            
            # Add batch offset
            batch_offset = b * num_nodes
            src = src + batch_offset
            dst = dst + batch_offset
            
            edge_index = torch.stack([src, dst], dim=0)
            edge_indices.append(edge_index)
            
        return torch.cat(edge_indices, dim=1) 