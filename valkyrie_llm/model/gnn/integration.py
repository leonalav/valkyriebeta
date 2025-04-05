"""
Integration modules for combining GNNs with transformers.

This module provides components for integrating Graph Neural Networks with
transformer-based language models for joint reasoning over text and graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from .graph_encoder import GraphEncoder
from .tree_gnn import TreeGNN
from .contrastive import GraphCL, InfoGraph


class ModelRegistry:
    """
    Registry of available GNN models for easy access and instantiation.
    
    This registry maintains a mapping of model names to their respective classes
    and provides utility methods for creating model instances with proper arguments.
    """
    
    # Mapping from model names to model classes
    _models = {
        "gat": "GAT",
        "gin": "GIN",
        "mpnn": "MPNN",
        "egnn": "EGNN",
        "edgegat": "EdgeGAT",
        "graphtransformer": "GraphTransformer",
        "diffpool": "DiffPool",
        "hgt": "HGT",
        "treegnn": "TreeGNN"
    }
    
    @classmethod
    def get_model_class(cls, model_name: str) -> str:
        """
        Get the model class name corresponding to the given model name.
        
        Args:
            model_name: Name of the model (lowercase)
            
        Returns:
            Class name of the model
        """
        if model_name.lower() not in cls._models:
            raise ValueError(f"Unknown model name: {model_name}. "
                             f"Available models: {list(cls._models.keys())}")
        
        return cls._models[model_name.lower()]
    
    @classmethod
    def create_model(cls, model_name: str, **kwargs) -> nn.Module:
        """
        Create a model instance with the given name and arguments.
        
        Args:
            model_name: Name of the model (case-insensitive)
            **kwargs: Arguments to pass to the model constructor
            
        Returns:
            Instantiated model
        """
        # Import here to avoid circular imports
        from .layers import (
            GAT, GIN, MPNN, EGNN, EdgeGAT, GraphTransformer, DiffPool, HGT
        )
        from .tree_gnn import TreeGNN
        
        model_class_name = cls.get_model_class(model_name)
        
        # Get the model class
        if model_class_name == "GAT":
            return GAT(**kwargs)
        elif model_class_name == "GIN":
            return GIN(**kwargs)
        elif model_class_name == "MPNN":
            return MPNN(**kwargs)
        elif model_class_name == "EGNN":
            return EGNN(**kwargs)
        elif model_class_name == "EdgeGAT":
            return EdgeGAT(**kwargs)
        elif model_class_name == "GraphTransformer":
            return GraphTransformer(**kwargs)
        elif model_class_name == "DiffPool":
            return DiffPool(**kwargs)
        elif model_class_name == "HGT":
            return HGT(**kwargs)
        elif model_class_name == "TreeGNN":
            return TreeGNN(**kwargs)
        else:
            raise ValueError(f"Model class {model_class_name} not implemented in create_model")


class TransformerGNNIntegration(nn.Module):
    """
    Integration between transformer-based language models and Graph Neural Networks.
    
    This module provides bidirectional interactions between transformers and GNNs,
    allowing information to flow from text to graphs and from graphs to text.
    """
    
    def __init__(
        self,
        transformer_dim: int,
        graph_dim: int,
        hidden_dim: int = 256,
        num_graph_encoder_layers: int = 3,
        num_fusion_layers: int = 2,
        dropout: float = 0.1,
        gnn_type: str = "gat",
        use_graph_attention: bool = True,
        use_tree_structure: bool = False,
        use_contrastive: bool = False,
        contrastive_type: str = "graphcl",
        gnn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the transformer-GNN integration module.
        
        Args:
            transformer_dim: Dimension of transformer representations
            graph_dim: Dimension of graph node features
            hidden_dim: Hidden dimension for integration components
            num_graph_encoder_layers: Number of layers in the graph encoder
            num_fusion_layers: Number of transformer-graph fusion layers
            dropout: Dropout rate
            gnn_type: Type of GNN to use ("gat", "gin", "mpnn", "egnn", "graphtransformer", etc.)
            use_graph_attention: Whether to use attention mechanism for integrating graph into transformer
            use_tree_structure: Whether to use tree-structured GNN for hierarchical data
            use_contrastive: Whether to use contrastive learning for the graph encoder
            contrastive_type: Type of contrastive learning to use ("graphcl" or "infograph")
            gnn_kwargs: Additional keyword arguments for the GNN
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.graph_dim = graph_dim
        self.hidden_dim = hidden_dim
        self.num_fusion_layers = num_fusion_layers
        self.use_graph_attention = use_graph_attention
        self.use_tree_structure = use_tree_structure
        self.use_contrastive = use_contrastive
        self.contrastive_type = contrastive_type
        
        # Default GNN kwargs if none provided
        if gnn_kwargs is None:
            gnn_kwargs = {}
        
        # Transform node features to common hidden dimension
        self.node_projection = nn.Linear(graph_dim, hidden_dim)
        
        # Transform transformer features to common hidden dimension
        self.transformer_projection = nn.Linear(transformer_dim, hidden_dim)
        
        # Special handling for tree-structured data
        if use_tree_structure:
            self.graph_encoder = TreeGNN(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=num_graph_encoder_layers,
                dropout=dropout,
                **gnn_kwargs
            )
        else:
            # Create graph encoder using the specified GNN type
            self.graph_encoder = GraphEncoder(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=num_graph_encoder_layers,
                gnn_type=gnn_type,
                dropout=dropout,
                **gnn_kwargs
            )
        
        # Optional contrastive learning wrapper
        if use_contrastive:
            if contrastive_type.lower() == "graphcl":
                self.contrastive_model = GraphCL(
                    encoder=self.graph_encoder,
                    proj_hidden_dim=hidden_dim,
                    proj_output_dim=hidden_dim // 2,
                    dropout=dropout
                )
            elif contrastive_type.lower() == "infograph":
                self.contrastive_model = InfoGraph(
                    encoder=self.graph_encoder,
                    proj_hidden_dim=hidden_dim,
                    proj_output_dim=hidden_dim // 2,
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown contrastive type: {contrastive_type}. "
                                 "Available types: 'graphcl', 'infograph'")
        
        # Fusion layers for integrating graph information into transformer
        self.graph_to_transformer_layers = nn.ModuleList()
        for _ in range(num_fusion_layers):
            if use_graph_attention:
                # Cross-attention fusion
                self.graph_to_transformer_layers.append(
                    CrossAttentionFusion(
                        query_dim=hidden_dim,
                        key_dim=hidden_dim,
                        value_dim=hidden_dim,
                        num_heads=8,
                        dropout=dropout
                    )
                )
            else:
                # Concat-and-project fusion
                self.graph_to_transformer_layers.append(
                    ConcatFusion(
                        dim1=hidden_dim,
                        dim2=hidden_dim,
                        output_dim=hidden_dim,
                        dropout=dropout
                    )
                )
        
        # Fusion layers for integrating transformer information into graph
        self.transformer_to_graph_layers = nn.ModuleList()
        for _ in range(num_fusion_layers):
            if use_graph_attention:
                # Cross-attention fusion
                self.transformer_to_graph_layers.append(
                    CrossAttentionFusion(
                        query_dim=hidden_dim,
                        key_dim=hidden_dim,
                        value_dim=hidden_dim,
                        num_heads=8,
                        dropout=dropout
                    )
                )
            else:
                # Concat-and-project fusion
                self.transformer_to_graph_layers.append(
                    ConcatFusion(
                        dim1=hidden_dim,
                        dim2=hidden_dim,
                        output_dim=hidden_dim,
                        dropout=dropout
                    )
                )
        
        # Output projections to original dimensions
        self.graph_output_projection = nn.Linear(hidden_dim, graph_dim)
        self.transformer_output_projection = nn.Linear(hidden_dim, transformer_dim)
        
        # Layer norm for stabilizing training
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.transformer_norm = nn.LayerNorm(hidden_dim)
    
    def verify_integration(self, llm_model, gnn_model):
        """
        Verify that the integration model is compatible with the provided LLM and GNN models.
        
        Args:
            llm_model: The language model to integrate with
            gnn_model: The graph neural network model to integrate with
        
        Raises:
            ValueError: If the models are not compatible
        """
        # Check LLM model compatibility
        if not hasattr(llm_model, 'config') or not hasattr(llm_model.config, 'hidden_size'):
            raise ValueError("LLM model must have a config attribute with hidden_size")
        
        if llm_model.config.hidden_size != self.transformer_dim:
            raise ValueError(
                f"LLM model hidden size ({llm_model.config.hidden_size}) does not match "
                f"integration transformer_dim ({self.transformer_dim})"
            )
        
        # Check GNN model compatibility - verify it has the required input/output dimensions
        if not hasattr(gnn_model, 'in_channels') or not hasattr(gnn_model, 'out_channels'):
            raise ValueError("GNN model must have in_channels and out_channels attributes")
        
        if gnn_model.in_channels != self.graph_dim:
            raise ValueError(
                f"GNN model input dimension ({gnn_model.in_channels}) does not match "
                f"integration graph_dim ({self.graph_dim})"
            )
        
        if gnn_model.out_channels != self.transformer_dim:
            raise ValueError(
                f"GNN model output dimension ({gnn_model.out_channels}) does not match "
                f"transformer dimension ({self.transformer_dim}). "
                f"These should match for proper integration."
            )
        
        # Check forward method compatibility
        if not hasattr(gnn_model, 'forward'):
            raise ValueError("GNN model must have a forward method")
        
        # Check if the integration model has all required components
        if self.use_tree_structure and not isinstance(self.graph_encoder, TreeGNN):
            raise ValueError("use_tree_structure is enabled but graph_encoder is not a TreeGNN")
        
        if self.use_contrastive and not hasattr(self, 'contrastive_model'):
            raise ValueError("use_contrastive is enabled but contrastive_model is not initialized")
        
        print("Integration verification successful. LLM and GNN models are compatible.")
    
    def forward(
        self,
        transformer_features: torch.Tensor,
        transformer_mask: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        node_depth: Optional[torch.Tensor] = None,
        node_parent: Optional[torch.Tensor] = None,
        is_leaf: Optional[torch.Tensor] = None,
        return_contrastive_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the transformer-GNN integration.
        
        Args:
            transformer_features: Transformer token features [batch_size, seq_len, transformer_dim]
            transformer_mask: Attention mask for transformer tokens [batch_size, seq_len]
            node_features: Graph node features [num_nodes, graph_dim]
            edge_index: Graph edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            batch: Batch indices for node features [num_nodes] (optional)
            node_depth: Depth of each node in the tree [num_nodes] (for tree-structured GNN)
            node_parent: Parent index for each node [num_nodes] (for tree-structured GNN)
            is_leaf: Boolean mask indicating leaf nodes [num_nodes] (for tree-structured GNN)
            return_contrastive_loss: Whether to return contrastive loss (if contrastive learning is enabled)
            
        Returns:
            Dictionary containing updated transformer and graph features, and optionally contrastive loss
        """
        # Project node features to hidden dimension
        h_graph = self.node_projection(node_features)
        h_graph = F.relu(h_graph)
        
        # Project transformer features to hidden dimension
        batch_size, seq_len, _ = transformer_features.shape
        h_transformer = self.transformer_projection(transformer_features)
        h_transformer = F.relu(h_transformer)
        
        # Process graph with the graph encoder or tree GNN
        if self.use_contrastive and return_contrastive_loss:
            # Use contrastive learning wrapper for graph encoding
            contrastive_output = self.contrastive_model(
                h_graph, edge_index, batch, edge_attr
            )
            contrastive_loss = contrastive_output["contrastive_loss"]
            
            # Get graph node and graph embeddings
            if self.use_tree_structure:
                graph_output = self.graph_encoder(
                    h_graph, edge_index, edge_attr, batch,
                    node_depth, node_parent, is_leaf
                )
                node_embeddings = graph_output["node_embeddings"]
                graph_embedding = graph_output.get("tree_embedding", None)
            else:
                graph_output = self.graph_encoder(
                    h_graph, edge_index, edge_attr, batch
                )
                node_embeddings = graph_output["node_embeddings"]
                graph_embedding = graph_output.get("graph_embedding", None)
        else:
            # Regular graph encoding without contrastive learning
            if self.use_tree_structure:
                graph_output = self.graph_encoder(
                    h_graph, edge_index, edge_attr, batch,
                    node_depth, node_parent, is_leaf
                )
                node_embeddings = graph_output["node_embeddings"]
                graph_embedding = graph_output.get("tree_embedding", None)
            else:
                graph_output = self.graph_encoder(
                    h_graph, edge_index, edge_attr, batch
                )
                node_embeddings = graph_output["node_embeddings"]
                graph_embedding = graph_output.get("graph_embedding", None)
            
            contrastive_loss = None
        
        # Apply layer norm
        node_embeddings = self.graph_norm(node_embeddings)
        h_transformer = self.transformer_norm(h_transformer)
        
        # Integrate graph information into transformer (graph → transformer)
        for layer in self.graph_to_transformer_layers:
            # For each token, attend to all nodes
            # Reshape transformer features: [batch_size, seq_len, hidden_dim] → [batch_size * seq_len, hidden_dim]
            h_transformer_flat = h_transformer.reshape(-1, self.hidden_dim)
            
            # Create token to batch mapping
            if batch is not None:
                # Map tokens to their corresponding graph
                token_to_graph = torch.arange(batch_size, device=h_transformer.device).repeat_interleave(seq_len)
                
                # For each token, only attend to nodes in the same graph
                h_transformer_updated = []
                for token_idx in range(batch_size * seq_len):
                    token_graph = token_to_graph[token_idx]
                    graph_node_mask = (batch == token_graph)
                    
                    if graph_node_mask.sum() > 0:
                        token_features = h_transformer_flat[token_idx:token_idx+1]
                        graph_nodes = node_embeddings[graph_node_mask]
                        
                        # Apply fusion
                        if isinstance(layer, CrossAttentionFusion):
                            updated_token = layer(token_features, graph_nodes, graph_nodes)
                        else:
                            # For ConcatFusion, we need to aggregate graph nodes first
                            graph_agg = torch.mean(graph_nodes, dim=0, keepdim=True)
                            updated_token = layer(token_features, graph_agg)
                        
                        h_transformer_updated.append(updated_token)
                    else:
                        # If no nodes in this graph, keep token unchanged
                        h_transformer_updated.append(h_transformer_flat[token_idx:token_idx+1])
                
                # Combine updated tokens
                h_transformer_flat = torch.cat(h_transformer_updated, dim=0)
            else:
                # If no batch information, treat all nodes as part of one graph
                if isinstance(layer, CrossAttentionFusion):
                    h_transformer_flat = layer(h_transformer_flat, node_embeddings, node_embeddings)
                else:
                    # For ConcatFusion, we need to aggregate graph nodes first
                    graph_agg = torch.mean(node_embeddings, dim=0, keepdim=True)
                    graph_agg = graph_agg.expand(h_transformer_flat.size(0), -1)
                    h_transformer_flat = layer(h_transformer_flat, graph_agg)
            
            # Reshape back: [batch_size * seq_len, hidden_dim] → [batch_size, seq_len, hidden_dim]
            h_transformer = h_transformer_flat.view(batch_size, seq_len, self.hidden_dim)
        
        # Integrate transformer information into graph (transformer → graph)
        for layer in self.transformer_to_graph_layers:
            # For each node, attend to all tokens in its corresponding graph
            # Create mapping from graph to tokens
            if batch is not None:
                # Map nodes to their corresponding graph's tokens
                h_node_updated = []
                for node_idx in range(node_embeddings.size(0)):
                    node_graph = batch[node_idx].item() if batch[node_idx].item() < batch_size else 0
                    
                    # Get token features for this graph
                    graph_tokens = h_transformer[node_graph]
                    
                    # Apply fusion
                    node_features = node_embeddings[node_idx:node_idx+1]
                    if isinstance(layer, CrossAttentionFusion):
                        updated_node = layer(node_features, graph_tokens, graph_tokens)
                    else:
                        # For ConcatFusion, we need to aggregate tokens first
                        token_agg = torch.mean(graph_tokens, dim=0, keepdim=True)
                        updated_node = layer(node_features, token_agg)
                    
                    h_node_updated.append(updated_node)
                
                # Combine updated nodes
                node_embeddings = torch.cat(h_node_updated, dim=0)
            else:
                # If no batch information, treat all tokens as part of one graph
                if isinstance(layer, CrossAttentionFusion):
                    # Flatten transformer features for attention
                    h_transformer_flat = h_transformer.view(-1, self.hidden_dim)
                    node_embeddings = layer(node_embeddings, h_transformer_flat, h_transformer_flat)
                else:
                    # For ConcatFusion, we need to aggregate tokens first
                    token_agg = torch.mean(h_transformer, dim=1)  # [batch_size, hidden_dim]
                    token_agg = token_agg[0:1]  # Use the first batch's tokens if no batch info
                    token_agg = token_agg.expand(node_embeddings.size(0), -1)
                    node_embeddings = layer(node_embeddings, token_agg)
        
        # Project back to original dimensions
        transformer_features_updated = self.transformer_output_projection(h_transformer)
        node_features_updated = self.graph_output_projection(node_embeddings)
        
        # Prepare output dictionary
        output = {
            "transformer_features": transformer_features_updated,
            "node_features": node_features_updated,
        }
        
        if graph_embedding is not None:
            output["graph_embedding"] = graph_embedding
        
        if contrastive_loss is not None and return_contrastive_loss:
            output["contrastive_loss"] = contrastive_loss
        
        return output


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion module for integrating features from one modality into another.
    
    This module uses multi-head attention where queries come from the target modality
    and keys/values come from the source modality.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-attention fusion module.
        
        Args:
            query_dim: Dimension of query features
            key_dim: Dimension of key features
            value_dim: Dimension of value features
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(key_dim, query_dim)
        self.to_v = nn.Linear(value_dim, query_dim)
        
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(query_dim)
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention fusion.
        
        Args:
            queries: Query features from target modality [batch_size, seq_len_q, query_dim]
            keys: Key features from source modality [batch_size, seq_len_k, key_dim]
            values: Value features from source modality [batch_size, seq_len_v, value_dim]
            mask: Attention mask [batch_size, seq_len_q, seq_len_k] (optional)
            
        Returns:
            Updated query features [batch_size, seq_len_q, query_dim]
        """
        residual = queries
        
        # Project queries, keys, and values
        q = self.to_q(queries)
        k = self.to_k(keys)
        v = self.to_v(values)
        
        # Reshape for multi-head attention
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        # Calculate attention weights and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, q.size(-1) * self.num_heads)
        
        # Project to output
        output = self.to_out(attn_output)
        
        # Add residual connection and layer norm
        output = self.layer_norm(output + residual)
        
        return output


class ConcatFusion(nn.Module):
    """
    Concatenation-based fusion module for integrating features from one modality into another.
    
    This module concatenates features from two modalities and projects them to the output dimension.
    """
    
    def __init__(
        self,
        dim1: int,
        dim2: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        """
        Initialize concatenation fusion module.
        
        Args:
            dim1: Dimension of first modality features
            dim2: Dimension of second modality features
            output_dim: Output dimension after fusion
            dropout: Dropout rate
        """
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim1 + dim2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for concatenation fusion.
        
        Args:
            features1: Features from first modality [batch_size, ..., dim1]
            features2: Features from second modality [batch_size, ..., dim2]
            
        Returns:
            Fused features [batch_size, ..., output_dim]
        """
        # Ensure features2 has the same size as features1 in all dimensions except the last
        if features1.shape[:-1] != features2.shape[:-1]:
            raise ValueError(f"Feature shapes do not match: {features1.shape} vs {features2.shape}")
        
        # Concatenate features along the last dimension
        concat = torch.cat([features1, features2], dim=-1)
        
        # Apply fusion
        output = self.fusion_layer(concat)
        
        # Apply layer norm with residual connection
        # For the residual, we use features1 since we want to keep the identity of the first modality
        if features1.shape[-1] == output.shape[-1]:
            output = self.layer_norm(output + features1)
        else:
            output = self.layer_norm(output)
        
        return output 