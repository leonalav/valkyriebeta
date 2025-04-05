"""
Graph Contrastive Learning for ValkyrieLLM.

This module implements graph contrastive learning approaches like GraphCL and InfoGraph
to learn better graph representations in self-supervised or semi-supervised settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from .graph_encoder import GraphEncoder


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    This projects embeddings to a space where contrastive loss is applied.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        """
        Initialize projection head.
        
        Args:
            in_channels: Input dimension
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # First layer
        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_channels, out_channels))
        
        self.layers = nn.Sequential(*layers)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for projection head."""
        return self.layers(x)


class GraphCL(nn.Module):
    """
    Graph Contrastive Learning (GraphCL) as described in 
    "Graph Contrastive Learning with Augmentations" (You et al., NeurIPS 2020).
    
    GraphCL learns graph representations by maximizing agreement between different 
    augmented views of the same graph.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        proj_hidden_dim: int = 256,
        proj_output_dim: int = 128,
        temperature: float = 0.1,
        dropout: float = 0.1,
        augmentation_types: List[str] = ["node_dropping", "edge_perturbation"],
        node_drop_rate: float = 0.2,
        edge_perturb_rate: float = 0.2,
        subgraph_rate: float = 0.8,
        feature_mask_rate: float = 0.2,
    ):
        """
        Initialize GraphCL model.
        
        Args:
            encoder: Graph encoder model (typically GraphEncoder)
            proj_hidden_dim: Hidden dimension for projection head
            proj_output_dim: Output dimension for projection head
            temperature: Temperature parameter for InfoNCE loss
            dropout: Dropout rate for projection head
            augmentation_types: List of augmentation types to apply
                              Options: ["node_dropping", "edge_perturbation", 
                                       "subgraph", "feature_masking"]
            node_drop_rate: Rate of nodes to drop in node dropping augmentation
            edge_perturb_rate: Rate of edges to perturb in edge perturbation
            subgraph_rate: Rate of nodes to keep in subgraph augmentation
            feature_mask_rate: Rate of features to mask in feature masking
        """
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.augmentation_types = augmentation_types
        self.node_drop_rate = node_drop_rate
        self.edge_perturb_rate = edge_perturb_rate
        self.subgraph_rate = subgraph_rate
        self.feature_mask_rate = feature_mask_rate
        
        # Get embedding dimension from encoder
        if hasattr(encoder, "out_channels"):
            in_channels = encoder.out_channels
        else:
            # Default fallback
            in_channels = 128
        
        # Projection head for contrastive learning
        self.projection = ProjectionHead(
            in_channels=in_channels,
            hidden_channels=proj_hidden_dim,
            out_channels=proj_output_dim,
            dropout=dropout,
            use_batch_norm=True,
        )
    
    def augment(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        augmentation_type: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply augmentation to graph data.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]
            augmentation_type: Type of augmentation to apply
                             If None, a random one from augmentation_types is selected
                           
        Returns:
            Augmented x, edge_index, and batch
        """
        # If no specific augmentation is provided, choose randomly
        if augmentation_type is None:
            augmentation_type = torch.choice(self.augmentation_types)
        
        # Apply the selected augmentation
        if augmentation_type == "node_dropping":
            return self._node_dropping(x, edge_index, batch)
        elif augmentation_type == "edge_perturbation":
            return self._edge_perturbation(x, edge_index, batch)
        elif augmentation_type == "subgraph":
            return self._subgraph(x, edge_index, batch)
        elif augmentation_type == "feature_masking":
            return self._feature_masking(x, edge_index, batch)
        else:
            # Return original if no valid augmentation
            return x, edge_index, batch
    
    def _node_dropping(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Randomly drop nodes from the graph.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]
            
        Returns:
            Augmented x, edge_index, and batch
        """
        num_nodes = x.size(0)
        
        # Create mask for nodes to keep (True = keep)
        keep_mask = torch.rand(num_nodes, device=x.device) > self.node_drop_rate
        
        # Always keep at least one node
        if keep_mask.sum() == 0:
            keep_mask[torch.randint(0, num_nodes, (1,))] = True
        
        # Filter nodes
        x_new = x[keep_mask]
        
        # Update batch indices
        batch_new = None
        if batch is not None:
            batch_new = batch[keep_mask]
        
        # Create mapping from old indices to new indices
        node_idx_map = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
        node_idx_map[keep_mask] = torch.arange(keep_mask.sum(), device=x.device)
        
        # Filter edges that have both source and target nodes in the new graph
        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        edge_index_new = edge_index[:, edge_mask]
        
        # Remap edge indices to new node indices
        edge_index_new = node_idx_map[edge_index_new]
        
        return x_new, edge_index_new, batch_new
    
    def _edge_perturbation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Randomly perturb edges by adding and removing some edges.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]
            
        Returns:
            Augmented x, edge_index, and batch
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        # Randomly remove some edges
        keep_mask = torch.rand(num_edges, device=edge_index.device) > self.edge_perturb_rate
        edge_index_new = edge_index[:, keep_mask]
        
        # Add random edges (number of edges to add = number of edges removed)
        num_edges_to_add = num_edges - edge_index_new.size(1)
        
        if num_edges_to_add > 0 and num_nodes > 1:
            # Create random edges between nodes
            src_nodes = torch.randint(0, num_nodes, (num_edges_to_add,), device=edge_index.device)
            dst_nodes = torch.randint(0, num_nodes, (num_edges_to_add,), device=edge_index.device)
            
            # Ensure src != dst to avoid self-loops
            same_mask = src_nodes == dst_nodes
            dst_nodes[same_mask] = (src_nodes[same_mask] + 1) % num_nodes
            
            # Create new edges
            new_edges = torch.stack([src_nodes, dst_nodes], dim=0)
            
            # If we have batch information, ensure edges only connect nodes within same graph
            if batch is not None:
                # Keep only edges where src and dst are in the same graph
                same_graph_mask = batch[new_edges[0]] == batch[new_edges[1]]
                new_edges = new_edges[:, same_graph_mask]
            
            # Combine with existing edges
            edge_index_new = torch.cat([edge_index_new, new_edges], dim=1)
        
        return x, edge_index_new, batch
    
    def _subgraph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Extract subgraphs by randomly selecting starting nodes and taking their neighbors.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]
            
        Returns:
            Augmented x, edge_index, and batch
        """
        num_nodes = x.size(0)
        
        # Determine number of seed nodes
        num_seeds = max(1, int(num_nodes * self.subgraph_rate))
        
        # Randomly select seed nodes
        seed_nodes = torch.randperm(num_nodes, device=x.device)[:num_seeds]
        
        # Create initial mask for nodes to keep (seed nodes)
        keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        keep_mask[seed_nodes] = True
        
        # If we have batch information, ensure we have seeds from each graph
        if batch is not None:
            num_graphs = batch.max().item() + 1
            for g in range(num_graphs):
                graph_mask = batch == g
                graph_nodes = torch.where(graph_mask)[0]
                
                if graph_nodes.size(0) > 0:
                    # Ensure at least one seed from each graph
                    graph_seed = graph_nodes[torch.randint(0, graph_nodes.size(0), (1,))]
                    keep_mask[graph_seed] = True
        
        # Find neighbors of seed nodes (1-hop neighborhood)
        neighbors = edge_index[1, torch.isin(edge_index[0], seed_nodes)]
        keep_mask[neighbors] = True
        
        # Filter nodes
        x_new = x[keep_mask]
        
        # Update batch indices
        batch_new = None
        if batch is not None:
            batch_new = batch[keep_mask]
        
        # Create mapping from old indices to new indices
        node_idx_map = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
        node_idx_map[keep_mask] = torch.arange(keep_mask.sum(), device=x.device)
        
        # Filter edges that have both source and target nodes in the new graph
        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        edge_index_new = edge_index[:, edge_mask]
        
        # Remap edge indices to new node indices
        edge_index_new = node_idx_map[edge_index_new]
        
        return x_new, edge_index_new, batch_new
    
    def _feature_masking(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Randomly mask (set to zero) some node features.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes]
            
        Returns:
            Augmented x, edge_index, and batch
        """
        num_features = x.size(1)
        
        # Create mask for features to keep (False = mask/set to zero)
        mask_features = torch.rand(num_features, device=x.device) < self.feature_mask_rate
        
        # Create a copy of x and mask selected features
        x_new = x.clone()
        x_new[:, mask_features] = 0.0
        
        return x_new, edge_index, batch
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for GraphCL.
        
        This creates two augmented views of each graph and computes the contrastive loss.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes] for multiple graphs
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            
        Returns:
            Dictionary containing representations and loss
        """
        if batch is None:
            # If no batch is provided, assume a single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Create two augmented views of the graphs
        aug_types = torch.random.choice(self.augmentation_types, size=2, replace=True)
        
        x1, edge_index1, batch1 = self.augment(x, edge_index, batch, aug_types[0])
        x2, edge_index2, batch2 = self.augment(x, edge_index, batch, aug_types[1])
        
        # Compute representations for both views
        h1 = self.encoder(x1, edge_index1, edge_attr, batch1)["graph_embedding"]
        h2 = self.encoder(x2, edge_index2, edge_attr, batch2)["graph_embedding"]
        
        # Project representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        
        # Normalize projections
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # Compute contrastive loss
        loss = self.contrastive_loss(z1, z2)
        
        return {
            "representations": h1,  # Use representations from first view
            "projections": z1,
            "contrastive_loss": loss,
        }
    
    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            z1: Projections from first view [batch_size, proj_dim]
            z2: Projections from second view [batch_size, proj_dim]
            
        Returns:
            InfoNCE loss
        """
        batch_size = z1.size(0)
        
        # Calculate cosine similarity matrix
        similarity_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # Labels are the diagonal elements (positive pairs)
        labels = torch.arange(batch_size, device=z1.device)
        
        # Calculate loss for both directions (z1->z2 and z2->z1)
        loss_1 = F.cross_entropy(similarity_matrix, labels)
        loss_2 = F.cross_entropy(similarity_matrix.t(), labels)
        
        # Average both directions
        loss = (loss_1 + loss_2) / 2
        
        return loss


class InfoGraph(nn.Module):
    """
    InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via 
    Mutual Information Maximization (Sun et al., ICLR 2020).
    
    InfoGraph learns graph-level representations by maximizing the mutual information between 
    graph-level representations and substructure-level representations.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        proj_hidden_dim: int = 256,
        proj_output_dim: int = 128,
        dropout: float = 0.1,
    ):
        """
        Initialize InfoGraph model.
        
        Args:
            encoder: Graph encoder model (typically GraphEncoder)
            proj_hidden_dim: Hidden dimension for projection head
            proj_output_dim: Output dimension for projection head
            dropout: Dropout rate for projection head
        """
        super().__init__()
        self.encoder = encoder
        
        # Get embedding dimension from encoder
        if hasattr(encoder, "out_channels"):
            graph_dim = encoder.out_channels
        else:
            # Default fallback
            graph_dim = 128
        
        # Get node embedding dimension from encoder
        if hasattr(encoder, "hidden_channels"):
            node_dim = encoder.hidden_channels
        else:
            # Default fallback
            node_dim = 128
        
        # Projection head for graph-level representations
        self.graph_projection = ProjectionHead(
            in_channels=graph_dim,
            hidden_channels=proj_hidden_dim,
            out_channels=proj_output_dim,
            dropout=dropout,
            use_batch_norm=True,
        )
        
        # Projection head for node-level representations
        self.node_projection = ProjectionHead(
            in_channels=node_dim,
            hidden_channels=proj_hidden_dim,
            out_channels=proj_output_dim,
            dropout=dropout,
            use_batch_norm=True,
        )
        
        # Discriminator (bilinear layer) for mutual information estimation
        self.discriminator = nn.Bilinear(proj_output_dim, proj_output_dim, 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        self.graph_projection.reset_parameters()
        self.node_projection.reset_parameters()
        nn.init.xavier_uniform_(self.discriminator.weight)
        nn.init.zeros_(self.discriminator.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for InfoGraph.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            batch: Batch indices [num_nodes] for multiple graphs
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            
        Returns:
            Dictionary containing representations and loss
        """
        # Get node and graph embeddings from encoder
        encoder_output = self.encoder(
            x, edge_index, edge_attr, batch, return_node_embeddings=True
        )
        
        # Extract graph and node embeddings
        graph_embeddings = encoder_output["graph_embedding"]
        node_embeddings = encoder_output["node_embeddings"]
        
        # Project embeddings
        graph_projections = self.graph_projection(graph_embeddings)
        node_projections = self.node_projection(node_embeddings)
        
        # Normalize projections
        graph_projections = F.normalize(graph_projections, p=2, dim=1)
        node_projections = F.normalize(node_projections, p=2, dim=1)
        
        # Compute loss
        loss = self.mutual_info_loss(graph_projections, node_projections, batch)
        
        return {
            "representations": graph_embeddings,
            "graph_projections": graph_projections,
            "node_projections": node_projections,
            "contrastive_loss": loss,
        }
    
    def mutual_info_loss(
        self,
        graph_projections: torch.Tensor,
        node_projections: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mutual information loss between graph and node representations.
        
        Args:
            graph_projections: Graph-level projections [num_graphs, proj_dim]
            node_projections: Node-level projections [num_nodes, proj_dim]
            batch: Batch indices [num_nodes] for mapping nodes to graphs
            
        Returns:
            Mutual information loss
        """
        num_graphs = graph_projections.size(0)
        num_nodes = node_projections.size(0)
        
        # Expand graph projections to match corresponding nodes
        graph_projections_expanded = graph_projections[batch]  # [num_nodes, proj_dim]
        
        # Compute positive scores for node-graph pairs from the same graph
        pos_scores = self.discriminator(node_projections, graph_projections_expanded)  # [num_nodes, 1]
        
        # Sample negative pairs (nodes and graphs from different graphs)
        neg_scores_list = []
        
        # For each node, sample a graph from a different graph
        for i in range(num_nodes):
            node_graph_idx = batch[i].item()
            
            # Sample a random graph different from the node's graph
            negative_graph_indices = [j for j in range(num_graphs) if j != node_graph_idx]
            
            if negative_graph_indices:
                # If there are negative graphs, sample one
                neg_idx = torch.randint(0, len(negative_graph_indices), (1,)).item()
                neg_graph_idx = negative_graph_indices[neg_idx]
                
                # Compute score for this negative pair
                neg_score = self.discriminator(
                    node_projections[i:i+1], graph_projections[neg_graph_idx:neg_graph_idx+1]
                )
                neg_scores_list.append(neg_score)
        
        if neg_scores_list:
            # Combine negative scores
            neg_scores = torch.cat(neg_scores_list, dim=0)  # [num_sampled_negs, 1]
            
            # Compute Binary Cross Entropy loss
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            
            loss = (pos_loss + neg_loss) / 2
        else:
            # Fallback if no negative pairs could be created (e.g., only one graph in batch)
            loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            )
        
        return loss 