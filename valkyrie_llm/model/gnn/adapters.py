"""
Adapters for different graph formats and GNN models.

This module provides adapter classes that help convert between different
graph formats or provide compatibility layers between different GNN models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any

class HGTAdapter(nn.Module):
    """
    Adapter for HGT (Heterogeneous Graph Transformer) model.
    
    This adapter converts standard graph data (single node/edge type) to the
    heterogeneous format expected by HGT. In a real heterogeneous graph, you would
    have multiple node and edge types with their own features, but this adapter
    allows using HGT in a homogeneous setting for testing or when node/edge types
    are not explicitly available.
    """
    
    def __init__(
        self,
        hgt_layer: nn.Module,
        default_node_type: str = "default",
        default_edge_type: str = "default"
    ):
        """
        Initialize the HGT adapter.
        
        Args:
            hgt_layer: The HGT layer to adapt
            default_node_type: Name for the default node type
            default_edge_type: Name for the default edge type
        """
        super().__init__()
        self.hgt_layer = hgt_layer
        self.default_node_type = default_node_type
        self.default_edge_type = default_edge_type
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert standard graph data to heterogeneous format and process through HGT.
        
        Args:
            x: Node features [num_nodes, channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch indices [num_nodes]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Convert to heterogeneous format
        node_features = {self.default_node_type: x}
        edge_indices = {self.default_edge_type: edge_index}
        
        # Create node type indices dictionary
        node_type_indices = {
            self.default_node_type: torch.arange(x.size(0), device=x.device)
        }
        
        # Process through HGT
        outputs = self.hgt_layer(node_features, edge_indices, node_type_indices)
        
        # Return the default node type results
        return outputs[self.default_node_type] 