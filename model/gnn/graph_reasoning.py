"""
Graph Reasoning module for ValkyrieLLM.

This module implements graph-based reasoning capabilities for the language model,
enabling it to perform structured reasoning over text by constructing and processing
implicit or explicit knowledge graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from .graph_encoder import GraphEncoder
from .layers import GraphConvolution, GraphAttention


class GraphReasoningModule(nn.Module):
    """
    Graph reasoning module for enhancing transformer models with graph-based reasoning
    capabilities. This module can be integrated into the hybrid RWKV-Transformer model
    to provide structured reasoning over tokens.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gating: bool = True,
        use_edge_features: bool = False,
        attention_heads: int = 4,
        graph_residual: bool = True,
        integration_type: str = "additive"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gating = use_gating
        self.use_edge_features = use_edge_features
        self.integration_type = integration_type
        
        # Graph layers
        self.graph_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # First layer uses GraphConvolution
                self.graph_layers.append(GraphConvolution(hidden_size, hidden_size))
            else:
                # Subsequent layers use GraphAttention for more expressive power
                self.graph_layers.append(
                    GraphAttention(
                        hidden_size, 
                        hidden_size, 
                        attention_heads, 
                        dropout=dropout,
                        use_edge_features=use_edge_features
                    )
                )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Optional gating mechanism to control graph information flow
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
        
        # Integration projection
        self.integration_projection = nn.Linear(hidden_size, hidden_size)
        
        # Residual connection
        self.graph_residual = graph_residual
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def build_dynamic_graph(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        window_size: int = 512
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Dynamically construct a graph from hidden states.
        
        For long sequences, this uses a sliding window approach to build local graphs
        that can then be connected with sparse global edges.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            window_size: Size of local attention window for graph construction
            
        Returns:
            edge_list: List of tensor pairs representing edges
            edge_weights: Edge weights tensor
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Initialize edge lists and weights
        edge_sources = []
        edge_targets = []
        edge_weights_list = []
        
        for b in range(batch_size):
            # Process each sequence in the batch
            sequence_length = attention_mask[b].sum().item() if attention_mask is not None else seq_len
            
            # Build local window edges
            for i in range(sequence_length):
                # Create edges within window_size neighborhood
                window_start = max(0, i - window_size // 2)
                window_end = min(sequence_length, i + window_size // 2)
                
                # Connect to all nodes in window (excluding self)
                for j in range(window_start, window_end):
                    if i != j:  # Exclude self-loops
                        edge_sources.append(torch.tensor([b, i], device=device))
                        edge_targets.append(torch.tensor([b, j], device=device))
                        
                        # Calculate edge weight based on position difference
                        distance = abs(i - j)
                        edge_weight = 1.0 / (1.0 + distance)  # Decay with distance
                        edge_weights_list.append(edge_weight)
        
        # Convert lists to tensors
        edge_indices = torch.stack([
            torch.stack(edge_sources, dim=0),
            torch.stack(edge_targets, dim=0)
        ], dim=0)
        
        edge_weights = torch.tensor(edge_weights_list, device=device)
        
        return edge_indices, edge_weights
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        edge_indices: Optional[torch.Tensor] = None,
        edge_weights: Optional[torch.Tensor] = None,
        return_graph_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the graph reasoning module.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            edge_indices: Optional pre-computed edge indices
            edge_weights: Optional pre-computed edge weights
            return_graph_states: Whether to return graph states
            
        Returns:
            enhanced_hidden_states: Hidden states enhanced with graph reasoning
            graph_states: Graph states if return_graph_states is True
        """
        # Store original hidden states for residual connection
        original_states = hidden_states
        
        # Build graph if not provided
        if edge_indices is None or edge_weights is None:
            edge_indices, edge_weights = self.build_dynamic_graph(hidden_states, attention_mask)
        
        # Process through graph layers
        graph_states = hidden_states
        for layer in self.graph_layers:
            graph_states = layer(graph_states, edge_indices, edge_weights)
            
            if self.graph_residual:
                graph_states = graph_states + hidden_states
            
            graph_states = self.layer_norm(graph_states)
            graph_states = self.dropout(graph_states)
        
        # Integrate graph information back into original hidden states
        if self.use_gating:
            # Use gating mechanism to control information flow
            gate_value = self.gate(torch.cat([original_states, graph_states], dim=-1))
            enhanced_states = original_states + gate_value * self.integration_projection(graph_states)
        else:
            if self.integration_type == "additive":
                enhanced_states = original_states + self.integration_projection(graph_states)
            elif self.integration_type == "multiplicative":
                enhanced_states = original_states * self.integration_projection(graph_states)
            else:  # "concat"
                enhanced_states = torch.cat([original_states, graph_states], dim=-1)
                enhanced_states = self.integration_projection(enhanced_states)
        
        if return_graph_states:
            return enhanced_states, graph_states
        
        return enhanced_states 