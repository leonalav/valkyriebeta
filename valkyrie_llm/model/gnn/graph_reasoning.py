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


class GraphReasoner(nn.Module):
    """
    Graph-based reasoning module for language understanding and generation.
    
    This module enables the model to:
    1. Construct knowledge graphs from text
    2. Reason over these graphs using GNNs
    3. Incorporate graph-based reasoning into the language model
    
    The reasoning can be done at different granularities:
    - Word-level: Each token is a node in the graph
    - Entity-level: Entities are nodes, relationships are edges
    - Concept-level: Abstract concepts form the graph structure
    """
    
    def __init__(
        self,
        hidden_size: int,
        gnn_hidden_size: int = 256,
        num_gnn_layers: int = 3,
        gnn_type: str = "gat",
        granularity: str = "entity",
        max_nodes: int = 100,
        max_edges: int = 500,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        edge_feature_dim: int = 64,
        num_edge_types: int = 8,
        num_relation_heads: int = 4,
        **kwargs
    ):
        """
        Initialize the GraphReasoner.
        
        Args:
            hidden_size: Hidden dimension of the language model
            gnn_hidden_size: Hidden dimension for the GNN
            num_gnn_layers: Number of GNN layers
            gnn_type: Type of GNN to use
            granularity: Level of graph construction ("word", "entity", "concept")
            max_nodes: Maximum number of nodes in the graph
            max_edges: Maximum number of edges in the graph
            dropout: Dropout probability
            use_edge_features: Whether to use edge features
            edge_feature_dim: Dimension of edge features
            num_edge_types: Number of edge types for edge features
            num_relation_heads: Number of relation heads for entity extraction
            **kwargs: Additional arguments for the GNN
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.gnn_hidden_size = gnn_hidden_size
        self.num_gnn_layers = num_gnn_layers
        self.gnn_type = gnn_type
        self.granularity = granularity
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        self.edge_feature_dim = edge_feature_dim if use_edge_features else None
        self.num_edge_types = num_edge_types
        self.num_relation_heads = num_relation_heads
        
        # Node feature transformation
        self.node_feature_proj = nn.Linear(hidden_size, gnn_hidden_size)
        
        # Edge features
        if use_edge_features:
            self.edge_embedding = nn.Embedding(num_edge_types, edge_feature_dim)
            
        # Graph construction components based on granularity
        if granularity == "word":
            # For word-level granularity, we use attention to construct the graph
            self.attention_scores = nn.Linear(hidden_size, hidden_size)
            self.edge_threshold = nn.Parameter(torch.tensor(0.1))
            
        elif granularity == "entity":
            # For entity-level granularity, we need entity extraction and relation extraction
            self.entity_extractor = self._create_entity_extractor(hidden_size)
            self.relation_extractor = self._create_relation_extractor(hidden_size, num_relation_heads)
            
        elif granularity == "concept":
            # For concept-level granularity, we use a more sophisticated extraction mechanism
            self.concept_extractor = self._create_concept_extractor(hidden_size)
            self.concept_relation_extractor = self._create_concept_relation_extractor(hidden_size)
            
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")
        
        # Graph Neural Network for reasoning
        self.graph_encoder = GraphEncoder(
            in_channels=gnn_hidden_size,
            hidden_channels=gnn_hidden_size,
            out_channels=hidden_size,
            num_layers=num_gnn_layers,
            gnn_type=gnn_type,
            dropout=dropout,
            edge_dim=edge_feature_dim if use_edge_features else None,
            heads=kwargs.get("heads", 8) if gnn_type == "gat" else None,
            attention_dropout=kwargs.get("attention_dropout", 0.1) if gnn_type == "gat" else None,
            readout="mean"
        )
        
        # Output projection to transform GNN outputs back to hidden size
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization for combining with original hidden states
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def _create_entity_extractor(self, hidden_size: int) -> nn.Module:
        """
        Create an entity extraction module.
        
        Args:
            hidden_size: Hidden dimension of the language model
            
        Returns:
            Entity extraction module
        """
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2)  # Binary classification: entity or not
        )
    
    def _create_relation_extractor(self, hidden_size: int, num_heads: int) -> nn.Module:
        """
        Create a relation extraction module.
        
        Args:
            hidden_size: Hidden dimension of the language model
            num_heads: Number of relation heads
            
        Returns:
            Relation extraction module
        """
        return nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_heads)  # Multi-class classification for relation types
        )
    
    def _create_concept_extractor(self, hidden_size: int) -> nn.Module:
        """
        Create a concept extraction module.
        
        Args:
            hidden_size: Hidden dimension of the language model
            
        Returns:
            Concept extraction module
        """
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)  # Concept score (higher means more likely to be a concept)
        )
    
    def _create_concept_relation_extractor(self, hidden_size: int) -> nn.Module:
        """
        Create a concept relation extraction module.
        
        Args:
            hidden_size: Hidden dimension of the language model
            
        Returns:
            Concept relation extraction module
        """
        return nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, self.num_edge_types)  # Multi-class for relation types
        )
    
    def _construct_word_level_graph(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Construct a word-level graph from hidden states.
        
        Args:
            hidden_states: Hidden states from the language model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (node_features, edge_index, edge_attr)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Use the first example in the batch for simplicity
        # In a real implementation, you would process each example in the batch
        hidden_states = hidden_states[0]  # [seq_len, hidden_size]
        attention_mask = attention_mask[0]  # [seq_len]
        
        # Filter out padding tokens
        valid_mask = attention_mask.bool()
        valid_hidden_states = hidden_states[valid_mask]  # [valid_seq_len, hidden_size]
        valid_seq_len = valid_hidden_states.size(0)
        
        # Limit to max_nodes
        valid_seq_len = min(valid_seq_len, self.max_nodes)
        valid_hidden_states = valid_hidden_states[:valid_seq_len]
        
        # Node features are just the hidden states
        node_features = self.node_feature_proj(valid_hidden_states)  # [valid_seq_len, gnn_hidden_size]
        
        # Compute attention scores between all pairs of tokens
        # This is a simple way to construct edges based on similarity
        scores = torch.matmul(
            self.attention_scores(valid_hidden_states),
            valid_hidden_states.transpose(0, 1)
        )  # [valid_seq_len, valid_seq_len]
        
        # Apply softmax to get attention weights
        scores = F.softmax(scores, dim=-1)
        
        # Threshold the scores to get sparse connections
        edge_mask = scores > self.edge_threshold
        
        # Convert to edge index
        edge_index = edge_mask.nonzero(as_tuple=False).t()  # [2, num_edges]
        
        # Limit to max_edges
        if edge_index.size(1) > self.max_edges:
            # Keep the edges with highest scores
            edge_scores = scores[edge_index[0], edge_index[1]]
            _, top_indices = edge_scores.topk(self.max_edges)
            edge_index = edge_index[:, top_indices]
        
        # Edge features (if enabled)
        edge_attr = None
        if self.use_edge_features:
            # For word-level, we use a simple heuristic based on the score
            # to assign edge types (discretize the range [0, 1] into num_edge_types bins)
            edge_scores = scores[edge_index[0], edge_index[1]]
            edge_types = (edge_scores * self.num_edge_types).long().clamp_(0, self.num_edge_types - 1)
            edge_attr = self.edge_embedding(edge_types)
        
        return node_features, edge_index, edge_attr
    
    def _construct_entity_level_graph(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Construct an entity-level graph from hidden states.
        
        Args:
            hidden_states: Hidden states from the language model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (node_features, edge_index, edge_attr)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Use the first example in the batch for simplicity
        hidden_states = hidden_states[0]  # [seq_len, hidden_size]
        attention_mask = attention_mask[0]  # [seq_len]
        
        # Filter out padding tokens
        valid_mask = attention_mask.bool()
        valid_hidden_states = hidden_states[valid_mask]  # [valid_seq_len, hidden_size]
        valid_seq_len = valid_hidden_states.size(0)
        
        # Entity extraction (binary classification)
        entity_scores = self.entity_extractor(valid_hidden_states)  # [valid_seq_len, 2]
        entity_probs = F.softmax(entity_scores, dim=-1)  # [valid_seq_len, 2]
        entity_mask = entity_probs[:, 1] > 0.5  # [valid_seq_len]
        
        # Extract entity hidden states
        entity_hidden_states = valid_hidden_states[entity_mask]  # [num_entities, hidden_size]
        num_entities = entity_hidden_states.size(0)
        
        # Limit to max_nodes
        num_entities = min(num_entities, self.max_nodes)
        entity_hidden_states = entity_hidden_states[:num_entities]
        
        # Node features are the entity hidden states
        node_features = self.node_feature_proj(entity_hidden_states)  # [num_entities, gnn_hidden_size]
        
        # Construct entity pairs for relation extraction
        sources, targets = torch.meshgrid(
            torch.arange(num_entities, device=device),
            torch.arange(num_entities, device=device),
            indexing='ij'
        )
        
        # Exclude self-loops
        mask = sources != targets
        sources, targets = sources[mask], targets[mask]
        
        # Limit to max_edges
        num_edges = min(sources.size(0), self.max_edges)
        sources, targets = sources[:num_edges], targets[:num_edges]
        
        # Create edge index
        edge_index = torch.stack([sources, targets], dim=0)  # [2, num_edges]
        
        # Extract relation features
        source_features = entity_hidden_states[sources]  # [num_edges, hidden_size]
        target_features = entity_hidden_states[targets]  # [num_edges, hidden_size]
        relation_features = torch.cat([source_features, target_features], dim=-1)  # [num_edges, hidden_size*2]
        
        # Relation classification
        relation_scores = self.relation_extractor(relation_features)  # [num_edges, num_relation_heads]
        relation_probs = F.softmax(relation_scores, dim=-1)  # [num_edges, num_relation_heads]
        relation_types = relation_probs.argmax(dim=-1)  # [num_edges]
        
        # Create edge attributes
        edge_attr = None
        if self.use_edge_features:
            # Map relation types to edge features
            edge_attr = self.edge_embedding(relation_types)  # [num_edges, edge_feature_dim]
        
        return node_features, edge_index, edge_attr
    
    def _construct_concept_level_graph(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Construct a concept-level graph from hidden states.
        
        Args:
            hidden_states: Hidden states from the language model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (node_features, edge_index, edge_attr)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Use the first example in the batch for simplicity
        hidden_states = hidden_states[0]  # [seq_len, hidden_size]
        attention_mask = attention_mask[0]  # [seq_len]
        
        # Filter out padding tokens
        valid_mask = attention_mask.bool()
        valid_hidden_states = hidden_states[valid_mask]  # [valid_seq_len, hidden_size]
        valid_seq_len = valid_hidden_states.size(0)
        
        # Concept extraction (continuous score)
        concept_scores = self.concept_extractor(valid_hidden_states).squeeze(-1)  # [valid_seq_len]
        
        # Select top-k nodes as concepts
        num_concepts = min(valid_seq_len, self.max_nodes)
        _, top_indices = concept_scores.topk(num_concepts)
        
        # Extract concept hidden states
        concept_hidden_states = valid_hidden_states[top_indices]  # [num_concepts, hidden_size]
        
        # Node features are the concept hidden states
        node_features = self.node_feature_proj(concept_hidden_states)  # [num_concepts, gnn_hidden_size]
        
        # Construct concept pairs for relation extraction
        sources, targets = torch.meshgrid(
            torch.arange(num_concepts, device=device),
            torch.arange(num_concepts, device=device),
            indexing='ij'
        )
        
        # Exclude self-loops
        mask = sources != targets
        sources, targets = sources[mask], targets[mask]
        
        # Limit to max_edges
        num_edges = min(sources.size(0), self.max_edges)
        sources, targets = sources[:num_edges], targets[:num_edges]
        
        # Create edge index
        edge_index = torch.stack([sources, targets], dim=0)  # [2, num_edges]
        
        # Extract concept relation features
        source_features = concept_hidden_states[sources]  # [num_edges, hidden_size]
        target_features = concept_hidden_states[targets]  # [num_edges, hidden_size]
        relation_features = torch.cat([source_features, target_features], dim=-1)  # [num_edges, hidden_size*2]
        
        # Concept relation classification
        relation_scores = self.concept_relation_extractor(relation_features)  # [num_edges, num_edge_types]
        relation_probs = F.softmax(relation_scores, dim=-1)  # [num_edges, num_edge_types]
        relation_types = relation_probs.argmax(dim=-1)  # [num_edges]
        
        # Create edge attributes
        edge_attr = None
        if self.use_edge_features:
            # Map relation types to edge features
            edge_attr = self.edge_embedding(relation_types)  # [num_edges, edge_feature_dim]
        
        return node_features, edge_index, edge_attr
    
    def construct_graph(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Construct a graph from hidden states based on the configured granularity.
        
        Args:
            hidden_states: Hidden states from the language model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (node_features, edge_index, edge_attr)
        """
        if self.granularity == "word":
            return self._construct_word_level_graph(hidden_states, attention_mask)
        elif self.granularity == "entity":
            return self._construct_entity_level_graph(hidden_states, attention_mask)
        elif self.granularity == "concept":
            return self._construct_concept_level_graph(hidden_states, attention_mask)
        else:
            raise ValueError(f"Unsupported granularity: {self.granularity}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        return_graph_outputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the graph reasoning module.
        
        Args:
            hidden_states: Hidden states from the language model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            return_graph_outputs: Whether to return the internal graph outputs
            
        Returns:
            Dictionary containing updated hidden states and optional graph outputs
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Construct the graph
        node_features, edge_index, edge_attr = self.construct_graph(hidden_states, attention_mask)
        
        # Apply GNN reasoning
        graph_outputs = self.graph_encoder(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_node_embeddings=True
        )
        
        # Get graph embedding (represents the reasoning result)
        # For now, we'll just use mean pooling over node embeddings
        graph_embedding = graph_outputs["node_embeddings"].mean(dim=0)  # [hidden_size]
        
        # Project to hidden size
        graph_embedding = self.output_proj(graph_embedding)  # [hidden_size]
        
        # Create a "reasoning vector" to be added to all hidden states
        reasoning_vector = graph_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size]
        reasoning_vector = reasoning_vector.expand(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        
        # Add the reasoning vector to the original hidden states
        updated_hidden_states = self.layer_norm(hidden_states + self.dropout_layer(reasoning_vector))
        
        # Prepare output
        outputs = {
            "hidden_states": updated_hidden_states
        }
        
        if return_graph_outputs:
            outputs["graph_outputs"] = graph_outputs
            outputs["graph_embedding"] = graph_embedding
            outputs["node_features"] = node_features
            outputs["edge_index"] = edge_index
            if edge_attr is not None:
                outputs["edge_attr"] = edge_attr
        
        return outputs 