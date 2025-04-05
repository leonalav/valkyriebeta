"""
Integration module for combining transformer models with graph neural networks.
Provides components for bridging transformers with GNNs to enhance structural reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import logging
import weakref

from model.gnn.graph_encoder import GraphEncoder
from model.gnn.gnn_model import GNNEncoder

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Registry for model components that enables dynamic registration and retrieval.
    Allows models to reference and interact with each other without circular dependencies.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one registry exists"""
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._components = {}
            cls._instance._weak_refs = {}  # Store weak references to avoid memory leaks
            cls._instance._hooks = {}  # Store hooks for component interactions
        return cls._instance
    
    def register(self, name: str, component: Any, overwrite: bool = False) -> bool:
        """
        Register a component with the registry
        
        Args:
            name: Unique identifier for the component
            component: The component to register
            overwrite: Whether to overwrite an existing component with the same name
            
        Returns:
            True if registration was successful, False otherwise
        """
        if name in self._components and not overwrite:
            logger.warning(f"Component '{name}' already exists in registry and overwrite=False")
            return False
        
        # Store weak reference to avoid memory leaks
        self._weak_refs[name] = weakref.ref(component)
        self._components[name] = component
        logger.debug(f"Registered component '{name}' in ModelRegistry")
        return True
    
    def get(self, name: str, default: Any = None) -> Any:
        """
        Retrieve a component from the registry
        
        Args:
            name: Identifier of the component to retrieve
            default: Default value to return if component doesn't exist
            
        Returns:
            The component or default if not found
        """
        # Clean up any weak references that have been garbage collected
        self._cleanup_refs()
        
        component = self._components.get(name, default)
        if component is None and default is None:
            logger.warning(f"Component '{name}' not found in registry")
        return component
    
    def list_components(self) -> List[str]:
        """
        List all registered component names
        
        Returns:
            List of component names
        """
        self._cleanup_refs()
        return list(self._components.keys())
    
    def register_hook(self, trigger_name: str, hook_fn: callable) -> int:
        """
        Register a hook function to be called when a trigger is activated
        
        Args:
            trigger_name: Name of the trigger that activates the hook
            hook_fn: Function to call when triggered
            
        Returns:
            Hook ID for later removal
        """
        if trigger_name not in self._hooks:
            self._hooks[trigger_name] = []
        
        hook_id = len(self._hooks[trigger_name])
        self._hooks[trigger_name].append(hook_fn)
        return hook_id
    
    def trigger_hooks(self, trigger_name: str, *args, **kwargs) -> List[Any]:
        """
        Trigger all hooks registered for a specific trigger name
        
        Args:
            trigger_name: Name of the trigger to activate
            *args, **kwargs: Arguments to pass to hook functions
            
        Returns:
            List of results from hook functions
        """
        if trigger_name not in self._hooks:
            return []
        
        results = []
        for hook_fn in self._hooks[trigger_name]:
            try:
                results.append(hook_fn(*args, **kwargs))
            except Exception as e:
                logger.error(f"Error in hook for trigger '{trigger_name}': {e}")
                results.append(None)
        
        return results
    
    def remove_hook(self, trigger_name: str, hook_id: int) -> bool:
        """
        Remove a registered hook
        
        Args:
            trigger_name: Name of the trigger
            hook_id: ID of the hook to remove
            
        Returns:
            True if hook was removed, False otherwise
        """
        if trigger_name not in self._hooks or hook_id >= len(self._hooks[trigger_name]):
            return False
        
        self._hooks[trigger_name].pop(hook_id)
        return True
    
    def unregister(self, name: str) -> bool:
        """
        Remove a component from the registry
        
        Args:
            name: Identifier of the component to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self._components:
            del self._components[name]
            if name in self._weak_refs:
                del self._weak_refs[name]
            return True
        return False
    
    def _cleanup_refs(self):
        """Clean up any weak references that have been garbage collected"""
        to_remove = []
        
        for name, ref in self._weak_refs.items():
            if ref() is None:  # Object has been garbage collected
                to_remove.append(name)
        
        for name in to_remove:
            if name in self._components:
                del self._components[name]
            del self._weak_refs[name]


class TransformerGNNIntegration(nn.Module):
    """
    Integration module that bridges transformer models with graph neural networks.
    Enables transformers to reason over graph-structured data by interleaving
    sequence processing with graph message passing.
    """
    
    def __init__(
        self,
        hidden_size: int,
        gnn_hidden_size: Optional[int] = None,
        num_gnn_layers: int = 3,
        gnn_dropout: float = 0.1,
        graph_residual: bool = True,
        graph_attention: bool = True,
        graph_message_passing_steps: int = 2,
        graph_readout_mode: str = "attention",
        gnn_model_type: str = "gcn",
        use_node_features: bool = True,
        use_edge_features: bool = True,
        bidirectional_messages: bool = True,
        text_to_graph_mode: str = "entities",
        register_with_model_registry: bool = True
    ):
        """
        Initialize the transformer-GNN integration module.
        
        Args:
            hidden_size: Hidden dimension of the transformer model
            gnn_hidden_size: Hidden dimension for the GNN (defaults to hidden_size)
            num_gnn_layers: Number of GNN layers
            gnn_dropout: Dropout rate for GNN layers
            graph_residual: Whether to use residual connections in the GNN
            graph_attention: Whether to use attention mechanism in the GNN
            graph_message_passing_steps: Number of message passing steps per GNN layer
            graph_readout_mode: How to convert graph representations back to sequence ("attention", "mean", "max")
            gnn_model_type: Type of GNN model to use ("gcn", "gat", "gin")
            use_node_features: Whether to use node features
            use_edge_features: Whether to use edge features
            bidirectional_messages: Whether to use bidirectional message passing
            text_to_graph_mode: Method for converting text to graphs ("entities", "syntax", "semantic")
            register_with_model_registry: Whether to register with the model registry
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.gnn_hidden_size = gnn_hidden_size or hidden_size
        self.graph_residual = graph_residual
        self.graph_attention = graph_attention
        self.graph_message_passing_steps = graph_message_passing_steps
        self.graph_readout_mode = graph_readout_mode
        self.text_to_graph_mode = text_to_graph_mode
        
        # Projection from transformer hidden size to GNN hidden size
        if self.hidden_size != self.gnn_hidden_size:
            self.hidden_to_gnn = nn.Linear(self.hidden_size, self.gnn_hidden_size)
            self.gnn_to_hidden = nn.Linear(self.gnn_hidden_size, self.hidden_size)
        
        # Create the GNN encoder
        self.gnn_encoder = GNNEncoder(
            hidden_size=self.gnn_hidden_size,
            num_layers=num_gnn_layers,
            dropout=gnn_dropout,
            use_node_features=use_node_features,
            use_edge_features=use_edge_features,
            residual=graph_residual,
            use_attention=graph_attention,
            message_passing_steps=graph_message_passing_steps,
            model_type=gnn_model_type,
            bidirectional=bidirectional_messages
        )
        
        # Graph encoder for converting node representations to graph representation
        self.graph_encoder = GraphEncoder(
            hidden_size=self.gnn_hidden_size,
            readout_mode=graph_readout_mode
        )
        
        # Determine if and how to combine the graph and transformer representations
        self.integration_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Sigmoid()
        )
        
        # Layer normalization for the integrated representation
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Register with model registry if requested
        if register_with_model_registry:
            registry = ModelRegistry()
            registry.register("transformer_gnn_integration", self)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        graph_data: Optional[Dict[str, Any]] = None,
        return_graph_representations: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Process hidden states with GNN if graph data is available.
        
        Args:
            hidden_states: Transformer hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            graph_data: Optional dictionary with graph data:
                - node_indices: List of node indices for each batch item
                - edge_indices: Edge index tensor [2, num_edges]
                - batch_indices: Batch indices for nodes
                - node_features: Optional node features
                - edge_features: Optional edge features
            return_graph_representations: Whether to return the graph representations
            
        Returns:
            If return_graph_representations is False:
                - Enhanced hidden states after graph processing
            Otherwise:
                - Tuple of (enhanced hidden states, graph representation dict)
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # If no graph data, return original hidden states
        if graph_data is None:
            if return_graph_representations:
                return hidden_states, {"graph_embedding": None, "node_embeddings": None}
            return hidden_states
        
        # Project to GNN dimension if needed
        gnn_inputs = hidden_states
        if hasattr(self, 'hidden_to_gnn'):
            gnn_inputs = self.hidden_to_gnn(hidden_states)
        
        # Extract graph data
        node_indices = graph_data.get("node_indices", None)
        edge_indices = graph_data.get("edge_indices", None)
        batch_indices = graph_data.get("batch_indices", None)
        node_features = graph_data.get("node_features", None)
        edge_features = graph_data.get("edge_features", None)
        
        # If node indices not provided, use all sequence positions
        if node_indices is None:
            node_indices = torch.arange(seq_len, device=device).expand(batch_size, -1)
        
        # If edge indices not provided, create a fully connected graph
        if edge_indices is None:
            # Create edges between all nodes in each batch item
            edge_indices = []
            offset = 0
            for b in range(batch_size):
                nodes = node_indices[b]
                num_nodes = len(nodes)
                
                # Create all pairwise edges
                src = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
                dst = torch.arange(num_nodes, device=device).repeat(num_nodes)
                
                # Add batch offset
                src += offset
                dst += offset
                
                # Stack to get edge indices
                batch_edges = torch.stack([src, dst], dim=0)
                edge_indices.append(batch_edges)
                
                # Update offset
                offset += num_nodes
            
            # Concatenate all batch edges
            edge_indices = torch.cat(edge_indices, dim=1)
        
        # Collect node features for the GNN
        node_embeddings = []
        for b in range(batch_size):
            # Get indices for this batch item
            indices = node_indices[b]
            if indices is not None:
                # Extract node embeddings from sequence
                node_embedding = gnn_inputs[b, indices]
                node_embeddings.append(node_embedding)
        
        # Stack all node embeddings
        node_embeddings = torch.cat(node_embeddings, dim=0)
        
        # Create batch indices if not provided
        if batch_indices is None:
            batch_indices = []
            for b in range(batch_size):
                indices = node_indices[b]
                if indices is not None:
                    batch_indices.append(torch.full((len(indices),), b, device=device))
            batch_indices = torch.cat(batch_indices, dim=0)
        
        # Apply GNN to node representations
        gnn_output = self.gnn_encoder(
            node_embeddings,
            edge_indices,
            batch_indices,
            node_features,
            edge_features
        )
        
        # Get graph-level representation
        graph_embedding = self.graph_encoder(
            gnn_output,
            batch_indices,
            batch_size
        )
        
        # Map back to transformer hidden dimension if needed
        if hasattr(self, 'gnn_to_hidden'):
            gnn_output = self.gnn_to_hidden(gnn_output)
            graph_embedding = self.gnn_to_hidden(graph_embedding)
        
        # Distribute node embeddings back to their positions in the sequence
        enhanced_hidden_states = hidden_states.clone()
        
        node_offset = 0
        for b in range(batch_size):
            indices = node_indices[b]
            if indices is not None:
                num_nodes = len(indices)
                # Update hidden states at node positions
                enhanced_hidden_states[b, indices] = gnn_output[node_offset:node_offset+num_nodes]
                node_offset += num_nodes
        
        # Apply graph embedding to the entire sequence through gating
        graph_embedding_expanded = graph_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute integration gate
        gate = self.integration_gate(
            torch.cat([enhanced_hidden_states, graph_embedding_expanded], dim=-1)
        )
        
        # Apply gated integration
        integrated_hidden_states = (
            enhanced_hidden_states * gate + graph_embedding_expanded * (1 - gate)
        )
        
        # Apply layer normalization
        output = self.layer_norm(integrated_hidden_states + hidden_states)  # Add residual connection
        
        if return_graph_representations:
            graph_repr = {
                "graph_embedding": graph_embedding,
                "node_embeddings": gnn_output,
                "batch_indices": batch_indices
            }
            return output, graph_repr
        
        return output
    
    def create_graph_from_text(self, texts: List[str], tokenizer=None):
        """
        Create graph data from text for use with the integration module.
        
        Args:
            texts: List of text inputs
            tokenizer: Optional tokenizer for mapping text to tokens
            
        Returns:
            Dictionary with graph data
        """
        try:
            from model.gnn.utils import TextToGraphExtractor
            
            extractor = TextToGraphExtractor(
                graph_min_entities=2,
                graph_max_entities=50,
                create_reverse_edges=True,
                extract_attributes=True
            )
            
            graphs = []
            for text in texts:
                graph = extractor.extract_graph(text)
                graphs.append(graph)
            
            # Convert to batched graph format
            batched_graph = self._batch_graphs(graphs)
            return batched_graph
            
        except ImportError:
            logger.warning("TextToGraphExtractor not found, returning empty graph data")
            return {
                "node_indices": None,
                "edge_indices": None,
                "batch_indices": None
            }
    
    def _batch_graphs(self, graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert list of individual graphs to batched graph format.
        
        Args:
            graphs: List of graph dictionaries
            
        Returns:
            Batched graph data
        """
        if not graphs or all(g is None for g in graphs):
            return {
                "node_indices": None,
                "edge_indices": None,
                "batch_indices": None
            }
        
        # Remove None graphs
        graphs = [g for g in graphs if g is not None]
        if not graphs:
            return {
                "node_indices": None,
                "edge_indices": None, 
                "batch_indices": None
            }
        
        # Collect node indices
        node_indices = [g.get("node_indices", list(range(g["num_nodes"]))) for g in graphs]
        
        # Collect edge indices with offsets
        edge_indices = []
        batch_indices = []
        offset = 0
        
        for i, graph in enumerate(graphs):
            edges = graph.get("edges", [])
            num_nodes = graph["num_nodes"]
            
            # Convert edges to tensor format
            if edges:
                if isinstance(edges[0], (list, tuple)) and len(edges[0]) == 2:
                    # Edges are already in (src, dst) format
                    src_nodes = [e[0] + offset for e in edges]
                    dst_nodes = [e[1] + offset for e in edges]
                else:
                    # Assume edges are in COO format
                    src_nodes = [edges[0][j] + offset for j in range(len(edges[0]))]
                    dst_nodes = [edges[1][j] + offset for j in range(len(edges[1]))]
                
                edge_indices.extend(list(zip(src_nodes, dst_nodes)))
            
            # Create batch indices
            batch_indices.extend([i] * num_nodes)
            
            # Update offset
            offset += num_nodes
        
        # Convert to tensors
        if edge_indices:
            edge_indices = torch.tensor(edge_indices, dtype=torch.long).t()  # [2, num_edges]
        else:
            edge_indices = torch.zeros((2, 0), dtype=torch.long)
            
        batch_indices = torch.tensor(batch_indices, dtype=torch.long)
        
        # Collect node features if available
        node_features = None
        if any(g.get("node_features") is not None for g in graphs):
            # Concatenate node features
            features = []
            for graph in graphs:
                if graph.get("node_features") is not None:
                    features.append(graph["node_features"])
                else:
                    # Create dummy features with same size as others
                    feat_size = next((g["node_features"].shape[1] for g in graphs if g.get("node_features") is not None), 0)
                    features.append(torch.zeros(graph["num_nodes"], feat_size))
            
            if features:
                node_features = torch.cat(features, dim=0)
        
        return {
            "node_indices": node_indices,
            "edge_indices": edge_indices,
            "batch_indices": batch_indices,
            "node_features": node_features
        }

# Create singleton registry instance for global access
registry = ModelRegistry() 