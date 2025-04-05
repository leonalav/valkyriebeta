"""
Comprehensive Graph Reasoning System for ValkyrieLLM.

This module provides high-level interfaces for graph-based reasoning within
the language model, bringing together all the graph neural network components
and integration mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from .graph_encoder import GraphEncoder
from .graph_reasoning import GraphReasoner
from .integration import TransformerGNNIntegration
from .gnn_model import GraphTransformer, GNNEncoder


class ReasoningSystem(nn.Module):
    """
    End-to-end graph-based reasoning system for language understanding and generation.
    
    This class orchestrates the process of:
    1. Graph construction from language model hidden states
    2. Graph-based reasoning using GNNs
    3. Integration of graph reasoning results back into the language model
    4. Multi-step reasoning capabilities
    """
    
    def __init__(
        self,
        hidden_size: int,
        gnn_hidden_size: int = 256,
        num_gnn_layers: int = 3,
        gnn_type: str = "gat",
        integration_strategy: str = "sequential",
        granularity: str = "entity",
        num_reasoning_steps: int = 1,
        use_graph_transformer: bool = True,
        num_transformer_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        edge_feature_dim: int = 64,
        num_edge_types: int = 8,
        reasoning_mode: str = "multi_step"
    ):
        """
        Initialize the reasoning system.
        
        Args:
            hidden_size: Hidden dimension of the language model
            gnn_hidden_size: Hidden dimension for the GNN
            num_gnn_layers: Number of GNN layers
            gnn_type: Type of GNN to use
            integration_strategy: How to integrate GNN with transformer ("sequential", "parallel", "iterative")
            granularity: Level of graph construction ("word", "entity", "concept")
            num_reasoning_steps: Number of reasoning steps to perform
            use_graph_transformer: Whether to use the GraphTransformer model
            num_transformer_layers: Number of transformer layers in the GraphTransformer
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            edge_feature_dim: Dimension of edge features
            num_edge_types: Number of edge types for edge features
            reasoning_mode: Mode of reasoning ("multi_step", "recursive", "adaptive")
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.gnn_hidden_size = gnn_hidden_size
        self.num_gnn_layers = num_gnn_layers
        self.gnn_type = gnn_type
        self.integration_strategy = integration_strategy
        self.granularity = granularity
        self.num_reasoning_steps = num_reasoning_steps
        self.use_graph_transformer = use_graph_transformer
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.edge_feature_dim = edge_feature_dim
        self.num_edge_types = num_edge_types
        self.reasoning_mode = reasoning_mode
        
        # Create the graph reasoner
        self.graph_reasoner = GraphReasoner(
            hidden_size=hidden_size,
            gnn_hidden_size=gnn_hidden_size,
            num_gnn_layers=num_gnn_layers,
            gnn_type=gnn_type,
            granularity=granularity,
            dropout=dropout,
            use_edge_features=True,
            edge_feature_dim=edge_feature_dim,
            num_edge_types=num_edge_types
        )
        
        # Create the integration module
        self.integration = TransformerGNNIntegration(
            hidden_size=hidden_size,
            integration_strategy=integration_strategy,
            gnn_hidden_size=gnn_hidden_size,
            num_gnn_layers=num_gnn_layers,
            gnn_type=gnn_type,
            granularity=granularity,
            num_iterations=num_reasoning_steps,
            dropout=dropout,
            edge_feature_dim=edge_feature_dim,
            num_edge_types=num_edge_types
        )
        
        # Optional graph transformer for more sophisticated reasoning
        if use_graph_transformer:
            self.graph_transformer = GraphTransformer(
                input_dim=hidden_size,
                hidden_dim=gnn_hidden_size,
                output_dim=hidden_size,
                num_gnn_layers=num_gnn_layers,
                num_transformer_layers=num_transformer_layers,
                num_heads=num_attention_heads,
                dropout=dropout,
                gnn_type=gnn_type,
                use_edge_features=True,
                edge_dim=edge_feature_dim,
                readout_type="attention"
            )
        
        # For multi-step reasoning
        if reasoning_mode == "multi_step" or reasoning_mode == "adaptive":
            self.reasoning_gates = nn.ModuleList([
                nn.Linear(hidden_size, 1) for _ in range(num_reasoning_steps)
            ])
        
        # For adaptive reasoning
        if reasoning_mode == "adaptive":
            self.adaptive_controller = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_reasoning_steps)
            )
        
        # Layer normalization for output
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def _single_reasoning_step(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a single reasoning step.
        
        Args:
            hidden_states: Hidden states from the language model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Updated hidden states [batch_size, seq_len, hidden_size]
        """
        # Use the graph reasoner to extract and reason over graphs
        outputs = self.graph_reasoner(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_graph_outputs=True
        )
        
        return outputs["hidden_states"]
    
    def _multi_step_reasoning(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform multi-step reasoning.
        
        Args:
            hidden_states: Hidden states from the language model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Updated hidden states [batch_size, seq_len, hidden_size]
        """
        current_states = hidden_states
        
        for i in range(self.num_reasoning_steps):
            # Get gate value to control how much reasoning to apply
            if hasattr(self, 'reasoning_gates'):
                # Compute average representation
                pooled_states = torch.mean(current_states * attention_mask.unsqueeze(-1), dim=1)
                gate = torch.sigmoid(self.reasoning_gates[i](pooled_states)).unsqueeze(1)
            else:
                gate = 1.0
            
            # Single reasoning step
            reasoning_output = self._single_reasoning_step(current_states, attention_mask)
            
            # Apply gating
            current_states = gate * reasoning_output + (1 - gate) * current_states
            
            # Apply layer norm
            current_states = self.layer_norm(current_states)
        
        return current_states
    
    def _adaptive_reasoning(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform adaptive reasoning where the model decides how many steps to take.
        
        Args:
            hidden_states: Hidden states from the language model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Updated hidden states [batch_size, seq_len, hidden_size]
        """
        current_states = hidden_states
        
        # Get adaptive weights for each reasoning step
        pooled_states = torch.mean(hidden_states * attention_mask.unsqueeze(-1), dim=1)
        step_weights = F.softmax(self.adaptive_controller(pooled_states), dim=-1)
        
        # Perform reasoning for each step and combine with weights
        weighted_output = 0
        
        for i in range(self.num_reasoning_steps):
            # Single reasoning step
            step_output = self._single_reasoning_step(current_states, attention_mask)
            
            # Update running weighted output
            step_weight = step_weights[:, i].unsqueeze(1).unsqueeze(2)
            weighted_output = weighted_output + step_weight * step_output
            
            # Update current states for next step
            current_states = step_output
        
        return weighted_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        return_graph_outputs: bool = False,
        layer_hidden_states: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the reasoning system.
        
        Args:
            hidden_states: Hidden states from the language model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            return_graph_outputs: Whether to return the internal graph outputs
            layer_hidden_states: Optional list of hidden states from each transformer layer
            
        Returns:
            Dictionary containing updated hidden states and optional outputs
        """
        # Choose reasoning strategy based on mode
        if self.reasoning_mode == "multi_step":
            updated_hidden_states = self._multi_step_reasoning(hidden_states, attention_mask)
        elif self.reasoning_mode == "adaptive":
            updated_hidden_states = self._adaptive_reasoning(hidden_states, attention_mask)
        else:  # Default to using the integration module
            outputs = self.integration(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                transformer_outputs=None,
                layer_hidden_states=layer_hidden_states,
                return_graph_outputs=return_graph_outputs
            )
            updated_hidden_states = outputs["hidden_states"]
        
        # Apply dropout
        updated_hidden_states = self.dropout_layer(updated_hidden_states)
        
        # Create output dictionary
        outputs = {"hidden_states": updated_hidden_states}
        
        if return_graph_outputs:
            # Extract graph from hidden states
            graph_outputs = self.graph_reasoner(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                return_graph_outputs=True
            )
            
            # Add graph outputs to the return dictionary
            for key, value in graph_outputs.items():
                if key != "hidden_states":
                    outputs[f"graph_{key}"] = value
        
        return outputs


class ReasoningLayers(nn.Module):
    """
    Reasoning layers that can be inserted into a transformer model.
    
    This module provides graph-based reasoning capabilities that can be inserted
    between transformer layers to enhance the model's reasoning abilities.
    """
    
    def __init__(
        self,
        config,
        layer_index: int = -1,
        num_layers: int = 1,
        insertion_mode: str = "fixed",
        reasoning_system_params: Optional[Dict] = None
    ):
        """
        Initialize the reasoning layers.
        
        Args:
            config: Model configuration
            layer_index: Which transformer layer to insert after (-1 means after the last layer)
            num_layers: Number of reasoning layers to insert
            insertion_mode: Mode of insertion ("fixed", "every_n", "adaptive")
            reasoning_system_params: Parameters for the reasoning system
        """
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.num_layers = num_layers
        self.insertion_mode = insertion_mode
        
        # Default reasoning system parameters
        default_params = {
            "hidden_size": config.hidden_size,
            "gnn_hidden_size": config.hidden_size // 2,
            "num_gnn_layers": 3,
            "gnn_type": "gat",
            "integration_strategy": "sequential",
            "granularity": "entity",
            "num_reasoning_steps": 1,
            "use_graph_transformer": False,
            "dropout": config.hidden_dropout_prob
        }
        
        # Update with user-provided parameters
        if reasoning_system_params:
            default_params.update(reasoning_system_params)
        
        # Create reasoning systems
        self.reasoning_systems = nn.ModuleList([
            ReasoningSystem(**default_params) for _ in range(num_layers)
        ])
        
        # Adaptive insertion parameters
        if insertion_mode == "adaptive":
            self.insertion_controller = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_index: int = -1,
        all_hidden_states: Optional[List[torch.Tensor]] = None,
        return_graph_outputs: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the reasoning layers.
        
        Args:
            hidden_states: Hidden states from the transformer layer [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            layer_index: Current transformer layer index
            all_hidden_states: List of all hidden states from previous layers
            return_graph_outputs: Whether to return the internal graph outputs
            
        Returns:
            Updated hidden states or dictionary with outputs
        """
        # Determine if reasoning should be applied based on the insertion mode
        apply_reasoning = False
        
        if self.insertion_mode == "fixed":
            # Apply at the specified layer index
            apply_reasoning = (layer_index == self.layer_index) or (self.layer_index == -1 and layer_index == -1)
        elif self.insertion_mode == "every_n":
            # Apply every n layers
            apply_reasoning = (layer_index % self.num_layers == 0)
        elif self.insertion_mode == "adaptive":
            # Decide adaptively whether to apply reasoning
            pooled_states = torch.mean(hidden_states, dim=1)
            insertion_prob = self.insertion_controller(pooled_states).squeeze(-1)
            apply_reasoning = (torch.rand_like(insertion_prob) < insertion_prob).any().item()
        
        # Apply reasoning if determined
        if apply_reasoning:
            outputs = {}
            current_states = hidden_states
            
            for reasoning_system in self.reasoning_systems:
                system_outputs = reasoning_system(
                    hidden_states=current_states,
                    attention_mask=attention_mask,
                    return_graph_outputs=return_graph_outputs,
                    layer_hidden_states=all_hidden_states
                )
                
                current_states = system_outputs["hidden_states"]
                
                # Collect outputs for return
                if return_graph_outputs:
                    for key, value in system_outputs.items():
                        if key != "hidden_states":
                            outputs[key] = value
            
            if return_graph_outputs:
                outputs["hidden_states"] = current_states
                return outputs
            else:
                return current_states
        else:
            # No reasoning applied, return as is
            if return_graph_outputs:
                return {"hidden_states": hidden_states}
            else:
                return hidden_states


class GraphReasoningModel(nn.Module):
    """
    A complete model that combines a transformer with graph-based reasoning.
    
    This class implements a end-to-end model that can process input text,
    construct and reason over knowledge graphs, and generate outputs with
    enhanced reasoning capabilities.
    """
    
    def __init__(
        self,
        config,
        base_model: nn.Module,
        reasoning_mode: str = "integrated",
        reasoning_params: Optional[Dict] = None
    ):
        """
        Initialize the graph reasoning model.
        
        Args:
            config: Model configuration
            base_model: Base transformer model
            reasoning_mode: How to integrate reasoning ("integrated", "layer_insert", "output_only")
            reasoning_params: Parameters for the reasoning components
        """
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.reasoning_mode = reasoning_mode
        
        # Default reasoning parameters
        default_params = {
            "hidden_size": config.hidden_size,
            "gnn_hidden_size": config.hidden_size // 2,
            "num_gnn_layers": 3,
            "gnn_type": "gat",
            "granularity": "entity",
            "integration_strategy": "sequential",
            "num_reasoning_steps": 2
        }
        
        # Update with user-provided parameters
        if reasoning_params:
            default_params.update(reasoning_params)
        
        # Create reasoning components based on mode
        if reasoning_mode == "integrated":
            # Replace some transformer layers with graph-augmented ones
            self.reasoning_layers = ReasoningLayers(
                config=config,
                layer_index=config.num_hidden_layers // 2,  # Insert in the middle
                num_layers=1,
                insertion_mode="fixed",
                reasoning_system_params=default_params
            )
            
        elif reasoning_mode == "layer_insert":
            # Insert reasoning layers after specific transformer layers
            self.reasoning_layers = ReasoningLayers(
                config=config,
                layer_index=-1,  # After the last layer
                num_layers=1,
                insertion_mode="fixed",
                reasoning_system_params=default_params
            )
            
        elif reasoning_mode == "output_only":
            # Apply reasoning only on the final output
            self.reasoning_system = ReasoningSystem(**default_params)
        
        else:
            raise ValueError(f"Unknown reasoning mode: {reasoning_mode}")
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_graph_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the graph reasoning model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_graph_outputs: Whether to return graph reasoning outputs
            **kwargs: Additional arguments for the base model
            
        Returns:
            Output logits or dictionary with outputs
        """
        outputs = {}
        
        # Process through base model
        if self.reasoning_mode == "integrated":
            # Need to modify the base model to apply reasoning at specific layers
            # This implementation will depend on the specific base model structure
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )
            
            # Get the hidden states
            hidden_states = base_outputs.last_hidden_state
            all_hidden_states = base_outputs.hidden_states
            
            # Apply reasoning layers
            if return_graph_outputs:
                reasoning_outputs = self.reasoning_layers(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    all_hidden_states=all_hidden_states,
                    return_graph_outputs=True
                )
                hidden_states = reasoning_outputs["hidden_states"]
                
                # Collect graph outputs
                for key, value in reasoning_outputs.items():
                    if key != "hidden_states":
                        outputs[key] = value
            else:
                hidden_states = self.reasoning_layers(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    all_hidden_states=all_hidden_states
                )
            
        elif self.reasoning_mode == "layer_insert":
            # Run base model normally
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
            # Get the hidden states
            hidden_states = base_outputs.last_hidden_state
            
            # Apply reasoning layers
            if return_graph_outputs:
                reasoning_outputs = self.reasoning_layers(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    return_graph_outputs=True
                )
                hidden_states = reasoning_outputs["hidden_states"]
                
                # Collect graph outputs
                for key, value in reasoning_outputs.items():
                    if key != "hidden_states":
                        outputs[key] = value
            else:
                hidden_states = self.reasoning_layers(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask
                )
            
        elif self.reasoning_mode == "output_only":
            # Run base model normally
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            
            # Get the hidden states
            hidden_states = base_outputs.last_hidden_state
            
            # Apply reasoning system to the output
            reasoning_outputs = self.reasoning_system(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                return_graph_outputs=return_graph_outputs
            )
            
            hidden_states = reasoning_outputs["hidden_states"]
            
            # Collect graph outputs if requested
            if return_graph_outputs:
                for key, value in reasoning_outputs.items():
                    if key != "hidden_states":
                        outputs[key] = value
        
        # Apply output projection
        logits = self.output_projection(self.dropout(hidden_states))
        
        # Prepare final outputs
        outputs["logits"] = logits
        
        if not return_graph_outputs:
            return logits
        else:
            return outputs 