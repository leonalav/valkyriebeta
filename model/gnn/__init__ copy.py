"""
Graph Neural Network (GNN) implementations for Valkyrie LLM.

This module provides various GNN architectures for processing graph-structured data,
including Graph Attention Networks (GAT), Graph Convolutional Networks (GCN),
and more advanced models like GraphTransformer, HGT, TreeGNN, and contrastive learning approaches.
"""

# Basic GNN models
from valkyrie_llm.model.gnn.models import GCN, GAT, GIN, EGNN
from valkyrie_llm.model.gnn.graph_encoder import GraphEncoder
from valkyrie_llm.model.gnn.adapters import HGTAdapter

# Advanced GNN models
from valkyrie_llm.model.gnn.layers import GraphTransformer, GraphTransformerLayer, DiffPool, DiffPoolBlock
from valkyrie_llm.model.gnn.tree_gnn import TreeLSTMCell, RecursiveTreeGNN, TreeGNN
from valkyrie_llm.model.gnn.contrastive import ProjectionHead, GraphCL, InfoGraph

# Model integration utilities
from valkyrie_llm.model.gnn.integration import TransformerGNNIntegration, ModelRegistry

# Training utilities
from valkyrie_llm.model.gnn.training import TrainingConfig, GNNTrainer, IntegratedGNNTrainer, create_gnn_model

__all__ = [
    # Basic GNN models
    'GCN', 'GAT', 'GIN', 'EGNN',
    'GraphEncoder',
    'HGTAdapter',
    
    # Advanced GNN models
    'GraphTransformer', 'GraphTransformerLayer',
    'DiffPool', 'DiffPoolBlock',
    'TreeLSTMCell', 'RecursiveTreeGNN', 'TreeGNN',
    'ProjectionHead', 'GraphCL', 'InfoGraph',
    
    # Model integration utilities
    'TransformerGNNIntegration',
    'ModelRegistry',
    
    # Training utilities
    'TrainingConfig',
    'GNNTrainer',
    'IntegratedGNNTrainer',
    'create_gnn_model',
] 