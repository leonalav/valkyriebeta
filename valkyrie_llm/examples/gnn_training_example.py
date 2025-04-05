#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use the Valkyrie LLM GNN models and training utilities.

This script provides a complete example of configuring, training, and evaluating 
various GNN models implemented in the Valkyrie LLM framework, including:
- Graph Transformer (Graphormer)
- Heterogeneous Graph Transformer (HGT)
- Tree-Structured GNN
- Contrastive Learning with GraphCL
- Integrated GNN-Transformer model

Usage:
    python -m valkyrie_llm.examples.gnn_training_example --model graphormer
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures

from valkyrie_llm.model.gnn.layers import GraphTransformer
from valkyrie_llm.model.gnn.tree_gnn import TreeGNN
from valkyrie_llm.model.gnn.contrastive import GraphCL
from valkyrie_llm.model.gnn.integration import TransformerGNNIntegration, ModelRegistry
from valkyrie_llm.model.gnn.training import TrainingConfig, GNNTrainer, IntegratedGNNTrainer

# Setup argparse
parser = argparse.ArgumentParser(description='GNN Training Example')
parser.add_argument('--model', type=str, default='graphormer', 
                    choices=['graphormer', 'tree_gnn', 'graphcl', 'integrated'],
                    help='GNN model to train')
parser.add_argument('--dataset', type=str, default='PROTEINS',
                    help='Dataset to use for training')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--hidden_channels', type=int, default=64,
                    help='Number of hidden channels')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device to use for training')


def load_dataset(name):
    """Load a dataset from PyTorch Geometric."""
    # Apply normalization transform to features
    transform = NormalizeFeatures()
    
    # Load dataset
    dataset = TUDataset(root='data/TUDataset', name=name, transform=transform)
    
    # Get input dimension
    if hasattr(dataset, 'num_features'):
        in_channels = dataset.num_features
    else:
        in_channels = dataset[0].x.size(1) if dataset[0].x is not None else 1
    
    # Get output dimension
    if hasattr(dataset, 'num_classes'):
        out_channels = dataset.num_classes
    else:
        out_channels = int(dataset[0].y.max().item()) + 1 if dataset[0].y is not None else 2
    
    # Split dataset into train, validation, and test sets
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    return train_loader, val_loader, test_loader, in_channels, out_channels


def create_model(model_name, in_channels, out_channels, hidden_channels):
    """Create a GNN model based on the provided name."""
    if model_name == 'graphormer':
        return GraphTransformer(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=4,
            num_heads=4,
            dropout=args.dropout,
        )
    elif model_name == 'tree_gnn':
        return TreeGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=4,
            dropout=args.dropout,
        )
    elif model_name == 'graphcl':
        # Create a base GNN model for GraphCL
        base_gnn = ModelRegistry.create_model(
            'GIN',
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # GraphCL will handle the final classification
            num_layers=4,
            dropout=args.dropout,
        )
        
        return GraphCL(
            gnn=base_gnn,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            proj_dim=hidden_channels,
            dropout=args.dropout,
        )
    elif model_name == 'integrated':
        # Create a mock transformer for the integrated model
        transformer_dim = 768  # Common dimension for transformer models
        
        # Mock transformer for the example
        class MockTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10000, transformer_dim)
                self.layers = nn.ModuleList([
                    nn.Linear(transformer_dim, transformer_dim) 
                    for _ in range(2)
                ])
            
            def forward(self, input_ids):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = F.relu(layer(x))
                return x
        
        # Create the integrated model
        return TransformerGNNIntegration(
            transformer=MockTransformer(),
            transformer_dim=transformer_dim,
            graph_dim=in_channels,
            hidden_dim=hidden_channels,
            output_dim=out_channels,
            num_layers=3,
            use_graph_attention=True,
            use_tree_structure=False,
            use_contrastive=True,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def criterion(outputs, batch):
    """Define the loss function for the model."""
    if isinstance(outputs, dict):
        logits = outputs['logits']
    else:
        logits = outputs
    
    # Extract ground truth from batch
    if hasattr(batch, 'y'):
        y = batch.y
    else:
        y = batch['y']
    
    # Handle different label formats
    if len(y.shape) > 1 and y.shape[1] > 1:
        # Multi-class or multi-label classification
        return F.cross_entropy(logits, y)
    else:
        # Binary classification or single-label classification
        if len(y.shape) > 1:
            y = y.squeeze(1)
        return F.cross_entropy(logits, y.long())


def metric_fn(outputs, batch):
    """Define the evaluation metric for the model."""
    if isinstance(outputs, dict):
        logits = outputs['logits']
    else:
        logits = outputs
    
    # Extract ground truth from batch
    if hasattr(batch, 'y'):
        y = batch.y
    else:
        y = batch['y']
    
    # Handle different label formats
    if len(y.shape) > 1 and y.shape[1] > 1:
        # Multi-class or multi-label classification
        pred = logits.argmax(dim=1)
        y_true = y.argmax(dim=1)
    else:
        # Binary classification or single-label classification
        pred = logits.argmax(dim=1)
        if len(y.shape) > 1:
            y_true = y.squeeze(1)
        else:
            y_true = y
    
    # Calculate accuracy
    correct = (pred == y_true).sum()
    accuracy = correct.float() / y_true.size(0)
    
    return accuracy


def main(args):
    """Main function for training and evaluating GNN models."""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Load dataset
    train_loader, val_loader, test_loader, in_channels, out_channels = load_dataset(args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Number of input features: {in_channels}")
    print(f"Number of output classes: {out_channels}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    print(f"Number of test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = create_model(args.model, in_channels, out_channels, args.hidden_channels)
    print(f"Model: {args.model}")
    
    # Create training configuration
    config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_epochs=5,
        lr_scheduler="cosine",
        grad_clip=1.0,
        grad_accumulation_steps=1,
        early_stopping_patience=10,
        optimizer_type="adamw",
        mixed_precision=True,
        save_best_model=True,
        checkpoint_dir=f"checkpoints/{args.model}_{args.dataset}",
        log_interval=1,
        dropout=args.dropout,
        seed=args.seed,
    )
    
    # Create trainer based on model type
    if args.model == 'integrated':
        trainer = IntegratedGNNTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=args.device,
            transformer_lr_factor=0.1,
        )
    else:
        trainer = GNNTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=args.device,
        )
    
    # Train the model
    history = trainer.train(
        criterion=criterion,
        metric_fn=metric_fn,
        optimize_metric=True,
    )
    
    # Test the model
    test_loss, test_acc = trainer.test(
        criterion=criterion,
        metric_fn=metric_fn,
    )
    
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    
    # Plot training history if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        # Create directory for plots
        os.makedirs('plots', exist_ok=True)
        
        # Plot loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_metric'], label='Train Accuracy')
        plt.plot(history['val_metric'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"plots/{args.model}_{args.dataset}_history.png")
        plt.close()
        
        print(f"Training history plot saved to plots/{args.model}_{args.dataset}_history.png")
    except ImportError:
        print("Matplotlib not available. Skipping plotting.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args) 