#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to train an LLM with integrated GNN components.

This script provides a complete example of setting up and training a language model
enhanced with graph neural networks for improved reasoning over structured data.
"""

import argparse
import os
import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset

from valkyrie_llm.model.gnn_llm_trainer import GNNLLMTrainer
from valkyrie_llm.model.gnn.models import GIN  # Import from our models.py file
from valkyrie_llm.model.gnn.integration import TransformerGNNIntegration
from valkyrie_llm.model.gnn.training import TrainingConfig


# Define mock implementations for PyTorch Geometric and OGB dependencies
# These are simplified versions for demonstration purposes
class MockData:
    """Mock implementation of PyTorch Geometric Data object."""
    def __init__(self, x, edge_index, y=None, edge_attr=None):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.edge_attr = edge_attr


class MockDataLoader:
    """Mock implementation of PyTorch Geometric DataLoader."""
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            yield [self.dataset[j] for j in batch_indices]
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class MockGraphDataset:
    """Mock dataset for graph data."""
    def __init__(self, num_graphs=100, num_features=9, num_classes=2):
        self.data = []
        for i in range(num_graphs):
            # Create random graph with 5-10 nodes
            num_nodes = np.random.randint(5, 11)
            
            # Create random node features
            x = torch.randn(num_nodes, num_features)
            
            # Create random edges (ensuring connected graph)
            edges = []
            for j in range(num_nodes):
                # Connect to at least one other node
                target = (j + 1) % num_nodes
                edges.append((j, target))
                edges.append((target, j))
                
                # Add some random edges
                for _ in range(np.random.randint(0, 3)):
                    target = np.random.randint(0, num_nodes)
                    if target != j:
                        edges.append((j, target))
                        edges.append((target, j))
            
            # Convert edges to tensor
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            
            # Create random edge features
            edge_attr = torch.randn(edge_index.size(1), 3)
            
            # Create random label
            y = torch.tensor([np.random.randint(0, num_classes)], dtype=torch.long)
            
            # Create data object
            self.data.append(MockData(x, edge_index, y, edge_attr))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# Setup argument parser
parser = argparse.ArgumentParser(description='Train an LLM with integrated GNN components')
parser.add_argument('--llm_model', type=str, default='gpt2', 
                    help='Pre-trained language model to use')
parser.add_argument('--gnn_type', type=str, default='gin',
                    choices=['gin', 'gat', 'graphormer', 'tree_gnn'],
                    help='GNN architecture to use')
parser.add_argument('--dataset', type=str, default='ogbg-molhiv',
                    help='Graph dataset to use')
parser.add_argument('--text_dataset', type=str, default='wikitext',
                    help='Text dataset to use for language modeling')
parser.add_argument('--output_dir', type=str, default='./outputs',
                    help='Directory to save model checkpoints')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=3,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='Learning rate')
parser.add_argument('--gnn_lr', type=float, default=1e-4,
                    help='Learning rate for GNN component')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed')
parser.add_argument('--fp16', action='store_true',
                    help='Use mixed precision training')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device to use for training')


def prepare_combined_dataset(text_dataset, graph_dataset):
    """
    Prepare a combined dataset of text and graph data.
    
    This function creates a synthetic dataset that pairs text data with graph data,
    simulating a scenario where language models need to reason over both textual
    and graph-structured information.
    """
    # Load graph data using our mock implementation
    print(f"Loading graph dataset: {graph_dataset}")
    graph_data = MockGraphDataset(num_graphs=100, num_features=9, num_classes=2)
    
    # Extract graph features
    graphs = []
    for i in range(min(len(graph_data), 100)):  # Limit to 100 graphs for demo
        g = graph_data[i]
        graphs.append({
            'node_features': g.x,
            'edge_index': g.edge_index,
            'edge_attr': g.edge_attr,
            'y': g.y,
        })
    
    # Load text data (using first 100 examples for demo)
    print(f"Loading text dataset: {text_dataset}")
    if text_dataset.startswith('wikitext'):
        # For demonstration, generate mock text data if unable to load dataset
        try:
            text_data = load_dataset('wikitext', 'wikitext-103-v1')['train']
            texts = text_data['text'][:100]  # Take subset for demonstration
        except Exception as e:
            print(f"Error loading wikitext dataset: {e}")
            # Generate mock text data
            texts = [f"This is a sample text document {i} for demonstration." for i in range(100)]
    else:
        # Generate mock text data
        texts = [f"This is a sample text document {i} for the {text_dataset} dataset." for i in range(100)]
    
    # Create combined examples by pairing text with graphs
    # In a real application, there would be a natural connection between text and graphs
    combined_data = []
    for i in range(min(len(texts), len(graphs))):
        text = texts[i]
        graph = graphs[i]
        
        # Add synthetic task instruction for demonstration
        if graph_dataset.startswith('ogbg'):
            instruction = f"Analyze this molecule: {text}\n" + \
                          f"Predict if it is active against HIV."
        else:
            instruction = f"Analyze this graph structure: {text}\n" + \
                          f"Classify the graph based on its properties."
        
        combined_data.append({
            'instruction': instruction,
            'node_features': graph['node_features'],
            'edge_index': graph['edge_index'],
            'edge_attr': graph['edge_attr'],
            'label': graph['y'],
        })
    
    return combined_data


def create_gnn_model(gnn_type, in_channels, hidden_channels, out_channels):
    """Create a GNN model based on the specified type."""
    if gnn_type == 'gin':
        return GIN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=3,
            dropout=0.1,
        )
    elif gnn_type == 'gat':
        from valkyrie_llm.model.gnn.models import GAT
        return GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
        )
    elif gnn_type == 'graphormer':
        from valkyrie_llm.model.gnn.layers import GraphTransformer
        return GraphTransformer(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
        )
    elif gnn_type == 'tree_gnn':
        from valkyrie_llm.model.gnn.tree_gnn import TreeGNN
        return TreeGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=3,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}")


def tokenize_function(examples, tokenizer):
    """Tokenize text inputs for the language model."""
    return tokenizer(
        examples['instruction'],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )


def collate_fn(batch):
    """
    Custom collate function to handle both text and graph data.
    
    This function processes a batch of data containing both tokenized text
    and graph components, preparing them for input to the integrated model.
    """
    # Extract text inputs and tokenize
    text_inputs = {
        'input_ids': [],
        'attention_mask': [],
    }
    
    # Extract graph inputs
    graph_inputs = {
        'node_features': [],
        'edge_index': [],
        'edge_attr': [],
        'ptr': [0],  # For batching in PyG (node pointers)
    }
    
    # Extract labels
    labels = []
    
    # Track node offsets for batching graphs
    node_offset = 0
    
    for item in batch:
        # Add text inputs
        for k in text_inputs:
            if k in item:
                text_inputs[k].append(item[k])
        
        # Process graph inputs
        if 'node_features' in item and 'edge_index' in item:
            # Add node features
            graph_inputs['node_features'].append(item['node_features'])
            
            # Update edge indices with offset
            edge_index = item['edge_index'].clone()
            edge_index += node_offset
            graph_inputs['edge_index'].append(edge_index)
            
            # Add edge attributes if available
            if 'edge_attr' in item and item['edge_attr'] is not None:
                graph_inputs['edge_attr'].append(item['edge_attr'])
            
            # Update node offset and ptr
            num_nodes = item['node_features'].size(0)
            node_offset += num_nodes
            graph_inputs['ptr'].append(node_offset)
        
        # Add labels
        if 'label' in item:
            labels.append(item['label'])
    
    # Concatenate text inputs
    for k in text_inputs:
        if text_inputs[k]:
            text_inputs[k] = torch.stack(text_inputs[k])
    
    # Concatenate graph inputs
    if graph_inputs['node_features']:
        graph_inputs['node_features'] = torch.cat(graph_inputs['node_features'], dim=0)
    if graph_inputs['edge_index']:
        graph_inputs['edge_index'] = torch.cat(graph_inputs['edge_index'], dim=1)
    if graph_inputs['edge_attr'] and all(x is not None for x in graph_inputs['edge_attr']):
        graph_inputs['edge_attr'] = torch.cat(graph_inputs['edge_attr'], dim=0)
    else:
        graph_inputs.pop('edge_attr')
    
    graph_inputs['ptr'] = torch.tensor(graph_inputs['ptr'], dtype=torch.long)
    
    # Concatenate labels
    if labels:
        labels = torch.stack(labels)
    
    # Combine all inputs
    combined_inputs = {**text_inputs, **graph_inputs}
    if labels:
        combined_inputs['labels'] = labels
    
    return combined_inputs


def main(args):
    """Main function to train an LLM with integrated GNN components."""
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load tokenizer and language model
    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model)
    
    # Determine dimensions for GNN
    transformer_dim = llm_model.config.hidden_size
    gnn_in_channels = 9  # Example value, depends on your graph data
    gnn_hidden_channels = 128
    gnn_out_channels = transformer_dim  # Match transformer dimension for fusion
    
    # Create GNN model
    gnn_model = create_gnn_model(
        args.gnn_type,
        in_channels=gnn_in_channels,
        hidden_channels=gnn_hidden_channels,
        out_channels=gnn_out_channels,
    )
    
    # Create integration model
    integration_model = TransformerGNNIntegration(
        transformer_dim=transformer_dim,
        graph_dim=gnn_in_channels,
        hidden_dim=gnn_hidden_channels,
        use_graph_attention=args.gnn_type in ['gat', 'graphormer'],
        use_tree_structure=args.gnn_type == 'tree_gnn',
        use_contrastive=True,
    )
    
    # Prepare combined dataset
    combined_data = prepare_combined_dataset(args.text_dataset, args.dataset)
    
    # Tokenize dataset
    tokenized_data = []
    for example in combined_data:
        # Tokenize text
        tokens = tokenizer(
            example['instruction'],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        # Create tokenized example with graph data
        tokenized_example = {
            'input_ids': tokens['input_ids'][0],
            'attention_mask': tokens['attention_mask'][0],
            'node_features': example['node_features'],
            'edge_index': example['edge_index'],
            'edge_attr': example['edge_attr'],
            'label': example['label'],
        }
        
        tokenized_data.append(tokenized_example)
    
    # Split data into train and eval sets
    train_size = int(0.9 * len(tokenized_data))
    train_data = tokenized_data[:train_size]
    eval_data = tokenized_data[train_size:]
    
    # Create PyTorch datasets
    class CombinedDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = CombinedDataset(train_data)
    eval_dataset = CombinedDataset(eval_data)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=args.fp16,
        dataloader_num_workers=0,  # Needed for custom collate_fn
        gradient_accumulation_steps=4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )
    
    # Define GNN training config
    gnn_config = TrainingConfig(
        learning_rate=args.gnn_lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_epochs=int(0.1 * args.epochs),
        lr_scheduler="cosine",
        grad_clip=1.0,
        grad_accumulation_steps=4,
        optimizer_type="adamw",
        mixed_precision=args.fp16,
        save_best_model=True,
        checkpoint_dir=os.path.join(args.output_dir, "gnn_checkpoints"),
        dropout=0.1,
        seed=args.seed,
    )
    
    # Create GNN-LLM trainer
    trainer = GNNLLMTrainer(
        llm_model=llm_model,
        gnn_model=gnn_model,
        integration_model=integration_model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        gnn_config=gnn_config,
        tokenizer=tokenizer,
        device=args.device,
    )
    
    # Train the model
    print("Starting GNN-LLM joint training...")
    train_output = trainer.train()
    
    print(f"Training completed with final loss: {train_output.training_loss:.6f}")
    print(f"Best evaluation loss: {train_output.best_eval_loss:.6f}")
    
    # Example inference with trained model
    print("\nRunning inference with trained model...")
    
    # Prepare a sample input
    sample_text = "Analyze this molecule with a benzene ring and hydroxyl group."
    inputs = tokenizer(sample_text, return_tensors="pt").to(args.device)
    
    # Create a small sample graph
    sample_nodes = torch.randn(5, gnn_in_channels).to(args.device)
    sample_edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                                [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long).to(args.device)
    
    # Set models to evaluation mode
    integration_model.eval()
    
    # Generate text with graph context
    with torch.no_grad():
        outputs = integration_model(
            transformer_features=llm_model(inputs['input_ids']).last_hidden_state,
            transformer_mask=inputs['attention_mask'],
            node_features=sample_nodes,
            edge_index=sample_edges,
        )
        
        # Generate continuation
        generated_ids = llm_model.generate(
            inputs['input_ids'],
            max_length=100,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
        )
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Input: {sample_text}")
        print(f"Generated: {generated_text}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args) 