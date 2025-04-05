# Valkyrie LLM Examples

This directory contains examples demonstrating how to use various components of the Valkyrie LLM framework.

## GNN Model Training

The `gnn_training_example.py` script demonstrates how to train and evaluate various GNN models implemented in Valkyrie LLM.

### Available Models

- **GraphTransformer (Graphormer)**: A transformer-based GNN model incorporating attention on graphs.
- **Tree-Structured GNN**: A model specialized for hierarchical tree-structured data.
- **GraphCL**: A contrastive learning approach for GNNs with graph augmentations.
- **Integrated GNN-Transformer**: A model that integrates GNNs with transformers for joint reasoning.

### Training Example

To train a GraphTransformer model on the PROTEINS dataset:

```bash
python -m valkyrie_llm.examples.gnn_training_example --model graphormer --dataset PROTEINS
```

### Command-line Options

- `--model`: Model to train (`graphormer`, `tree_gnn`, `graphcl`, `integrated`)
- `--dataset`: Dataset to use (any TUDataset, default: PROTEINS)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--hidden_channels`: Hidden dimension size (default: 64)
- `--dropout`: Dropout rate (default: 0.5)
- `--seed`: Random seed (default: 42)
- `--device`: Training device (default: cuda if available, else cpu)

## Using GNNs with LLMs

The following is a basic example of how to use the integrated GNN-Transformer model:

```python
from valkyrie_llm.model.gnn import TransformerGNNIntegration
from transformers import AutoModel
import torch

# Load a pretrained transformer model
transformer = AutoModel.from_pretrained("bert-base-uncased")

# Create the integrated model
model = TransformerGNNIntegration(
    transformer=transformer,
    transformer_dim=768,  # BERT hidden dimension
    graph_dim=64,         # Graph node feature dimension
    hidden_dim=256,       # Hidden dimension for the integration
    output_dim=10,        # Output dimension
    num_layers=3,
    use_graph_attention=True,
    use_tree_structure=False,
    use_contrastive=True,
)

# Prepare inputs
input_ids = torch.randint(0, 30522, (2, 128))  # Batch of 2, sequence length 128
attention_mask = torch.ones_like(input_ids)
node_features = torch.randn(20, 64)  # 20 nodes with 64 features
edge_index = torch.randint(0, 20, (2, 30))  # 30 edges connecting the 20 nodes
batch = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])  # Node-to-graph assignment

# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    node_features=node_features,
    edge_index=edge_index,
    batch=batch,
)
```

## Training Configuration

Valkyrie LLM provides a `TrainingConfig` class for configuring training settings:

```python
from valkyrie_llm.model.gnn import TrainingConfig, GNNTrainer

# Create training configuration
config = TrainingConfig(
    learning_rate=1e-3,
    batch_size=32,
    epochs=100,
    warmup_epochs=10,
    lr_scheduler="cosine",
    grad_clip=1.0,
    grad_accumulation_steps=1,
    early_stopping_patience=15,
    optimizer_type="adamw",
    mixed_precision=True,
    save_best_model=True,
    checkpoint_dir="./checkpoints",
    log_interval=5,
    contrastive_loss_weight=0.1,
    dropout=0.1,
)

# Train a model with this configuration
trainer = GNNTrainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
)

# For integrated models, use the specialized trainer
trainer = IntegratedGNNTrainer(
    model=model,
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    transformer_lr_factor=0.1,  # Use lower learning rate for transformer
)
```

## Model Registry

The `ModelRegistry` allows easy creation of models by name:

```python
from valkyrie_llm.model.gnn import ModelRegistry, create_gnn_model

# Get available model names
model_names = ModelRegistry.available_models()
print(f"Available models: {model_names}")

# Create a model by name
model = create_gnn_model(
    model_name="GraphTransformer",
    model_config={
        "in_channels": 64,
        "hidden_channels": 128,
        "out_channels": 10,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.1,
    },
)
```

This provides a convenient way to experiment with different model architectures. 