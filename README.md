# ValkyrieLLM: LLM Framework with Advanced GNN Integration

ValkyrieLLM is a powerful framework for integrating Large Language Models (LLMs) with advanced Graph Neural Networks (GNNs) for enhanced structured reasoning capabilities.

## Key Features

- **Advanced GNN Models**: Comprehensive suite of state-of-the-art GNN architectures
  - Transformer-GNN (Graphormer)
  - Heterogeneous Graph Transformer (HGT)
  - DiffPool (Differentiable Pooling)
  - Tree-Structured GNNs
  - Edge-Enhanced GNN (EGNN)
  - Contrastive Learning (GraphCL, InfoGraph)

- **Seamless LLM Integration**: Bidirectional information flow between graphs and text
  - Combined representation learning
  - Joint training infrastructure
  - Flexible architecture adapters

- **Training Utilities**: Comprehensive tools for efficient model training
  - Learning rate scheduling with warmup
  - Gradient accumulation for stable training
  - Mixed precision support
  - Early stopping and checkpointing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/valkyrie-llm.git
cd valkyrie-llm

# Install dependencies
pip install -r requirements.txt
```

## Dependencies and Environment Setup

ValkyrieLLM relies on several external libraries. The core dependencies include:

- PyTorch for neural network operations
- PyTorch Geometric for graph operations
- Transformers for language models
- OGB for graph benchmarks

### Handling External Dependencies

The example scripts include mock implementations to handle cases where PyTorch Geometric or OGB might not be installed:

```python
# Mock implementations for PyTorch Geometric objects
class MockData:
    """Mock implementation of PyTorch Geometric Data object."""
    def __init__(self, x, edge_index, y=None, edge_attr=None):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.edge_attr = edge_attr

# Generate synthetic graph data for demos
class MockGraphDataset:
    """Mock dataset for graph data."""
    def __init__(self, num_graphs=100, num_features=9, num_classes=2):
        # Creates synthetic graph dataset for demonstration
        ...
```

If you encounter import errors with PyTorch Geometric or OGB, you have two options:

1. **Install the missing dependencies:**
   ```bash
   pip install torch-geometric
   pip install ogb
   ```

2. **Use the mock implementations:**
   The example scripts will automatically fall back to using mock implementations when the external libraries are not available.

## Quick Start

### Training a Standalone GNN Model

```python
from valkyrie_llm.model.gnn import GraphTransformer, TrainingConfig, GNNTrainer

# Create model
model = GraphTransformer(
    in_channels=64,
    hidden_channels=128,
    out_channels=10,
    num_layers=4,
    num_heads=4,
)

# Configure training
config = TrainingConfig(
    learning_rate=1e-3,
    batch_size=32,
    epochs=100,
    warmup_epochs=10,
)

# Train model
trainer = GNNTrainer(model, config, train_loader, val_loader)
history = trainer.train(criterion=loss_fn, metric_fn=accuracy_fn)
```

### Training an Integrated GNN-LLM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from valkyrie_llm.model.gnn import GIN, TransformerGNNIntegration
from valkyrie_llm.model.gnn_llm_trainer import GNNLLMTrainer

# Load LLM
tokenizer = AutoTokenizer.from_pretrained("gpt2")
llm_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Create GNN model
gnn_model = GIN(
    in_channels=64,
    hidden_channels=128,
    out_channels=768,  # Match transformer dimension
    num_layers=3,
)

# Create integration model
integration_model = TransformerGNNIntegration(
    transformer_dim=768,
    graph_dim=64,
    hidden_dim=128,
    use_graph_attention=True,
)

# Setup training arguments
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=3,
)

# Create trainer and train model
trainer = GNNLLMTrainer(
    llm_model=llm_model,
    gnn_model=gnn_model,
    integration_model=integration_model,
    training_args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## Examples

See the `valkyrie_llm/examples/` directory for complete training examples:

- **gnn_training_example.py**: Training standalone GNN models
- **gnn_llm_training_example.py**: Training integrated GNN-LLM models

Run an example:

```bash
python -m valkyrie_llm.examples.gnn_training_example --model graphormer --dataset PROTEINS
```

## Supported GNN Architectures

| Model | Description | Paper |
|-------|-------------|-------|
| GraphTransformer | Transformer-style self-attention for graphs | [Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2106.05234) |
| HGT | Specialized for heterogeneous graphs | [Heterogeneous Graph Transformer](https://arxiv.org/abs/2003.01332) |
| DiffPool | Hierarchical graph representation learning | [Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/abs/1806.08804) |
| TreeGNN | Specialized for tree-structured data | [Tree-LSTM](https://arxiv.org/abs/1503.00075) |
| GraphCL | Contrastive learning with graph augmentations | [Graph Contrastive Learning with Augmentations](https://arxiv.org/abs/2010.13902) |
| EGNN | Edge-enhanced graph neural networks | [Edge-enhanced GNN for Molecule Properties](https://arxiv.org/abs/2010.09559) |

## License

[MIT License](LICENSE)

## Citation

If you use ValkyrieLLM in your research, please cite:

```
@software{valkyrie_llm,
  title = {ValkyrieLLM: LLM Framework with Advanced GNN Integration},
  author = {Your Name},
  year = {2023},
  url = {https://github.com/yourusername/valkyrie-llm},
}
```

## Acknowledgments

This project builds upon numerous open-source libraries and research papers in the fields of graph neural networks and language models.
