# NanoGPT Enhanced LLM Training Manual

This manual provides comprehensive instructions for building, training, optimizing, and deploying enhanced Large Language Models using the NanoGPT framework.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Complete Workflow](#complete-workflow)
  - [Pre-Training Workflow](#pre-training-workflow)
  - [Post-Training (Student Model) Workflow](#post-training-student-model-workflow)
- [Configuration](#configuration)
  - [Model Configuration](#model-configuration)
  - [Training Configuration](#training-configuration)
  - [Data Configuration](#data-configuration)
- [Advanced Features](#advanced-features)
  - [Domain-Specific Training](#domain-specific-training)
  - [Knowledge Distillation](#knowledge-distillation)
  - [Computational Efficiency](#computational-efficiency)
  - [Adaptive Reasoning](#adaptive-reasoning)
- [Evaluation and Benchmarking](#evaluation-and-benchmarking)
- [Deployment Options](#deployment-options)
- [Troubleshooting](#troubleshooting)
- [Reference](#reference)

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (32GB+ recommended for larger models)
- 100GB+ storage for datasets and checkpoints

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/nanogpt.git
   cd nanogpt
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Quick Start

Run a complete training and distillation workflow with default settings:

```bash
# All-in-one training with all advanced features
python train_aio.py --use_knowledge_distillation --use_domain_specific_data --use_computational_efficiency --use_adaptive_reasoning

# Process the model for deployment
python scripts/process_data.py --input_dir data/processed --output_dir data/inference --mode inference --student_model_path models/final/model.safetensors
```

## Complete Workflow

### Pre-Training Workflow

1. **Download and prepare datasets**:
   ```bash
   python scripts/download_datasets.py --save-format parquet --output-dir data/raw
   ```

2. **Process raw data into splits**:
   ```bash
   python scripts/prepare_data.py --data_root data/raw --output_root data/processed
   ```

3. **Final data preparation**:
   ```bash
   python scripts/process_data.py --input_dir data/processed --output_dir data/ready --mode train
   ```

4. **Prepare domain-specific datasets** (optional but recommended):
   ```bash
   python scripts/prepare_domain_datasets.py --config config/domain_bridge_config.json
   ```

5. **Train the model**:
   ```bash
   python train_aio.py --config config/training_config.json --output_dir models/my_model
   ```

### Post-Training (Student Model) Workflow

1. **Ensure your custom LLM is trained and saved as .safetensors**

2. **Process data for student model**:
   ```bash
   python scripts/process_data.py \
       --input_dir data/processed \
       --output_dir data/student_ready \
       --mode student \
       --student_model_path models/my_model/final_model/model.safetensors
   ```

3. **Run inference validation** (optional):
   ```bash
   python scripts/process_data.py \
       --input_dir data/student_ready \
       --output_dir data/inference \
       --mode inference \
       --student_model_path models/my_model/final_model/model.safetensors
   ```

4. **Deploy the model**:
   ```bash
   python serve.py --model_path models/my_model/final_model
   ```

## Configuration

### Model Configuration

Create a file `config/model_config.json`:

```json
{
  "model": {
    "model_type": "enhanced",
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "max_position_embeddings": 1024,
    "vocab_size": 50257,
    "use_adapter_modules": true,
    "use_domain_adaptation": true
  }
}
```

Model size recommendations:

| Model Size | Parameters | hidden_size | layers | heads |
|------------|------------|-------------|--------|-------|
| Nano       | 125M       | 768         | 12     | 12    |
| Small      | 350M       | 1024        | 24     | 16    |
| Medium     | 760M       | 1536        | 24     | 16    |
| Large      | 1.3B       | 2048        | 24     | 16    |
| XL         | 2.7B       | 2560        | 32     | 32    |

### Training Configuration

Create a file `config/training_config.json`:

```json
{
  "training": {
    "batch_size": 16,
    "eval_batch_size": 32,
    "num_epochs": 3,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 500,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 3,
    "fp16": true,
    "num_workers": 4,
    "seed": 42,
    "use_domain_specific_data": true,
    "domain_data_dir": "data/domain_specific",
    "mixed_precision": true,
    "mixed_precision_dtype": "float16",
    "use_knowledge_distillation": true
  }
}
```

### Data Configuration

Create a file `config/domain_bridge_config.json`:

```json
{
  "loaders": {
    "load_math": {
      "max_samples": 3000
    },
    "load_numinamath": {},
    "load_scieval": {
      "n_examples": 2000
    },
    "load_olympiad_bench": {
      "filter_hard": true
    },
    "load_livecodebench": {
      "difficulty": "medium"
    },
    "load_quant": {},
    "load_curated_thoughts": {
      "include_cot": true
    },
    "load_wildchat": {
      "min_length": 200
    }
  },
  "domain_weights": {
    "math": 1.5,
    "science": 1.2,
    "coding": 1.3,
    "logic": 1.1,
    "reasoning": 1.4,
    "general": 1.0
  },
  "train_ratio": 0.8,
  "val_ratio": 0.1,
  "test_ratio": 0.1,
  "max_examples_per_domain": 5000,
  "extract_vocab": true,
  "vocab_size": 1000,
  "seed": 42,
  "output_dir": "data/domain_specific"
}
```

## Advanced Features

### Domain-Specific Training

Enable specialized training for different domains (math, science, coding, etc.):

1. **List available data loaders**:
   ```bash
   python scripts/prepare_domain_datasets.py --list_loaders
   ```

2. **Prepare domain-specific datasets**:
   ```bash
   python scripts/prepare_domain_datasets.py --config config/domain_bridge_config.json
   ```

3. **Train with domain-specific data**:
   ```bash
   python train_aio.py --use_domain_specific_data --config config/domain_training_config.json
   ```

### Knowledge Distillation

Train a smaller, efficient model that retains the knowledge of a larger teacher model:

1. **Configure distillation settings**:
   ```json
   {
     "knowledge_distillation": {
       "alpha": 0.5,
       "temperature": 2.0,
       "distill_logits": true,
       "distill_hidden_states": true,
       "distill_attention": true,
       "teacher_model_path": "path/to/teacher/model",
       "adaptation_type": "domain_specific"
     }
   }
   ```

2. **Train with knowledge distillation**:
   ```bash
   python train_aio.py --use_knowledge_distillation --teacher_model_path path/to/teacher/model
   ```

### Computational Efficiency

Optimize memory usage and training/inference speed:

1. **Configure efficiency settings**:
   ```json
   {
     "computational_efficiency": {
       "use_activation_checkpointing": true,
       "use_efficient_attention": true,
       "use_quantization": false,
       "quantization_bits": 8,
       "use_kv_cache": true,
       "use_model_compiler": true,
       "use_adaptive_batch_size": true,
       "min_batch_size": 8,
       "max_batch_size": 32
     }
   }
   ```

2. **Train with efficiency optimizations**:
   ```bash
   python train_aio.py --use_computational_efficiency
   ```

### Adaptive Reasoning

Enable the model to adjust computation based on input complexity:

1. **Configure adaptive reasoning**:
   ```json
   {
     "adaptive_reasoning": {
       "enabled": true,
       "low_complexity_threshold": 0.3,
       "medium_complexity_threshold": 0.7,
       "max_computation_budget": 1.0,
       "min_computation_budget": 0.5,
       "use_early_exit": true
     }
   }
   ```

2. **Train with adaptive reasoning**:
   ```bash
   python train_aio.py --use_adaptive_reasoning
   ```

## Evaluation and Benchmarking

1. **Evaluate the model**:
   ```bash
   python evaluate.py --model_path models/my_model/final_model --eval_datasets math,science
   ```

2. **Run tests**:
   ```bash
   python scripts/run_tests.py --model_path models/my_model/final_model --tests all
   ```

3. **Run specific reasoning tests**:
   ```bash
   python scripts/test_all_reasoning.py --model_path models/my_model/final_model
   ```

## Deployment Options

1. **Command-line interface**:
   ```bash
   python generate.py --model_path models/my_model/final_model --prompt "Explain quantum computing"
   ```

2. **API server**:
   ```bash
   python serve.py --model_path models/my_model/final_model --port 8000
   ```

3. **Python library usage**:
   ```python
   from model import load_model
   
   model, tokenizer = load_model("models/my_model/final_model")
   output = model.generate("Explain quantum computing", max_tokens=200)
   print(output)
   ```

## Troubleshooting

### Common Issues and Solutions

1. **Out of memory errors**:
   - Reduce batch size
   - Enable gradient accumulation
   - Use activation checkpointing
   - Enable mixed precision training

2. **Slow training**:
   - Enable computational efficiency options
   - Use a more efficient attention mechanism
   - Increase number of workers for data loading

3. **Poor model quality**:
   - Use domain-specific training
   - Implement knowledge distillation
   - Check data quality and diversity
   - Extend training time

4. **Deployment issues**:
   - Convert model to optimized format (ONNX, TorchScript)
   - Use quantization for smaller size
   - Check hardware compatibility

### Debugging Tools

1. **Memory analysis**:
   ```bash
   python debug_memory.py --model_path models/my_model/final_model
   ```

2. **Performance profiling**:
   ```bash
   python profile_model.py --model_path models/my_model/final_model --batch_size 16
   ```

## Reference

### Command-Line Arguments

- `--config`: Path to configuration file
- `--output_dir`: Directory to save model checkpoints
- `--data_dir`: Directory containing training data
- `--model_path`: Path to pretrained model
- `--resume`: Resume training from checkpoint
- `--seed`: Random seed for reproducibility
- `--use_domain_specific_data`: Enable domain-specific training
- `--use_knowledge_distillation`: Enable knowledge distillation
- `--use_computational_efficiency`: Enable efficiency optimizations
- `--use_adaptive_reasoning`: Enable adaptive reasoning

### Directory Structure

```
nanogpt/
├── config/                # Configuration files
├── data/                  # Data directory
│   ├── raw/               # Raw downloaded datasets
│   ├── processed/         # Processed data splits
│   ├── ready/             # Training-ready data
│   └── domain_specific/   # Domain-specific datasets
├── model/                 # Model implementation
├── scripts/               # Utility scripts
├── training/              # Training modules
├── utils/                 # Utility functions
├── tests/                 # Test files
└── models/                # Saved model checkpoints
```

### Additional Resources

- Documentation: `docs/`
- Domain data guide: `docs/domain_data_bridge.md`
- Dataset preparation: `docs/dataset_preparation.md`
- Knowledge distillation: `docs/knowledge_distillation.md`
- Computational efficiency: `docs/computational_efficiency.md` 