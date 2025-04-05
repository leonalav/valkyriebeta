# Modular Training System for ValkyrieLLM

This repository contains a modular training system for ValkyrieLLM, a language model with advanced reasoning capabilities.

## Directory Structure

The codebase is organized into the following modules:

- `config/`: Configuration classes for model, training, and optimization
- `model/`: Model architecture and components
- `data/`: Data processing and loading utilities
- `training/`: Training engine and utilities
- `validators/`: Model validation utilities
- `train.py`: Main training script

## Usage

### Basic Training

To train a basic model:

```bash
python train.py --experiment_name basic_model --output_dir output/basic
```

### Training with Advanced Features

To train a model with advanced features:

```bash
python train.py \
    --experiment_name advanced_model \
    --output_dir output/advanced \
    --hidden_size 1024 \
    --num_layers 24 \
    --num_heads 16 \
    --use_mixed_precision \
    --use_gradient_checkpointing \
    --use_flash_attention \
    --use_reasoning \
    --reasoning_type adaptive
```

### Distributed Training

To train with distributed data parallel:

```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py \
    --experiment_name distributed_model \
    --output_dir output/distributed \
    --use_distributed \
    --use_mixed_precision
```

### Training with RLHF

To train with RLHF:

```bash
python train.py \
    --experiment_name rlhf_model \
    --output_dir output/rlhf \
    --use_rlhf \
    --rlhf_type ppo
```

### Domain-Specific Training

To train with domain-specific data:

```bash
python train.py \
    --experiment_name domain_model \
    --output_dir output/domain \
    --use_domain_training \
    --domains science medicine finance
```

## Configuration

The training system supports extensive configuration options. See `train.py` for all available command-line arguments.

## Components

### Model

The `ValkyrieLLM` class in `model/valkyrie_llm.py` implements the core model architecture with support for:

- Mixture of Experts (MoE)
- Advanced reasoning capabilities (tree reasoning, recursive reasoning, etc.)
- RLHF integration
- Memory augmentation

### Training Engine

The `TrainingEngine` class in `training/training_engine.py` implements the training loop with support for:

- Mixed precision training
- Distributed training
- Gradient accumulation
- Knowledge distillation
- Domain-specific training
- RLHF training

### Data Handling

The data handling modules in `data/nanogpt_data.py` provide utilities for:

- Efficient data loading
- Domain-specific datasets
- Reasoning evaluation

### Validation

The `ModelValidator` class in `validators/model_validator.py` provides comprehensive model validation to ensure model integrity and compatibility.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 