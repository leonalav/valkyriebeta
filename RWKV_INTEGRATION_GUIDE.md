# RWKV Integration Guide

This guide explains how to use the RWKV components implemented in the nanogpt codebase for efficient training and inference with RWKV architecture.

## Table of Contents

1. [Introduction to RWKV](#introduction-to-rwkv)
2. [RWKV Components Overview](#rwkv-components-overview)
3. [Basic Training with RWKV](#basic-training-with-rwkv)
4. [Advanced Configuration](#advanced-configuration)
5. [Hybrid Model Architecture](#hybrid-model-architecture)
6. [Memory Optimization](#memory-optimization)
7. [Long Context Processing](#long-context-processing)
8. [Inference with RWKV](#inference-with-rwkv)
9. [Component Explanations](#component-explanations)

## Introduction to RWKV

RWKV (Receptance Weighted Key Value) is an architecture that combines the best aspects of RNNs and Transformers:

- **Linear scaling** with sequence length (O(n) vs O(n²) for standard transformers)
- **Parallelizable training** like transformers (unlike traditional RNNs)
- **State-based processing** for infinitely long contexts
- **Token-by-token inference** for efficient generation

RWKV achieves this through its time-mixing mechanism that replaces the standard self-attention with a more efficient computation that can be run either in parallel (for training) or recurrently (for inference).

## RWKV Components Overview

The following components have been implemented:

1. **RWKVIntegrator**: Core class that integrates RWKV-specific optimizations into the model
2. **HybridModelConfigurator**: Configures which layers use RWKV vs transformer architecture
3. **RWKVTimeFirst/RWKVChannelMix layers**: Core RWKV computation primitives in training/layers/rwkv_layer.py
4. **RWKV-specific optimizer settings**: Parameter group configurations for RWKV models
5. **Memory optimizations**: Chunking and state compression for efficient processing
6. **Hybrid models**: Combining RWKV and transformer layers in a single model

## Basic Training with RWKV

### Quick Start

To train a 3B RWKV model, use the provided `train_rwkv.py` script:

```bash
python train_rwkv.py \
  --output_dir ./outputs/rwkv_3b \
  --experiment_name valkyrie_rwkv_3b \
  --train_file data/train.txt \
  --val_file data/val.txt \
  --max_seq_length 16384 \
  --batch_size 8 \
  --use_rwkv_chunking \
  --rwkv_chunk_size 1024
```

### Core Components Usage

The training script already integrates the necessary components:

```python
# Configure the hybrid RWKV-Transformer model
hybrid_configurator = HybridModelConfigurator(
    model_config=model_config, 
    architecture_params=architecture_params
)

# Set up which layers use RWKV vs traditional attention
hybrid_configurator.configure_layer_architecture(
    rwkv_layer_indices=[i for i in range(model_config.num_layers) if i % 2 == 0],
    transformer_layer_indices=[i for i in range(model_config.num_layers) if i % 2 == 1]
)

# Set up model
model = setup_model(args, model_config, tokenizer, training_config, 
                   architecture_params, hybrid_configurator)

# Integrate RWKV-specific components
rwkv_integrator = RWKVIntegrator(
    model=model,
    model_config=model_config,
    training_config=training_config
)

# Apply RWKV optimizations and integration
model = rwkv_integrator.apply_rwkv_optimizations()
```

## Advanced Configuration

### Layer Architecture

You can customize which layers use RWKV vs transformer architecture:

```python
# Example: Use RWKV for deeper layers (better for long-range dependencies)
hybrid_configurator.configure_layer_architecture(
    rwkv_layer_indices=list(range(16, 32)),  # Last 16 layers use RWKV
    transformer_layer_indices=list(range(0, 16))  # First 16 layers use transformer
)

# Example: Optimize automatically for sequence length
hybrid_configurator.optimize_for_sequence_length(seq_length=8192)
```

### Training Optimization

The RWKV model benefits from specific optimizer configurations:

```python
# Set up RWKV-specific optimizers
engine.setup_rwkv_optimizer(
    optimizer=optimizer,
    rwkv_lr_multiplier=training_params["rwkv_lr_multiplier"],
    att_weight_decay=training_params["rwkv_att_weight_decay"],
    ffn_weight_decay=training_params["rwkv_ffn_weight_decay"],
)

# RWKV-specific optimizations
engine.optimize_rwkv_training_setup()
```

## Hybrid Model Architecture

The hybrid model architecture combines RWKV and transformer layers in a single model. By default, it alternates between RWKV and transformer layers for a balanced approach:

```python
# Default configuration (alternating layers)
hybrid_configurator.configure_layer_architecture(
    rwkv_layer_indices=[i for i in range(model_config.num_layers) if i % 2 == 0],
    transformer_layer_indices=[i for i in range(model_config.num_layers) if i % 2 == 1]
)

# Get distribution statistics
distribution = hybrid_configurator.get_layer_distribution()
print(f"RWKV layers: {distribution['rwkv_percentage']}%, Transformer: {distribution['transformer_percentage']}%")
```

### When to Use Different Configurations

- **More RWKV layers**: Better for very long sequences (>8k tokens), lower memory usage
- **More transformer layers**: Better for shorter sequences with complex dependencies
- **RWKV in deeper layers**: Better for capturing long-range patterns
- **Transformers in deeper layers**: Better for complex reasoning tasks

## Memory Optimization

RWKV models are memory-efficient but can be further optimized:

```python
# Enable chunked processing for long sequences
if hasattr(model, 'set_chunk_size'):
    model.set_chunk_size(chunk_size=1024)

# Apply chunking to dataloaders
train_dataloader = rwkv_integrator.apply_rwkv_chunking(
    dataloader=train_dataloader,
    chunk_size=memory_config.rwkv_chunk_size
)
```

### State Compression

For very long sequences, state compression can be enabled:

```python
if hasattr(model, 'enable_state_compression'):
    model.enable_state_compression()
```

## Long Context Processing

RWKV is particularly good at processing long contexts efficiently:

```python
# Process a long sequence efficiently
def process_long_sequence(model, tokenizer, text, chunk_size=1024):
    # Tokenize the full text
    tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    
    # Reset model state
    model.reset_state(batch_size=1)
    
    # Process in chunks
    outputs = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size].unsqueeze(0).to(model.device)
        with torch.no_grad():
            # Process with state passing
            chunk_output, new_state = model.process_with_state(chunk, model.state)
            model.state = new_state
            outputs.append(chunk_output)
    
    # Combine outputs
    return torch.cat(outputs, dim=1)
```

## Inference with RWKV

RWKV models have two inference modes:

### Parallel Mode (for batch processing)

```python
# Process a batch of sequences in parallel
outputs = model(input_ids)
```

### Recurrent Mode (for token-by-token generation)

```python
# Initialize state
model.reset_state(batch_size=1)

# Generate tokens one by one
for i in range(max_length):
    # Get next token prediction
    next_token_logits, new_state = model.process_with_state(
        current_token.unsqueeze(0).unsqueeze(0),
        model.state
    )
    model.state = new_state
    
    # Sample next token
    next_token = sample_from_logits(next_token_logits[0, -1])
    generated_tokens.append(next_token)
    current_token = next_token
```

## Component Explanations

### RWKVIntegrator

This component integrates RWKV-specific optimizations into the model:

- **apply_rwkv_optimizations()**: Main method to apply all RWKV optimizations
- **setup_rwkv_optimizer()**: Configures optimizer for RWKV parameters
- **export_rwkv_weights()**: Exports RWKV-specific weights for deployment
- **optimize_for_inference()**: Applies inference-specific optimizations
- **apply_rwkv_chunking()**: Configures chunking for dataloader

### HybridModelConfigurator

This component configures which layers use RWKV vs transformer:

- **configure_layer_architecture()**: Sets which layers use which architecture
- **get_layer_distribution()**: Gets statistics about layer allocation
- **optimize_for_sequence_length()**: Optimizes architecture for a given sequence length

### TrainingEngine Extensions

The training engine has been extended with RWKV-specific methods:

- **setup_rwkv_optimizer()**: Sets up parameter groups for RWKV
- **optimize_rwkv_training_setup()**: Applies RWKV-specific training optimizations

### RWKV Model Configuration

The configuration in `training/configs/model_3b_rwkv.py` contains comprehensive settings:

- **ModelConfig3BRWKV**: Main model configuration
- **RWKVMemoryConfig**: Memory-specific settings
- **RWKVTrainingEfficiencyConfig**: Training efficiency settings
- **RWKVArchitectureParams**: Architecture-specific parameters 