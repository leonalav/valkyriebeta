# Valkyrie LLM with RWKV Architecture

This repository contains an implementation of the Valkyrie LLM model with support for RWKV (Receptance Weighted Key Value) architecture. This README explains how to train and use the RWKV hybrid model.

## What is RWKV?

RWKV (Receptance Weighted Key Value) is a novel architecture that combines the best of both RNN and Transformer architectures:

- Linear scaling with sequence length (like RNNs) instead of quadratic scaling in standard transformers
- Parallelizable training like transformers (unlike traditional RNNs)
- Support for infinitely long context through state-based processing
- Can process sequences token by token during inference (like RNNs)

## Hybrid Architecture

The Valkyrie LLM implementation uses a hybrid architecture that combines both RWKV and traditional transformer layers:

- RWKV layers handle efficient sequence processing with O(n) complexity
- Transformer layers provide strong contextual understanding
- By default, even-indexed layers use RWKV and odd-indexed layers use transformers
- The layer allocation can be configured through the `HybridModelConfigurator`

## Directory Structure

```
training/
│
├── configs/
│   ├── model_3b.py               # 3B standard transformer configuration
│   └── model_3b_rwkv.py          # 3B RWKV hybrid configuration
│
├── layers/
│   ├── rwkv_layer.py             # RWKV layer implementation
│   └── hybrid_model.py           # Hybrid RWKV-Transformer model
│
├── training_engine.py            # Training engine with RWKV support
├── model_setup.py                # Model setup with RWKV support
├── data_loaders.py               # Data loaders with RWKV chunking support
└── evaluation.py                 # Evaluation with RWKV-specific metrics
```

## Training Scripts

- `train.py`: Standard training script for transformer-based models
- `train_rwkv.py`: Training script for RWKV-based models with state caching and chunked processing

## Getting Started

### Training a 3B RWKV Model

```bash
python train_rwkv.py \
  --output_dir ./outputs/rwkv_3b \
  --experiment_name valkyrie_rwkv_3b \
  --train_file data/train.txt \
  --val_file data/val.txt \
  --max_seq_length 16384 \
  --seed 42
```

### Configuration

The RWKV model configuration is defined in `training/configs/model_3b_rwkv.py`. You can adjust the following key parameters:

- `use_rwkv`: Enable RWKV architecture
- `rwkv_time_mix_ratio`: Time mixing ratio for RWKV
- `rwkv_use_linear_att`: Use linear attention approximation
- `rwkv_att_scale`: Attention scale factor
- `rwkv_chunk_size`: Chunk size for efficient processing

### Hybrid Configuration

You can customize which layers use RWKV vs. transformer by modifying the `HybridModelConfigurator` in `train_rwkv.py`:

```python
# Set up which layers use RWKV vs traditional attention
hybrid_configurator.configure_layer_architecture(
    rwkv_layer_indices=[0, 2, 4, 6, ...],  # Layers that use RWKV
    transformer_layer_indices=[1, 3, 5, 7, ...]  # Layers that use transformers
)
```

## RWKV-Specific Features

### State Caching

RWKV supports efficient state caching for processing very long sequences:

```python
# Reset state
model.reset_state(batch_size=1)

# Process sequence in chunks
for chunk in chunks:
    outputs, new_state = model.process_with_state(chunk, model.state)
    model.state = new_state
```

### Chunked Processing

For memory-efficient processing of long sequences:

```python
# Set chunk size
model.set_chunk_size(1024)

# Process long sequence with chunking
outputs = model.forward_chunked(input_ids, labels)
```

### Evaluating RWKV Models

The repository includes RWKV-specific evaluation methods:

```python
# Evaluate sequence modeling capabilities
results = evaluate_sequence_modeling(
    model=model,
    tokenizer=tokenizer,
    chunk_size=1024
)
```

## Performance Considerations

- RWKV models generally use less memory than transformer models for the same sequence length
- Processing very long sequences (>16k tokens) is more efficient with RWKV
- For shorter sequences, the hybrid architecture provides a good balance of efficiency and performance
- The training script automatically handles memory optimizations like state compression and chunked processing

## License

[MIT License] 