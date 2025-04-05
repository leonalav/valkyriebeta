# Training with HuggingFaceFW/fineweb Dataset

This guide explains how to train a model using the HuggingFaceFW/fineweb dataset with our enhanced training pipeline.

## Dataset Information

The [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb) is a massive web corpus collected by Hugging Face. It contains high-quality, filtered web content suitable for large language model training.

Key features:
- Multiple subsets available (we use "CC-MAIN-2024-10" by default)
- Streaming support for processing massive amounts of data without loading it all into memory
- High-quality filtered web content

## Requirements

Before starting, ensure you have installed the following dependencies:

```bash
pip install torch transformers datasets
```

And any other dependencies required by the training code.

## Quick Start

To train a model using the FineWeb dataset, run:

```bash
python train_with_fineweb.py --tokenizer_path PATH_TO_TOKENIZER
```

Where `PATH_TO_TOKENIZER` is the path to a pretrained tokenizer (can be a local path or Hugging Face model ID).

## Configuration Options

### Basic Parameters

- `--output_dir`: Directory for saving checkpoints and logs (default: "output/fineweb")
- `--experiment_name`: Name of the experiment (default: "fineweb_training")
- `--tokenizer_path`: Path to pretrained tokenizer (required)

### Model Configuration

- `--model_type`: Base model architecture (default: "gpt2")
- `--hidden_size`: Hidden dimension size (default: 768)
- `--num_layers`: Number of transformer layers (default: 12)
- `--num_attention_heads`: Number of attention heads (default: 12)
- `--max_seq_length`: Maximum sequence length (default: 1024)

### Dataset Configuration

- `--dataset_subset`: FineWeb subset to use (default: "CC-MAIN-2024-10")
- `--batch_size`: Training batch size (default: 8)

### Training Parameters

- `--learning_rate`: Learning rate (default: 5e-5)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)
- `--use_mixed_precision`: Enable mixed precision training
- `--use_activation_checkpointing`: Use activation checkpointing to save memory
- `--save_steps`: Save checkpoint every X steps (default: 10000)
- `--eval_steps`: Evaluate every X steps (default: 5000)

## Advanced Usage

For more advanced configurations, you can directly use the modified `maintrain.py` with additional parameters:

```bash
python -m training.maintrain --huggingface_dataset=HuggingFaceFW/fineweb --huggingface_subset=CC-MAIN-2024-10 --use_streaming [...other args]
```

## Sample Configurations

### Small Model Training

```bash
python train_with_fineweb.py --tokenizer_path gpt2 --model_type=gpt2 --hidden_size=768 --num_layers=12 --num_attention_heads=12 --max_seq_length=1024 --batch_size=16 --use_mixed_precision
```

### Medium Model Training

```bash
python train_with_fineweb.py --tokenizer_path gpt2-medium --model_type=gpt2 --hidden_size=1024 --num_layers=24 --num_attention_heads=16 --max_seq_length=1024 --batch_size=8 --gradient_accumulation_steps=4 --use_mixed_precision --use_activation_checkpointing
```

### Training with Smaller Subset

To use the smaller 10BT sample:

```bash
python train_with_fineweb.py --tokenizer_path gpt2 --dataset_subset=sample-10BT
```

## Notes

- The training uses streaming mode by default, which is ideal for large datasets like FineWeb
- If you have limited memory, consider reducing batch size and using gradient accumulation
- For very long training runs, increase the save_steps parameter to reduce storage requirements 