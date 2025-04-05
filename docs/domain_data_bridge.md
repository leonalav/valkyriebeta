# Domain Data Bridge

This document explains how to use the domain data bridge to connect existing data loaders with the enhanced training pipeline. The bridge allows you to use domain-specific data for training with knowledge distillation and other advanced features.

## Overview

The domain data bridge serves as a connector between your existing data loaders (in `data/collect_data.py` and `data/additionalcollect.py`) and the enhanced training pipeline's domain-specific data management. It provides:

1. A method to transform data from your loaders into a domain-specific format
2. Functions to extract domain-specific vocabulary
3. Utilities to split datasets into train/validation/test sets
4. Integration with the DomainDataManager for training

## Usage

### 1. Preparing Domain-Specific Datasets

Use the provided script to prepare domain-specific datasets:

```bash
python scripts/prepare_domain_datasets.py --config config/domain_bridge_config.json
```

This will:
- Load data from your existing loaders
- Format it for domain-specific training
- Split it into train/validation/test sets
- Extract domain-specific vocabulary
- Save it in a format compatible with the DomainDataManager

### 2. List Available Loaders

To see which loaders are available for which domains:

```bash
python scripts/prepare_domain_datasets.py --list_loaders
```

### 3. Process Specific Domains

To process only certain domains:

```bash
python scripts/prepare_domain_datasets.py --domains math science
```

### 4. Configure Data Preparation

Create or modify a configuration file (e.g., `config/domain_bridge_config.json`) with the following structure:

```json
{
  "loaders": {
    "load_math": {
      "max_samples": 3000
    },
    "load_scieval": {
      "n_examples": 2000
    }
  },
  "domain_weights": {
    "math": 1.5,
    "science": 1.2
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

### 5. Using the Prepared Data for Training

Use the prepared data with the enhanced training pipeline:

```bash
python train_aio.py --config config/domain_training_config.json
```

The training script will automatically:
- Load domain-specific datasets
- Create mixed dataloaders
- Configure knowledge distillation for domains
- Train with domain-specific optimizations

## Domain Mapping

The bridge maps loaders to domains as follows:

```
Math:
  - load_math
  - load_numinamath
  - load_olympic_arena
  - load_theoremqa
  - load_olympiad_bench
  - load_jeebench
  - load_statsqual

Science:
  - load_scieval
  - load_agieval

Coding:
  - load_livecodebench
  - load_usaco

Logic:
  - load_quant
  - load_gpqa_extended

Reasoning:
  - load_curated_thoughts
  - load_xword

General:
  - load_wildchat
```

You can modify this mapping in `data/domain_data_bridge.py` if needed.

## Extending the Bridge

### Adding New Loaders

1. Add your loader functions to `data/collect_data.py` or `data/additionalcollect.py`
2. Update the `DOMAIN_MAPPINGS` dictionary in `data/domain_data_bridge.py`
3. Update the `LOADER_FUNCTIONS` dictionary to include your new loaders

### Customizing Example Formatting

The bridge formats examples based on their domain and fields. You can customize this by modifying the `format_example_for_domain` function in `data/domain_data_bridge.py`.

## API Reference

### Key Functions

- `prepare_all_domains(config_file, output_dir, ...)`: Main function to prepare all domains
- `process_domain_data(loader_name, output_dir, ...)`: Process data for a specific domain
- `load_domain_data_for_training(data_dir, tokenizer, ...)`: Load prepared data for training
- `get_available_loaders()`: Get a dictionary of available loaders by domain

### Input Data Format

The bridge expects your loader functions to return a list of dictionaries. Each dictionary can have various fields, but the bridge looks for standard fields like:

- `question` and `solution` for math/science/logic domains
- `input` and `output` for coding domains
- `prompt` and `response` for conversational domains
- `text` for direct text examples

### Output Data Format

The bridge produces data in the following format:

```json
{
  "domain": "math",
  "text": "Question: What is 2+2?\n\nSolution: 4",
  "metadata": {...},
  "original_question": "What is 2+2?",
  "original_solution": "4"
}
```

## Troubleshooting

### Common Issues

1. **Loader not found**: Ensure your loader function is properly imported in the bridge
2. **Empty datasets**: Check if your loader function returns data in the expected format
3. **Missing domains**: Verify the domain mapping for your loaders
4. **Training errors**: Ensure the prepared data follows the format expected by DomainDataManager

### Logging

The bridge logs information to both the console and to `domain_preparation.log`. Check this file for detailed information about the data preparation process.