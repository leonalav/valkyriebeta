# Valkyrie LLM - FineWeb Training Package

This package provides tools for training the Valkyrie LLM model on the FineWeb dataset, with special support for TPUs on Kaggle.

## Package Structure

The package is organized as follows:

```
valkyrie_llm/
├── __init__.py               # Main package initialization
├── cli/                      # Command-line interface tools
│   ├── __init__.py
│   └── train_fineweb.py      # CLI for FineWeb training
├── data/                     # Data loading utilities
│   ├── __init__.py
│   └── fineweb.py            # FineWeb dataset handling
├── utils/                    # Utility functions
│   ├── tpu_utils.py          # TPU support utilities
│   └── ...
└── scripts/                  # Utility scripts
    └── kaggle_tpu_setup.py   # Setup script for Kaggle TPU
```

## Installation

Install the package using pip:

```bash
pip install valkyrie-llm
```

Or install in development mode from the source:

```bash
git clone https://github.com/valkyrie-llm/valkyrie-llm.git
cd valkyrie-llm
pip install -e .
```

## Training on FineWeb Dataset

### Command-line Interface

After installation, you can use the `valkyrie-train-fineweb` command to train the model:

```bash
valkyrie-train-fineweb \
    --dataset_name HuggingFaceFW/fineweb \
    --dataset_config sample-10BT \
    --output_dir output/fineweb \
    --experiment_name valkyrie_fineweb_3b \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 6e-4 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --num_train_epochs 3 \
    --max_grad_norm 1.0 \
    --val_split 0.05 \
    --use_mixed_precision \
    --use_flash_attention \
    --use_gradient_checkpointing \
    --evaluate_reasoning
```

### Python API

You can also train the model using the Python API:

```python
from valkyrie_llm.cli.train_fineweb import main as train_fineweb

# Call the training function
train_fineweb()
```

Or import individual components:

```python
import torch
from valkyrie_llm.data.fineweb import setup_fineweb_dataloader
from valkyrie_llm.training.training_engine import TrainingEngine
from transformers import AutoTokenizer

# Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create dataloader
dataloader = setup_fineweb_dataloader(
    tokenizer=tokenizer,
    block_size=1024,
    batch_size=8,
    dataset_name="HuggingFaceFW/fineweb",
    config_name="sample-10BT"
)

# Continue with model setup and training...
```

## Training on Kaggle TPU

The package includes special support for training on Kaggle TPU. Use the provided Kaggle notebook template:

1. Upload `kaggle_tpu_train.ipynb` to Kaggle
2. Select TPU accelerator in the notebook settings
3. Run the notebook cells to train the model

Or set up your own TPU training with:

```python
import valkyrie_llm
from valkyrie_llm.utils.tpu_utils import setup_tpu_strategy, is_tpu_available

# Set up TPU strategy
strategy, is_tpu_available = setup_tpu_strategy()

if is_tpu_available:
    print("TPU is available for training")
    
    # Run training with TPU flag
    from valkyrie_llm.cli.train_fineweb import main as train_fineweb
    import sys
    
    sys.argv = [
        "train_fineweb.py",
        "--use_tpu",
        "--dataset_name=HuggingFaceFW/fineweb",
        "--dataset_config=sample-10BT",
        # Add other training parameters
    ]
    
    train_fineweb()
```

## Additional Resources

- See `README_FINEWEB.md` for detailed instructions on training with FineWeb
- Check `kaggle_tpu_train.ipynb` for a Kaggle TPU example
- See `valkyrie_llm/scripts/kaggle_tpu_setup.py` for TPU setup utilities

## Development

For development, we recommend installing in development mode:

```bash
pip install -e .
```

### Building the Package

To build the package:

```bash
pip install build
python -m build
```

This will create both a source distribution and a wheel in the `dist/` directory. 