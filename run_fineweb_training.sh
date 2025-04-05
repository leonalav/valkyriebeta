#!/bin/bash

# Script to train Valkyrie LLM on FineWeb sample-10BT dataset

# Set paths and dataset info
DATASET_NAME="HuggingFaceFW/fineweb"     # HuggingFace dataset name
DATASET_CONFIG="sample-10BT"             # Dataset config (sample-10BT for the 10B tokens sample)
OUTPUT_DIR="output/fineweb"              # Output directory for model checkpoints and logs
EXPERIMENT_NAME="valkyrie_fineweb_3b"    # Name of the experiment

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Set training hyperparameters
BATCH_SIZE=8                     # Per-device batch size
GRADIENT_ACCUMULATION_STEPS=8    # Gradient accumulation steps (effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
LEARNING_RATE=6e-4               # Peak learning rate
WEIGHT_DECAY=0.1                 # Weight decay for AdamW optimizer
WARMUP_STEPS=2000                # Linear warmup steps
NUM_TRAIN_EPOCHS=3               # Number of training epochs
MAX_GRAD_NORM=1.0                # Max gradient norm for gradient clipping
VAL_SPLIT=0.05                   # Validation split ratio

# Check if running on Kaggle TPU
if [ -n "$KAGGLE_KERNEL_RUN_TYPE" ]; then
    echo "Running on Kaggle. Checking for TPU..."
    if [ -d "/kaggle/input/tpu" ]; then
        echo "TPU detected. Enabling TPU training."
        TPU_FLAG="--use_tpu"
    else
        echo "No TPU detected. Using CPU/GPU."
        TPU_FLAG=""
    fi
else
    echo "Not running on Kaggle. Using CPU/GPU."
    TPU_FLAG=""
fi

# Run training
python train_fineweb.py \
    --dataset_name $DATASET_NAME \
    --dataset_config $DATASET_CONFIG \
    --output_dir $OUTPUT_DIR \
    --experiment_name $EXPERIMENT_NAME \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --max_grad_norm $MAX_GRAD_NORM \
    --val_split $VAL_SPLIT \
    --use_mixed_precision \
    --use_flash_attention \
    --use_gradient_checkpointing \
    --evaluate_reasoning \
    $TPU_FLAG

echo "Training completed!" 