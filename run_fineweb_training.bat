@echo off
:: Script to train Valkyrie LLM on FineWeb sample-10BT dataset

:: Set paths and dataset info
set DATASET_NAME=HuggingFaceFW/fineweb
set DATASET_CONFIG=sample-10BT
set OUTPUT_DIR=output\fineweb
set EXPERIMENT_NAME=valkyrie_fineweb_3b

:: Create output directory if it doesn't exist
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

:: Set training hyperparameters
set BATCH_SIZE=8
set GRADIENT_ACCUMULATION_STEPS=8
set LEARNING_RATE=6e-4
set WEIGHT_DECAY=0.1
set WARMUP_STEPS=2000
set NUM_TRAIN_EPOCHS=3
set MAX_GRAD_NORM=1.0
set VAL_SPLIT=0.05

:: Check if running on Kaggle
set TPU_FLAG=

:: Run training
python train_fineweb.py ^
    --dataset_name %DATASET_NAME% ^
    --dataset_config %DATASET_CONFIG% ^
    --output_dir %OUTPUT_DIR% ^
    --experiment_name %EXPERIMENT_NAME% ^
    --batch_size %BATCH_SIZE% ^
    --gradient_accumulation_steps %GRADIENT_ACCUMULATION_STEPS% ^
    --learning_rate %LEARNING_RATE% ^
    --weight_decay %WEIGHT_DECAY% ^
    --warmup_steps %WARMUP_STEPS% ^
    --num_train_epochs %NUM_TRAIN_EPOCHS% ^
    --max_grad_norm %MAX_GRAD_NORM% ^
    --val_split %VAL_SPLIT% ^
    --use_mixed_precision ^
    --use_flash_attention ^
    --use_gradient_checkpointing ^
    --evaluate_reasoning %TPU_FLAG%

echo Training completed! 