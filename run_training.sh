#!/bin/bash

# Simple script to run training with different configurations

# Basic training
function run_basic_training() {
    echo "Running basic training..."
    python train.py \
        --experiment_name basic_model \
        --output_dir output/basic
}

# Advanced training with reasoning
function run_advanced_training() {
    echo "Running advanced training with reasoning..."
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
}

# Training with RLHF
function run_rlhf_training() {
    echo "Running training with RLHF..."
    python train.py \
        --experiment_name rlhf_model \
        --output_dir output/rlhf \
        --use_rlhf \
        --rlhf_type ppo
}

# Training with domain-specific data
function run_domain_training() {
    echo "Running domain-specific training..."
    python train.py \
        --experiment_name domain_model \
        --output_dir output/domain \
        --use_domain_training \
        --domains science medicine finance
}

# Parse command-line arguments
case "$1" in
    basic)
        run_basic_training
        ;;
    advanced)
        run_advanced_training
        ;;
    rlhf)
        run_rlhf_training
        ;;
    domain)
        run_domain_training
        ;;
    all)
        run_basic_training
        run_advanced_training
        run_rlhf_training
        run_domain_training
        ;;
    *)
        echo "Usage: $0 {basic|advanced|rlhf|domain|all}"
        exit 1
        ;;
esac

echo "Training complete!" 