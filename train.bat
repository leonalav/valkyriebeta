# Basic training
python train.py \
    --output_dir output \
    --experiment_name valkyrie_test \
    --hidden_size 768 \
    --num_layers 12 \
    --num_attention_heads 12 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --train_batch_size 8

# Training with advanced features
python train.py \
    --output_dir output \
    --experiment_name valkyrie_advanced \
    --use_moe \
    --num_experts 8 \
    --use_tree_reasoning \
    --use_neural_symbolic \
    --use_memory_augmentation \
    --use_mixed_precision \
    --use_flash_attention \
    --evaluate_reasoning

# Training with RLHF
python train.py \
    --output_dir output \
    --experiment_name valkyrie_rlhf \
    --use_rlhf \
    --rlhf_algorithm ppo \
    --reward_model_path path/to/reward/model \
    --reference_model_path path/to/reference/model