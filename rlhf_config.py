#!/usr/bin/env python3
"""
RLHF Configuration Module

This module provides configuration management for advanced RLHF training
to enhance model reasoning capabilities.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union, Any
import torch

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for the model to be enhanced."""
    name: str
    tokenizer: str = "auto"
    model_type: str = "causal_lm"  # causal_lm, seq2seq, etc.
    max_length: int = 2048
    use_fp16: bool = True
    use_8bit: bool = False
    gradient_checkpointing: bool = True
    padding_side: str = "right"
    trust_remote_code: bool = False
    device_map: str = "auto"

@dataclass
class RLHFConfig:
    """Configuration for RLHF algorithm."""
    algorithm: str = "PPO"  # PPO, REINFORCE, DPO, IPO
    kl_coef: float = 0.1
    gamma: float = 0.99
    lam: float = 0.95
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    horizon: int = 10000
    reward_model: str = "auto"
    reward_model_type: str = "default"  # default, learned, external
    reward_normalization: bool = True
    ppo_epochs: int = 4
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    use_advantage_whitening: bool = True
    critic_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    update_epochs: int = 1
    update_mini_batch_size: int = 1
    dpo_beta: float = 0.1  # For DPO algorithm

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 8
    micro_batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    optimizer: str = "AdamW"  # AdamW, Adam, etc.
    scheduler: str = "cosine"  # cosine, linear, constant, etc.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    num_warmup_steps: int = 0
    seed: int = 42
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 3
    early_stopping_patience: int = 3
    checkpoint_dir: Optional[str] = None

@dataclass
class DataConfig:
    """Configuration for data handling."""
    use_synthetic: bool = True
    custom_data_path: Optional[str] = None
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    synthetic_data_size: int = 1000
    synthetic_template_path: Optional[str] = None
    use_augmentation: bool = True
    augmentation_factor: int = 2
    filter_nsfw: bool = True
    cache_dir: Optional[str] = None
    load_in_8bit: bool = False
    preprocessing_num_workers: int = 4
    dataloader_num_workers: int = 2
    dataset_format: str = "json"
    prompt_column_name: str = "prompt"
    response_column_name: str = "response"
    preference_column_name: str = "preference"

@dataclass
class ComponentConfig:
    """Configuration for reasoning components."""
    enabled: List[str] = field(default_factory=lambda: ["all"])
    weights: Dict[str, float] = field(default_factory=lambda: {
        "math": 1.0,
        "logical": 1.0,
        "causal": 1.0,
        "nlu": 1.0,
        "constitutional": 1.0
    })
    enhancement_level: str = "medium"  # basic, medium, advanced
    custom_component_path: Optional[str] = None

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "consistency", "reasoning_steps"
    ])
    test_datasets: List[str] = field(default_factory=lambda: ["GSM8K", "LogiQA"])
    custom_eval_path: Optional[str] = None
    eval_batch_size: int = 16
    max_eval_samples: Optional[int] = None
    eval_accumulation_steps: int = 1
    predict_with_generate: bool = True
    num_beams: int = 1
    max_new_tokens: int = 512
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    generation_do_sample: bool = True
    metric_for_best_model: str = "average"

@dataclass
class InfraConfig:
    """Configuration for infrastructure and resources."""
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    mixed_precision: str = "fp16"  # no, fp16, bf16
    cpu_only: bool = False
    distributed_training: bool = False
    world_size: int = 1
    local_rank: int = -1
    master_port: int = 29500
    deepspeed_config: Optional[str] = None
    num_worker_threads: int = 4
    log_level: str = "INFO"
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

@dataclass
class RLHFPipelineConfig:
    """Main configuration class that contains all configuration components."""
    model: ModelConfig = field(default_factory=ModelConfig)
    rlhf: RLHFConfig = field(default_factory=RLHFConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    components: ComponentConfig = field(default_factory=ComponentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    infra: InfraConfig = field(default_factory=InfraConfig)
    
    output_dir: str = "output"
    experiment_name: str = "rlhf_enhancement"
    run_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save config to a file."""
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RLHFPipelineConfig':
        """Create config from dictionary."""
        # Create base config
        config = cls()
        
        # Update nested dataclasses
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        
        if "rlhf" in config_dict:
            config.rlhf = RLHFConfig(**config_dict["rlhf"])
        
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        
        if "components" in config_dict:
            config.components = ComponentConfig(**config_dict["components"])
        
        if "evaluation" in config_dict:
            config.evaluation = EvaluationConfig(**config_dict["evaluation"])
        
        if "infra" in config_dict:
            config.infra = InfraConfig(**config_dict["infra"])
        
        # Update top-level attributes
        for key, value in config_dict.items():
            if key not in ["model", "rlhf", "training", "data", "components", "evaluation", "infra"]:
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RLHFPipelineConfig':
        """Create config from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'RLHFPipelineConfig':
        """Load config from a file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        logger.info(f"Configuration loaded from {path}")
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'RLHFPipelineConfig':
        """Create config from argparse namespace."""
        config = cls()
        
        # Map arguments to config
        if hasattr(args, "model_path") and args.model_path:
            config.model.name = args.model_path
        
        if hasattr(args, "output_dir") and args.output_dir:
            config.output_dir = args.output_dir
        
        if hasattr(args, "batch_size") and args.batch_size:
            config.training.batch_size = args.batch_size
        
        if hasattr(args, "learning_rate") and args.learning_rate:
            config.training.learning_rate = args.learning_rate
        
        if hasattr(args, "num_epochs") and args.num_epochs:
            config.training.num_epochs = args.num_epochs
        
        if hasattr(args, "components") and args.components:
            config.components.enabled = args.components
        
        if hasattr(args, "data_dir") and args.data_dir:
            config.data.custom_data_path = args.data_dir
            config.data.use_synthetic = False
        
        if hasattr(args, "use_synthetic_data") and args.use_synthetic_data:
            config.data.use_synthetic = True
        
        if hasattr(args, "gpu_id") and args.gpu_id:
            # Parse comma-separated GPU IDs
            gpu_ids = [int(id.strip()) for id in args.gpu_id.split(",") if id.strip()]
            config.infra.gpu_ids = gpu_ids
        
        return config

def get_default_config() -> RLHFPipelineConfig:
    """Get default configuration."""
    return RLHFPipelineConfig()

def get_small_model_config() -> RLHFPipelineConfig:
    """Get configuration optimized for small models (<= 7B parameters)."""
    config = RLHFPipelineConfig()
    config.model.use_fp16 = True
    config.training.batch_size = 16
    config.training.learning_rate = 2e-5
    config.rlhf.kl_coef = 0.05
    return config

def get_medium_model_config() -> RLHFPipelineConfig:
    """Get configuration optimized for medium models (7B-13B parameters)."""
    config = RLHFPipelineConfig()
    config.model.use_fp16 = True
    config.model.gradient_checkpointing = True
    config.training.batch_size = 8
    config.training.gradient_accumulation_steps = 2
    config.training.learning_rate = 1e-5
    return config

def get_large_model_config() -> RLHFPipelineConfig:
    """Get configuration optimized for large models (>13B parameters)."""
    config = RLHFPipelineConfig()
    config.model.use_fp16 = True
    config.model.use_8bit = True
    config.model.gradient_checkpointing = True
    config.training.batch_size = 4
    config.training.gradient_accumulation_steps = 4
    config.training.learning_rate = 5e-6
    config.data.load_in_8bit = True
    return config

def auto_config(model_path: str) -> RLHFPipelineConfig:
    """Automatically determine the best configuration based on model size and available resources."""
    config = get_default_config()
    config.model.name = model_path
    
    # Try to determine model size
    try:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Estimate parameters
        if hasattr(model_config, "num_parameters"):
            num_params = model_config.num_parameters
        elif hasattr(model_config, "n_params"):
            num_params = model_config.n_params
        elif hasattr(model_config, "hidden_size") and hasattr(model_config, "num_hidden_layers"):
            # Rough estimate for transformer models
            hidden_size = model_config.hidden_size
            num_layers = model_config.num_hidden_layers
            num_params = (12 * hidden_size * hidden_size * num_layers) // 1_000_000_000  # in billions
        else:
            # Default to medium if we can't determine
            logger.warning("Could not determine model size, defaulting to medium configuration")
            return get_medium_model_config()
        
        # Select configuration based on size
        if num_params < 7_000_000_000:  # < 7B
            logger.info(f"Detected small model (~{num_params/1_000_000_000:.1f}B parameters), using optimized configuration")
            return get_small_model_config()
        elif num_params <= 13_000_000_000:  # 7B-13B
            logger.info(f"Detected medium model (~{num_params/1_000_000_000:.1f}B parameters), using optimized configuration")
            return get_medium_model_config()
        else:  # > 13B
            logger.info(f"Detected large model (~{num_params/1_000_000_000:.1f}B parameters), using optimized configuration")
            return get_large_model_config()
    
    except Exception as e:
        logger.warning(f"Error determining model size: {e}")
        logger.warning("Defaulting to medium configuration")
        return get_medium_model_config()

def create_guided_config(model_path: str) -> RLHFPipelineConfig:
    """Interactive guided configuration creation."""
    config = get_default_config()
    config.model.name = model_path
    
    print("\n=== RLHF Enhancement Guided Configuration ===\n")
    
    # Model configuration
    print("\n--- Model Configuration ---")
    print(f"Model: {model_path}")
    
    use_8bit = input("Use 8-bit quantization? [y/N]: ").lower() == 'y'
    config.model.use_8bit = use_8bit
    
    # Components selection
    print("\n--- Components Configuration ---")
    components = []
    
    if input("Include Mathematical Reasoning enhancement? [Y/n]: ").lower() != 'n':
        components.append("math")
    
    if input("Include Logical Reasoning enhancement? [Y/n]: ").lower() != 'n':
        components.append("logical")
    
    if input("Include Causal Reasoning enhancement? [Y/n]: ").lower() != 'n':
        components.append("causal")
    
    if input("Include Natural Language Understanding enhancement? [Y/n]: ").lower() != 'n':
        components.append("nlu")
    
    if input("Include Constitutional AI alignment? [Y/n]: ").lower() != 'n':
        components.append("constitutional")
    
    if not components:
        print("No components selected, defaulting to 'all'")
        components = ["all"]
    
    config.components.enabled = components
    
    # RLHF algorithm
    print("\n--- RLHF Algorithm ---")
    algorithms = ["PPO", "REINFORCE", "DPO", "IPO"]
    for i, alg in enumerate(algorithms, 1):
        print(f"{i}. {alg}")
    
    alg_choice = input("Select algorithm [1-4, default=1]: ").strip()
    if alg_choice and alg_choice.isdigit() and 1 <= int(alg_choice) <= 4:
        config.rlhf.algorithm = algorithms[int(alg_choice) - 1]
    else:
        config.rlhf.algorithm = "PPO"
    
    # Training configuration
    print("\n--- Training Configuration ---")
    
    try:
        batch_size = input("Batch size [default=8]: ").strip()
        if batch_size and batch_size.isdigit():
            config.training.batch_size = int(batch_size)
        
        epochs = input("Number of epochs [default=3]: ").strip()
        if epochs and epochs.isdigit():
            config.training.num_epochs = int(epochs)
        
        lr = input("Learning rate [default=1e-5]: ").strip()
        if lr:
            try:
                config.training.learning_rate = float(lr)
            except ValueError:
                pass
    except Exception as e:
        logger.warning(f"Error parsing training parameters: {e}")
        logger.warning("Using default values")
    
    # Data configuration
    print("\n--- Data Configuration ---")
    
    use_synthetic = input("Use synthetic data? [Y/n]: ").lower() != 'n'
    config.data.use_synthetic = use_synthetic
    
    if not use_synthetic:
        data_path = input("Path to custom data: ").strip()
        if data_path:
            config.data.custom_data_path = data_path
        else:
            print("No custom data path provided, falling back to synthetic data")
            config.data.use_synthetic = True
    
    # Infrastructure configuration
    print("\n--- Infrastructure Configuration ---")
    
    gpus = input("Comma-separated GPU IDs to use [default=0]: ").strip()
    if gpus:
        try:
            gpu_ids = [int(id.strip()) for id in gpus.split(",") if id.strip()]
            if gpu_ids:
                config.infra.gpu_ids = gpu_ids
        except Exception:
            pass
    
    # Evaluation configuration
    print("\n--- Evaluation Configuration ---")
    
    eval_datasets = []
    
    if input("Evaluate on GSM8K (math)? [Y/n]: ").lower() != 'n':
        eval_datasets.append("GSM8K")
    
    if input("Evaluate on LogiQA (logical)? [Y/n]: ").lower() != 'n':
        eval_datasets.append("LogiQA")
    
    if input("Evaluate on BIG-Bench Hard? [y/N]: ").lower() == 'y':
        eval_datasets.append("BBH")
    
    if input("Evaluate on MMLU? [y/N]: ").lower() == 'y':
        eval_datasets.append("MMLU")
    
    if eval_datasets:
        config.evaluation.test_datasets = eval_datasets
    
    # Finish
    print("\n=== Configuration Complete ===\n")
    
    # Ask to save configuration
    if input("Save this configuration to file? [Y/n]: ").lower() != 'n':
        save_path = input("Save path [default=rlhf_config.json]: ").strip() or "rlhf_config.json"
        config.save(save_path)
        print(f"Configuration saved to {save_path}")
    
    return config

if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    print(config.to_json()) 