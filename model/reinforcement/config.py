import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable
import os
import json

@dataclass
class RLConfig:
    """Configuration for reinforcement learning components."""
    # General RL parameters
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # PPO specific parameters
    ppo_epochs: int = 4
    ppo_batch_size: int = 64
    target_kl: float = 0.01
    
    # RLHF specific parameters
    use_rlhf: bool = True
    reward_model_path: Optional[str] = None
    reward_batch_size: int = 32
    human_feedback_path: Optional[str] = None
    preference_learning_lr: float = 1e-5
    
    # Expert Iteration parameters
    use_expert_iteration: bool = True
    num_expert_iterations: int = 10
    mcts_iterations_per_expert_step: int = 100
    expert_temperature: float = 1.0
    expert_batch_size: int = 16
    
    # DPO (Direct Preference Optimization) parameters
    use_dpo: bool = True
    dpo_beta: float = 0.1
    reference_model_path: Optional[str] = None
    
    # KTO (KL-constrained Preference Optimization) parameters
    use_kto: bool = False
    kto_alpha: float = 0.1
    
    # Rejection sampling parameters
    use_rejection_sampling: bool = True
    rejection_threshold: float = 0.7
    max_rejection_attempts: int = 5
    
    # Constitutional AI parameters
    use_constitutional_ai: bool = True
    constitution_path: Optional[str] = None
    constitution_weight: float = 1.0
    
    # Optimization parameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 100
    eval_frequency: int = 10
    save_frequency: int = 50
    log_frequency: int = 1
    
    # Environment parameters
    env_type: str = "text"  # "text", "code", "math", "reasoning"
    env_config: Dict[str, Any] = field(default_factory=dict)
    
    # Reward shaping
    use_reward_shaping: bool = True
    reward_shaping_fn: Optional[Callable] = None
    
    # Exploration strategies
    exploration_strategy: str = "entropy"  # "entropy", "epsilon_greedy", "ucb"
    exploration_decay: float = 0.995
    initial_exploration: float = 1.0
    min_exploration: float = 0.01
    
    # Distributed training
    use_distributed: bool = False
    world_size: int = 1
    
    # Logging and monitoring
    use_wandb: bool = True
    project_name: str = "nanogpt-rl"
    experiment_name: Optional[str] = None
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "RLConfig":
        """Load configuration from pretrained model path"""
        config_path = os.path.join(model_path, "rl_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"RL config file not found at {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
            
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_") and not callable(v)}
    
    def save_pretrained(self, save_directory: str) -> None:
        """Save configuration to directory"""
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "rl_config.json")
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ExpertIterationConfig:
    """Configuration for Expert Iteration (ExIt)."""
    # General ExIt parameters
    num_iterations: int = 10
    batch_size: int = 16
    temperature: float = 1.0
    
    # MCTS parameters for expert
    mcts_iterations: int = 100
    mcts_c_puct: float = 1.0
    mcts_dirichlet_alpha: float = 0.3
    mcts_dirichlet_weight: float = 0.25
    
    # Policy improvement parameters
    policy_lr: float = 1e-5
    policy_weight_decay: float = 0.01
    policy_epochs: int = 4
    policy_batch_size: int = 32
    
    # Value network parameters
    value_lr: float = 1e-5
    value_weight_decay: float = 0.01
    value_epochs: int = 4
    value_batch_size: int = 32
    
    # Training parameters
    eval_frequency: int = 1
    save_frequency: int = 1
    log_frequency: int = 1
    
    # Optimization parameters
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    
    # Logging and monitoring
    use_wandb: bool = True
    project_name: str = "nanogpt-exit"
    experiment_name: Optional[str] = None


@dataclass
class RLHFConfig:
    """Configuration for Reinforcement Learning from Human Feedback (RLHF)."""
    # Reward model parameters
    reward_model_hidden_size: int = 768
    reward_model_num_layers: int = 2
    reward_model_lr: float = 1e-5
    reward_model_weight_decay: float = 0.01
    reward_model_epochs: int = 3
    reward_model_batch_size: int = 16
    
    # Human feedback parameters
    feedback_collection_method: str = "pairwise"  # "pairwise", "rating", "ranking"
    num_comparisons_per_prompt: int = 3
    feedback_batch_size: int = 32
    
    # Preference learning parameters
    preference_learning_method: str = "bradley_terry"  # "bradley_terry", "thurstone_mosteller"
    preference_learning_epochs: int = 3
    preference_learning_batch_size: int = 16
    
    # RL fine-tuning parameters
    rl_algorithm: str = "ppo"  # "ppo", "dpo", "kto"
    rl_epochs: int = 4
    rl_batch_size: int = 16
    
    # KL penalty parameters
    kl_penalty_method: str = "adaptive"  # "fixed", "adaptive"
    initial_kl_coef: float = 0.2
    target_kl: float = 0.1
    
    # Optimization parameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Training parameters
    num_train_epochs: int = 3
    eval_frequency: int = 1
    save_frequency: int = 1
    log_frequency: int = 1
    
    # Logging and monitoring
    use_wandb: bool = True
    project_name: str = "nanogpt-rlhf"
    experiment_name: Optional[str] = None 