import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from tqdm import tqdm
import time
from dataclasses import dataclass, field

from .config import RLConfig, RLHFConfig
from .ppo import PPOTrainer
from .dpo import DPOTrainer
from .expert_iteration import ExpertIterationTrainer
from ..math_reasoning import MathReasoningConfig, SymbolicMathTransformer
from ..numerical_precision import NumericallyStableOperations, NumericalPrecisionConfig
from ..transformer import TransformerModel
from ..tree_reasoning_mcts import MonteCarloTreeSearch, MCTSConfig

logger = logging.getLogger(__name__)

@dataclass
class RLHFMathConfig:
    """Configuration for RLHF with mathematical reasoning integration."""
    # Base configurations
    rl_config: RLConfig = field(default_factory=RLConfig)
    rlhf_config: RLHFConfig = field(default_factory=RLHFConfig)
    math_config: MathReasoningConfig = field(default_factory=MathReasoningConfig)
    
    # Integration parameters
    use_math_reward_bonus: bool = True
    math_reward_weight: float = 0.5
    use_symbolic_verification: bool = True
    use_numerical_verification: bool = True
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_difficulty_levels: int = 5
    curriculum_steps_per_level: int = 1000
    
    # Multi-task learning
    use_multi_task: bool = True
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        "general": 0.4,
        "math": 0.3,
        "reasoning": 0.3
    })
    
    # Specialized RLHF components
    use_math_critic: bool = True
    use_reasoning_verifier: bool = True
    
    # Ablation and experimentation
    ablation_mode: Optional[str] = None  # None, "no_math", "no_rl", "no_verification"
    experimental_features: List[str] = field(default_factory=list)


class MathRewardModel(nn.Module):
    """
    Specialized reward model for mathematical reasoning tasks.
    
    This model evaluates the quality of mathematical reasoning steps and solutions,
    providing rewards based on correctness, clarity, and efficiency.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_symbolic_verification: bool = True,
        use_numerical_verification: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_symbolic_verification = use_symbolic_verification
        self.use_numerical_verification = use_numerical_verification
        
        # Encoder for mathematical expressions
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Layers for different aspects of mathematical reasoning
        self.correctness_layer = nn.Linear(hidden_size, hidden_size // 2)
        self.clarity_layer = nn.Linear(hidden_size, hidden_size // 2)
        self.efficiency_layer = nn.Linear(hidden_size, hidden_size // 2)
        
        # Final reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size * 3 // 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        
        # Numerical operations for verification
        self.numerical_ops = NumericallyStableOperations(NumericalPrecisionConfig())
        
        # Symbolic verification components (if enabled)
        if use_symbolic_verification:
            self.symbolic_transformer = SymbolicMathTransformer(
                hidden_size=hidden_size,
                num_heads=8,
                intermediate_size=hidden_size * 4
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        reasoning_steps: Optional[List[str]] = None,
        solution: Optional[str] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rewards for mathematical reasoning.
        
        Args:
            hidden_states: Hidden states from the language model
            reasoning_steps: Optional list of reasoning steps as strings
            solution: Optional final solution as string
            ground_truth: Optional ground truth answer for verification
            
        Returns:
            Dictionary containing rewards and component scores
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Encode hidden states
        encoded = self.encoder(hidden_states)
        
        # Extract features for different aspects of mathematical reasoning
        correctness_features = self.correctness_layer(encoded)
        clarity_features = self.clarity_layer(encoded)
        efficiency_features = self.efficiency_layer(encoded)
        
        # Concatenate features
        combined_features = torch.cat(
            [correctness_features, clarity_features, efficiency_features],
            dim=-1
        )
        
        # Compute base reward
        reward = self.reward_head(combined_features).squeeze(-1)
        
        # Apply symbolic verification if enabled and reasoning steps provided
        symbolic_score = torch.zeros_like(reward[:, 0])
        if self.use_symbolic_verification and reasoning_steps is not None:
            # This would involve more complex symbolic verification
            # For now, we'll use a placeholder implementation
            symbolic_score = torch.ones_like(reward[:, 0]) * 0.5
        
        # Apply numerical verification if enabled and solution + ground truth provided
        numerical_score = torch.zeros_like(reward[:, 0])
        if self.use_numerical_verification and solution is not None and ground_truth is not None:
            # This would involve numerical comparison between solution and ground truth
            # For now, we'll use a placeholder implementation
            numerical_score = torch.ones_like(reward[:, 0]) * 0.5
        
        # Compute final reward as weighted sum of components
        final_reward = reward.mean(dim=1) + symbolic_score + numerical_score
        
        return {
            "reward": final_reward,
            "base_reward": reward.mean(dim=1),
            "symbolic_score": symbolic_score,
            "numerical_score": numerical_score
        }


class RLHFMathIntegration:
    """
    Integrates RLHF with mathematical reasoning capabilities.
    
    This class combines reinforcement learning from human feedback with
    specialized mathematical reasoning components to create a model that
    excels at both general language tasks and mathematical reasoning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: RLHFMathConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize RLHF components based on configuration
        if config.rl_config.use_ppo:
            self.rl_trainer = PPOTrainer(
                model=model,
                tokenizer=tokenizer,
                config=config.rl_config,
                device=device
            )
        elif config.rl_config.use_dpo:
            self.rl_trainer = DPOTrainer(
                model=model,
                tokenizer=tokenizer,
                config=config.rl_config,
                device=device
            )
        elif config.rl_config.use_expert_iteration:
            self.rl_trainer = ExpertIterationTrainer(
                model=model,
                tokenizer=tokenizer,
                config=config.rl_config,
                device=device
            )
        else:
            raise ValueError("No RL algorithm specified in config")
        
        # Initialize math reward model if enabled
        if config.use_math_reward_bonus:
            self.math_reward_model = MathRewardModel(
                hidden_size=model.config.hidden_size,
                use_symbolic_verification=config.use_symbolic_verification,
                use_numerical_verification=config.use_numerical_verification
            ).to(device)
        else:
            self.math_reward_model = None
        
        # Initialize curriculum learning if enabled
        if config.use_curriculum:
            self.curriculum = MathCurriculum(
                difficulty_levels=config.curriculum_difficulty_levels,
                steps_per_level=config.curriculum_steps_per_level
            )
        else:
            self.curriculum = None
        
        # Track training progress
        self.global_step = 0
        self.metrics = {
            "rl_metrics": {},
            "math_metrics": {},
            "integration_metrics": {}
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        math_eval_dataloader: Optional[DataLoader] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train the model using integrated RLHF and mathematical reasoning.
        
        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of epochs to train for
            eval_dataloader: Optional DataLoader for general evaluation
            math_eval_dataloader: Optional DataLoader for math-specific evaluation
            callback: Optional callback function called after each epoch
            
        Returns:
            metrics: Dictionary of training metrics
        """
        logger.info(f"Starting integrated RLHF+Math training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Update curriculum if enabled
            if self.curriculum is not None:
                current_difficulty = self.curriculum.get_current_difficulty()
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Curriculum difficulty: {current_difficulty}")
                train_dataloader = self.curriculum.get_data_loader(train_dataloader)
            
            # Train with RL algorithm
            rl_metrics = self.rl_trainer.train(
                train_dataloader=train_dataloader,
                num_epochs=1,  # We're already in an epoch loop
                eval_dataloader=eval_dataloader
            )
            
            # Evaluate on math-specific tasks if provided
            math_metrics = {}
            if math_eval_dataloader is not None:
                math_metrics = self.evaluate_math(math_eval_dataloader)
            
            # Update metrics
            self.metrics["rl_metrics"][f"epoch_{epoch}"] = rl_metrics
            self.metrics["math_metrics"][f"epoch_{epoch}"] = math_metrics
            
            # Update curriculum
            if self.curriculum is not None:
                self.curriculum.step(math_metrics.get("accuracy", 0))
            
            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
            
            # Call callback if provided
            if callback is not None:
                callback(epoch, self.model, {**rl_metrics, **math_metrics})
        
        logger.info("Integrated RLHF+Math training completed")
        return self.metrics
    
    def evaluate_math(self, math_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on mathematical reasoning tasks.
        
        Args:
            math_dataloader: DataLoader for math evaluation data
            
        Returns:
            metrics: Dictionary of math evaluation metrics
        """
        self.model.eval()
        
        metrics = {
            "accuracy": 0.0,
            "symbolic_score": 0.0,
            "numerical_score": 0.0,
            "reasoning_steps": 0.0
        }
        
        total_samples = 0
        correct_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(math_dataloader, desc="Evaluating math"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch.get("labels")
                ground_truth = batch.get("ground_truth")
                
                # Generate responses
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=512,
                    do_sample=False
                )
                
                # Decode outputs
                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Extract solutions and reasoning steps
                solutions = []
                reasoning_steps_list = []
                
                for text in generated_texts:
                    solution, steps = self._extract_solution_and_steps(text)
                    solutions.append(solution)
                    reasoning_steps_list.append(steps)
                
                # Compute accuracy if ground truth available
                if ground_truth is not None:
                    for solution, gt in zip(solutions, ground_truth):
                        if self._is_correct_solution(solution, gt):
                            correct_samples += 1
                        total_samples += 1
                
                # Compute other metrics
                # This would involve more complex evaluation of reasoning steps
                # For now, we'll use placeholder values
                metrics["symbolic_score"] += 0.5 * len(solutions)
                metrics["numerical_score"] += 0.5 * len(solutions)
                metrics["reasoning_steps"] += 0.5 * len(solutions)
        
        # Compute final metrics
        if total_samples > 0:
            metrics["accuracy"] = correct_samples / total_samples
        
        # Normalize other metrics
        for key in ["symbolic_score", "numerical_score", "reasoning_steps"]:
            if total_samples > 0:
                metrics[key] /= total_samples
        
        return metrics
    
    def _extract_solution_and_steps(self, text: str) -> Tuple[str, List[str]]:
        """
        Extract the final solution and reasoning steps from generated text.
        
        Args:
            text: Generated text containing reasoning and solution
            
        Returns:
            solution: Extracted solution
            steps: List of reasoning steps
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated parsing
        
        # Try to find solution after "Answer:" or similar markers
        solution = ""
        if "Answer:" in text:
            solution = text.split("Answer:")[-1].strip()
        elif "Solution:" in text:
            solution = text.split("Solution:")[-1].strip()
        elif "Therefore," in text:
            solution = text.split("Therefore,")[-1].strip()
        
        # Extract reasoning steps
        steps = []
        if "Step" in text:
            parts = text.split("Step")
            for part in parts[1:]:  # Skip the first part (before "Step 1")
                if ":" in part:
                    step_text = part.split(":", 1)[1].strip()
                    steps.append(step_text)
        
        return solution, steps
    
    def _is_correct_solution(self, solution: str, ground_truth: str) -> bool:
        """
        Check if the solution is correct by comparing with ground truth.
        
        Args:
            solution: Model's solution
            ground_truth: Ground truth solution
            
        Returns:
            is_correct: Whether the solution is correct
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated comparison
        
        # Normalize both strings
        solution = solution.strip().lower()
        ground_truth = ground_truth.strip().lower()
        
        # Extract numbers if present
        solution_num = self._extract_number(solution)
        ground_truth_num = self._extract_number(ground_truth)
        
        # If both are numbers, compare numerically
        if solution_num is not None and ground_truth_num is not None:
            return abs(solution_num - ground_truth_num) < 1e-6
        
        # Otherwise, check if strings match
        return solution == ground_truth
    
    def _extract_number(self, text: str) -> Optional[float]:
        """
        Extract a number from text if present.
        
        Args:
            text: Text to extract number from
            
        Returns:
            number: Extracted number or None if no number found
        """
        import re
        
        # Try to find a number in the text
        match = re.search(r"[-+]?\d*\.\d+|\d+", text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass
        
        return None
    
    def save(self, save_dir: str) -> None:
        """
        Save the integrated model and training state.
        
        Args:
            save_dir: Directory to save to
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save RL trainer
        rl_save_dir = os.path.join(save_dir, "rl_trainer")
        os.makedirs(rl_save_dir, exist_ok=True)
        self.rl_trainer.save(rl_save_dir)
        
        # Save math reward model if exists
        if self.math_reward_model is not None:
            math_model_path = os.path.join(save_dir, "math_reward_model.pt")
            torch.save(self.math_reward_model.state_dict(), math_model_path)
        
        # Save configuration
        config_path = os.path.join(save_dir, "rlhf_math_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save metrics
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Integrated model and training state saved to {save_dir}")
    
    def load(self, load_dir: str) -> None:
        """
        Load the integrated model and training state.
        
        Args:
            load_dir: Directory to load from
        """
        # Load RL trainer
        rl_load_dir = os.path.join(load_dir, "rl_trainer")
        if os.path.exists(rl_load_dir):
            self.rl_trainer.load(rl_load_dir)
        
        # Load math reward model if exists
        math_model_path = os.path.join(load_dir, "math_reward_model.pt")
        if os.path.exists(math_model_path) and self.math_reward_model is not None:
            self.math_reward_model.load_state_dict(
                torch.load(math_model_path, map_location=self.device)
            )
        
        # Load metrics if exists
        metrics_path = os.path.join(load_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
        
        logger.info(f"Integrated model and training state loaded from {load_dir}")


class MathCurriculum:
    """
    Curriculum learning for mathematical reasoning tasks.
    
    This class manages the difficulty progression during training,
    starting with simpler problems and gradually increasing complexity.
    """
    
    def __init__(
        self,
        difficulty_levels: int = 5,
        steps_per_level: int = 1000,
        auto_progress: bool = True,
        progress_threshold: float = 0.8
    ):
        self.difficulty_levels = difficulty_levels
        self.steps_per_level = steps_per_level
        self.auto_progress = auto_progress
        self.progress_threshold = progress_threshold
        
        self.current_level = 0
        self.steps_at_current_level = 0
        self.performance_history = []
    
    def step(self, performance_metric: float = None) -> None:
        """
        Update curriculum state after a training step.
        
        Args:
            performance_metric: Optional performance metric to determine progression
        """
        self.steps_at_current_level += 1
        
        # Store performance metric if provided
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
            
            # Keep only recent history
            if len(self.performance_history) > 10:
                self.performance_history = self.performance_history[-10:]
        
        # Check if we should progress to next level
        if self.auto_progress:
            # Progress based on steps
            if self.steps_at_current_level >= self.steps_per_level:
                self._progress_to_next_level()
            
            # Progress based on performance
            elif (performance_metric is not None and 
                  len(self.performance_history) >= 5 and
                  sum(self.performance_history[-5:]) / 5 >= self.progress_threshold):
                self._progress_to_next_level()
    
    def _progress_to_next_level(self) -> None:
        """Progress to the next difficulty level."""
        if self.current_level < self.difficulty_levels - 1:
            self.current_level += 1
            self.steps_at_current_level = 0
            self.performance_history = []
            logger.info(f"Curriculum progressed to level {self.current_level+1}/{self.difficulty_levels}")
    
    def get_current_difficulty(self) -> float:
        """Get the current difficulty level normalized to [0, 1]."""
        return self.current_level / (self.difficulty_levels - 1)
    
    def get_data_loader(self, base_dataloader: DataLoader) -> DataLoader:
        """
        Get a data loader filtered to the current difficulty level.
        
        In a real implementation, this would filter the dataset based on difficulty.
        For now, we'll return the original data loader.
        
        Args:
            base_dataloader: Original data loader
            
        Returns:
            filtered_dataloader: Data loader filtered to current difficulty
        """
        # This is a placeholder implementation
        # In a real system, this would filter the dataset based on difficulty
        return base_dataloader


class MathReasoningDataset(Dataset):
    """
    Dataset for mathematical reasoning tasks.
    
    This dataset contains mathematical problems with ground truth solutions
    and optional human demonstrations for RLHF.
    """
    
    def __init__(
        self,
        problems: List[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 512,
        difficulty_key: str = "difficulty"
    ):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.difficulty_key = difficulty_key
    
    def __len__(self) -> int:
        return len(self.problems)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        problem = self.problems[idx]
        
        # Tokenize problem statement
        tokens = self.tokenizer(
            problem["problem"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "ground_truth": problem["solution"],
            "difficulty": problem.get(self.difficulty_key, 0.0)
        }
        
        # Add human demonstration if available
        if "demonstration" in problem:
            demo_tokens = self.tokenizer(
                problem["demonstration"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            item["demonstration_input_ids"] = demo_tokens["input_ids"].squeeze(0)
            item["demonstration_attention_mask"] = demo_tokens["attention_mask"].squeeze(0)
        
        return item 