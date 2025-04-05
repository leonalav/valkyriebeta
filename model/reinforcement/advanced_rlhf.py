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
from .rlhf_math_integration import RLHFMathIntegration, RLHFMathConfig, MathRewardModel

# Import NLP components
from ..nlp.natural_language_understanding import NLUModule, NLUConfig
from ..nlp.semantic_parser import SemanticParser, SemanticParserConfig

# Import reasoning components
from ..reasoning.logical_reasoner import LogicalReasoner, LogicalReasoningConfig
from ..reasoning.causal_inference import CausalInferenceEngine, CausalInferenceConfig
from ..tree_reasoning_mcts import MonteCarloTreeSearch, MCTSConfig

# Import constitutional AI components
from ..constitutional_ai import ConstitutionalAI, ConstitutionalAIConfig

logger = logging.getLogger(__name__)

@dataclass
class AdvancedRLHFConfig:
    """
    Configuration for Advanced RLHF with comprehensive integration of reasoning components.
    """
    # Base configurations
    rl_config: RLConfig = field(default_factory=RLConfig)
    rlhf_config: RLHFConfig = field(default_factory=RLHFConfig)
    rlhf_math_config: RLHFMathConfig = field(default_factory=RLHFMathConfig)
    
    # Natural Language Understanding config
    nlu_config: NLUConfig = field(default_factory=NLUConfig)
    use_nlu: bool = True
    
    # Reasoning configurations
    logical_reasoning_config: LogicalReasoningConfig = field(default_factory=LogicalReasoningConfig)
    causal_inference_config: CausalInferenceConfig = field(default_factory=CausalInferenceConfig)
    use_logical_reasoning: bool = True
    use_causal_inference: bool = True
    
    # Constitutional AI configuration
    constitutional_ai_config: ConstitutionalAIConfig = field(default_factory=ConstitutionalAIConfig)
    use_constitutional_ai: bool = True
    
    # Recursive RLHF parameters
    use_recursive_rlhf: bool = True
    recursive_depth: int = 2
    recursive_kl_penalty: float = 0.1
    
    # Multi-agent debating
    use_multi_agent_debate: bool = True
    num_debate_agents: int = 3
    debate_iterations: int = 2
    
    # Reward model ensemble
    use_reward_ensemble: bool = True
    num_reward_models: int = 3
    ensemble_aggregation: str = "mean"  # "mean", "median", "max"
    
    # Advanced optimization
    use_population_based_training: bool = False
    population_size: int = 4
    migration_interval: int = 1000
    
    # Integration weights
    component_weights: Dict[str, float] = field(default_factory=lambda: {
        "language_modeling": 0.25,
        "mathematical_reasoning": 0.20,
        "logical_reasoning": 0.20,
        "causal_inference": 0.15,
        "constitutional_alignment": 0.20
    })
    
    # Training parameters
    num_epochs: int = 3
    eval_frequency: int = 1
    save_frequency: int = 1
    log_frequency: int = 10
    
    # Distributed training
    use_distributed: bool = False
    

class AdvancedRLHFIntegration:
    """
    Advanced RLHF Integration with multiple reasoning capabilities.
    
    This class integrates reinforcement learning from human feedback with:
    1. Mathematical reasoning
    2. Logical reasoning
    3. Natural language understanding
    4. Causal inference
    5. Constitutional AI alignment
    
    The integration supports multiple advanced techniques:
    - Recursive RLHF
    - Multi-agent debate
    - Reward model ensembles
    - Constitutional AI constraints
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: AdvancedRLHFConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        reference_model: Optional[nn.Module] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Create reference model (frozen) if not provided
        if reference_model is not None:
            self.reference_model = reference_model
        else:
            self.reference_model = self._create_reference_model()
        
        # Initialize RLHF components based on configuration
        self._initialize_rl_trainer()
        
        # Initialize mathematical reasoning integration
        if self.config.rlhf_math_config.use_math_reward_bonus:
            self.math_integration = RLHFMathIntegration(
                model=model,
                tokenizer=tokenizer,
                config=self.config.rlhf_math_config,
                device=device
            )
        else:
            self.math_integration = None
        
        # Initialize NLU module if enabled
        if self.config.use_nlu:
            self.nlu_module = NLUModule(
                config=self.config.nlu_config,
                device=device
            )
        else:
            self.nlu_module = None
        
        # Initialize reasoning components
        self._initialize_reasoning_components()
        
        # Initialize Constitutional AI if enabled
        if self.config.use_constitutional_ai:
            self.constitutional_ai = ConstitutionalAI(
                config=self.config.constitutional_ai_config,
                device=device
            )
        else:
            self.constitutional_ai = None
        
        # Initialize reward model ensemble if enabled
        if self.config.use_reward_ensemble:
            self.reward_models = self._initialize_reward_ensemble()
        else:
            self.reward_models = None
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.rl_config.learning_rate,
            weight_decay=config.rl_config.weight_decay
        )
        
        # Track training progress and metrics
        self.global_step = 0
        self.metrics = {
            "rl_metrics": {},
            "math_metrics": {},
            "logical_metrics": {},
            "nlu_metrics": {},
            "constitutional_metrics": {},
            "integration_metrics": {}
        }
    
    def _create_reference_model(self) -> nn.Module:
        """Create a copy of the current model to serve as reference model."""
        reference_model = type(self.model)(**self.model.config.to_dict())
        reference_model.load_state_dict(self.model.state_dict())
        reference_model.eval()  # Set to evaluation mode
        
        # Freeze parameters
        for param in reference_model.parameters():
            param.requires_grad = False
            
        return reference_model
    
    def _initialize_rl_trainer(self):
        """Initialize the appropriate RL trainer based on configuration."""
        if self.config.rl_config.use_ppo:
            self.rl_trainer = PPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                config=self.config.rl_config,
                device=self.device
            )
        elif self.config.rl_config.use_dpo:
            self.rl_trainer = DPOTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                config=self.config.rl_config,
                reference_model=self.reference_model,
                device=self.device
            )
        elif self.config.rl_config.use_expert_iteration:
            self.rl_trainer = ExpertIterationTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                config=self.config.rl_config,
                device=self.device
            )
        else:
            raise ValueError("No RL algorithm specified in config")
    
    def _initialize_reasoning_components(self):
        """Initialize logical reasoning and causal inference components."""
        # Initialize logical reasoning if enabled
        if self.config.use_logical_reasoning:
            self.logical_reasoner = LogicalReasoner(
                config=self.config.logical_reasoning_config,
                device=self.device
            )
        else:
            self.logical_reasoner = None
        
        # Initialize causal inference if enabled
        if self.config.use_causal_inference:
            self.causal_inference = CausalInferenceEngine(
                config=self.config.causal_inference_config,
                device=self.device
            )
        else:
            self.causal_inference = None
    
    def _initialize_reward_ensemble(self) -> List[nn.Module]:
        """Initialize ensemble of reward models."""
        reward_models = []
        
        for i in range(self.config.num_reward_models):
            # Create reward model with slightly different architectures or initializations
            reward_model = self._create_diverse_reward_model(i)
            reward_model.to(self.device)
            reward_models.append(reward_model)
        
        return reward_models
    
    def _create_diverse_reward_model(self, index: int) -> nn.Module:
        """Create a diverse reward model for the ensemble."""
        # Base model with variation in architecture based on index
        if index % 3 == 0:
            # Standard reward model
            return MathRewardModel(
                hidden_size=self.model.config.hidden_size,
                use_symbolic_verification=self.config.rlhf_math_config.use_symbolic_verification,
                use_numerical_verification=self.config.rlhf_math_config.use_numerical_verification
            )
        elif index % 3 == 1:
            # Reward model with deeper architecture
            return MathRewardModel(
                hidden_size=self.model.config.hidden_size,
                num_layers=4,  # More layers
                use_symbolic_verification=self.config.rlhf_math_config.use_symbolic_verification,
                use_numerical_verification=self.config.rlhf_math_config.use_numerical_verification
            )
        else:
            # Reward model with wider architecture
            return MathRewardModel(
                hidden_size=self.model.config.hidden_size * 2,  # Wider
                use_symbolic_verification=self.config.rlhf_math_config.use_symbolic_verification,
                use_numerical_verification=self.config.rlhf_math_config.use_numerical_verification
            )
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: Optional[int] = None,
        eval_dataloader: Optional[DataLoader] = None,
        math_eval_dataloader: Optional[DataLoader] = None,
        logical_eval_dataloader: Optional[DataLoader] = None,
        nlu_eval_dataloader: Optional[DataLoader] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Train the model with advanced RLHF techniques.
        
        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of epochs to train for (defaults to config value)
            eval_dataloader: DataLoader for general evaluation
            math_eval_dataloader: DataLoader for math-specific evaluation
            logical_eval_dataloader: DataLoader for logical reasoning evaluation
            nlu_eval_dataloader: DataLoader for NLU evaluation
            callback: Optional callback function called after each epoch
            
        Returns:
            Dictionary containing training metrics
        """
        epochs = num_epochs if num_epochs is not None else self.config.num_epochs
        logger.info(f"Starting advanced RLHF training for {epochs} epochs")
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Train with recursive RLHF if enabled
            if self.config.use_recursive_rlhf:
                self._train_recursive_rlhf(train_dataloader, epoch)
            else:
                # Standard training with integration
                self._train_integrated_rlhf(train_dataloader, epoch)
            
            # Evaluate on specialized tasks if dataloaders provided
            self._evaluate_all_tasks(
                epoch=epoch,
                eval_dataloader=eval_dataloader,
                math_eval_dataloader=math_eval_dataloader,
                logical_eval_dataloader=logical_eval_dataloader,
                nlu_eval_dataloader=nlu_eval_dataloader
            )
            
            # Log epoch metrics
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s")
            
            # Save checkpoint if configured
            if (epoch + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}")
            
            # Call callback if provided
            if callback is not None:
                callback(epoch, self.model, self.metrics)
        
        logger.info("Advanced RLHF training completed")
        return self.metrics
    
    def _train_recursive_rlhf(self, train_dataloader: DataLoader, epoch: int):
        """
        Implement recursive RLHF training.
        
        Recursive RLHF iteratively refines the policy by using the current model
        to generate better demonstrations for further training.
        """
        logger.info(f"Epoch {epoch+1}: Starting recursive RLHF (depth={self.config.recursive_depth})")
        
        # Initial dataset is the provided one
        current_dataloader = train_dataloader
        
        # Perform recursive training
        for depth in range(self.config.recursive_depth):
            logger.info(f"Recursive RLHF iteration {depth+1}/{self.config.recursive_depth}")
            
            # Train model on current dataset
            self._train_integrated_rlhf(current_dataloader, epoch, is_recursive=True)
            
            # Generate improved demonstrations with current model
            if depth < self.config.recursive_depth - 1:  # Skip generation on last iteration
                current_dataloader = self._generate_improved_demonstrations(current_dataloader)
    
    def _train_integrated_rlhf(self, train_dataloader: DataLoader, epoch: int, is_recursive: bool = False):
        """
        Train with integrated RLHF combining all components.
        
        This integrates language modeling, mathematical reasoning, logical reasoning,
        causal inference, and constitutional AI in a single training loop.
        """
        rl_metrics = {}
        
        # Multi-agent debate if enabled
        if self.config.use_multi_agent_debate and not is_recursive:
            train_dataloader = self._perform_multi_agent_debate(train_dataloader)
        
        # Main RL training
        rl_metrics = self.rl_trainer.train(
            train_dataloader=train_dataloader,
            num_epochs=1  # We're already in an epoch loop
        )
        
        # Integrate mathematical reasoning if enabled
        math_metrics = {}
        if self.math_integration is not None:
            math_outputs = self.math_integration.train_step(train_dataloader)
            math_metrics = math_outputs.get("metrics", {})
            
            # Apply math gradients with weight
            if "gradients" in math_outputs:
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in math_outputs["gradients"]:
                        weight = self.config.component_weights["mathematical_reasoning"]
                        param.grad += weight * math_outputs["gradients"][name]
        
        # Integrate logical reasoning if enabled
        logical_metrics = {}
        if self.logical_reasoner is not None:
            logical_outputs = self.logical_reasoner.train_step(
                model=self.model,
                dataloader=train_dataloader
            )
            logical_metrics = logical_outputs.get("metrics", {})
            
            # Apply logical reasoning gradients with weight
            if "gradients" in logical_outputs:
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in logical_outputs["gradients"]:
                        weight = self.config.component_weights["logical_reasoning"]
                        param.grad += weight * logical_outputs["gradients"][name]
        
        # Integrate NLU if enabled
        nlu_metrics = {}
        if self.nlu_module is not None:
            nlu_outputs = self.nlu_module.train_step(
                model=self.model,
                dataloader=train_dataloader
            )
            nlu_metrics = nlu_outputs.get("metrics", {})
            
            # Apply NLU gradients with weight
            if "gradients" in nlu_outputs:
                weight = self.config.component_weights["language_modeling"]
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in nlu_outputs["gradients"]:
                        param.grad += weight * nlu_outputs["gradients"][name]
        
        # Apply constitutional constraints if enabled
        constitutional_metrics = {}
        if self.constitutional_ai is not None:
            constitutional_outputs = self.constitutional_ai.apply_constraints(
                model=self.model,
                dataloader=train_dataloader
            )
            constitutional_metrics = constitutional_outputs.get("metrics", {})
            
            # Apply constitutional gradients with weight
            if "gradients" in constitutional_outputs:
                weight = self.config.component_weights["constitutional_alignment"]
                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in constitutional_outputs["gradients"]:
                        param.grad += weight * constitutional_outputs["gradients"][name]
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update metrics
        self.metrics["rl_metrics"][f"epoch_{epoch}"] = rl_metrics
        self.metrics["math_metrics"][f"epoch_{epoch}"] = math_metrics
        self.metrics["logical_metrics"][f"epoch_{epoch}"] = logical_metrics
        self.metrics["nlu_metrics"][f"epoch_{epoch}"] = nlu_metrics
        self.metrics["constitutional_metrics"][f"epoch_{epoch}"] = constitutional_metrics
        
        # Log step
        self.global_step += 1
        if self.global_step % self.config.log_frequency == 0:
            self._log_training_progress(epoch)
    
    def _evaluate_all_tasks(
        self,
        epoch: int,
        eval_dataloader: Optional[DataLoader] = None,
        math_eval_dataloader: Optional[DataLoader] = None,
        logical_eval_dataloader: Optional[DataLoader] = None,
        nlu_eval_dataloader: Optional[DataLoader] = None
    ):
        """Evaluate on all specialized tasks."""
        if epoch % self.config.eval_frequency != 0:
            return
        
        # General evaluation
        if eval_dataloader is not None:
            eval_metrics = self.rl_trainer.evaluate(eval_dataloader)
            self.metrics["integration_metrics"][f"epoch_{epoch}_eval"] = eval_metrics
        
        # Mathematical reasoning evaluation
        if math_eval_dataloader is not None and self.math_integration is not None:
            math_eval_metrics = self.math_integration.evaluate(math_eval_dataloader)
            self.metrics["math_metrics"][f"epoch_{epoch}_eval"] = math_eval_metrics
        
        # Logical reasoning evaluation
        if logical_eval_dataloader is not None and self.logical_reasoner is not None:
            logical_eval_metrics = self.logical_reasoner.evaluate(
                model=self.model,
                dataloader=logical_eval_dataloader
            )
            self.metrics["logical_metrics"][f"epoch_{epoch}_eval"] = logical_eval_metrics
        
        # NLU evaluation
        if nlu_eval_dataloader is not None and self.nlu_module is not None:
            nlu_eval_metrics = self.nlu_module.evaluate(
                model=self.model,
                dataloader=nlu_eval_dataloader
            )
            self.metrics["nlu_metrics"][f"epoch_{epoch}_eval"] = nlu_eval_metrics
    
    def _perform_multi_agent_debate(self, dataloader: DataLoader) -> DataLoader:
        """
        Perform multi-agent debate to improve training data.
        
        This simulates multiple agents debating to refine responses,
        resulting in higher quality training data.
        """
        logger.info(f"Performing multi-agent debate with {self.config.num_debate_agents} agents")
        
        # Implementation would create multiple model instances (agents)
        # and have them debate on responses to improve quality
        # Returns enhanced dataloader with debate-improved responses
        
        # For now, just return the original dataloader as a placeholder
        return dataloader
    
    def _generate_improved_demonstrations(self, dataloader: DataLoader) -> DataLoader:
        """
        Generate improved demonstrations with the current model.
        
        This is used in recursive RLHF to iteratively improve the training data.
        """
        logger.info("Generating improved demonstrations for recursive RLHF")
        
        # Implementation would use the current model to generate better
        # responses to the prompts in the dataloader
        
        # For now, just return the original dataloader as a placeholder
        return dataloader
    
    def _log_training_progress(self, epoch: int):
        """Log training progress metrics."""
        # Extract latest metrics
        rl_metrics = self.metrics["rl_metrics"].get(f"epoch_{epoch}", {})
        math_metrics = self.metrics["math_metrics"].get(f"epoch_{epoch}", {})
        logical_metrics = self.metrics["logical_metrics"].get(f"epoch_{epoch}", {})
        nlu_metrics = self.metrics["nlu_metrics"].get(f"epoch_{epoch}", {})
        
        # Log to console
        logger.info(f"Step {self.global_step}, Epoch {epoch+1}")
        if rl_metrics:
            logger.info(f"RL metrics: {rl_metrics}")
        if math_metrics:
            logger.info(f"Math metrics: {math_metrics}")
        if logical_metrics:
            logger.info(f"Logical metrics: {logical_metrics}")
        if nlu_metrics:
            logger.info(f"NLU metrics: {nlu_metrics}")
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        # Create output directory if it doesn't exist
        os.makedirs("checkpoints", exist_ok=True)
        
        # Save model
        checkpoint_path = os.path.join("checkpoints", f"{name}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "metrics": self.metrics
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}") 