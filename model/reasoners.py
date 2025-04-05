"""
Reasoners module for creating and configuring different reasoner types.
This module serves as a factory for reasoner implementations,
allowing for easy selection and configuration of reasoning strategies.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type, Union

from model.reasoning import (
    ReasoningStrategy,
    ReasoningConfig,
    BaseReasoner,
    ChainOfThoughtReasoner,
    TreeReasoner,
    RecursiveReasoner,
    NeuralSymbolicReasoner,
    KnowledgeReasoner,
    MCTSReasoner
)

logger = logging.getLogger(__name__)

# Registry of reasoner implementations
REASONER_REGISTRY: Dict[ReasoningStrategy, Type[BaseReasoner]] = {
    ReasoningStrategy.DIRECT: BaseReasoner,
    ReasoningStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtReasoner,
    ReasoningStrategy.STEP_BY_STEP: ChainOfThoughtReasoner,  # Same implementation
    ReasoningStrategy.TREE_OF_THOUGHT: TreeReasoner,
    ReasoningStrategy.RECURSIVE: RecursiveReasoner,
    ReasoningStrategy.NEURAL_SYMBOLIC: NeuralSymbolicReasoner,
    ReasoningStrategy.KNOWLEDGE_AUGMENTED: KnowledgeReasoner,
    ReasoningStrategy.MCTS: MCTSReasoner,
}

def create_reasoner(
    strategy: Union[str, ReasoningStrategy],
    config: Optional[Union[Dict[str, Any], ReasoningConfig]] = None,
    **kwargs
) -> BaseReasoner:
    """
    Create a reasoner based on the specified strategy
    
    Args:
        strategy: Reasoning strategy (string or enum)
        config: Configuration for the reasoner
        **kwargs: Additional keyword arguments for reasoner initialization
        
    Returns:
        An initialized reasoner instance
    """
    # Convert string to enum if needed
    if isinstance(strategy, str):
        strategy = ReasoningStrategy.from_string(strategy)
    
    # Ensure we have a config object
    if config is None:
        config = ReasoningConfig()
    elif isinstance(config, dict):
        config = ReasoningConfig(**config)
    
    # Update config with kwargs
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    # Get reasoner class
    if strategy not in REASONER_REGISTRY:
        logger.warning(f"Unknown reasoning strategy: {strategy}. Using DIRECT strategy.")
        reasoner_cls = REASONER_REGISTRY[ReasoningStrategy.DIRECT]
    else:
        reasoner_cls = REASONER_REGISTRY[strategy]
    
    # Create and return reasoner instance
    logger.info(f"Creating reasoner: {reasoner_cls.__name__}")
    return reasoner_cls(config)

def get_available_reasoners() -> Dict[str, ReasoningStrategy]:
    """
    Get a dictionary of available reasoners
    
    Returns:
        Dictionary mapping names to reasoning strategies
    """
    return {
        "direct": ReasoningStrategy.DIRECT,
        "cot": ReasoningStrategy.CHAIN_OF_THOUGHT,
        "chain_of_thought": ReasoningStrategy.CHAIN_OF_THOUGHT, 
        "step_by_step": ReasoningStrategy.STEP_BY_STEP,
        "tot": ReasoningStrategy.TREE_OF_THOUGHT,
        "tree_of_thought": ReasoningStrategy.TREE_OF_THOUGHT,
        "recursive": ReasoningStrategy.RECURSIVE,
        "neural_symbolic": ReasoningStrategy.NEURAL_SYMBOLIC,
        "knowledge": ReasoningStrategy.KNOWLEDGE_AUGMENTED,
        "knowledge_augmented": ReasoningStrategy.KNOWLEDGE_AUGMENTED,
        "mcts": ReasoningStrategy.MCTS,
    }

def get_best_reasoner_for_task(task_type: str) -> ReasoningStrategy:
    """
    Get the recommended reasoning strategy for a specific task type
    
    Args:
        task_type: The type of task (e.g., "qa", "math", "planning")
        
    Returns:
        The recommended reasoning strategy for the task
    """
    task_to_strategy = {
        "general": ReasoningStrategy.CHAIN_OF_THOUGHT,
        "qa": ReasoningStrategy.CHAIN_OF_THOUGHT,
        "math": ReasoningStrategy.RECURSIVE,
        "planning": ReasoningStrategy.TREE_OF_THOUGHT,
        "logic": ReasoningStrategy.NEURAL_SYMBOLIC,
        "knowledge_intensive": ReasoningStrategy.KNOWLEDGE_AUGMENTED,
        "search": ReasoningStrategy.MCTS,
        "step_by_step": ReasoningStrategy.STEP_BY_STEP,
    }
    
    if task_type not in task_to_strategy:
        logger.warning(f"Unknown task type: {task_type}. Using general strategy.")
        return task_to_strategy["general"]
    
    return task_to_strategy[task_type]

class AdaptiveReasoner(nn.Module):
    """
    Adaptive reasoner that selects the appropriate reasoning strategy based on the task
    """
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
        # Create reasoners for each strategy
        self.reasoners = nn.ModuleDict({
            strategy.name: REASONER_REGISTRY[strategy](config) 
            for strategy in REASONER_REGISTRY
        })
        
        # Task classifier to predict the best reasoning strategy
        self.task_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, len(REASONER_REGISTRY))
        )
        
    def forward(self, hidden_states, attention_mask=None, task_type=None, return_dict=False, **kwargs):
        """
        Apply adaptive reasoning
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            task_type: Optional explicit task type
            return_dict: Whether to return a dictionary
            **kwargs: Additional arguments
            
        Returns:
            Reasoning results
        """
        # If task type is specified, use the corresponding reasoner
        if task_type is not None:
            strategy = get_best_reasoner_for_task(task_type)
            return self.reasoners[strategy.name](
                hidden_states, 
                attention_mask, 
                return_dict,
                **kwargs
            )
        
        # Otherwise, predict the task type from the input
        batch_size = hidden_states.size(0)
        task_logits = self.task_classifier(hidden_states[:, 0])  # Use first token
        task_probs = torch.nn.functional.softmax(task_logits, dim=-1)
        
        # Select the best strategy
        best_strategy_idx = task_probs.argmax(dim=-1)
        
        # For simplicity, we'll use the same strategy for the whole batch
        # In a more advanced implementation, we could use different strategies per example
        strategy_idx = best_strategy_idx[0].item()
        strategy = list(REASONER_REGISTRY.keys())[strategy_idx]
        
        # Apply the selected reasoner
        return self.reasoners[strategy.name](
            hidden_states, 
            attention_mask, 
            return_dict,
            **kwargs
        ) 