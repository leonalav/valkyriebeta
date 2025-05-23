import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass, field
from transformers import PreTrainedModel, AutoConfig, AutoTokenizer
import os
import logging
import math
from enum import Enum, auto
import random
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ReasoningStrategy(Enum):
    """Enumeration of available reasoning strategies"""
    DIRECT = auto()
    CHAIN_OF_THOUGHT = auto()
    STEP_BY_STEP = auto()
    TREE_OF_THOUGHT = auto()
    NEURAL_SYMBOLIC = auto()
    RECURSIVE = auto()
    KNOWLEDGE_AUGMENTED = auto()
    COMPOSITIONAL = auto()
    MCTS = auto()
    
    @classmethod
    def from_string(cls, strategy_str: str) -> 'ReasoningStrategy':
        strategy_map = {
            "direct": cls.DIRECT,
            "cot": cls.CHAIN_OF_THOUGHT,
            "chain_of_thought": cls.CHAIN_OF_THOUGHT,
            "step_by_step": cls.STEP_BY_STEP,
            "tot": cls.TREE_OF_THOUGHT,
            "tree_of_thought": cls.TREE_OF_THOUGHT,
            "neural_symbolic": cls.NEURAL_SYMBOLIC,
            "recursive": cls.RECURSIVE,
            "knowledge": cls.KNOWLEDGE_AUGMENTED,
            "knowledge_augmented": cls.KNOWLEDGE_AUGMENTED,
            "compositional": cls.COMPOSITIONAL,
            "mcts": cls.MCTS,
            "monte_carlo_tree_search": cls.MCTS
        }
        
        strategy_str = strategy_str.lower().strip()
        if strategy_str in strategy_map:
            return strategy_map[strategy_str]
        else:
            logger.warning(f"Unknown reasoning strategy: {strategy_str}. Defaulting to DIRECT.")
            return cls.DIRECT

@dataclass
class ReasoningConfig:
    """Configuration for reasoning components"""
    enabled: bool = True
    strategy: Union[ReasoningStrategy, str] = ReasoningStrategy.CHAIN_OF_THOUGHT
    max_reasoning_steps: int = 5
    reasoning_depth: int = 3
    tree_width: int = 3
    mcts_simulations: int = 100
    uncertainty_threshold: float = 0.2
    confidence_threshold: float = 0.7
    use_verification: bool = True
    use_symbolic_reasoning: bool = True
    use_neural_reasoning: bool = True
    knowledge_graph_size: int = 1000
    recursive_depth: int = 3
    
    def __post_init__(self):
        # Convert string strategy to enum if needed
        if isinstance(self.strategy, str):
            self.strategy = ReasoningStrategy.from_string(self.strategy)

@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process"""
    step_type: str  # 'parse', 'process', or 'resolve'
    intermediate_result: torch.Tensor
    confidence: float
    logical_predicates: Optional[Dict[str, torch.Tensor]] = None
    uncertainty: Optional[float] = None
    reasoning_path: Optional[List[str]] = None
    
class ReasoningOutput:
    """Output from a reasoning process"""
    def __init__(
        self,
        hidden_states: torch.Tensor,
        steps: List[ReasoningStep],
        confidence: float,
        strategy: ReasoningStrategy,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.hidden_states = hidden_states
        self.steps = steps
        self.confidence = confidence
        self.strategy = strategy
        self.metadata = metadata or {}
        
    def get_final_state(self) -> torch.Tensor:
        """Get the final hidden state"""
        return self.hidden_states
    
    def get_confidence(self) -> float:
        """Get the confidence score"""
        return self.confidence
    
    def get_reasoning_path(self) -> List[str]:
        """Get the reasoning path as strings"""
        path = []
        for step in self.steps:
            if step.reasoning_path:
                path.extend(step.reasoning_path)
        return path

class BaseReasoner(nn.Module):
    """Base class for all reasoning components"""
    def __init__(self, config: ReasoningConfig):
        super().__init__()
        self.config = config
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = False,
                **kwargs) -> Union[torch.Tensor, ReasoningOutput]:
        """
        Perform reasoning on hidden states
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dict: If True, returns a ReasoningOutput object. If False, returns just hidden_states.
            
        Returns:
            If return_dict=True: ReasoningOutput object
            If return_dict=False: hidden_states tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def estimate_uncertainty(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Estimate uncertainty using Monte Carlo Dropout
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple of (adjusted_hidden_states, uncertainty)
        """
        dropout = nn.Dropout(p=0.1)
        num_samples = 5
        samples = []
        
        for _ in range(num_samples):
            sample = dropout(hidden_states)
            samples.append(sample)
            
        # Calculate mean and variance
        samples_tensor = torch.stack(samples)
        mean = samples_tensor.mean(0)
        uncertainty = samples_tensor.var(0).mean().item()
        
        # Adjust output based on uncertainty
        adjusted = mean * (1 - uncertainty)
        return adjusted, uncertainty

class ChainOfThoughtReasoner(PreTrainedModel):
    def __init__(self, config):
        super().__init__(AutoConfig.from_pretrained(config.model_name))
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        hidden_size = config.hidden_size
        
        # Three-step reasoning components
        self.parser = nn.ModuleDict({
            'encoder': nn.Linear(hidden_size, hidden_size),
            'predicate_extractor': nn.Linear(hidden_size, config.num_predicates),
            'uncertainty': nn.Linear(hidden_size, 1)
        })
        
        self.processor = nn.ModuleDict({
            'thought_controller': nn.LSTM(hidden_size, hidden_size, batch_first=True),
            'dynamic_router': nn.Linear(hidden_size, config.num_reasoning_steps),
            'neural_logic': NeuralLogicMachine(config)
        })
        
        self.resolver = nn.ModuleDict({
            'consistency_checker': nn.Linear(hidden_size, hidden_size),
            'output_voter': nn.Linear(hidden_size, config.num_output_classes),
            'uncertainty_estimator': nn.Dropout(p=0.1)
        })
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[ReasoningStep]]:
        batch_size = hidden_states.size(0)
        reasoning_steps = []
        
        # Step 1: Parse
        parsed = self._parse_input(hidden_states)
        reasoning_steps.append(parsed)
        
        # Step 2: Process with dynamic thought allocation
        processed = self._process_with_dynamics(parsed.intermediate_result)
        reasoning_steps.append(processed)
        
        # Step 3: Resolve with consistency checking
        resolved = self._resolve_with_consistency(processed.intermediate_result)
        reasoning_steps.append(resolved)
        
        # Perform Monte Carlo Dropout for uncertainty estimation
        final_output = self._estimate_uncertainty(resolved.intermediate_result)
        
        return final_output, reasoning_steps
    
    def _parse_input(self, hidden_states: torch.Tensor) -> ReasoningStep:
        """Parse input and extract logical predicates"""
        encoded = self.parser['encoder'](hidden_states)
        predicates = torch.sigmoid(self.parser['predicate_extractor'](encoded))
        uncertainty = torch.sigmoid(self.parser['uncertainty'](encoded))
        
        return ReasoningStep(
            step_type='parse',
            intermediate_result=encoded,
            confidence=predicates.mean().item(),
            logical_predicates={'base_predicates': predicates},
            uncertainty=uncertainty.mean().item()
        )
    
    def _process_with_dynamics(self, hidden_states: torch.Tensor) -> ReasoningStep:
        """Process with dynamic thought allocation"""
        # Dynamic routing based on complexity
        routing_weights = torch.softmax(self.processor['dynamic_router'](hidden_states), dim=-1)
        
        # LSTM for thought processing
        thought_output, _ = self.processor['thought_controller'](hidden_states)
        
        # Apply Neural Logic Machine
        logic_output = self.processor['neural_logic'](thought_output, routing_weights)
        
        return ReasoningStep(
            step_type='process',
            intermediate_result=logic_output,
            confidence=routing_weights.max().item()
        )
    
    def _resolve_with_consistency(self, hidden_states: torch.Tensor) -> ReasoningStep:
        """Resolve with self-consistency checking"""
        # Generate multiple decoding paths
        num_paths = 5
        decoding_paths = []
        
        for _ in range(num_paths):
            path = self.resolver['consistency_checker'](hidden_states)
            decoding_paths.append(path)

        # Check consistency across paths
        paths_tensor = torch.stack(decoding_paths)
        consistency_scores = torch.cosine_similarity(paths_tensor, paths_tensor.mean(0, keepdim=True), dim=-1)
        
        # Weighted voting
        votes = self.resolver['output_voter'](torch.stack(decoding_paths))
        weighted_votes = votes * consistency_scores.unsqueeze(-1)
        final_output = weighted_votes.mean(0)
        
        return ReasoningStep(
            step_type='resolve',
            intermediate_result=final_output,
            confidence=consistency_scores.mean().item()
        )
    
    def _estimate_uncertainty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Estimate uncertainty using Monte Carlo Dropout"""
        num_samples = 10
        dropout_samples = []
        
        for _ in range(num_samples):
            sample = self.resolver['uncertainty_estimator'](hidden_states)
            dropout_samples.append(sample)
            
        # Calculate mean and variance
        samples_tensor = torch.stack(dropout_samples)
        mean = samples_tensor.mean(0)
        uncertainty = samples_tensor.var(0)
        
        # Adjust output based on uncertainty
        final_output = mean * (1 - uncertainty)
        return final_output

    def save_pretrained(self, save_directory):
        """Save model in Hugging Face format"""
        # Save the model configuration
        self.config.save_pretrained(save_directory)
        
        # Save model weights
        model_to_save = self.module if hasattr(self, 'module') else self
        state_dict = model_to_save.state_dict()
        torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory)

class NeuralLogicMachine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Neural predicates
        self.predicate_embeddings = nn.Parameter(
            torch.randn(config.num_predicates, config.hidden_size)
        )
        
        # Logic operations
        self.and_gate = nn.Bilinear(config.hidden_size, config.hidden_size, config.hidden_size)
        self.or_gate = nn.Bilinear(config.hidden_size, config.hidden_size, config.hidden_size)
        self.not_gate = nn.Linear(config.hidden_size, config.hidden_size)
        
        # SAT solver integration
        self.sat_projection = nn.Linear(config.hidden_size, config.num_predicates)
        
    def forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        
        # Extract predicates
        predicate_scores = torch.matmul(hidden_states, self.predicate_embeddings.t())
        
        # Apply logical operations
        logical_states = []
        for i in range(batch_size):
            # AND operation
            and_result = self.and_gate(hidden_states[i:i+1], predicate_scores[i:i+1])
            
            # OR operation
            or_result = self.or_gate(hidden_states[i:i+1], predicate_scores[i:i+1])
            
            # NOT operation
            not_result = self.not_gate(hidden_states[i:i+1])
            
            # Combine based on routing weights
            combined = (routing_weights[i:i+1, 0:1] * and_result +
                       routing_weights[i:i+1, 1:2] * or_result +
                       routing_weights[i:i+1, 2:3] * not_result)
            
            logical_states.append(combined)
        
        logical_output = torch.cat(logical_states, dim=0)
        
        # Project to predicate space for SAT solver
        sat_predicates = self.sat_projection(logical_output)
        
        return logical_output

class TreeReasoner(BaseReasoner):
    """Tree-based reasoning with multiple paths of exploration"""
    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        self.hidden_size = getattr(config, "hidden_size", 768)
        
        # Tree structure parameters
        self.max_depth = config.reasoning_depth
        self.branching_factor = config.tree_width
        
        # Components
        self.node_expander = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size * self.branching_factor)
        )
        
        self.node_evaluator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        self.path_aggregator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Dropout(p=0.1)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = False,
                **kwargs) -> Union[torch.Tensor, ReasoningOutput]:
        batch_size, seq_len, _ = hidden_states.size()
        reasoning_steps = []
        
        # Use the first token's hidden state as the root node
        root_node = hidden_states[:, 0]
        
        # Explore reasoning tree
        leaf_nodes, node_scores, reasoning_path = self._explore_tree(root_node)
        reasoning_steps.append(ReasoningStep(
            step_type='tree_exploration',
            intermediate_result=root_node,
            confidence=node_scores.max().item(),
            reasoning_path=reasoning_path
        ))
        
        # Aggregate leaf nodes based on scores
        normalized_scores = torch.softmax(node_scores, dim=-1)
        weighted_leaves = torch.sum(leaf_nodes * normalized_scores.unsqueeze(-1), dim=1)
        
        # Process aggregated result
        result = self.path_aggregator(weighted_leaves)
        
        # Estimate uncertainty
        adjusted_result, uncertainty = self.estimate_uncertainty(result)
        
        # Create final reasoning step
        final_step = ReasoningStep(
            step_type='tree_aggregation',
            intermediate_result=result,
            confidence=normalized_scores.max().item(),
            uncertainty=uncertainty
        )
        reasoning_steps.append(final_step)
        
        # Shape output back to batch_size, seq_len, hidden_size
        # by replacing the first token with the result
        output = hidden_states.clone()
        output[:, 0] = adjusted_result
        
        if return_dict:
            return ReasoningOutput(
                hidden_states=output,
                steps=reasoning_steps,
                confidence=normalized_scores.max().item(),
                strategy=ReasoningStrategy.TREE_OF_THOUGHT,
                metadata={"num_leaf_nodes": len(leaf_nodes)}
            )
        else:
            return output
    
    def _explore_tree(self, root_node: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Explore the reasoning tree starting from root node
        
        Args:
            root_node: Root node tensor [batch_size, hidden_size]
            
        Returns:
            Tuple of (leaf_nodes, node_scores, reasoning_path)
        """
        batch_size = root_node.size(0)
        
        # Initialize with root node
        current_nodes = [root_node]
        node_paths = [["Root"]]
        
        for depth in range(self.max_depth):
            next_nodes = []
            next_paths = []
            
            # Expand each current node
            for node_idx, node in enumerate(current_nodes):
                # Expand node to create children
                expanded = self.node_expander(node)
                
                # Reshape to get branching_factor children
                children = expanded.view(batch_size, self.branching_factor, self.hidden_size)
                
                # Add each child with its path
                for branch in range(self.branching_factor):
                    child = children[:, branch]
                    child_path = node_paths[node_idx] + [f"Branch_{depth}_{branch}"]
                    
                    next_nodes.append(child)
                    next_paths.append(child_path)
            
            # Update current nodes
            current_nodes = next_nodes
            node_paths = next_paths
        
        # Stack leaf nodes
        leaf_nodes = torch.stack(current_nodes, dim=1)  # [batch_size, num_leaves, hidden_size]
        
        # Evaluate leaf nodes
        num_leaves = leaf_nodes.size(1)
        leaf_evaluations = []
        
        for i in range(num_leaves):
            score = self.node_evaluator(leaf_nodes[:, i]).squeeze(-1)  # [batch_size]
            leaf_evaluations.append(score)
        
        # Stack scores
        node_scores = torch.stack(leaf_evaluations, dim=1)  # [batch_size, num_leaves]
        
        # Find best path for logging
        best_path_idx = torch.argmax(node_scores, dim=1)[0].item()
        best_path = node_paths[best_path_idx]
        
        return leaf_nodes, node_scores, best_path

class RecursiveReasoner(BaseReasoner):
    """Recursive reasoning component that applies self-similar reasoning patterns at multiple levels"""
    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        self.hidden_size = getattr(config, "hidden_size", 768)
        self.max_recursion_depth = getattr(config, "max_recursion_depth", 3)
        self.recursion_temperature = getattr(config, "recursion_temperature", 0.8)
        
        # Subproblem decomposition network
        self.decomposer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        # Recursive processing network
        self.recursive_processor = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=self.hidden_size * 4,
            dropout=0.1,
            activation=F.gelu,
            batch_first=True
        )
        
        # Solution aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = False,
                **kwargs) -> Union[torch.Tensor, ReasoningOutput]:
        """
        Apply recursive reasoning to the input hidden states
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dict: If True, return a ReasoningOutput object with intermediate steps
            
        Returns:
            Either the transformed hidden states or a ReasoningOutput object
        """
        batch_size, seq_len, _ = hidden_states.size()
        reasoning_steps = []
        
        # 1. Initial problem representation
        problem_repr = hidden_states[:, 0].clone()  # Use first token as problem representation
        
        reasoning_steps.append(ReasoningStep(
            step_type='recursive_init',
            intermediate_result=problem_repr,
            confidence=0.9,
            reasoning_path=["Initializing recursive reasoning"]
        ))
        
        # 2. Apply recursive reasoning
        subproblems, recursion_path = self._decompose_problem(problem_repr, attention_mask, depth=0)
        
        reasoning_steps.append(ReasoningStep(
            step_type='recursive_decomposition',
            intermediate_result=subproblems,
            confidence=0.85,
            reasoning_path=recursion_path
        ))
        
        # 3. Aggregate solutions
        solution_repr = self._aggregate_solutions(problem_repr, subproblems)
        
        # 4. Estimate uncertainty
        uncertainty = self.uncertainty_estimator(solution_repr).squeeze(-1)
        
        # Create final reasoning step
        reasoning_steps.append(ReasoningStep(
            step_type='recursive_solution',
            intermediate_result=solution_repr,
            confidence=1.0 - uncertainty.mean().item(),
            reasoning_path=["Completed recursive reasoning with aggregated solutions"]
        ))
        
        # Construct output by replacing the first token
        output_hidden = hidden_states.clone()
        output_hidden[:, 0] = solution_repr
        
        if return_dict:
            return ReasoningOutput(
                hidden_states=output_hidden,
                steps=reasoning_steps,
                confidence=1.0 - uncertainty.mean().item(),
                strategy=ReasoningStrategy.RECURSIVE,
                metadata={"max_recursion_depth": self.max_recursion_depth}
            )
        else:
            return output_hidden
    
    def _decompose_problem(self, 
                          problem_repr: torch.Tensor, 
                          attention_mask: Optional[torch.Tensor],
                          depth: int) -> Tuple[torch.Tensor, List[str]]:
        """
        Recursively decompose a problem into subproblems
        
        Args:
            problem_repr: Problem representation [batch_size, hidden_size]
            attention_mask: Attention mask
            depth: Current recursion depth
            
        Returns:
            Tuple of (subproblem_reprs, recursion_path)
        """
        # Generate reasoning path
        recursion_path = [f"Recursion depth {depth}: Decomposing problem"]
        
        # Base case: max recursion depth reached
        if depth >= self.max_recursion_depth:
            recursion_path.append(f"Reached max recursion depth {depth}, solving directly")
            # Direct solution at max depth
            return self.decomposer(problem_repr), recursion_path
        
        # Decompose problem into subproblems
        subproblem_repr = self.decomposer(problem_repr)
        
        # Create pseudo-sequence for transformer processing
        batch_size = problem_repr.size(0)
        device = problem_repr.device
        
        # Create a small sequence of 3 tokens: original problem, subproblem, and a context token
        sequence = torch.cat([
            problem_repr.unsqueeze(1),
            subproblem_repr.unsqueeze(1),
            torch.zeros(batch_size, 1, self.hidden_size, device=device)
        ], dim=1)
        
        # Process with transformer
        if attention_mask is None:
            seq_mask = torch.ones(batch_size, 3, device=device)
        else:
            seq_mask = torch.ones(batch_size, 3, device=device)
            seq_mask = seq_mask * attention_mask[:, 0].unsqueeze(1)
            
        processed = self.recursive_processor(sequence, src_key_padding_mask=(1 - seq_mask).bool())
        
        # Extract subproblem representation
        refined_subproblem = processed[:, 1]
        
        # Decide whether to recurse further based on temperature
        if random.random() < self.recursion_temperature:
            recursion_path.append(f"Further decomposing subproblem at depth {depth}")
            # Recursive case: decompose subproblem further
            sub_subproblems, sub_path = self._decompose_problem(
                refined_subproblem, attention_mask, depth + 1
            )
            recursion_path.extend(sub_path)
            return sub_subproblems, recursion_path
        else:
            recursion_path.append(f"Solving subproblem directly at depth {depth}")
            return refined_subproblem, recursion_path
    
    def _aggregate_solutions(self, 
                           original_problem: torch.Tensor,
                           subproblem_solutions: torch.Tensor) -> torch.Tensor:
        """
        Aggregate solutions from subproblems
        
        Args:
            original_problem: Original problem representation [batch_size, hidden_size]
            subproblem_solutions: Solutions to subproblems [batch_size, hidden_size]
            
        Returns:
            Aggregated solution representation [batch_size, hidden_size]
        """
        # Concatenate original problem and subproblem solutions
        combined = torch.cat([original_problem, subproblem_solutions], dim=1)
        
        # Aggregate solutions
        aggregated = self.aggregator(combined)
        
        return aggregated

class NeuralSymbolicReasoner(BaseReasoner):
    """Neural-symbolic reasoning component that combines neural network processing with symbolic reasoning"""
    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        self.hidden_size = getattr(config, "hidden_size", 768)
        self.num_symbols = getattr(config, "num_symbols", 100)
        self.num_rules = getattr(config, "num_rules", 50)
        self.num_heads = getattr(config, "num_heads", 8)
        self.dropout = getattr(config, "dropout", 0.1)
        
        # Symbol embeddings
        self.symbol_embeddings = nn.Parameter(
            torch.randn(self.num_symbols, self.hidden_size)
        )
        
        # Rule embeddings
        self.rule_embeddings = nn.Parameter(
            torch.randn(self.num_rules, self.hidden_size)
        )
        
        # Neural encoder
        self.neural_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Symbol extractor
        self.symbol_extractor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.num_symbols)
        )
        
        # Rule selector
        self.rule_selector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.num_rules)
        )
        
        # Symbolic reasoning module
        self.symbolic_reasoner = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.dropout,
                activation=F.gelu,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Neural-symbolic integration
        self.integration_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the neural-symbolic reasoner"""
        nn.init.normal_(self.symbol_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.rule_embeddings, mean=0.0, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = False,
                **kwargs) -> Union[torch.Tensor, ReasoningOutput]:
        """
        Apply neural-symbolic reasoning to input hidden states
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            return_dict: Whether to return a dictionary with reasoning steps
            
        Returns:
            Either transformed hidden states or a ReasoningOutput object
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        reasoning_steps = []
        
        # 1. Neural encoding
        neural_encoded = self.neural_encoder(hidden_states)
        
        reasoning_steps.append(ReasoningStep(
            step_type="neural_encoding",
            intermediate_result=neural_encoded.mean(dim=1),
            confidence=0.95,
            reasoning_path=["Applied neural encoding to input representation"]
        ))
        
        # 2. Extract symbols
        symbol_logits = self.symbol_extractor(neural_encoded)
        symbol_probs = F.softmax(symbol_logits, dim=-1)
        
        # Get symbol representations
        symbol_repr = torch.matmul(symbol_probs, self.symbol_embeddings)
        
        reasoning_steps.append(ReasoningStep(
            step_type="symbol_extraction",
            intermediate_result=symbol_repr.mean(dim=1),
            confidence=0.9,
            reasoning_path=["Extracted symbolic representations"]
        ))
        
        # 3. Select rules
        rule_logits = self.rule_selector(neural_encoded.mean(dim=1))
        rule_probs = F.softmax(rule_logits, dim=-1)
        
        # Get rule representations
        rule_repr = torch.matmul(rule_probs, self.rule_embeddings).unsqueeze(1).expand(-1, seq_len, -1)
        
        reasoning_steps.append(ReasoningStep(
            step_type="rule_selection",
            intermediate_result=rule_repr[:, 0],
            confidence=0.85,
            reasoning_path=["Selected reasoning rules"]
        ))
        
        # 4. Combine symbols and rules
        symbolic_input = symbol_repr + rule_repr
        
        # 5. Apply symbolic reasoning
        if attention_mask is not None:
            symbolic_reasoned = self.symbolic_reasoner(
                symbolic_input,
                src_key_padding_mask=(~attention_mask.bool())
            )
        else:
            symbolic_reasoned = self.symbolic_reasoner(symbolic_input)
        
        reasoning_steps.append(ReasoningStep(
            step_type="symbolic_reasoning",
            intermediate_result=symbolic_reasoned.mean(dim=1),
            confidence=0.8,
            reasoning_path=["Applied symbolic reasoning with transformer"]
        ))
        
        # 6. Integrate neural and symbolic representations
        integrated = self.integration_layer(
            torch.cat([neural_encoded, symbolic_reasoned], dim=-1)
        )
        
        reasoning_steps.append(ReasoningStep(
            step_type="neural_symbolic_integration",
            intermediate_result=integrated.mean(dim=1),
            confidence=0.85,
            reasoning_path=["Integrated neural and symbolic representations"]
        ))
        
        # 7. Apply output projection
        output = self.output_projection(integrated)
        
        # 8. Estimate uncertainty
        uncertainty = self.uncertainty_estimator(output[:, 0]).squeeze(-1)
        
        reasoning_steps.append(ReasoningStep(
            step_type="output_generation",
            intermediate_result=output.mean(dim=1),
            confidence=1.0 - uncertainty.mean().item(),
            reasoning_path=["Generated final output with neural-symbolic reasoning"]
        ))
        
        if return_dict:
            return ReasoningOutput(
                hidden_states=output,
                steps=reasoning_steps,
                confidence=1.0 - uncertainty.mean().item(),
                strategy=ReasoningStrategy.NEURAL_SYMBOLIC,
                metadata={
                    "num_symbols_used": int(torch.sum(symbol_probs > 0.1).item()),
                    "num_rules_used": int(torch.sum(rule_probs > 0.1).item())
                }
            )
        else:
            return output

class KnowledgeReasoner(BaseReasoner):
    """Reasoner that utilizes external knowledge and facts to enhance reasoning"""
    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        self.hidden_size = getattr(config, "hidden_size", 768)
        self.knowledge_size = getattr(config, "knowledge_size", 1024)
        self.num_knowledge_heads = getattr(config, "num_knowledge_heads", 4)
        self.max_knowledge_items = getattr(config, "max_knowledge_items", 100)
        
        # Knowledge lookup
        self.knowledge_query_projector = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Knowledge retrieval attention
        self.knowledge_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_knowledge_heads,
            batch_first=True
        )
        
        # Knowledge integration
        self.knowledge_integrator = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        # Knowledge relevance estimator
        self.relevance_estimator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Fact verifier
        self.fact_verifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Initialize knowledge base (in practice, this would be loaded externally)
        self.register_buffer(
            "knowledge_base",
            torch.randn(self.max_knowledge_items, self.hidden_size)
        )
        
        # Knowledge embeddings to store factual knowledge
        self.register_buffer(
            "knowledge_embeddings", 
            torch.randn(self.max_knowledge_items, self.hidden_size)
        )
        
        # Knowledge metadata (for interpretability)
        self.knowledge_metadata = [f"Knowledge item {i}" for i in range(self.max_knowledge_items)]
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = False,
                **kwargs) -> Union[torch.Tensor, ReasoningOutput]:
        batch_size, seq_len, _ = hidden_states.size()
        reasoning_steps = []
        
        # Extract query representation (what we're reasoning about)
        query_repr = hidden_states[:, 0]
        
        # Project query for knowledge lookup
        knowledge_query = self.knowledge_query_projector(query_repr)
        
        # Retrieve relevant knowledge using attention
        retrieved_knowledge, attention_weights = self._retrieve_knowledge(knowledge_query)
        
        reasoning_steps.append(ReasoningStep(
            step_type='knowledge_retrieval',
            intermediate_result=retrieved_knowledge,
            confidence=attention_weights.max(dim=-1)[0].mean().item(),
            reasoning_path=[f"Retrieved {self.num_knowledge_heads} knowledge items"]
        ))
        
        # Estimate relevance of retrieved knowledge
        relevance_scores = self.relevance_estimator(retrieved_knowledge)
        
        # Filter knowledge based on relevance
        relevant_mask = (relevance_scores > 0.5).float()
        filtered_knowledge = retrieved_knowledge * relevant_mask
        
        # Get top-k relevant knowledge items
        top_k = 3
        _, top_indices = torch.topk(relevance_scores.squeeze(-1), min(top_k, self.max_knowledge_items))
        
        # Create reasoning path with metadata about retrieved knowledge
        knowledge_path = ["Knowledge-based reasoning:"]
        for idx in top_indices[0][:top_k].cpu().numpy():
            knowledge_path.append(f"Used {self.knowledge_metadata[idx]}")
        
        reasoning_steps.append(ReasoningStep(
            step_type='knowledge_filtering',
            intermediate_result=filtered_knowledge,
            confidence=relevant_mask.mean().item(),
            reasoning_path=knowledge_path
        ))
        
        # Verify facts based on knowledge
        fact_verification = self._verify_facts(query_repr, filtered_knowledge)
        
        reasoning_steps.append(ReasoningStep(
            step_type='fact_verification',
            intermediate_result=fact_verification,
            confidence=fact_verification.mean().item(),
            reasoning_path=["Verified facts against knowledge base"]
        ))
        
        # Integrate knowledge with query
        knowledge_enhanced = torch.cat([query_repr, filtered_knowledge], dim=-1)
        integrated_repr = self.knowledge_integrator(knowledge_enhanced)
        
        reasoning_steps.append(ReasoningStep(
            step_type='knowledge_integration',
            intermediate_result=integrated_repr,
            confidence=0.9,  # Fixed confidence for integration
            reasoning_path=["Integrated verified knowledge with query"]
        ))
        
        # Estimate uncertainty
        adjusted_solution, uncertainty = self.estimate_uncertainty(integrated_repr)
        
        # Create final output
        output = hidden_states.clone()
        output[:, 0] = adjusted_solution
        
        if return_dict:
            return ReasoningOutput(
                hidden_states=output,
                steps=reasoning_steps,
                confidence=1.0 - uncertainty,
                strategy=ReasoningStrategy.KNOWLEDGE_AUGMENTED,
                metadata={"knowledge_items_used": int(relevant_mask.sum().item())}
            )
        else:
            return output
    
    def _retrieve_knowledge(self, 
                           query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant knowledge using attention mechanism
        
        Args:
            query: Query representation [batch_size, hidden_size]
            
        Returns:
            Tuple of (retrieved_knowledge, attention_weights)
        """
        batch_size = query.size(0)
        
        # Expand knowledge base for batch
        expanded_kb = self.knowledge_base.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Perform attention to retrieve knowledge
        query_expanded = query.unsqueeze(1)  # [batch_size, 1, hidden_size]
        retrieved, attention_weights = self.knowledge_attention(
            query_expanded, 
            expanded_kb, 
            expanded_kb
        )
        
        return retrieved.squeeze(1), attention_weights
    
    def _verify_facts(self, 
                     query: torch.Tensor, 
                     knowledge: torch.Tensor) -> torch.Tensor:
        """
        Verify facts based on knowledge
        
        Args:
            query: Query representation [batch_size, hidden_size]
            knowledge: Retrieved knowledge [batch_size, hidden_size]
            
        Returns:
            Fact verification scores [batch_size, 1]
        """
        # Concatenate query and knowledge
        combined = torch.cat([query, knowledge], dim=-1)
        
        # Compute verification score
        verification_score = self.fact_verifier(combined)
        
        return verification_score

class MCTSReasoner(BaseReasoner):
    """Monte Carlo Tree Search based reasoning component"""
    def __init__(self, config: ReasoningConfig):
        super().__init__(config)
        self.hidden_size = getattr(config, "hidden_size", 768)
        self.num_simulations = getattr(config, "mcts_simulations", 100)
        self.c_puct = getattr(config, "c_puct", 1.0)
        self.exploration_factor = getattr(config, "exploration_factor", 0.5)
        
        # State representation network
        self.state_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        # Action prediction network
        self.action_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Value prediction network
        self.value_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Tanh()  # Value in [-1, 1]
        )
        
        # Policy network (for action probabilities)
        self.policy_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 10)  # 10 possible actions by default
        )
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = False,
                **kwargs) -> Union[torch.Tensor, ReasoningOutput]:
        batch_size, seq_len, _ = hidden_states.size()
        reasoning_steps = []
        
        # Extract state representation (use first token)
        state_repr = hidden_states[:, 0]
        encoded_state = self.state_encoder(state_repr)
        
        # Store initial state
        reasoning_steps.append(ReasoningStep(
            step_type='mcts_init',
            intermediate_result=encoded_state,
            confidence=0.9,
            reasoning_path=["Initializing MCTS reasoning"]
        ))
        
        # Run MCTS simulations
        final_state, visit_counts, reasoning_path = self._run_mcts(encoded_state)
        
        reasoning_steps.append(ReasoningStep(
            step_type='mcts_search',
            intermediate_result=final_state,
            confidence=0.8,
            reasoning_path=reasoning_path
        ))
        
        # Evaluate final state
        value = self.value_predictor(final_state)
        
        # Create final reasoning step
        final_step = ReasoningStep(
            step_type='mcts_result',
            intermediate_result=final_state,
            confidence=value.sigmoid().item(),  # Convert from [-1,1] to [0,1]
            reasoning_path=["Completed MCTS reasoning"]
        )
        reasoning_steps.append(final_step)
        
        # Construct output hidden states by replacing the first token
        output = hidden_states.clone()
        output[:, 0] = final_state
        
        if return_dict:
            return ReasoningOutput(
                hidden_states=output,
                steps=reasoning_steps,
                confidence=value.sigmoid().item(),
                strategy=ReasoningStrategy.MCTS,
                metadata={"num_simulations": self.num_simulations}
            )
        else:
            return output
    
    def _run_mcts(self, 
                  state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Run Monte Carlo Tree Search simulations
        
        Args:
            state: Initial state representation [batch_size, hidden_size]
            
        Returns:
            Tuple of (final_state, visit_counts, reasoning_path)
        """
        batch_size = state.size(0)
        device = state.device
        
        # For simplicity, we'll create a mock MCTS implementation
        # In a real implementation, this would build a search tree
        reasoning_path = ["Starting MCTS with initial state"]
        
        # Simulate the improvement of the state through MCTS iterations
        current_state = state
        visit_counts = torch.zeros(batch_size, 10, device=device)  # 10 actions
        
        for i in range(self.num_simulations):
            # Get action probabilities from policy network
            logits = self.policy_network(current_state)
            probs = F.softmax(logits, dim=-1)
            
            # Select action (in practice, would use UCB formula)
            if random.random() < self.exploration_factor:
                # Explore: sample from distribution
                action_idx = torch.multinomial(probs, 1).squeeze(-1)
            else:
                # Exploit: pick highest probability
                action_idx = torch.argmax(probs, dim=-1)
            
            # Convert to action embedding
            action_embed = self.action_predictor(current_state)
            
            # Update state based on action (simplified transition)
            next_state = current_state + action_embed * 0.1
            
            # Evaluate new state
            value = self.value_predictor(next_state)
            
            # Update visit counts for the chosen action
            for b in range(batch_size):
                visit_counts[b, action_idx[b]] += 1
            
            # Update current state if value improves
            if i % 10 == 0:  # Log every 10 iterations
                reasoning_path.append(f"MCTS iteration {i}, value: {value.mean().item():.3f}")
            
            current_state = next_state
        
        reasoning_path.append(f"Completed {self.num_simulations} MCTS simulations")
        
        return current_state, visit_counts, reasoning_path