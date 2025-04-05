"""
Integration Layer for Advanced Model Components

This module provides a central hub for integrating all advanced components:
- Linear Attention
- Just Read Twice (JRT)
- Monte Carlo Tree Search (MCTS)
- Function Calling
- Quantization

It ensures components are properly connected, configurations are compatible,
and all systems work together during training and inference.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Type
import copy
import inspect
import warnings
import torch.nn.functional as F

from .attention import create_attention_layer, LinearAttention
from .transformer import EfficientTransformer, TransformerConfig
from .tree_reasoning_mcts import MCTSEnhancedTreeReasoningModule, MCTSConfig
from .quantization import QuantizationConfig, quantize_model
from ..utils.jrt_processor import apply_jrt, JRTProcessor
from ..utils.function_calling import FunctionCallingManager
from .gnn.integration import TransformerGNNIntegration
from .valkyrie_llm import ValkyrieLLM
from model.adaptive_reasoning import MetaReasoningOptimizer
from model.valkyrie_llm import SelfReflectivePromptAugmenter
from model.memory.memory_bank import StrategySequenceMemory
from model.computational_efficiency import ComputeTrackerModule

logger = logging.getLogger(__name__)

class ComponentRegistry:
    """Registry for managing and accessing model components"""
    
    def __init__(self):
        self.components = {}
        self.configs = {}
        self.dependencies = {}
        self.conflicts = {}
        
    def register(
        self, 
        name: str, 
        component: Any, 
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        conflicts: Optional[List[str]] = None
    ) -> None:
        """Register a component with its configuration and relationships"""
        self.components[name] = component
        self.configs[name] = config or {}
        self.dependencies[name] = dependencies or []
        self.conflicts[name] = conflicts or []
        
    def get(self, name: str) -> Optional[Any]:
        """Get a registered component by name"""
        return self.components.get(name)
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """Get a component's configuration"""
        return self.configs.get(name, {})
    
    def is_registered(self, name: str) -> bool:
        """Check if a component is registered"""
        return name in self.components
    
    def check_compatibility(self, component_names: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if a set of components are compatible with each other.
        
        Returns:
            is_compatible: Whether the components are compatible
            issues: List of compatibility issues if any
        """
        issues = []
        
        # Check dependencies
        for name in component_names:
            if not self.is_registered(name):
                issues.append(f"Component '{name}' is not registered")
                continue
                
            # Check that all dependencies are also enabled
            for dependency in self.dependencies[name]:
                if dependency not in component_names:
                    issues.append(f"Component '{name}' requires '{dependency}' but it's not enabled")
        
        # Check conflicts
        for i, name1 in enumerate(component_names):
            if not self.is_registered(name1):
                continue
                
            for name2 in component_names[i+1:]:
                if not self.is_registered(name2):
                    continue
                    
                if name2 in self.conflicts[name1] or name1 in self.conflicts[name2]:
                    issues.append(f"Components '{name1}' and '{name2}' conflict with each other")
        
        return len(issues) == 0, issues
    
    def merge_configs(self, component_names: List[str]) -> Dict[str, Any]:
        """Merge configurations from multiple components"""
        merged_config = {}
        
        for name in component_names:
            if not self.is_registered(name):
                continue
                
            component_config = self.configs[name]
            for key, value in component_config.items():
                if key in merged_config and merged_config[key] != value:
                    logger.warning(f"Config conflict for key '{key}': '{merged_config[key]}' vs '{value}'. Using '{value}'")
                merged_config[key] = value
        
        return merged_config

class IntegrationManager:
    """
    Central manager for integrating all advanced components.
    
    This class:
    1. Coordinates component initialization and interaction
    2. Ensures configuration compatibility
    3. Provides hooks for training and inference
    4. Manages component lifecycle
    """
    
    def __init__(self):
        self.registry = ComponentRegistry()
        self.active_components = []
        self.global_config = {}
        self.model = None
        self.tokenizer = None
        
        # Initialize component registry with core components
        self._register_core_components()
    
    def _register_core_components(self):
        """Register core components with their dependencies and conflicts"""
        # Linear Attention
        self.registry.register(
            "linear_attention",
            LinearAttention,
            config={"use_linear_attention": True},
            dependencies=[],
            conflicts=[]
        )
        
        # JRT Processing
        self.registry.register(
            "jrt",
            JRTProcessor,
            config={"enable_jrt": True},
            dependencies=[],
            conflicts=[]
        )
        
        # MCTS Reasoning
        self.registry.register(
            "mcts",
            MCTSEnhancedTreeReasoningModule,
            config={"enable_mcts": True},
            dependencies=[],
            conflicts=[]
        )
        
        # Function Calling
        self.registry.register(
            "function_calling",
            FunctionCallingManager,
            config={"enable_function_calling": True},
            dependencies=[],
            conflicts=[]
        )
        
        # Quantization
        self.registry.register(
            "quantization",
            QuantizationConfig,
            config={"enable_quantization": True},
            dependencies=[],
            conflicts=[]
        )
        
        # MCTS + Function Calling Integration
        self.registry.register(
            "mcts_function_integration",
            None,  # Special integration component
            config={"enable_mcts_function_integration": True},
            dependencies=["mcts", "function_calling"],
            conflicts=[]
        )
        
        # JRT + Linear Attention Integration
        self.registry.register(
            "jrt_linear_integration",
            None,  # Special integration component
            config={"enable_jrt_linear_integration": True},
            dependencies=["jrt", "linear_attention"],
            conflicts=[]
        )
        
        # Register new components
        
        # Task-targeted fine-tuning
        self.registry.register(
            "targeted_finetuning",
            TaskTargetedFineTuner,
            config={"enable_targeted_finetuning": True},
            dependencies=[],
            conflicts=[]
        )
        
        # Meta-reasoning optimizer for strategy selection
        self.registry.register(
            "meta_reasoning",
            MetaReasoningOptimizer,
            config={"enable_meta_reasoning": True},
            dependencies=[],
            conflicts=[]
        )
        
        # Self-reflective prompt augmentation
        self.registry.register(
            "prompt_augmentation",
            SelfReflectivePromptAugmenter,
            config={"enable_prompt_augmentation": True},
            dependencies=[],
            conflicts=[]
        )
        
        # Strategy sequence memory
        self.registry.register(
            "strategy_memory",
            StrategySequenceMemory,
            config={"enable_strategy_memory": True},
            dependencies=[],
            conflicts=[]
        )
        
        # Compute tracker
        self.registry.register(
            "compute_tracker",
            ComputeTrackerModule,
            config={"enable_compute_tracker": True},
            dependencies=[],
            conflicts=[]
        )
        
        # Advanced reasoning integration
        self.registry.register(
            "advanced_reasoning_integration",
            None,  # Special integration component
            config={"enable_advanced_reasoning_integration": True},
            dependencies=["meta_reasoning", "prompt_augmentation", "strategy_memory"],
            conflicts=[]
        )
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the integration manager with a set of components to enable.
        
        Args:
            config: Configuration dictionary with component settings
        """
        self.global_config = config
        
        # Determine which components to enable
        components_to_enable = []
        for component_name in self.registry.components:
            config_key = f"enable_{component_name}"
            if config.get(config_key, False):
                components_to_enable.append(component_name)
        
        # Check compatibility
        is_compatible, issues = self.registry.check_compatibility(components_to_enable)
        if not is_compatible:
            error_msg = "Incompatible component configuration:\n" + "\n".join(issues)
            raise ValueError(error_msg)
        
        # Store active components
        self.active_components = components_to_enable
        
        # Merge configurations
        component_configs = self.registry.merge_configs(components_to_enable)
        self.global_config.update(component_configs)
        
        logger.info(f"Configured integration manager with components: {', '.join(components_to_enable)}")
    
    def initialize_model(self, model: nn.Module, tokenizer=None) -> nn.Module:
        """
        Initialize a model with all active components.
        
        Args:
            model: Base model to enhance
            tokenizer: Optional tokenizer
            
        Returns:
            Enhanced model with all active components
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Apply each active component to the model
        enhanced_model = model
        
        # Apply quantization if enabled
        if "quantization" in self.active_components:
            bits = self.global_config.get("quantization_bits", 8)
            quant_config = QuantizationConfig(
                bits=bits,
                group_size=self.global_config.get("quantization_group_size", 128),
                sym=self.global_config.get("quantization_symmetric", True),
                per_channel=self.global_config.get("quantization_per_channel", True),
                quant_method=self.global_config.get("quantization_method", "absmax")
            )
            enhanced_model = quantize_model(enhanced_model, quant_config)
            logger.info(f"Applied {bits}-bit quantization to model")
        
        # Enhance tree reasoning with MCTS if enabled
        if "mcts" in self.active_components and hasattr(enhanced_model, 'tree_reasoning'):
            mcts_config = MCTSConfig(
                max_iterations=self.global_config.get("mcts_max_iterations", 100),
                exploration_weight=self.global_config.get("mcts_exploration_weight", 1.0),
                max_depth=self.global_config.get("mcts_max_depth", 5),
                rollout_depth=self.global_config.get("mcts_rollout_depth", 3),
                early_stopping_threshold=self.global_config.get("mcts_early_stopping_threshold", 0.95)
            )
            enhanced_model.tree_reasoning = MCTSEnhancedTreeReasoningModule(
                config=mcts_config,
                base_tree_reasoning=enhanced_model.tree_reasoning
            )
            logger.info("Enhanced tree reasoning with MCTS")
        
        # Integrate MCTS with function calling if both are enabled
        if "mcts_function_integration" in self.active_components:
            self._integrate_mcts_with_functions(enhanced_model)
            logger.info("Integrated MCTS reasoning with function calling")
        
        # Setup function calling if enabled
        if "function_calling" in self.active_components:
            if not hasattr(enhanced_model, '_function_manager'):
                enhanced_model._function_manager = FunctionCallingManager()
            logger.info("Set up function calling framework")
        
        # Store reference to JRT processor if enabled
        if "jrt" in self.active_components:
            jrt_config = {
                "repetitions": self.global_config.get("jrt_repetitions", 2),
                "instruction_aware": self.global_config.get("jrt_instruction_aware", True),
                "reverse_order": self.global_config.get("jrt_reverse_order", True),
                "preserve_task": self.global_config.get("jrt_preserve_task", True)
            }
            enhanced_model._jrt_processor = JRTProcessor(**jrt_config)
            logger.info("Set up Just Read Twice processing")
        
        # Set linear attention config if enabled
        if "linear_attention" in self.active_components:
            if hasattr(enhanced_model, 'config'):
                enhanced_model.config.use_linear_attention = True
                enhanced_model.config.linear_attention_feature_dim = self.global_config.get("linear_attention_feature_dim", 16)
                enhanced_model.config.linear_attention_kernel_size = self.global_config.get("linear_attention_kernel_size", 4)
            logger.info("Configured model to use linear attention")
        
        # Integrate JRT with linear attention if both are enabled
        if "jrt_linear_integration" in self.active_components:
            self._integrate_jrt_with_linear_attention(enhanced_model)
            logger.info("Optimized JRT for linear attention")
            
        # Initialize new components if active
        if "targeted_finetuning" in self.active_components:
            self.targeted_finetuner = TaskTargetedFineTuner(self)
            
        if "meta_reasoning" in self.active_components:
            config = model.config if hasattr(model, "config") else None
            hidden_size = getattr(config, "hidden_size", 768)
            adaptive_config = AdaptiveReasoningConfig(
                enabled=True,
                default_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                strategy_selection_threshold=0.7,
                max_reasoning_steps=10,
                strategy_embeddings_size=hidden_size
            )
            self.meta_reasoning = MetaReasoningOptimizer(adaptive_config)
            model.add_module("meta_reasoning", self.meta_reasoning)
            
        if "prompt_augmentation" in self.active_components:
            self.prompt_augmenter = SelfReflectivePromptAugmenter(
                model=model,
                tokenizer=tokenizer,
                confidence_threshold=0.7,
                max_retries=3
            )
            
        if "strategy_memory" in self.active_components:
            config = model.config if hasattr(model, "config") else None
            hidden_size = getattr(config, "hidden_size", 768)
            self.strategy_memory = StrategySequenceMemory(
                hidden_size=hidden_size,
                memory_size=1000,
                strategy_embedding_size=128,
                num_strategy_types=10
            )
            model.add_module("strategy_memory", self.strategy_memory)
            
        if "compute_tracker" in self.active_components:
            self.compute_tracker = ComputeTrackerModule(
                max_budget=1.0,
                track_tokens=True,
                track_time=True,
                track_memory=True,
                adaptive_budget=True
            )
            
        # If advanced reasoning integration is enabled, connect components
        if "advanced_reasoning_integration" in self.active_components:
            self._integrate_advanced_reasoning(model)
            
        # Store component metadata on the model
        enhanced_model._active_components = self.active_components
        enhanced_model._integration_config = self.global_config
        
        return enhanced_model
    
    def _integrate_mcts_with_functions(self, model: nn.Module) -> None:
        """Integrate MCTS reasoning with function calling"""
        if not hasattr(model, 'tree_reasoning') or not isinstance(model.tree_reasoning, MCTSEnhancedTreeReasoningModule):
            logger.warning("MCTS not properly set up, skipping integration with function calling")
            return
            
        if not hasattr(model, '_function_manager'):
            model._function_manager = FunctionCallingManager()
        
        # Extend MCTS state evaluation to consider function calls
        original_evaluate = model.tree_reasoning._evaluate_state
        
        def enhanced_evaluate_state(state):
            # Get base evaluation
            action_probs, value = original_evaluate(state)
            
            # Check if any actions might benefit from function calls
            for action_id, (prob, text) in list(action_probs.items()):
                # Add function calling capability if the action suggests using external tools
                function_keywords = ['search', 'lookup', 'calculate', 'api', 'retrieve', 'query']
                if any(keyword in text.lower() for keyword in function_keywords):
                    # Increase probability for actions that could use functions
                    action_probs[action_id] = (prob * 1.2, text + " (with function calling)")
            
            return action_probs, value
        
        # Replace the evaluation method
        model.tree_reasoning._evaluate_state = enhanced_evaluate_state
        
        # Extend state simulator to incorporate function results
        original_simulator = model.tree_reasoning._simulate_from_state
        
        def enhanced_simulator(state, depth):
            # Basic simulation value
            value = original_simulator(state, depth)
            
            # Enhance value if state can use functions effectively
            state_text = "Simulated state"  # In a real implementation, we'd convert state to text
            
            # Check if state suggests function use would be valuable
            function_keywords = ['search', 'lookup', 'calculate', 'api', 'retrieve', 'query']
            if any(keyword in state_text.lower() for keyword in function_keywords):
                # Boost value to encourage function use in appropriate contexts
                value *= 1.1
            
            return value
        
        # Replace the simulator
        model.tree_reasoning._simulate_from_state = enhanced_simulator
    
    def _integrate_jrt_with_linear_attention(self, model: nn.Module) -> None:
        """Optimize JRT processing specifically for linear attention"""
        if not hasattr(model, '_jrt_processor'):
            logger.warning("JRT not properly set up, skipping integration with linear attention")
            return
        
        # Customize JRT parameters for optimal use with linear attention
        # Linear attention benefits from more fine-grained chunks and different repetition patterns
        jrt_processor = model._jrt_processor
        
        # Set optimal chunk size for linear attention (smaller chunks work better)
        jrt_processor.chunk_size = self.global_config.get("jrt_linear_chunk_size", 64)
        
        # Use interleaving for linear attention (works better than sequential repetition)
        jrt_processor.interleave = self.global_config.get("jrt_linear_interleave", True)
        
        # Increase repetitions for linear attention (linear attention benefits from more repetition)
        jrt_processor.repetitions = self.global_config.get("jrt_linear_repetitions", 3)
    
    def _integrate_advanced_reasoning(self, model):
        """Connect advanced reasoning components for seamless interaction"""
        # Skip if not all required components are active
        required_components = ["meta_reasoning", "prompt_augmentation", "strategy_memory"]
        if not all(comp in self.active_components for comp in required_components):
            logger.warning("Advanced reasoning integration requires meta_reasoning, prompt_augmentation, and strategy_memory")
            return
            
        # Store original generate method
        original_generate = model.generate
        
        # Create a compute tracker if not already available
        if "compute_tracker" not in self.active_components:
            self.compute_tracker = ComputeTrackerModule(
                max_budget=1.0,
                track_tokens=True,
                track_time=True,
                adaptive_budget=True
            )
        
        # Create a wrapper for the generate method that integrates all components
        def enhanced_generate(input_ids=None, attention_mask=None, **kwargs):
            # Extract prompt if available
            prompt = kwargs.pop("prompt", None)
            if prompt is None and input_ids is not None and hasattr(self, "tokenizer") and self.tokenizer is not None:
                prompt = self.tokenizer.decode(input_ids[0])
            
            # Extract task type if available
            task_type = kwargs.pop("task_type", None)
            
            # Start compute tracking
            if hasattr(self, "compute_tracker"):
                self.compute_tracker.start_tracking(strategy=None, task_type=task_type)
            
            # Check if we have a stored strategy sequence for this task
            strategy_sequence = None
            if hasattr(self, "strategy_memory") and input_ids is not None:
                # Get hidden states for the input
                with torch.no_grad():
                    hidden_states = model.get_input_embeddings()(input_ids)
                    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                        # Pass through first few layers to get better representation
                        for i in range(min(3, len(model.transformer.h))):
                            hidden_states = model.transformer.h[i](hidden_states)[0]
                
                # Retrieve strategy sequence
                strategy_sequence, similarity = self.strategy_memory.retrieve_sequence(hidden_states)
                
                # If found with high similarity, use the stored sequence
                if strategy_sequence is not None and similarity > 0.8:
                    logger.info(f"Using stored strategy sequence with similarity {similarity:.3f}")
                    
                    # Start tracking the first strategy
                    if hasattr(self, "compute_tracker") and len(strategy_sequence) > 0:
                        self.compute_tracker.start_tracking(
                            strategy=strategy_sequence[0], 
                            task_type=task_type
                        )
            
            # If no stored sequence, ask meta-reasoning to select a strategy
            selected_strategy = None
            if strategy_sequence is None and hasattr(self, "meta_reasoning") and input_ids is not None:
                # Get hidden states if not already computed
                if "hidden_states" not in locals():
                    with torch.no_grad():
                        hidden_states = model.get_input_embeddings()(input_ids)
                        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                            for i in range(min(3, len(model.transformer.h))):
                                hidden_states = model.transformer.h[i](hidden_states)[0]
                
                # Select strategy
                selected_strategy, confidence = self.meta_reasoning.select_strategy(hidden_states)
                logger.info(f"Selected reasoning strategy: {selected_strategy.name} with confidence {confidence:.3f}")
                
                # Start tracking the selected strategy
                if hasattr(self, "compute_tracker"):
                    self.compute_tracker.start_tracking(
                        strategy=selected_strategy.name, 
                        task_type=task_type
                    )
            
            # Apply prompt augmentation if available
            if prompt is not None and hasattr(self, "prompt_augmenter"):
                # Choose augmentation based on selected strategy
                if selected_strategy is not None:
                    # Map strategy to template
                    strategy_template_map = {
                        ReasoningStrategy.STEP_BY_STEP: "step_by_step",
                        ReasoningStrategy.CHAIN_OF_THOUGHT: "chain_of_thought",
                        ReasoningStrategy.SYMBOLIC: "structured_reasoning",
                        ReasoningStrategy.VERIFICATION: "verification",
                        ReasoningStrategy.COMPARATIVE: "alternative_approaches",
                        ReasoningStrategy.DEFAULT: None
                    }
                    
                    template = strategy_template_map.get(selected_strategy)
                    
                    if template is not None:
                        # Add template to kwargs to guide prompt augmenter
                        kwargs["template"] = template
                
                # Perform multi-strategy reasoning with retries if confidence is low
                outputs, confidence, augmented_prompt = self.prompt_augmenter.multi_strategy_reasoning(prompt)
                
                # If prompt was augmented, update inputs
                if augmented_prompt != prompt:
                    logger.info("Prompt was augmented for better reasoning")
                    if input_ids is not None and hasattr(self, "tokenizer") and self.tokenizer is not None:
                        input_ids = self.tokenizer(augmented_prompt, return_tensors="pt").input_ids.to(input_ids.device)
            
            # Update compute tracking
            if hasattr(self, "compute_tracker"):
                self.compute_tracker.update_tracking()
            
            # Call original generate
            outputs = original_generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            
            # Store successful reasoning path
            if hasattr(self, "strategy_memory") and input_ids is not None and selected_strategy is not None:
                # TODO: Determine success based on outputs
                success_rate = 0.8  # Placeholder
                with torch.no_grad():
                    if "hidden_states" not in locals():
                        hidden_states = model.get_input_embeddings()(input_ids)
                        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                            for i in range(min(3, len(model.transformer.h))):
                                hidden_states = model.transformer.h[i](hidden_states)[0]
                
                # Store the sequence
                self.strategy_memory.store_sequence(
                    hidden_states,
                    [selected_strategy.name],
                    success_rate,
                    prompt if prompt is not None else ""
                )
            
            # End compute tracking
            if hasattr(self, "compute_tracker"):
                self.compute_tracker.end_tracking(success=True)  # Placeholder
            
            return outputs
        
        # Replace model's generate method
        model.generate = enhanced_generate
        
        # Store reference to original method
        model._original_generate = original_generate
        
        logger.info("Advanced reasoning components integrated into model generation")
    
    def process_input(self, 
                     input_text: str, 
                     max_tokens: int = 100,
                     **kwargs) -> Dict[str, Any]:
        """
        Process input text using all active components before model inference.
        
        Args:
            input_text: Input text to process
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing processed input and metadata
        """
        processed_input = input_text
        metadata = {}
        
        # Apply JRT processing if enabled
        if "jrt" in self.active_components and hasattr(self.model, '_jrt_processor'):
            original_length = len(input_text)
            processed_input = self.model._jrt_processor.process(input_text)
            metadata["jrt_applied"] = True
            metadata["original_length"] = original_length
            metadata["processed_length"] = len(processed_input)
        
        # Add function calling instructions if enabled
        if "function_calling" in self.active_components and hasattr(self.model, '_function_manager'):
            function_prompt = self.model._function_manager.create_function_prompt()
            if function_prompt:
                processed_input = function_prompt + "\n\n" + processed_input
                metadata["function_prompt_added"] = True
                metadata["function_prompt_length"] = len(function_prompt)
        
        # Prepare result
        result = {
            "processed_input": processed_input,
            "max_tokens": max_tokens,
            "metadata": metadata
        }
        
        # Add additional parameters
        result.update(kwargs)
        
        return result
    
    def process_output(self, output_text: str, **metadata) -> Dict[str, Any]:
        """
        Process model output using all active components after inference.
        
        Args:
            output_text: Raw model output
            **metadata: Additional metadata from input processing
            
        Returns:
            Dictionary containing processed output and results
        """
        processed_output = output_text
        results = {"text": processed_output}
        
        # Process function calls if enabled
        if "function_calling" in self.active_components and hasattr(self.model, '_function_manager'):
            function_results = self.model._function_manager.process_text(processed_output)
            if function_results["function_calls_detected"]:
                results["function_results"] = function_results
        
        # Extract reasoning trace from MCTS if available
        if "mcts" in self.active_components and hasattr(self.model, '_mcts_reasoning_trace'):
            results["reasoning_trace"] = self.model._mcts_reasoning_trace
        
        return results
    
    def get_training_hooks(self) -> Dict[str, Callable]:
        """
        Get hooks to integrate components into the training process.
        
        Returns:
            Dictionary of hook functions for different training stages
        """
        hooks = {}
        
        # Pre-forward hook: Applied before model forward pass
        def pre_forward_hook(batch, model):
            modified_batch = batch
            
            # Apply JRT to training samples if enabled
            if "jrt" in self.active_components and "input_texts" in batch:
                jrt_processor = getattr(model, '_jrt_processor', None) or JRTProcessor()
                input_texts = batch["input_texts"]
                processed_texts = [jrt_processor.process(text) for text in input_texts]
                modified_batch = batch.copy()
                modified_batch["input_texts"] = processed_texts
                modified_batch["_original_texts"] = input_texts
            
            return modified_batch
        
        # Post-forward hook: Applied after model forward pass
        def post_forward_hook(outputs, batch, model):
            modified_outputs = outputs
            
            # Process function calls in outputs if enabled
            if "function_calling" in self.active_components and "generated_texts" in outputs:
                function_manager = getattr(model, '_function_manager', None) or FunctionCallingManager()
                generated_texts = outputs["generated_texts"]
                function_results = [function_manager.process_text(text) for text in generated_texts]
                modified_outputs = outputs.copy()
                modified_outputs["function_results"] = function_results
            
            return modified_outputs
        
        # Loss modification hook: Customize loss computation
        def loss_modification_hook(loss, outputs, batch, model):
            modified_loss = loss
            
            # Add regularization for quantization if enabled
            if "quantization" in self.active_components:
                quant_penalty = 0.0
                for name, param in model.named_parameters():
                    if 'qweight' in name:
                        # Add small penalty to encourage weight clustering
                        quant_penalty += 0.0001 * torch.sum(torch.abs(param))
                
                modified_loss = loss + quant_penalty
            
            return modified_loss
        
        # Optimizer configuration hook: Customize optimizer
        def optimizer_config_hook(optimizer_class, optimizer_kwargs, model):
            modified_kwargs = optimizer_kwargs.copy()
            
            # Adjust learning rate for quantized models
            if "quantization" in self.active_components:
                if "lr" in modified_kwargs:
                    # Reduce learning rate for quantized models
                    modified_kwargs["lr"] *= 0.8
            
            return optimizer_class, modified_kwargs
        
        hooks["pre_forward"] = pre_forward_hook
        hooks["post_forward"] = post_forward_hook
        hooks["loss_modification"] = loss_modification_hook
        hooks["optimizer_config"] = optimizer_config_hook
        
        return hooks
    
    def register_custom_component(
        self,
        name: str,
        component: Any,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        conflicts: Optional[List[str]] = None
    ) -> None:
        """
        Register a custom component for integration.
        
        Args:
            name: Component name
            component: Component class or instance
            config: Component configuration
            dependencies: List of required components
            conflicts: List of conflicting components
        """
        self.registry.register(
            name=name,
            component=component,
            config=config,
            dependencies=dependencies,
            conflicts=conflicts
        )
        logger.info(f"Registered custom component: {name}")

# Integration presets for common use cases
INTEGRATION_PRESETS = {
    "inference_speed": {
        "enable_linear_attention": True,
        "enable_quantization": True,
        "quantization_bits": 8,
        "linear_attention_feature_dim": 8,
        "enable_mcts": False,
        "enable_jrt": False,
    },
    "memory_efficiency": {
        "enable_linear_attention": True,
        "enable_quantization": True,
        "quantization_bits": 4,
        "linear_attention_feature_dim": 16,
        "enable_mcts": False,
        "enable_jrt": True,
        "jrt_repetitions": 2,
    },
    "reasoning_quality": {
        "enable_linear_attention": True,
        "enable_mcts": True,
        "enable_function_calling": True,
        "enable_mcts_function_integration": True,
        "mcts_max_iterations": 150,
        "mcts_exploration_weight": 1.2,
    },
    "full_capabilities": {
        "enable_linear_attention": True,
        "enable_mcts": True,
        "enable_function_calling": True,
        "enable_jrt": True,
        "enable_quantization": True,
        "enable_mcts_function_integration": True,
        "enable_jrt_linear_integration": True,
        "quantization_bits": 8,
        "jrt_repetitions": 3,
        "mcts_max_iterations": 100,
    }
}

def apply_integration_preset(model: nn.Module, preset_name: str, tokenizer=None, **overrides) -> nn.Module:
    """
    Apply an integration preset to a model.
    
    Args:
        model: Base model to enhance
        preset_name: Name of the preset to apply
        tokenizer: Optional tokenizer
        **overrides: Override specific configuration values
        
    Returns:
        Enhanced model with preset integration
    """
    if preset_name not in INTEGRATION_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(INTEGRATION_PRESETS.keys())}")
    
    # Get preset configuration
    config = INTEGRATION_PRESETS[preset_name].copy()
    
    # Apply overrides
    config.update(overrides)
    
    # Initialize integration manager
    manager = IntegrationManager()
    manager.configure(config)
    
    # Apply integration
    enhanced_model = manager.initialize_model(model, tokenizer)
    
    # Attach the manager to the model for later use
    enhanced_model._integration_manager = manager
    
    return enhanced_model

class TransformerGNNIntegration(nn.Module):
    """
    Integration module for combining Transformer and GNN architectures.
    This is a wrapper that delegates to the GNN implementation.
    """
    
    def __init__(self, hidden_size, **kwargs):
        super().__init__()
        from .gnn.integration import TransformerGNNIntegration as GNNIntegration
        self.integration = GNNIntegration(hidden_size=hidden_size, **kwargs)
        
    def forward(self, hidden_states, gnn_output, attention_mask=None):
        return self.integration(hidden_states, gnn_output, attention_mask)

class ModelIntegrator:
    """
    Model integrator for combining different model components
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.components = {}
        
    def register_component(self, name, component):
        """Register a model component"""
        self.components[name] = component
        self.logger.info(f"Registered component: {name}")
        return component
    
    def get_component(self, name):
        """Get a registered component by name"""
        if name not in self.components:
            raise ValueError(f"Component not found: {name}")
        return self.components[name]
    
    def integrate_gnn(self, model, gnn_config):
        """
        Integrate GNN components with a transformer model
        
        Args:
            model: Base transformer model
            gnn_config: GNN configuration
            
        Returns:
            Integrated model
        """
        # Only support ValkyrieLLM models for now
        if not isinstance(model, ValkyrieLLM):
            raise TypeError("GNN integration currently only supports ValkyrieLLM models")
        
        # Create GNN components
        from .gnn.graph_encoder import GraphEncoder
        from .gnn.gnn_model import GNNEncoder
        
        # Create graph encoder
        graph_encoder = GraphEncoder(
            hidden_size=model.hidden_size,
            gnn_hidden_size=gnn_config.get("gnn_hidden_size", model.hidden_size),
            num_gnn_layers=gnn_config.get("num_layers", 3),
            num_heads=gnn_config.get("num_heads", 8),
            dropout=gnn_config.get("dropout", 0.1),
            use_edge_features=gnn_config.get("use_edge_features", True),
            graph_construction=gnn_config.get("graph_construction", "dynamic")
        )
        
        # Create GNN encoder
        gnn_encoder = GNNEncoder(
            hidden_size=model.hidden_size,
            gnn_hidden_size=gnn_config.get("gnn_hidden_size", model.hidden_size),
            gnn_type=gnn_config.get("gnn_type", "gcn"),
            num_layers=gnn_config.get("num_layers", 3),
            num_heads=gnn_config.get("num_heads", 8),
            dropout=gnn_config.get("dropout", 0.1),
            use_edge_features=gnn_config.get("use_edge_features", True),
            residual=gnn_config.get("residual", True),
            layer_norm=gnn_config.get("layer_norm", True),
            pooling_type=gnn_config.get("pooling_type", "mean")
        )
        
        # Create integration module
        integration_module = TransformerGNNIntegration(
            hidden_size=model.hidden_size,
            gnn_hidden_size=gnn_config.get("gnn_hidden_size", model.hidden_size),
            integration_type=gnn_config.get("integration_type", "concat")
        )
        
        # Attach components to model
        model.graph_encoder = graph_encoder
        model.gnn_encoder = gnn_encoder
        model.gnn_integration = integration_module
        
        # Modify model's forward method to include GNN processing
        # We do this by creating a new forward method that wraps the original
        original_forward = model.forward
        
        def forward_with_gnn(self, input_ids, attention_mask=None, **kwargs):
            # Call original forward method to get transformer outputs
            outputs = original_forward(input_ids, attention_mask, **kwargs)
            
            # Get hidden states
            hidden_states = outputs["hidden_states"]
            
            # If GNN integration is disabled or for training/dropout
            if not hasattr(self, "graph_encoder") or not hasattr(self, "gnn_encoder") or torch.rand(1).item() < 0.1:
                return outputs
                
            # Create graph representation
            node_features, edge_index, edge_attr = self.graph_encoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask
            )
            
            # Apply GNN encoder
            gnn_output = self.gnn_encoder(
                node_features=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr
            )
            
            # Integrate GNN output with transformer hidden states
            integrated_hidden_states = self.gnn_integration(
                hidden_states=hidden_states,
                gnn_output=gnn_output,
                attention_mask=attention_mask
            )
            
            # Update outputs with integrated hidden states
            outputs["hidden_states"] = integrated_hidden_states
            
            # Update logits if present
            if "logits" in outputs:
                # Assuming the model has a projection layer from hidden states to logits
                if hasattr(self, "lm_head"):
                    outputs["logits"] = self.lm_head(integrated_hidden_states)
                    
            return outputs
        
        # Replace model's forward method
        # This uses a bound method to ensure 'self' is passed correctly
        import types
        model.forward = types.MethodType(forward_with_gnn, model)
        
        self.logger.info("GNN integration completed")
        
        return model

    def integrate_enhanced_tokenizer(self, model, tokenizer_config=None, use_gnn=True):
        """
        Integrate enhanced tokenizer with model, optionally with GNN components
        
        Args:
            model: Base transformer model
            tokenizer_config: Configuration for enhanced tokenizer
            use_gnn: Whether to also integrate GNN components
            
        Returns:
            Integrated model
        """
        from model.nlp.tokenizer_adapter import TokenizerAdapter
        
        # Create default config if none provided
        if tokenizer_config is None:
            tokenizer_config = {
                'vocab_size': getattr(model, 'vocab_size', 50000),
                'hidden_size': getattr(model, 'hidden_size', 768),
                'max_position_embeddings': getattr(model, 'max_seq_length', 2048),
                'dropout': 0.1,
                'use_factorized_embeddings': True
            }
        
        # Create tokenizer adapter
        tokenizer_adapter = TokenizerAdapter(
            vocab_size=tokenizer_config.get('vocab_size', 50000),
            hidden_size=tokenizer_config.get('hidden_size', 768),
            max_position_embeddings=tokenizer_config.get('max_position_embeddings', 2048),
            dropout=tokenizer_config.get('dropout', 0.1),
            use_factorized_embeddings=tokenizer_config.get('use_factorized_embeddings', False),
            config=model.config if hasattr(model, 'config') else None
        )
        
        # Attach tokenizer adapter to model
        model.tokenizer_adapter = tokenizer_adapter
        
        # Store original embedding module for potential rollback
        if hasattr(model, 'token_embedding'):
            model._original_token_embedding = model.token_embedding
            model.token_embedding = tokenizer_adapter.module
        
        # If requested, also integrate GNN components
        if use_gnn:
            # Default GNN configuration
            gnn_config = {
                'gnn_hidden_size': tokenizer_config.get('hidden_size', 768),
                'num_layers': 3,
                'gnn_type': 'gcn',
                'use_edge_features': True,
                'integration_type': 'gating'
            }
            
            # Integrate GNN components
            self.integrate_gnn(model, gnn_config)
            
            # Create integration between tokenizer and GNN
            original_embed_tokens = tokenizer_adapter.embed_tokens
            
            def embed_tokens_with_gnn(input_ids, token_type_ids=None, position_ids=None, **kwargs):
                # Get standard embeddings
                embeddings = original_embed_tokens(input_ids, token_type_ids, position_ids)
                
                # Apply GNN processing if available and probability check passes
                if hasattr(model, 'graph_encoder') and hasattr(model, 'gnn_encoder') and torch.rand(1).item() > 0.2:
                    # Create graph from embeddings
                    batch_size, seq_len = input_ids.shape
                    attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
                    if 'attention_mask' in kwargs:
                        attention_mask = kwargs['attention_mask']
                    
                    # Process through GNN
                    node_features, edge_index, edge_attr = model.graph_encoder(
                        hidden_states=embeddings,
                        attention_mask=attention_mask
                    )
                    
                    gnn_output = model.gnn_encoder(
                        node_features=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_attr
                    )
                    
                    # Integrate GNN output with standard embeddings
                    embeddings = model.gnn_integration(
                        hidden_states=embeddings,
                        gnn_output=gnn_output,
                        attention_mask=attention_mask
                    )
                
                return embeddings
            
            # Replace tokenizer's embed_tokens method
            tokenizer_adapter.embed_tokens = embed_tokens_with_gnn
        
        self.logger.info(f"Enhanced tokenizer integration completed (with GNN: {use_gnn})")
        
        return model 

class TaskTargetedFineTuner:
    """
    Enables targeted fine-tuning of specific reasoning components.
    
    This class wraps the IntegrationManager to provide:
    1. Component-specific parameter freezing/unfreezing
    2. Targeted dataset filtering for component-specific training
    3. Custom loss functions for specific reasoning capabilities
    """
    
    def __init__(self, integration_manager: IntegrationManager):
        """Initialize with an existing IntegrationManager instance"""
        self.integration_manager = integration_manager
        self.frozen_components = set()
        self.trainable_components = set()
        self.component_optimizers = {}
        self.component_loss_weights = {}
        self.default_lr = 1e-5
    
    def prepare_for_targeted_finetuning(self, 
                                        target_components: List[str], 
                                        freeze_others: bool = True,
                                        learning_rates: Optional[Dict[str, float]] = None):
        """
        Prepare model for targeted fine-tuning on specific components.
        
        Args:
            target_components: List of component names to fine-tune
            freeze_others: Whether to freeze parameters of other components
            learning_rates: Optional dict mapping component names to learning rates
        """
        if not self.integration_manager.model:
            raise ValueError("No model initialized in the IntegrationManager")
        
        model = self.integration_manager.model
        
        # Set trainable components
        self.trainable_components = set(target_components)
        
        # Validate component names
        all_components = set(self.integration_manager.registry.components.keys())
        for component in self.trainable_components:
            if component not in all_components:
                raise ValueError(f"Unknown component: {component}")
        
        # Set up component-specific learning rates
        learning_rates = learning_rates or {}
        
        # Freeze or unfreeze parameters based on component membership
        for name, param in model.named_parameters():
            should_train = False
            component_name = None
            
            # Determine which component this parameter belongs to
            for comp in self.trainable_components:
                # Handle component-specific parameter naming patterns
                if (f"{comp}" in name or 
                    f"{comp}_layer" in name or 
                    (comp == "neural_symbolic" and "rule_embeddings" in name) or
                    (comp == "mcts" and "mcts" in name) or 
                    (comp == "recursive_reasoning" and "recursive" in name)):
                    should_train = True
                    component_name = comp
                    break
            
            # Set requires_grad based on whether we're training this component
            param.requires_grad = should_train
            
            # Track which components are frozen
            if not should_train and freeze_others:
                module_name = name.split('.')[0]
                self.frozen_components.add(module_name)
        
        # Set up component-specific optimizers
        optimizers = []
        for comp in self.trainable_components:
            # Get parameters for this component
            params = [p for n, p in model.named_parameters() 
                     if p.requires_grad and any(c in n for c in [comp, f"{comp}_layer"])]
            
            if params:
                lr = learning_rates.get(comp, self.default_lr)
                optimizer = torch.optim.AdamW(params, lr=lr)
                self.component_optimizers[comp] = optimizer
                optimizers.append(optimizer)
        
        return optimizers
    
    def filter_dataset_for_component(self, dataset, component_name: str):
        """
        Filter a dataset to focus on examples relevant to a specific component.
        
        Args:
            dataset: The original dataset
            component_name: Name of the component to filter for
            
        Returns:
            Filtered dataset
        """
        # Define filtering criteria for each component type
        component_filters = {
            "neural_symbolic": lambda x: "symbolic" in x.get("task_type", "") or 
                                         "math" in x.get("task_type", "") or
                                         "equation" in x.get("input", "").lower(),
            
            "mcts": lambda x: "search" in x.get("task_type", "") or
                              "planning" in x.get("task_type", "") or
                              "game" in x.get("task_type", ""),
            
            "recursive_reasoning": lambda x: "recursive" in x.get("task_type", "") or
                                            "multi_step" in x.get("task_type", "") or
                                            "step by step" in x.get("input", "").lower(),
            
            # Add more filters for other components
        }
        
        # Use the appropriate filter or return the original dataset
        if component_name in component_filters:
            filter_fn = component_filters[component_name]
            filtered_data = [item for item in dataset if filter_fn(item)]
            return filtered_data
        else:
            return dataset
    
    def component_specific_loss(self, outputs, targets, component_name: str):
        """
        Compute component-specific loss functions tailored to each reasoning capability.
        
        Args:
            outputs: Model outputs
            targets: Target values
            component_name: Name of the component to compute loss for
            
        Returns:
            Component-specific loss
        """
        # Default to standard cross-entropy loss
        standard_loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                        targets.view(-1))
        
        # Component-specific loss adjustments
        if component_name == "neural_symbolic":
            # Add auxiliary losses for rule application and verification
            if hasattr(outputs, "rule_losses"):
                return standard_loss + outputs.rule_losses * self.component_loss_weights.get("rule_loss", 0.1)
            
        elif component_name == "mcts":
            # Add policy and value losses similar to AlphaZero
            if hasattr(outputs, "policy_loss") and hasattr(outputs, "value_loss"):
                policy_weight = self.component_loss_weights.get("policy_loss", 1.0)
                value_weight = self.component_loss_weights.get("value_loss", 1.0)
                return standard_loss + policy_weight * outputs.policy_loss + value_weight * outputs.value_loss
            
        elif component_name == "recursive_reasoning":
            # Add intermediate step prediction losses
            if hasattr(outputs, "intermediate_losses"):
                intermediate_weight = self.component_loss_weights.get("intermediate_loss", 0.5)
                return standard_loss + intermediate_weight * outputs.intermediate_losses
        
        # Return standard loss for other components
        return standard_loss
    
    def train_component(self, component_name: str, dataloader, num_epochs=3):
        """
        Train a specific component on filtered data with component-specific loss.
        
        Args:
            component_name: Name of the component to train
            dataloader: DataLoader containing training data
            num_epochs: Number of training epochs
            
        Returns:
            Training statistics
        """
        if component_name not in self.trainable_components:
            raise ValueError(f"Component {component_name} is not marked as trainable")
        
        if component_name not in self.component_optimizers:
            raise ValueError(f"No optimizer found for component {component_name}")
        
        model = self.integration_manager.model
        optimizer = self.component_optimizers[component_name]
        
        # Track statistics
        stats = {"loss": [], "epoch": []}
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch in dataloader:
                # Forward pass
                outputs = model(**batch)
                
                # Compute component-specific loss
                loss = self.component_specific_loss(outputs, batch["labels"], component_name)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Record statistics
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            stats["loss"].append(avg_loss)
            stats["epoch"].append(epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Component: {component_name}, Loss: {avg_loss:.4f}")
        
        return stats 