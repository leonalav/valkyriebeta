import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import gc

@dataclass
class ValidationResult:
    """
    Result of model validation
    """
    errors: List[str] = None
    warnings: List[str] = None
    is_valid: bool = None
    
    def __post_init__(self):
        self.is_valid = True if self.is_valid is None else self.is_valid
        self.errors = [] if self.errors is None else self.errors
        self.warnings = [] if self.warnings is None else self.warnings
        
    @property
    def is_valid(self):
        return self._is_valid and len(self.errors) == 0
        
    @is_valid.setter
    def is_valid(self, value):
        self._is_valid = value
        
    def add_error(self, message):
        self.errors.append(message)
        self._is_valid = False
        
    def add_warning(self, message):
        self.warnings.append(message)

class ModelValidator:
    """
    Comprehensive model validator for checking model integrity and compatibility
    """
    
    @staticmethod
    def validate_model(model):
        """
        Validate a model for training and inference
        
        Args:
            model: The model to validate
            
        Returns:
            result: ValidationResult with validation status and messages
        """
        # Create validation result
        result = ValidationResult()
        
        # Check model type
        if not isinstance(model, nn.Module):
            result.add_error("Model must be an instance of torch.nn.Module")
            return result
            
        # Check if model has required attributes
        required_attributes = ['forward']
        for attr in required_attributes:
            if not hasattr(model, attr):
                result.add_error(f"Model missing required attribute: {attr}")
                
        # Validate transformer components
        ModelValidator._validate_transformer_components(model, result)
        
        # Validate parameter counts
        ModelValidator._validate_parameter_counts(model, result)
        
        # Validate recursive reasoning if enabled
        if hasattr(model, 'recursive_reasoning') and model.recursive_reasoning is not None:
            ModelValidator._validate_recursive_reasoning(model, result)
            
        # Validate tree reasoning if enabled
        if hasattr(model, 'tree_reasoning') and model.tree_reasoning is not None:
            ModelValidator._validate_tree_reasoning(model, result)
            
        # Validate neural symbolic reasoning if enabled
        if hasattr(model, 'neural_symbolic_reasoning') and model.neural_symbolic_reasoning is not None:
            ModelValidator._validate_neural_symbolic(model, result)
            
        # Validate MoE if enabled
        if hasattr(model, 'moe_layer') and model.moe_layer is not None:
            ModelValidator._validate_moe(model, result)
            
        # Validate enhanced memory if enabled
        if hasattr(model, 'memory_bank') and model.memory_bank is not None:
            ModelValidator._validate_enhanced_memory(model, result)
            
        # Validate knowledge incorporation if enabled
        if hasattr(model, 'knowledge_reasoning') and model.knowledge_reasoning is not None:
            ModelValidator._validate_knowledge_incorporation(model, result)
            
        # Validate computational efficiency
        ModelValidator._validate_computational_efficiency(model, result)
        
        # Validate forward pass
        ModelValidator._validate_forward_pass(model, result)
        
        return result
        
    @staticmethod
    def _validate_transformer_components(model, result):
        """
        Validate transformer components
        
        Args:
            model: The model to validate
            result: ValidationResult to update
        """
        # Check if model has transformer attribute
        if not hasattr(model, 'transformer'):
            result.add_warning("Model does not have a 'transformer' attribute")
            return
            
        # Check transformer type
        transformer = model.transformer
        if not isinstance(transformer, nn.Module):
            result.add_error("Model transformer must be an instance of torch.nn.Module")
            return
            
        # Check if transformer has required attributes
        required_attributes = ['forward']
        for attr in required_attributes:
            if not hasattr(transformer, attr):
                result.add_error(f"Transformer missing required attribute: {attr}")
                
        # Check if transformer has attention layers
        has_attention = False
        for module in transformer.modules():
            if 'attention' in module.__class__.__name__.lower():
                has_attention = True
                break
                
        if not has_attention:
            result.add_warning("Transformer does not appear to have attention layers")
            
        # Check if model has token embeddings
        if not hasattr(model, 'token_embedding'):
            result.add_warning("Model does not have a 'token_embedding' attribute")
            
        # Check if model has output layer
        if not hasattr(model, 'lm_head'):
            result.add_warning("Model does not have a 'lm_head' attribute")
            
    @staticmethod
    def _validate_parameter_counts(model, result):
        """
        Validate parameter counts
        
        Args:
            model: The model to validate
            result: ValidationResult to update
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Check if model has reasonable parameter count
        if total_params < 1000:
            result.add_warning(f"Model has very few parameters ({total_params})")
            
        if trainable_params == 0:
            result.add_error("Model has no trainable parameters")
            
        # Log parameter counts
        result.parameter_count = total_params
        result.trainable_parameter_count = trainable_params
        
    @staticmethod
    def _validate_recursive_reasoning(model, result):
        """
        Validate recursive reasoning components
        
        Args:
            model: The model to validate
            result: ValidationResult to update
        """
        # Check recursive reasoning type
        if not isinstance(model.recursive_reasoning, nn.Module):
            result.add_error("Recursive reasoning must be an instance of torch.nn.Module")
            return
            
        # Check if recursive reasoning has required attributes
        required_attributes = ['forward']
        for attr in required_attributes:
            if not hasattr(model.recursive_reasoning, attr):
                result.add_error(f"Recursive reasoning missing required attribute: {attr}")
                
        # Check recursive depth
        if hasattr(model.recursive_reasoning, 'max_depth'):
            if model.recursive_reasoning.max_depth < 1:
                result.add_error("Recursive reasoning max_depth must be at least 1")
            elif model.recursive_reasoning.max_depth > 10:
                result.add_warning("Recursive reasoning max_depth is very high, which may cause performance issues")
                
    @staticmethod
    def _validate_tree_reasoning(model, result):
        """
        Validate tree reasoning components
        
        Args:
            model: The model to validate
            result: ValidationResult to update
        """
        # Check tree reasoning type
        if not isinstance(model.tree_reasoning, nn.Module):
            result.add_error("Tree reasoning must be an instance of torch.nn.Module")
            return
            
        # Check if tree reasoning has required attributes
        required_attributes = ['forward']
        for attr in required_attributes:
            if not hasattr(model.tree_reasoning, attr):
                result.add_error(f"Tree reasoning missing required attribute: {attr}")
                
        # Check tree depth
        if hasattr(model.tree_reasoning, 'max_depth'):
            if model.tree_reasoning.max_depth < 1:
                result.add_error("Tree reasoning max_depth must be at least 1")
            elif model.tree_reasoning.max_depth > 10:
                result.add_warning("Tree reasoning max_depth is very high, which may cause performance issues")
                
    @staticmethod
    def _validate_neural_symbolic(model, result):
        """
        Validate neural symbolic components
        
        Args:
            model: The model to validate
            result: ValidationResult to update
        """
        # Check if model has neural symbolic reasoning component
        if not hasattr(model, 'neural_symbolic_reasoning'):
            result.add_error("Model missing neural_symbolic_reasoning component")
            return result
            
        # Basic attribute checks
        required_attributes = [
            'forward', 'config', 'rule_embeddings'
        ]
        
        for attr in required_attributes:
            if not hasattr(model.neural_symbolic_reasoning, attr):
                result.add_error(f"Neural symbolic reasoning missing required attribute: {attr}")
                
        # Check rule embeddings
        if hasattr(model.neural_symbolic_reasoning, 'rule_embeddings'):
            if not isinstance(model.neural_symbolic_reasoning.rule_embeddings, nn.Parameter):
                result.add_error("rule_embeddings must be an instance of nn.Parameter")
                
        # Check memory if it's used
        if (hasattr(model.neural_symbolic_reasoning, 'config') and 
            hasattr(model.neural_symbolic_reasoning.config, 'use_memory_for_reasoning') and
            model.neural_symbolic_reasoning.config.use_memory_for_reasoning):
            
            if not hasattr(model.neural_symbolic_reasoning, 'memory'):
                result.add_error("Model uses memory for reasoning but has no memory attribute")
            elif not isinstance(model.neural_symbolic_reasoning.memory, nn.Parameter):
                result.add_error("memory must be an instance of nn.Parameter")
                
            # Check memory update methods
            memory_update_methods = ['memory_key', 'memory_value', 'memory_query']
            for method in memory_update_methods:
                if not hasattr(model.neural_symbolic_reasoning, method):
                    result.add_warning(f"Neural symbolic reasoning missing memory method: {method}")
                    
            # Check for memory cleanup functions
            if not hasattr(model.neural_symbolic_reasoning, '_cleanup_caches'):
                result.add_warning("Neural symbolic reasoning does not have a '_cleanup_caches' method for memory management")
                
        # Validate cache management
        cache_attributes = ['rule_cache', 'specialized_rules', 'rule_composition_cache']
        has_caches = False
        for attr in cache_attributes:
            if hasattr(model.neural_symbolic_reasoning, attr):
                has_caches = True
                break
                
        if has_caches:
            # Check if we have memory management capabilities
            memory_mgmt_attrs = ['_cleanup_caches', 'inference_steps_since_cleanup', 'cleanup_frequency', 'temporary_tensors']
            for attr in memory_mgmt_attrs:
                if not hasattr(model.neural_symbolic_reasoning, attr):
                    result.add_warning(f"Neural symbolic reasoning has caches but is missing memory management attribute: {attr}")
            
            # Check cache size limits
            cache_size_attrs = ['max_specialized_cache_size', 'max_composition_cache_size']
            for attr in cache_size_attrs:
                if not hasattr(model.neural_symbolic_reasoning, attr):
                    result.add_warning(f"Neural symbolic reasoning has caches but is missing cache size limit: {attr}")
                else:
                    # Check that cache size limit is reasonable
                    limit = getattr(model.neural_symbolic_reasoning, attr)
                    if limit > 10000:
                        result.add_warning(f"Neural symbolic reasoning has very large cache size limit: {attr}={limit}")
                        
            # Check usage tracking for cache pruning
            usage_attrs = ['specialized_rules_usage', 'rule_composition_usage'] 
            for attr in usage_attrs:
                if not hasattr(model.neural_symbolic_reasoning, attr):
                    result.add_warning(f"Neural symbolic reasoning has caches but is missing usage tracking: {attr}")
                    
        # Test forward pass with dummy data
        try:
            with torch.no_grad():
                batch_size, seq_len = 2, 8
                hidden_size = model.neural_symbolic_reasoning.config.hidden_size if hasattr(model.neural_symbolic_reasoning, 'config') else 768
                
                dummy_input = torch.randn(batch_size, seq_len, hidden_size, device=next(model.parameters()).device)
                dummy_mask = torch.ones(batch_size, seq_len, device=next(model.parameters()).device)
                
                outputs = model.neural_symbolic_reasoning(dummy_input, dummy_mask)
                
                # Check output type and contents
                if not isinstance(outputs, dict):
                    result.add_error("Neural symbolic reasoning forward pass should return a dictionary")
                else:
                    if 'hidden_states' not in outputs:
                        result.add_error("Neural symbolic reasoning outputs missing 'hidden_states'")
                    else:
                        # Check output shape
                        if outputs['hidden_states'].shape != dummy_input.shape:
                            result.add_error(f"Neural symbolic reasoning output shape mismatch: {outputs['hidden_states'].shape} vs {dummy_input.shape}")
        except Exception as e:
            result.add_error(f"Neural symbolic reasoning forward pass failed: {str(e)}")
            
        # Check for memory leaks by running multiple forward passes
        try:
            with torch.no_grad():
                # Run multiple passes and check memory usage
                device = next(model.parameters()).device
                if device.type == 'cuda':
                    # Record starting memory
                    start_memory = torch.cuda.memory_allocated()
                    
                    # Run several forward passes
                    for _ in range(5):
                        dummy_input = torch.randn(batch_size, seq_len, hidden_size, device=device)
                        dummy_mask = torch.ones(batch_size, seq_len, device=device)
                        _ = model.neural_symbolic_reasoning(dummy_input, dummy_mask)
                        
                        # Force cleanup
                        if hasattr(model.neural_symbolic_reasoning, '_cleanup_caches'):
                            model.neural_symbolic_reasoning._cleanup_caches()
                        
                        # Run garbage collection
                        gc.collect()
                        torch.cuda.empty_cache()
                    
                    # Check memory usage after multiple passes
                    end_memory = torch.cuda.memory_allocated()
                    memory_growth = end_memory - start_memory
                    
                    # Allow small growth but flag significant increase
                    # 10MB threshold is reasonable for test runs
                    if memory_growth > 10 * 1024 * 1024:  # 10 MB
                        result.add_warning(f"Potential memory leak detected: {memory_growth / (1024 * 1024):.2f} MB growth")
        except Exception as e:
            result.add_warning(f"Memory leak test failed: {str(e)}")
        
        return result
        
    @staticmethod
    def _validate_moe(model, result):
        """
        Validate Mixture of Experts components
        
        Args:
            model: The model to validate
            result: ValidationResult to update
        """
        # Check MoE layer type
        if not isinstance(model.moe_layer, nn.Module) and not isinstance(model.moe_layer, nn.ModuleList):
            result.add_error("MoE layer must be an instance of torch.nn.Module or torch.nn.ModuleList")
            return
            
        # Check if model has expert gating
        if not hasattr(model, 'expert_gating'):
            result.add_warning("Model with MoE does not have an 'expert_gating' attribute")
            return
            
        # Check expert gating type
        if not isinstance(model.expert_gating, nn.Module):
            result.add_error("Expert gating must be an instance of torch.nn.Module")
            return
            
        # Check if expert gating has required attributes
        required_attributes = ['forward']
        for attr in required_attributes:
            if not hasattr(model.expert_gating, attr):
                result.add_error(f"Expert gating missing required attribute: {attr}")
                
        # Check number of experts
        if hasattr(model.expert_gating, 'num_experts'):
            if model.expert_gating.num_experts < 2:
                result.add_error("Number of experts must be at least 2")
            elif model.expert_gating.num_experts > 128:
                result.add_warning("Number of experts is very high, which may cause performance issues")
                
    @staticmethod
    def _validate_enhanced_memory(model, result):
        """
        Validate enhanced memory components
        
        Args:
            model: The model to validate
            result: ValidationResult to update
        """
        # Check memory bank type
        if not isinstance(model.memory_bank, nn.Module):
            result.add_error("Memory bank must be an instance of torch.nn.Module")
            return
            
        # Check if memory bank has required attributes
        required_attributes = ['forward']
        for attr in required_attributes:
            if not hasattr(model.memory_bank, attr):
                result.add_error(f"Memory bank missing required attribute: {attr}")
                
        # Check memory size
        if hasattr(model.memory_bank, 'memory_size'):
            if model.memory_bank.memory_size < 1:
                result.add_error("Memory size must be at least 1")
            elif model.memory_bank.memory_size > 1000000:
                result.add_warning("Memory size is very high, which may cause performance issues")
                
    @staticmethod
    def _validate_knowledge_incorporation(model, result):
        """
        Validate knowledge incorporation components
        
        Args:
            model: The model to validate
            result: ValidationResult to update
        """
        # Check knowledge reasoning type
        if not isinstance(model.knowledge_reasoning, nn.Module):
            result.add_error("Knowledge reasoning must be an instance of torch.nn.Module")
            return
            
        # Check if knowledge reasoning has required attributes
        required_attributes = ['forward']
        for attr in required_attributes:
            if not hasattr(model.knowledge_reasoning, attr):
                result.add_error(f"Knowledge reasoning missing required attribute: {attr}")
                
        # Check knowledge graph size
        if hasattr(model.knowledge_reasoning, 'knowledge_graph_size'):
            if model.knowledge_reasoning.knowledge_graph_size < 1:
                result.add_error("Knowledge graph size must be at least 1")
            elif model.knowledge_reasoning.knowledge_graph_size > 1000000:
                result.add_warning("Knowledge graph size is very high, which may cause performance issues")
                
    @staticmethod
    def _validate_computational_efficiency(model, result):
        """
        Validate computational efficiency
        
        Args:
            model: The model to validate
            result: ValidationResult to update
        """
        # Check if model has gradient checkpointing
        if hasattr(model, 'config') and hasattr(model.config, 'use_gradient_checkpointing'):
            if model.config.use_gradient_checkpointing:
                # Check if transformer supports gradient checkpointing
                if hasattr(model.transformer, 'gradient_checkpointing_enable'):
                    result.add_warning("Model has gradient checkpointing enabled but transformer does not support it")
                    
        # Check if model has flash attention
        if hasattr(model, 'config') and hasattr(model.config, 'use_flash_attention'):
            if model.config.use_flash_attention:
                # Check if any attention module has flash attention
                has_flash_attention = False
                for module in model.modules():
                    if 'flash' in module.__class__.__name__.lower() and 'attention' in module.__class__.__name__.lower():
                        has_flash_attention = True
                        break
                        
                if not has_flash_attention:
                    result.add_warning("Model has flash attention enabled but no flash attention module was found")
                    
    @staticmethod
    def _validate_forward_pass(model, result):
        """
        Validate forward pass
        
        Args:
            model: The model to validate
            result: ValidationResult to update
        """
        # Try a forward pass with dummy inputs
        try:
            # Create dummy inputs
            batch_size = 2
            seq_length = 16
            vocab_size = getattr(model, 'vocab_size', 32000)
            
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
            attention_mask = torch.ones_like(input_ids)
            
            # Move to same device as model
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Set model to eval mode for forward pass
            model_mode = model.training
            model.eval()
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
            # Check outputs
            if outputs is None:
                result.add_error("Model forward pass returned None")
            elif isinstance(outputs, torch.Tensor):
                expected_shape = (batch_size, seq_length, vocab_size)
                if outputs.shape != expected_shape:
                    result.add_error(f"Model output shape {outputs.shape} does not match expected shape {expected_shape}")
            else:
                result.add_warning("Model forward pass returned non-tensor output")
                
            # Restore model mode
            model.train(model_mode)
            
        except Exception as e:
            result.add_error(f"Error during forward pass: {str(e)}")
            
        # Try generate method if available
        if hasattr(model, 'generate'):
            try:
                # Create dummy inputs
                batch_size = 1
                seq_length = 8
                vocab_size = getattr(model, 'vocab_size', 32000)
                
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
                
                # Move to same device as model
                device = next(model.parameters()).device
                input_ids = input_ids.to(device)
                
                # Set model to eval mode for generation
                model_mode = model.training
                model.eval()
                
                # Generate
                with torch.no_grad():
                    generated_ids = model.generate(input_ids=input_ids, max_length=seq_length + 4)
                    
                # Check outputs
                if generated_ids is None:
                    result.add_error("Model generate method returned None")
                elif not isinstance(generated_ids, torch.Tensor):
                    result.add_error("Model generate method returned non-tensor output")
                elif generated_ids.shape[0] != batch_size:
                    result.add_error(f"Generated output batch size {generated_ids.shape[0]} does not match input batch size {batch_size}")
                elif generated_ids.shape[1] <= seq_length:
                    result.add_error(f"Generated sequence length {generated_ids.shape[1]} is not greater than input sequence length {seq_length}")
                    
                # Restore model mode
                model.train(model_mode)
                
            except Exception as e:
                result.add_error(f"Error during generation: {str(e)}")
                
        return result
