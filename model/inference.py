import os
import time
import torch
import logging
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import json
import numpy as np
import torch.nn.functional as F

from .core_model import EnhancedLanguageModel
from ..config.model_config import ModelConfig
from ..monitoring.metrics_collector import MetricsCollector
from ..utils.cache_manager import ModelCache
from ..utils.jrt_processor import apply_jrt
from ..utils.function_calling import FunctionCallingManager
from .quantization import QuantizationConfig, quantize_model
from .tree_reasoning_mcts import MCTSEnhancedTreeReasoningModule
from .integration import IntegrationManager, apply_integration_preset

logger = logging.getLogger("model.inference")

class InferenceException(Exception):
    """Exception raised for errors during model inference"""
    pass

class ModelInference:
    """Production-ready inference wrapper for LLM models with robust error handling and observability"""
    
    def __init__(
        self, 
        config: ModelConfig, 
        model_path: Optional[str] = None,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """Initialize the inference wrapper
        
        Args:
            config: Model configuration
            model_path: Path to model weights
            model_id: Optional model identifier for the model cache
            device: Device to run inference on (cpu, cuda, cuda:0, etc.)
            metrics_collector: Optional metrics collector for monitoring
        """
        self.config = config
        self.model_path = model_path or os.environ.get("MODEL_PATH", "")
        self.model_id = model_id or os.path.basename(self.model_path)
        self.device = device or (
            "cuda" if torch.cuda.is_available() 
            else "mps" if hasattr(torch, "has_mps") and torch.has_mps 
            else "cpu"
        )
        self.metrics = metrics_collector or MetricsCollector()
        self.model = None
        self.tokenizer = None
        self.half_precision = getattr(self.config, 'use_half_precision', False) and self.device == "cuda"
        self.model_cache = ModelCache(max_models=2)
        
        logger.info(f"Inference initialized with device={self.device}, "
                   f"half_precision={self.half_precision}")
        
        # Load model if eager loading is enabled
        if getattr(self.config, 'eager_loading', False):
            self.load_model()
            
    def load_model(self) -> None:
        """Load model into memory
        
        Raises:
            InferenceException: If model loading fails
        """
        if self.model is not None:
            logger.debug("Model already loaded, skipping")
            return
            
        # Check if model is in cache
        cached_model = self.model_cache.get(self.model_id)
        if cached_model:
            logger.info(f"Loading model {self.model_id} from cache")
            self.model = cached_model
            return
            
        # Not in cache, load from disk
        try:
            start_time = time.time()
            logger.info(f"Loading model from {self.model_path or 'config'}")
            
            # Load the model
            self.model = EnhancedLanguageModel(self.config)
            
            # Load weights if path provided
            if self.model_path:
                try:
                    # Handle different weight formats
                    if self.model_path.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        state_dict = load_file(self.model_path)
                    else:
                        state_dict = torch.load(self.model_path, map_location='cpu')
                        
                    # Load weights into model
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Loaded weights from {self.model_path}")
                except Exception as e:
                    logger.error(f"Failed to load weights: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise InferenceException(f"Failed to load weights: {str(e)}")
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Use half precision if enabled and on CUDA
            if self.half_precision:
                self.model = self.model.half()
                logger.info("Using half precision (FP16)")
                
            # Set to evaluation mode
            self.model.eval()
            
            # Store in model cache
            self.model_cache.set(self.model_id, self.model)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            self.metrics.log_model_load_time(load_time)
                
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error(traceback.format_exc())
            raise InferenceException(f"Failed to load model: {str(e)}")
            
    def unload(self) -> None:
        """Unload model from memory to free resources"""
        if self.model is not None:
            # Remove from CUDA memory if applicable
            if hasattr(self.model, "cpu"):
                self.model = self.model.cpu()
                
            # Delete reference
            del self.model
            self.model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Model unloaded from memory")
            
    def generate(
        self, 
        text: str, 
        max_tokens: int = 100, 
        temperature: float = 0.7, 
        top_p: float = 0.9,
        top_k: int = 40,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text from the model
        
        Args:
            text: Input text for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: List of sequences to stop generation
            stream: Whether to stream the output
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation results
            
        Raises:
            InferenceException: If generation fails
        """
        try:
            # Ensure model is loaded
            if self.model is None:
                self.load_model()
                
            # Start timing
            start_time = time.time()
            
            # Prepare input
            inputs = self.prepare_input(text)
            input_tokens = inputs.get("input_tokens", 0)
            
            # Add generation parameters
            generation_params = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                **kwargs
            }
            
            # Add stop sequences if provided
            if stop_sequences:
                generation_params["stop_sequences"] = stop_sequences
                
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_params)
                
            # Decode output
            result_text = self.model.decode_output(outputs)
            
            # Calculate timing
            elapsed_time = time.time() - start_time
            output_tokens = outputs.get("output_tokens", len(result_text.split()))
            tokens_per_second = output_tokens / elapsed_time if elapsed_time > 0 else 0
            
            # Log metrics
            self.metrics.log_inference(elapsed_time, output_tokens)
            
            # Return result
            return {
                "text": result_text,
                "tokens_generated": output_tokens,
                "input_tokens": input_tokens, 
                "elapsed_time": elapsed_time,
                "tokens_per_second": tokens_per_second
            }
                
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise InferenceException(f"Generation failed: {str(e)}")
            
    def prepare_input(self, text: str) -> Dict[str, Any]:
        """Prepare input for the model
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prepared inputs
            
        Raises:
            InferenceException: If input preparation fails
        """
        try:
            # Ensure model is loaded
            if self.model is None:
                self.load_model()
                
            # Use model's input preparation method
            inputs = self.model.prepare_input(text)
            
            # Move inputs to the correct device
            if isinstance(inputs, dict):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
            
            return inputs
            
        except Exception as e:
            logger.error(f"Input preparation failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise InferenceException(f"Input preparation failed: {str(e)}")
            
    def generate_batch(
        self, 
        texts: List[str], 
        params: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """Generate text for a batch of inputs
        
        Args:
            texts: List of input texts
            params: List of parameter dictionaries for each input
            
        Returns:
            List of generated texts
            
        Raises:
            InferenceException: If batch generation fails
        """
        if not texts:
            return []
            
        # Default parameters for all inputs if not provided
        if params is None:
            params = [{} for _ in texts]
            
        # Make sure params length matches texts length
        if len(params) != len(texts):
            raise ValueError("params list must have same length as texts list")
            
        try:
            # Ensure model is loaded
            if self.model is None:
                self.load_model()
                
            # Start timing
            start_time = time.time()
            
            # Process each input
            results = []
            for i, (text, param) in enumerate(zip(texts, params)):
                try:
                    # Generate for this input
                    result = self.generate(text, **param)
                    results.append(result["text"])
                except Exception as e:
                    logger.error(f"Error generating for input {i}: {str(e)}")
                    # Add empty result or error placeholder
                    results.append("")
                    
            # Log batch metrics
            elapsed_time = time.time() - start_time
            self.metrics.log_batch_inference(elapsed_time, len(texts))
            
            return results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise InferenceException(f"Batch generation failed: {str(e)}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the model and inference
        
        Returns:
            Dictionary with model and inference statistics
        """
        stats = {
            "model_id": self.model_id,
            "device": self.device,
            "half_precision": self.half_precision,
            "is_loaded": self.model is not None
        }
        
        # Add CUDA info if available
        if torch.cuda.is_available():
            stats.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A",
                "cuda_memory_allocated": torch.cuda.memory_allocated() / (1024 ** 3),  # GB
                "cuda_memory_reserved": torch.cuda.memory_reserved() / (1024 ** 3)  # GB
            })
            
        # Add metrics if available
        if hasattr(self.metrics, "get_stats"):
            stats["metrics"] = self.metrics.get_stats()
            
        return stats

class EnhancedInferenceEngine:
    """
    Enhanced inference engine that incorporates multiple optimizations and advanced features:
    
    1. Support for linear attention for efficient processing of long contexts
    2. Just Read Twice (JRT) for improved in-context learning
    3. MCTS for enhanced reasoning
    4. Quantization for reduced memory usage and improved speed
    5. Function calling for interaction with external tools
    
    Uses the IntegrationManager to coordinate all components.
    """
    
    def __init__(
        self,
        model,
        tokenizer=None,
        config=None,
        device=None,
        integration_preset=None,
        **kwargs
    ):
        """
        Initialize the enhanced inference engine.
        
        Args:
            model: The model to use for inference
            tokenizer: Optional tokenizer for text processing
            config: Model configuration
            device: Device to run inference on
            integration_preset: Optional preset for component integration ("inference_speed", 
                                "memory_efficiency", "reasoning_quality", "full_capabilities")
            **kwargs: Additional configuration parameters
        """
        self.config = config or {}
        self.tokenizer = tokenizer
        
        # Determine device
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize integration manager
        self.integration_manager = IntegrationManager()
        
        # Apply configuration
        if integration_preset:
            self.model = apply_integration_preset(model, integration_preset, tokenizer, **kwargs)
        else:
            # Configure with explicit settings
            self.integration_manager.configure(kwargs)
            self.model = self.integration_manager.initialize_model(model, tokenizer)
            
        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Performance tracking
        self.last_inference_time = 0
        self.last_prompt_tokens = 0
        self.last_generated_tokens = 0
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt using the enhanced model.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Penalty for repeating tokens
            stop_sequences: Sequences that stop generation when generated
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing generation results
        """
        start_time = time.time()
        
        # Process input with all active components
        processed_input = self.integration_manager.process_input(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
            **kwargs
        )
        
        # Tokenize the processed input
        if self.tokenizer:
            input_ids = self.tokenizer.encode(processed_input["processed_input"], return_tensors="pt").to(self.device)
            attention_mask = torch.ones_like(input_ids)
            self.last_prompt_tokens = input_ids.shape[1]
        else:
            # Placeholder for when tokenizer is not available
            raise ValueError("Tokenizer is required for text generation")
        
        # Generation configuration
        gen_config = {
            "max_length": input_ids.shape[1] + max_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0,
            "eos_token_id": self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None,
        }
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_config
            )
            
            # Extract generated text
            output_ids = outputs[0][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            self.last_generated_tokens = len(output_ids)
        
        # Process output with all active components
        result = self.integration_manager.process_output(
            generated_text,
            **processed_input["metadata"]
        )
        
        # Calculate metrics
        self.last_inference_time = time.time() - start_time
        tokens_per_second = self.last_generated_tokens / self.last_inference_time if self.last_inference_time > 0 else 0
        
        # Add metrics to result
        result["prompt_tokens"] = self.last_prompt_tokens
        result["generated_tokens"] = self.last_generated_tokens
        result["total_tokens"] = self.last_prompt_tokens + self.last_generated_tokens
        result["inference_time"] = self.last_inference_time
        result["tokens_per_second"] = tokens_per_second
        
        return result
    
    def register_function(self, func=None, **kwargs):
        """Register a function for function calling"""
        if not hasattr(self.model, '_function_manager'):
            # This shouldn't happen if function calling is enabled
            self.model._function_manager = FunctionCallingManager()
            
        return self.model._function_manager.register_function(func, **kwargs)
    
    @classmethod
    def from_pretrained(
        cls,
        model_path,
        config=None,
        integration_preset=None,
        device=None,
        **kwargs
    ):
        """
        Create an enhanced inference engine from a pretrained model.
        
        Args:
            model_path: Path to the model
            config: Model configuration
            integration_preset: Integration preset to apply
            device: Device to run inference on
            **kwargs: Additional arguments
            
        Returns:
            Enhanced inference engine
        """
        # Import here to avoid circular imports
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Configure the model
        if config is None:
            # Create a default config
            config = {}
            
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(model_path, **config)
        
        # Create the inference engine
        return cls(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            integration_preset=integration_preset,
            **kwargs
        )
