import torch
import torch.nn as nn
import logging
import os
import gc
import psutil
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    
    # Memory management strategies
    optimize_memory_usage: bool = True
    gradient_checkpointing: bool = True
    cpu_offload: bool = False
    
    # Memory cleanup
    clear_keras_cache: bool = False
    clear_torch_cache: bool = True
    clear_jax_cache: bool = False
    gc_collect: bool = True
    
    # Fragmentation handling
    defragment_on_oom: bool = True
    release_memory_threshold: float = 0.8  # as fraction of total memory
    
    # Batching
    auto_adjust_batch_size: bool = False
    min_batch_size: int = 1
    max_batch_size: int = 32
    
    # Specific optimizations
    use_fp16: bool = True
    use_int8: bool = False
    use_4bit: bool = False
    
    # Flash attention settings
    use_flash_attention: bool = True
    flash_attention_block_size: int = 128
    
    # Sequence handling
    chunk_long_sequences: bool = True
    max_chunk_size: int = 2048
    
    def __post_init__(self):
        """Validate configuration."""
        if self.use_fp16 and self.use_int8:
            logger.warning("Both fp16 and int8 are enabled. This may lead to precision issues.")
            
        if self.use_4bit and (self.use_fp16 or self.use_int8):
            logger.warning("4-bit quantization is enabled alongside other precision formats.")
        
        if self.release_memory_threshold < 0 or self.release_memory_threshold > 1:
            logger.warning("release_memory_threshold should be between 0 and 1, setting to 0.8")
            self.release_memory_threshold = 0.8


class MemoryOptimizer:
    """
    Optimizes memory usage for training and inference.
    
    This class provides utilities for memory monitoring and optimization,
    including automatic batch size adjustment, memory defragmentation,
    and optimized sequence processing.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory optimizer.
        
        Args:
            config: Memory management configuration
        """
        self.config = config or MemoryConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = torch.cuda.is_available()
        
        # Memory usage tracking
        self.initial_cuda_memory = self._get_cuda_memory_used() if self.is_cuda else 0
        self.peak_cuda_memory = self.initial_cuda_memory
        self.last_memory_check = time.time()
        
        # Batch size tracking
        self.current_batch_size = self.config.max_batch_size
        
        if self.is_cuda:
            logger.info(f"CUDA memory initially allocated: {self.initial_cuda_memory / (1024**2):.2f} MB")
        
        logger.info(f"System memory available: {self._get_system_memory_available() / (1024**3):.2f} GB")
    
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """
        Apply memory optimizations to a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Optimized model
        """
        if not self.config.optimize_memory_usage:
            return model
        
        if self.config.gradient_checkpointing:
            model = self._apply_gradient_checkpointing(model)
        
        # Convert to half precision if requested
        if self.config.use_fp16 and self.is_cuda:
            model = model.half()
            logger.info("Converted model to half precision (FP16)")
        
        # Apply quantization if requested
        if self.config.use_int8:
            model = self._apply_int8_quantization(model)
        
        if self.config.use_4bit:
            model = self._apply_4bit_quantization(model)
        
        # Apply flash attention if available
        if self.config.use_flash_attention and self.is_cuda:
            model = self._apply_flash_attention(model)
        
        return model
    
    def _apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to model."""
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        else:
            logger.warning("Model doesn't support gradient_checkpointing_enable method")
        
        return model
    
    def _apply_int8_quantization(self, model: nn.Module) -> nn.Module:
        """Apply int8 quantization to model."""
        try:
            import torch.quantization as quant
            
            # Configure quantization
            model.eval()
            model.qconfig = quant.get_default_qconfig('fbgemm')
            
            # Prepare and quantize
            model_prepared = quant.prepare(model)
            model_quantized = quant.convert(model_prepared)
            
            logger.info("Applied INT8 quantization to model")
            return model_quantized
        except Exception as e:
            logger.warning(f"Failed to apply INT8 quantization: {e}")
            return model
    
    def _apply_4bit_quantization(self, model: nn.Module) -> nn.Module:
        """Apply 4-bit quantization to model."""
        try:
            import bitsandbytes as bnb
            
            # Convert linear layers to 4-bit
            for module_name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    module_path = module_name.split('.')
                    parent = model
                    for name in module_path[:-1]:
                        parent = getattr(parent, name)
                    
                    # Replace with 4-bit equivalent
                    in_features = module.in_features
                    out_features = module.out_features
                    bias = module.bias is not None
                    
                    # Create 4-bit module
                    quantized_module = bnb.nn.Linear4bit(
                        in_features, 
                        out_features, 
                        bias=bias
                    )
                    
                    # Copy weights
                    with torch.no_grad():
                        # Copy bias if exists
                        if bias:
                            quantized_module.bias.copy_(module.bias)
                    
                    # Replace module
                    setattr(parent, module_path[-1], quantized_module)
            
            logger.info("Applied 4-bit quantization to model")
            return model
        except Exception as e:
            logger.warning(f"Failed to apply 4-bit quantization: {e}")
            return model
    
    def _apply_flash_attention(self, model: nn.Module) -> nn.Module:
        """Apply flash attention to model if available."""
        try:
            # Check if we're using a transformer model
            if hasattr(model, "config") and hasattr(model.config, "attention_implementation"):
                model.config.attention_implementation = "flash_attention_2"
                logger.info("Enabled Flash Attention 2 for model")
            else:
                logger.warning("Model doesn't support Flash Attention configuration")
        except Exception as e:
            logger.warning(f"Failed to apply Flash Attention: {e}")
        
        return model
    
    def optimize_inference_memory(self, 
                                 model: nn.Module, 
                                 use_cache: bool = True,
                                 cpu_offload: Optional[bool] = None) -> nn.Module:
        """
        Optimize model for inference to reduce memory usage.
        
        Args:
            model: PyTorch model
            use_cache: Whether to use KV cache for generation
            cpu_offload: Whether to offload weights to CPU
            
        Returns:
            Optimized model
        """
        # Apply general memory optimizations first
        model = self.optimize_model_memory(model)
        
        # Put model in eval mode
        model.eval()
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
        
        # Configure model for inference
        if hasattr(model, "config"):
            # Set use_cache if available
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = use_cache
                logger.info(f"Set use_cache={use_cache} for inference")
        
        # Apply CPU offloading if requested
        cpu_offload = self.config.cpu_offload if cpu_offload is None else cpu_offload
        if cpu_offload:
            try:
                from accelerate import cpu_offload
                
                # Move model to CPU and offload to GPU only during forward pass
                model = model.to("cpu")
                cpu_offload(model, self.device)
                logger.info("Applied CPU offloading for inference")
            except ImportError:
                logger.warning("accelerate not found, CPU offloading not applied")
        
        return model
    
    def process_long_sequence(self, 
                             model_func: Callable, 
                             input_ids: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None,
                             **kwargs) -> torch.Tensor:
        """
        Process long sequences by chunking if needed.
        
        Args:
            model_func: Function to call with chunked inputs
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments for model_func
            
        Returns:
            Processed outputs
        """
        if not self.config.chunk_long_sequences:
            # Process normally without chunking
            return model_func(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        batch_size, seq_len = input_ids.shape
        
        # If sequence is short enough, process normally
        if seq_len <= self.config.max_chunk_size:
            return model_func(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        logger.info(f"Processing long sequence of length {seq_len} with chunking")
        
        # Determine chunk size and overlap
        chunk_size = self.config.max_chunk_size
        overlap = chunk_size // 10  # 10% overlap
        
        outputs = []
        
        # Process each sequence in the batch
        for b in range(batch_size):
            sequence_outputs = []
            
            # Process chunks with overlap
            for chunk_start in range(0, seq_len, chunk_size - overlap):
                chunk_end = min(chunk_start + chunk_size, seq_len)
                
                # Extract chunk
                chunk_input_ids = input_ids[b:b+1, chunk_start:chunk_end]
                
                # Extract chunk attention mask if provided
                chunk_attention_mask = None
                if attention_mask is not None:
                    chunk_attention_mask = attention_mask[b:b+1, chunk_start:chunk_end]
                
                # Process chunk
                with torch.no_grad():
                    chunk_output = model_func(
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                        **kwargs
                    )
                
                # Store chunk output
                sequence_outputs.append(chunk_output)
                
                # Free memory
                del chunk_input_ids, chunk_attention_mask
                if self.config.clear_torch_cache and self.is_cuda:
                    torch.cuda.empty_cache()
            
            # Combine chunks (implementation depends on model output format)
            # For simplicity, we'll assume the output is a tensor
            combined_output = self._combine_chunk_outputs(sequence_outputs, overlap)
            outputs.append(combined_output)
        
        # Combine batch outputs
        combined_batch_output = torch.cat(outputs, dim=0)
        return combined_batch_output
    
    def _combine_chunk_outputs(self, 
                              chunk_outputs: List[torch.Tensor], 
                              overlap: int) -> torch.Tensor:
        """
        Combine chunked outputs with overlap.
        
        Args:
            chunk_outputs: List of chunk outputs
            overlap: Overlap size between chunks
            
        Returns:
            Combined output
        """
        # This is a simplistic implementation and might need to be
        # adapted based on the actual output format
        if not chunk_outputs:
            return torch.tensor([])
        
        # If only one chunk, return it
        if len(chunk_outputs) == 1:
            return chunk_outputs[0]
        
        combined_outputs = []
        
        for i, output in enumerate(chunk_outputs):
            if i == 0:
                # For first chunk, keep everything
                combined_outputs.append(output[:, :-overlap] if overlap > 0 else output)
            elif i == len(chunk_outputs) - 1:
                # For last chunk, keep everything after the overlap
                combined_outputs.append(output[:, overlap:] if overlap > 0 else output)
            else:
                # For middle chunks, remove overlap from both sides
                combined_outputs.append(output[:, overlap:-overlap] if overlap > 0 else output)
        
        # Concatenate along sequence dimension
        return torch.cat(combined_outputs, dim=1)
    
    def optimize_batch_size(self, 
                           current_batch_size: int,
                           succeeded: bool) -> int:
        """
        Optimize batch size based on success/failure and memory usage.
        
        Args:
            current_batch_size: Current batch size
            succeeded: Whether current batch processing succeeded
            
        Returns:
            New recommended batch size
        """
        if not self.config.auto_adjust_batch_size:
            return current_batch_size
        
        # Update tracking
        self.current_batch_size = current_batch_size
        
        # Check if we need to adjust batch size
        if not succeeded:
            # Processing failed, reduce batch size
            new_batch_size = max(self.config.min_batch_size, current_batch_size // 2)
            logger.info(f"Batch processing failed, reducing batch size from {current_batch_size} to {new_batch_size}")
            return new_batch_size
        
        # Check current memory usage
        if self.is_cuda:
            memory_used = self._get_cuda_memory_used()
            total_memory = self._get_cuda_memory_total()
            memory_utilization = memory_used / total_memory
            
            # Update peak memory usage
            if memory_used > self.peak_cuda_memory:
                self.peak_cuda_memory = memory_used
            
            # If memory usage is high, reduce batch size
            if memory_utilization > 0.9:  # More than 90% used
                new_batch_size = max(self.config.min_batch_size, current_batch_size - 1)
                logger.info(f"High memory utilization ({memory_utilization:.2%}), reducing batch size from {current_batch_size} to {new_batch_size}")
                return new_batch_size
            
            # If memory usage is low, potentially increase batch size
            elif memory_utilization < 0.7 and current_batch_size < self.config.max_batch_size:  # Less than 70% used
                # Only try increasing every once in a while
                time_since_last_check = time.time() - self.last_memory_check
                if time_since_last_check > 60:  # 1 minute
                    self.last_memory_check = time.time()
                    new_batch_size = min(self.config.max_batch_size, current_batch_size + 1)
                    logger.info(f"Low memory utilization ({memory_utilization:.2%}), increasing batch size from {current_batch_size} to {new_batch_size}")
                    return new_batch_size
        
        # No change needed
        return current_batch_size
    
    def release_memory(self, full_cleanup: bool = False):
        """
        Release unused memory.
        
        Args:
            full_cleanup: Whether to perform a full cleanup
        """
        # Only check memory usage every few seconds to avoid overhead
        time_since_last_check = time.time() - self.last_memory_check
        if not full_cleanup and time_since_last_check < 5:  # Less than 5 seconds
            return
        
        self.last_memory_check = time.time()
        
        # Check if memory usage is above threshold
        if self.is_cuda:
            memory_used = self._get_cuda_memory_used()
            total_memory = self._get_cuda_memory_total()
            memory_utilization = memory_used / total_memory
            
            if memory_utilization < self.config.release_memory_threshold and not full_cleanup:
                # Memory usage is below threshold, no need to release
                return
            
            logger.info(f"Releasing memory (utilization: {memory_utilization:.2%})")
        else:
            logger.info("Releasing memory")
        
        if self.config.clear_torch_cache and self.is_cuda:
            torch.cuda.empty_cache()
            logger.debug("Cleared PyTorch CUDA cache")
        
        if self.config.clear_keras_cache:
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                logger.debug("Cleared Keras session")
            except ImportError:
                pass
        
        if self.config.clear_jax_cache:
            try:
                from jax.lib import xla_bridge
                xla_bridge.get_backend().clear_cache()
                logger.debug("Cleared JAX cache")
            except ImportError:
                pass
        
        if self.config.gc_collect:
            gc.collect()
            logger.debug("Ran garbage collection")
        
        # Defragment if OOM issues are likely
        if full_cleanup and self.config.defragment_on_oom and self.is_cuda:
            self._defragment_cuda_memory()
    
    def _defragment_cuda_memory(self):
        """Defragment CUDA memory by allocating and freeing a large tensor."""
        if not self.is_cuda:
            return
        
        # Estimate free memory by getting total - used
        free_memory = self._get_cuda_memory_total() - self._get_cuda_memory_used()
        
        # Use 90% of free memory for defragmentation
        size_to_allocate = int(0.9 * free_memory / 4)  # 4 bytes per float
        
        if size_to_allocate <= 0:
            logger.debug("Not enough free memory to defragment")
            return
        
        try:
            logger.info(f"Defragmenting CUDA memory with {size_to_allocate / (1024**2):.2f} MB tensor")
            # Allocate a large tensor to defragment memory
            x = torch.empty(size_to_allocate, device=self.device)
            del x
            torch.cuda.empty_cache()
            logger.debug("Completed CUDA memory defragmentation")
        except Exception as e:
            logger.warning(f"Failed to defragment CUDA memory: {e}")
    
    def _get_cuda_memory_used(self) -> int:
        """Get CUDA memory used in bytes."""
        if not self.is_cuda:
            return 0
        
        return torch.cuda.memory_allocated()
    
    def _get_cuda_memory_total(self) -> int:
        """Get total CUDA memory in bytes."""
        if not self.is_cuda:
            return 0
        
        device = torch.cuda.current_device()
        return torch.cuda.get_device_properties(device).total_memory
    
    def _get_system_memory_available(self) -> int:
        """Get system memory available in bytes."""
        return psutil.virtual_memory().available
    
    def log_memory_stats(self):
        """Log current memory statistics."""
        if self.is_cuda:
            used_memory = self._get_cuda_memory_used()
            total_memory = self._get_cuda_memory_total()
            logger.info(f"CUDA memory: {used_memory / (1024**2):.2f} MB used out of {total_memory / (1024**2):.2f} MB")
            logger.info(f"CUDA memory peak: {self.peak_cuda_memory / (1024**2):.2f} MB")
        
        system_memory = psutil.virtual_memory()
        logger.info(f"System memory: {system_memory.used / (1024**3):.2f} GB used out of {system_memory.total / (1024**3):.2f} GB")
        logger.info(f"System memory available: {system_memory.available / (1024**3):.2f} GB")


def optimize_transformer_memory(model: nn.Module) -> nn.Module:
    """
    Optimize transformer memory usage with common techniques.
    
    Args:
        model: PyTorch transformer model
        
    Returns:
        Optimized model
    """
    # Enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    # Enable attention implementation optimizations if available
    if hasattr(model, "config"):
        if hasattr(model.config, "attn_implementation"):
            try:
                model.config.attn_implementation = "flash_attention_2"
            except Exception:
                # Flash attention not available
                pass
        
        # Enable memory efficient attention if available
        if hasattr(model.config, "use_memory_efficient_attention"):
            model.config.use_memory_efficient_attention = True
        
        # Enable memory efficient attention if available
        if hasattr(model.config, "use_cache"):
            # Disable KV cache during training
            model.config.use_cache = False
    
    return model


def memory_efficient_inference(model_callable: Callable, *args, **kwargs) -> Any:
    """
    Run inference in a memory-efficient way.
    
    Args:
        model_callable: Function that runs the model
        *args: Arguments to pass to model_callable
        **kwargs: Keyword arguments to pass to model_callable
        
    Returns:
        Model outputs
    """
    # Clear memory before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Run inference
        with torch.no_grad():
            outputs = model_callable(*args, **kwargs)
        
        # Clear memory after inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return outputs
    except RuntimeError as e:
        # Check if it's an OOM error
        if "CUDA out of memory" in str(e):
            logger.warning("CUDA OOM during inference, trying aggressive memory cleanup")
            
            # Aggressive cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Try again with no_grad
            with torch.no_grad():
                outputs = model_callable(*args, **kwargs)
            
            return outputs
        else:
            # Re-raise other errors
            raise 