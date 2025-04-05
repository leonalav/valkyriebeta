import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel training."""
    
    # Distributed training
    distributed_backend: str = "nccl"  # or "gloo" for CPU
    mixed_precision: bool = True
    fp16: bool = True
    bf16: bool = False
    
    # Model parallelism
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Optimization
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    
    # Checkpointing
    save_interval: int = 1000
    save_total_limit: Optional[int] = 5
    
    # Communication
    all_reduce_strategy: str = "hierarchical"  # or "ring"
    bucket_cap_mb: int = 25
    broadcast_buffers: bool = True
    
    # ZeRO optimization
    zero_stage: int = 0  # 0, 1, 2, or 3
    zero_contiguous_gradients: bool = True
    
    # Activation checkpointing
    checkpoint_activations: bool = False
    cpu_checkpointing: bool = False
    
    # CPU offloading
    cpu_offload: bool = False
    cpu_offload_params: bool = False
    cpu_offload_use_pin_memory: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.tensor_parallel_size > 1 and self.pipeline_parallel_size > 1:
            logger.warning(
                "Using both tensor and pipeline parallelism. "
                "This is an advanced configuration that may require custom code."
            )
        
        if self.bf16 and self.fp16:
            logger.warning("Both bf16 and fp16 are enabled. Using bf16 as it takes precedence.")
            self.fp16 = False
        
        if self.zero_stage > 0 and self.tensor_parallel_size > 1:
            logger.warning(
                "Using ZeRO with tensor parallelism may have unexpected behaviors. "
                "Consider using only one of these optimization techniques."
            )


class ParallelManager:
    """
    Manages various parallelism strategies for efficient training.
    
    This class handles distributed training setup, mixed precision,
    and optimization strategies like gradient checkpointing and ZeRO.
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize parallel manager.
        
        Args:
            config: Configuration for parallel training
        """
        self.config = config or ParallelConfig()
        self.initialized = False
        self.world_size = 1
        self.local_rank = 0
        self.global_rank = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize DeepSpeed if available
        self.deepspeed_available = False
        try:
            import deepspeed
            self.deepspeed_available = True
        except ImportError:
            logger.info("DeepSpeed not available. Some features will be disabled.")
    
    def initialize_distributed(self):
        """Initialize distributed training environment."""
        if self.initialized:
            return
        
        # Check if distributed environment variables are set
        if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.global_rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            # Initialize distributed process group
            dist.init_process_group(
                backend=self.config.distributed_backend,
                world_size=self.world_size,
                rank=self.global_rank
            )
            
            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f"cuda:{self.local_rank}")
                
                # Empty CUDA cache
                torch.cuda.empty_cache()
            
            logger.info(
                f"Initialized distributed training with "
                f"world_size={self.world_size}, "
                f"rank={self.global_rank}, "
                f"local_rank={self.local_rank}"
            )
        else:
            logger.info("No distributed environment detected. Running in single process mode.")
        
        self.initialized = True
    
    def setup_mixed_precision(self):
        """Setup mixed precision training."""
        if self.config.mixed_precision:
            if self.config.bf16 and torch.cuda.is_bf16_supported():
                logger.info("Using bfloat16 mixed precision training")
                return torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True)
            elif self.config.fp16:
                logger.info("Using float16 mixed precision training")
                return torch.cuda.amp.autocast(dtype=torch.float16, enabled=True)
            else:
                logger.info("Mixed precision enabled but no precision format specified")
                return torch.cuda.amp.autocast(enabled=True)
        else:
            logger.info("Using full precision training")
            return torch.cuda.amp.autocast(enabled=False)
    
    def get_grad_scaler(self):
        """Get gradient scaler for mixed precision training."""
        if self.config.mixed_precision and self.config.fp16:
            return torch.cuda.amp.GradScaler()
        return None
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Prepare model for distributed training.
        
        Args:
            model: PyTorch model
            
        Returns:
            Prepared model
        """
        if not self.initialized:
            self.initialize_distributed()
        
        # Move model to device
        model = model.to(self.device)
        
        # Apply gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            elif hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
                
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            else:
                logger.warning(
                    "Model doesn't support gradient checkpointing. "
                    "Continuing without it."
                )
        
        # Apply tensor parallelism if configured
        if self.config.tensor_parallel_size > 1:
            model = self._apply_tensor_parallelism(model)
        
        # Apply pipeline parallelism if configured
        if self.config.pipeline_parallel_size > 1:
            model = self._apply_pipeline_parallelism(model)
        
        # Wrap with DDP if multiple processes
        if self.world_size > 1 and not self._is_deepspeed_used():
            ddp_model = DistributedDataParallel(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                broadcast_buffers=self.config.broadcast_buffers,
                bucket_cap_mb=self.config.bucket_cap_mb,
                gradient_as_bucket_view=self.config.zero_contiguous_gradients,
                static_graph=False
            )
            return ddp_model
        
        return model
    
    def _apply_tensor_parallelism(self, model: nn.Module) -> nn.Module:
        """Apply tensor parallelism to model."""
        # This is a simplified implementation
        # Real tensor parallelism would require significant model restructuring
        logger.warning(
            "Basic tensor parallelism implementation. "
            "For advanced use, consider using a specialized library like Megatron-LM."
        )
        
        # For now, we'll just return the model and provide guidance
        if self.is_main_process():
            logger.info(
                "To implement true tensor parallelism, you would need to: \n"
                "1. Split model layers across GPUs \n"
                "2. Implement custom communication for forward/backward passes \n"
                "3. Handle optimizer state splitting \n"
                "Consider using Megatron-LM or DeepSpeed for this functionality."
            )
        
        return model
    
    def _apply_pipeline_parallelism(self, model: nn.Module) -> nn.Module:
        """Apply pipeline parallelism to model."""
        # This is a simplified implementation
        # Real pipeline parallelism would require significant model restructuring
        logger.warning(
            "Basic pipeline parallelism implementation. "
            "For advanced use, consider using a specialized library like DeepSpeed."
        )
        
        # For now, we'll just return the model and provide guidance
        if self.is_main_process():
            logger.info(
                "To implement true pipeline parallelism, you would need to: \n"
                "1. Split model into stages \n"
                "2. Implement microbatching \n"
                "3. Handle bubble overhead and communication \n"
                "Consider using DeepSpeed or PiPPy for this functionality."
            )
        
        return model
    
    def prepare_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        """
        Prepare optimizer for distributed training.
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            Prepared optimizer
        """
        # For basic DDP, no special handling is needed
        if self.config.zero_stage > 0 and self.deepspeed_available:
            logger.warning(
                "ZeRO optimization requires DeepSpeed integration. "
                "Please use DeepSpeed's initialize method instead."
            )
        
        return optimizer
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "avg") -> torch.Tensor:
        """
        All-reduce operation across processes.
        
        Args:
            tensor: Tensor to all-reduce
            op: Operation ("sum", "avg", "min", "max", "product")
            
        Returns:
            All-reduced tensor
        """
        if not self.initialized or self.world_size == 1:
            return tensor
        
        # Clone tensor to avoid modifying the original
        result = tensor.clone().detach()
        
        # Choose reduction operation
        if op == "sum":
            dist_op = dist.ReduceOp.SUM
            dist.all_reduce(result, dist_op)
        elif op == "avg":
            dist_op = dist.ReduceOp.SUM
            dist.all_reduce(result, dist_op)
            result = result / self.world_size
        elif op == "min":
            dist_op = dist.ReduceOp.MIN
            dist.all_reduce(result, dist_op)
        elif op == "max":
            dist_op = dist.ReduceOp.MAX
            dist.all_reduce(result, dist_op)
        elif op == "product":
            dist_op = dist.ReduceOp.PRODUCT
            dist.all_reduce(result, dist_op)
        else:
            raise ValueError(f"Unknown all_reduce operation: {op}")
        
        return result
    
    def all_gather(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        All-gather operation across processes.
        
        Args:
            tensor: Tensor to all-gather
            
        Returns:
            List of gathered tensors
        """
        if not self.initialized or self.world_size == 1:
            return [tensor]
        
        gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return gathered
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """
        Broadcast tensor from source process.
        
        Args:
            tensor: Tensor to broadcast
            src: Source process rank
            
        Returns:
            Broadcasted tensor
        """
        if not self.initialized or self.world_size == 1:
            return tensor
        
        dist.broadcast(tensor, src=src)
        return tensor
    
    def barrier(self):
        """Synchronize all processes."""
        if self.initialized and self.world_size > 1:
            dist.barrier()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.global_rank == 0
    
    def _is_deepspeed_used(self) -> bool:
        """Check if DeepSpeed is being used."""
        return self.deepspeed_available and self.config.zero_stage > 0
    
    def cleanup(self):
        """Clean up distributed environment."""
        if self.initialized and self.world_size > 1:
            dist.destroy_process_group()
            self.initialized = False


class ZeroRedundancyOptimizer(torch.optim.Optimizer):
    """
    Simple implementation of ZeRO Stage 1 with parameter partitioning.
    
    This optimizer partitions optimizer states across processes to reduce
    memory usage. It's a simplified version of DeepSpeed's ZeRO optimizer.
    """
    
    def __init__(self, params, optim_class, parallel_manager: ParallelManager, **kwargs):
        """
        Initialize ZeRO optimizer.
        
        Args:
            params: Model parameters
            optim_class: Base optimizer class
            parallel_manager: Parallel manager instance
            **kwargs: Optimizer arguments
        """
        self.parallel_manager = parallel_manager
        self.world_size = parallel_manager.world_size
        self.rank = parallel_manager.global_rank
        
        if self.world_size == 1:
            self.optim = optim_class(params, **kwargs)
            super().__init__(params, kwargs)
        else:
            # Group parameters by rank for partitioning
            param_groups = list(params)
            if len(param_groups) == 0:
                raise ValueError("No parameters given for optimization")
            
            # Flatten parameter groups if needed
            if not isinstance(param_groups[0], dict):
                param_groups = [{'params': param_groups}]
            
            # Partition parameters
            partitioned_params = self._partition_parameters(param_groups)
            
            # Create optimizer for this partition
            self.optim = optim_class(partitioned_params, **kwargs)
            
            # Initialize parent class with all parameters for proper tracking
            super().__init__(params, kwargs)
            
            logger.info(f"Rank {self.rank}: Initialized ZeRO optimizer with parameter partition")
    
    def _partition_parameters(self, param_groups):
        """Partition parameters across processes."""
        partitioned_groups = []
        
        for group in param_groups:
            params = list(group['params'])
            
            # Count total parameters
            total_params = len(params)
            
            # Calculate partition size
            params_per_rank = (total_params + self.world_size - 1) // self.world_size
            
            # Get this rank's partition
            start_idx = min(self.rank * params_per_rank, total_params)
            end_idx = min((self.rank + 1) * params_per_rank, total_params)
            
            # Create new group with partitioned parameters
            new_group = {k: v for k, v in group.items() if k != 'params'}
            new_group['params'] = params[start_idx:end_idx]
            
            partitioned_groups.append(new_group)
        
        return partitioned_groups
    
    def step(self, closure=None):
        """Perform optimization step with all-reduce for gradients."""
        # All-reduce gradients across all processes
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.parallel_manager.all_reduce(p.grad)
        
        # Perform optimizer step
        return self.optim.step(closure)
    
    def zero_grad(self, set_to_none=False):
        """Zero gradients."""
        self.optim.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self):
        """Get optimizer state dict."""
        # Use the base optimizer's state dict
        return self.optim.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        # Load into the base optimizer
        self.optim.load_state_dict(state_dict) 