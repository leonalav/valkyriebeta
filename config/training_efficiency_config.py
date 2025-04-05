from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple, Union
import logging
import os
import torch

logger = logging.getLogger(__name__)

@dataclass
class TrainingEfficiencyConfig:
    """Configuration for training optimizations"""
    use_mixed_precision: bool = True
    optimize_cuda_kernels: bool = True
    optimize_grouping: bool = True
    compile_model: bool = False  # PyTorch 2.0+ feature
    dynamo_backend: Optional[str] = None
    use_fused_adam: bool = True
    use_fused_layer_norm: bool = True
    
    # Advanced efficiency options
    activation_checkpointing: bool = True
    checkpoint_every_n_layers: int = 2
    use_sharded_ddp: bool = False
    use_fsdp: bool = False
    use_offload: bool = False
    use_cpu_offload: bool = False
    gradient_accumulation_steps: int = 1
    
    # Precision settings
    mixed_precision: str = "bf16"  # ["no", "fp16", "bf16"]
    
    # Distributed training
    distributed_training: bool = False
    distributed_backend: str = "nccl"  # ["nccl", "gloo", "mpi"]
    distributed_world_size: int = 1
    distributed_rank: int = 0
    distributed_local_rank: int = 0
    distributed_init_method: Optional[str] = None
    
    # Parallelism strategies
    use_model_parallelism: bool = False
    model_parallel_size: int = 1
    use_pipeline_parallelism: bool = False
    pipeline_parallel_size: int = 1
    pipeline_chunks: int = 1
    use_tensor_parallelism: bool = False
    tensor_parallel_size: int = 1
    
    # Optimization techniques
    use_fused_optimizers: bool = True
    use_compile: bool = False  # torch.compile
    compile_mode: str = "default"  # ["default", "reduce-overhead", "max-autotune"]
    
    # Batch optimization
    dynamic_batch_size: bool = False
    min_batch_size: int = 1
    max_batch_size: int = 32
    target_batch_size: int = 32
    
    # CPU efficiency
    num_workers: int = 4
    pin_memory: bool = True
    
    # Profiling
    enable_profiling: bool = False
    profile_steps: int = 100
    profile_memory: bool = False
    
    def __post_init__(self):
        """Initialize configuration based on available resources"""
        # Auto-detect number of workers if set to 0
        if self.num_workers == 0:
            import multiprocessing
            self.num_workers = max(1, multiprocessing.cpu_count() // 2)
            logger.info(f"Auto-detected num_workers: {self.num_workers}")
            
        # Auto-detect distributed settings if using distributed training
        if self.distributed_training:
            # Try to get distributed settings from environment variables
            if "WORLD_SIZE" in os.environ:
                self.distributed_world_size = int(os.environ["WORLD_SIZE"])
            if "RANK" in os.environ:
                self.distributed_rank = int(os.environ["RANK"])
            if "LOCAL_RANK" in os.environ:
                self.distributed_local_rank = int(os.environ["LOCAL_RANK"])
                
            # Set default init method if not provided
            if self.distributed_init_method is None:
                if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
                    addr = os.environ["MASTER_ADDR"]
                    port = os.environ["MASTER_PORT"]
                    self.distributed_init_method = f"tcp://{addr}:{port}"
                else:
                    self.distributed_init_method = "env://"
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate training efficiency configuration"""
        valid = True
        errors = []
        
        # Validate precision settings
        valid_precision_types = ["no", "fp16", "bf16"]
        if self.mixed_precision not in valid_precision_types:
            valid = False
            errors.append(f"Mixed precision must be one of {valid_precision_types}, got {self.mixed_precision}")
            
        # Check if bf16 is supported
        if self.mixed_precision == "bf16" and not self._is_bf16_supported():
            valid = False
            errors.append("BF16 precision is enabled but not supported on this hardware")
            
        # Validate gradient accumulation
        if self.gradient_accumulation_steps < 1:
            valid = False
            errors.append(f"Gradient accumulation steps must be at least 1, got {self.gradient_accumulation_steps}")
            
        # Validate distributed settings
        if self.distributed_training:
            if self.distributed_world_size < 1:
                valid = False
                errors.append(f"Distributed world size must be at least 1, got {self.distributed_world_size}")
                
            if self.distributed_rank < 0 or self.distributed_rank >= self.distributed_world_size:
                valid = False
                errors.append(f"Distributed rank ({self.distributed_rank}) must be between 0 and world_size-1 ({self.distributed_world_size-1})")
                
            valid_backends = ["nccl", "gloo", "mpi"]
            if self.distributed_backend not in valid_backends:
                valid = False
                errors.append(f"Distributed backend must be one of {valid_backends}, got {self.distributed_backend}")
                
            # Check if NCCL is available when using it
            if self.distributed_backend == "nccl" and not torch.cuda.is_available():
                valid = False
                errors.append("NCCL backend requires CUDA, but CUDA is not available")
                
        # Validate parallelism settings
        if self.use_model_parallelism and self.model_parallel_size < 2:
            valid = False
            errors.append(f"Model parallel size must be at least 2 when model parallelism is enabled, got {self.model_parallel_size}")
            
        if self.use_pipeline_parallelism and self.pipeline_parallel_size < 2:
            valid = False
            errors.append(f"Pipeline parallel size must be at least 2 when pipeline parallelism is enabled, got {self.pipeline_parallel_size}")
            
        if self.use_tensor_parallelism and self.tensor_parallel_size < 2:
            valid = False
            errors.append(f"Tensor parallel size must be at least 2 when tensor parallelism is enabled, got {self.tensor_parallel_size}")
            
        # Validate batch size settings
        if self.dynamic_batch_size:
            if self.min_batch_size < 1:
                valid = False
                errors.append(f"Minimum batch size must be at least 1, got {self.min_batch_size}")
                
            if self.max_batch_size < self.min_batch_size:
                valid = False
                errors.append(f"Maximum batch size ({self.max_batch_size}) must be greater than or equal to minimum batch size ({self.min_batch_size})")
                
            if self.target_batch_size < self.min_batch_size or self.target_batch_size > self.max_batch_size:
                valid = False
                errors.append(f"Target batch size ({self.target_batch_size}) must be between min ({self.min_batch_size}) and max ({self.max_batch_size})")
                
        # Validate worker settings
        if self.num_workers < 0:
            valid = False
            errors.append(f"Number of workers must be non-negative, got {self.num_workers}")
            
        # Validate compilation settings
        if self.use_compile:
            if not self._is_compile_supported():
                valid = False
                errors.append("torch.compile is enabled but not supported in this PyTorch version")
                
            valid_compile_modes = ["default", "reduce-overhead", "max-autotune"]
            if self.compile_mode not in valid_compile_modes:
                valid = False
                errors.append(f"Compile mode must be one of {valid_compile_modes}, got {self.compile_mode}")
                
        return valid, errors
    
    def _is_bf16_supported(self) -> bool:
        """Check if BF16 precision is supported"""
        if not torch.cuda.is_available():
            return False
            
        try:
            # Check for NVIDIA Ampere or newer architecture (SM >= 8.0)
            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)
            major, minor = capability
            
            # Ampere or newer supports BF16
            if major >= 8:
                return True
                
            # Check for CPU support (Intel CPUs with AVX512_BF16 instruction)
            import torch.cpu
            if hasattr(torch.cpu, 'has_bf16') and torch.cpu.has_bf16():
                return True
                
            return False
        except Exception as e:
            logger.warning(f"Failed to check BF16 support: {e}")
            return False
    
    def _is_compile_supported(self) -> bool:
        """Check if torch.compile is supported"""
        try:
            return hasattr(torch, 'compile')
        except Exception:
            return False
    
    def setup_distributed(self) -> None:
        """Set up distributed training"""
        if not self.distributed_training:
            logger.info("Distributed training is disabled")
            return
            
        try:
            if not torch.distributed.is_initialized():
                logger.info(f"Initializing distributed training (backend={self.distributed_backend}, world_size={self.distributed_world_size}, rank={self.distributed_rank})")
                
                # Initialize process group
                torch.distributed.init_process_group(
                    backend=self.distributed_backend,
                    init_method=self.distributed_init_method,
                    world_size=self.distributed_world_size,
                    rank=self.distributed_rank
                )
                
                # Set device for this process
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.distributed_local_rank)
                    
                logger.info(f"Distributed training initialized (rank={torch.distributed.get_rank()}, world_size={torch.distributed.get_world_size()})")
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    def cleanup_distributed(self) -> None:
        """Clean up distributed training resources"""
        if self.distributed_training and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
                logger.info("Destroyed distributed process group")
            except Exception as e:
                logger.warning(f"Failed to destroy distributed process group: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingEfficiencyConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    def __str__(self):
        return str(asdict(self)) 