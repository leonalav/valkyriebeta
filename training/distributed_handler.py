import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload
)
from typing import Optional, Dict, Any, Tuple, Callable
import logging
from dataclasses import dataclass
import os
import torch.multiprocessing as mp
import datetime

@dataclass
class DistributedConfig:
    """Configuration for distributed training"""
    world_size: int
    rank: int
    local_rank: int
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "12355"
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_mixed_precision: bool = True
    fsdp_cpu_offload: bool = False

class DistributedHandler:
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def setup(self):
        """Initialize distributed training environment"""
        try:
            torch.cuda.set_device(self.config.local_rank)
            dist.init_process_group(
                backend=self.config.backend,
                init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            self.logger.info(f"Initialized process group: rank {self.config.rank}/{self.config.world_size}")
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed environment: {e}")
            raise

    def cleanup(self):
        """Cleanup distributed training resources"""
        if dist.is_initialized():
            dist.destroy_process_group()

    def sync_model(self, model: torch.nn.Module):
        """Synchronize model parameters across processes"""
        if self.config.use_fsdp and isinstance(model, FSDP):
            # FSDP handles synchronization internally
            return
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    def wrap_model_with_fsdp(
        self,
        model: torch.nn.Module,
        sharding_strategy: Optional[str] = None,
        mixed_precision: Optional[bool] = None,
        cpu_offload: Optional[bool] = None
    ) -> torch.nn.Module:
        """Wrap model with FSDP for memory-efficient training"""
        if not self.config.use_fsdp:
            return model
            
        # Set FSDP parameters
        sharding_strategy = sharding_strategy or self.config.fsdp_sharding_strategy
        mixed_precision = mixed_precision if mixed_precision is not None else self.config.fsdp_mixed_precision
        cpu_offload = cpu_offload if cpu_offload is not None else self.config.fsdp_cpu_offload
        
        # Convert strategy string to enum
        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD
        }
        
        try:
            strategy = strategy_map[sharding_strategy]
        except KeyError:
            raise ValueError(f"Invalid sharding strategy: {sharding_strategy}")
            
        # Configure mixed precision
        mp_policy = None
        if mixed_precision:
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
            
        # Configure CPU offload
        offload_policy = CPUOffload(enabled=cpu_offload)
        
        # Wrap model with FSDP
        model = FSDP(
            model,
            sharding_strategy=strategy,
            mixed_precision=mp_policy,
            cpu_offload=offload_policy,
            device_id=torch.cuda.current_device()
        )
        
        self.logger.info(f"Wrapped model with FSDP (strategy={sharding_strategy}, "
                        f"mixed_precision={mixed_precision}, cpu_offload={cpu_offload})")
        return model
            
    def reduce_metrics(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reduce metrics across all processes"""
        reduced_metrics = {}
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                reduced_metrics[name] = value / self.config.world_size
        return reduced_metrics

    @property
    def is_main_process(self) -> bool:
        """Check if current process is the main process"""
        return self.config.rank == 0

def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "12355",
    use_fsdp: bool = False,
    fsdp_sharding_strategy: str = "FULL_SHARD",
    fsdp_mixed_precision: bool = True,
    fsdp_cpu_offload: bool = False,
    timeout: int = 1800
) -> Tuple[int, int]:
    """
    Set up distributed training with enhanced error handling and logging.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Distributed backend ('nccl', 'gloo', etc.)
        master_addr: Master address for distributed coordination
        master_port: Master port for distributed coordination
        use_fsdp: Whether to use Fully Sharded Data Parallel
        fsdp_sharding_strategy: FSDP sharding strategy
        fsdp_mixed_precision: Whether to use mixed precision with FSDP
        fsdp_cpu_offload: Whether to enable CPU offload with FSDP
        timeout: Timeout in seconds for process group initialization
        
    Returns:
        rank: Local process rank
        world_size: Total number of processes
        
    Raises:
        RuntimeError: If distributed initialization fails
        ValueError: For invalid configurations
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Validate environment variables
        if not master_addr or not master_port:
            raise ValueError("Master address and port must be specified")
            
        if not master_port.isdigit():
            raise ValueError("Master port must be a valid number")
            
        # Set environment variables
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        
        # Initialize process group with timeout
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=timeout)
        )
        
        # Configure device
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        else:
            logger.warning("CUDA not available, running on CPU")
            
        if use_fsdp:
            # Validate FSDP configuration
            if backend != "nccl":
                raise ValueError("FSDP requires NCCL backend")
            if not torch.cuda.is_available():
                raise RuntimeError("FSDP requires CUDA")
                
            logger.info(f"Initialized process {rank}/{world_size} with FSDP support")
        else:
            logger.info(f"Initialized process {rank}/{world_size}")
            
        return rank, world_size
        
    except Exception as e:
        logger.error(f"Failed to initialize distributed environment: {e}")
        raise RuntimeError(f"Distributed setup failed: {e}") from e

def cleanup_distributed():
    """Clean up distributed training resources"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Destroyed process group")

def run_distributed(
    fn: Callable,
    world_size: int,
    args: Tuple = (),
    kwargs: Dict[str, Any] = None,
    backend: str = "nccl",
    use_fsdp: bool = False,
    fsdp_sharding_strategy: str = "FULL_SHARD",
    fsdp_mixed_precision: bool = True,
    fsdp_cpu_offload: bool = False
):
    """
    Run a function in a distributed setting.
    
    Args:
        fn: Function to run
        world_size: Total number of processes
        args: Args to pass to the function
        kwargs: Kwargs to pass to the function
        backend: Distributed backend
    """
    kwargs = kwargs or {}
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        backend = "gloo"
    
    # Define launcher function
    def _launcher(rank, world_size, args, kwargs):
        # Set up distributed
        rank, world_size = setup_distributed(
            rank,
            world_size,
            backend,
            use_fsdp=use_fsdp,
            fsdp_sharding_strategy=fsdp_sharding_strategy,
            fsdp_mixed_precision=fsdp_mixed_precision,
            fsdp_cpu_offload=fsdp_cpu_offload
        )
        
        try:
            # Run function
            fn(rank, world_size, *args, **{**kwargs, "rank": rank, "world_size": world_size})
        except Exception as e:
            print(f"Error in process {rank}: {e}")
            raise
        finally:
            # Clean up
            cleanup_distributed()
    
    # Launch processes
    if world_size > 1:
        mp.spawn(
            _launcher,
            args=(world_size, args, kwargs),
            nprocs=world_size,
            join=True
        )
    else:
        # Single process
        _launcher(0, 1, args, kwargs)

def distributed_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensor from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        Gathered tensor
    """
    if not dist.is_initialized():
        return tensor
        
    # Get world size
    world_size = dist.get_world_size()
    
    # Create output list
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Gather
    dist.all_gather(gathered, tensor)
    
    # Concatenate
    return torch.cat(gathered, dim=0)

def distributed_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source rank to all processes.
    
    Args:
        tensor: Tensor to broadcast
        src: Source rank
        
    Returns:
        Broadcasted tensor
    """
    if not dist.is_initialized():
        return tensor
        
    # Broadcast
    dist.broadcast(tensor, src=src)
    
    return tensor

def distributed_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation ("sum", "avg", "min", "max", "product")
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
        
    # Get reduce operation
    if op == "sum":
        reduce_op = dist.ReduceOp.SUM
    elif op == "avg":
        reduce_op = dist.ReduceOp.SUM
    elif op == "min":
        reduce_op = dist.ReduceOp.MIN
    elif op == "max":
        reduce_op = dist.ReduceOp.MAX
    elif op == "product":
        reduce_op = dist.ReduceOp.PRODUCT
    else:
        raise ValueError(f"Unsupported reduce operation: {op}")
    
    # Create output tensor
    output = torch.zeros_like(tensor)
    
    # Reduce
    dist.all_reduce(tensor, op=reduce_op, out=output)
    
    # Adjust for average
    if op == "avg":
        output = output / dist.get_world_size()
    
    return output
