from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
import logging
import os
import psutil
import torch

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for memory optimizations"""
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    activation_checkpointing: bool = True
    optimize_memory_use: bool = True
    mem_efficient_linear: bool = True
    cpu_offload: bool = False
    low_cpu_mem_usage: bool = True
    max_memory_MB: Optional[int] = None
    
    # Advanced memory mechanisms
    use_episodic_memory: bool = True
    episodic_memory_size: int = 1024
    use_working_memory: bool = True
    working_memory_size: int = 512
    use_long_term_memory: bool = True
    long_term_memory_size: int = 4096
    use_memory_router: bool = True
    memory_update_frequency: int = 10
    
    def __str__(self):
        return str(asdict(self))

    def __post_init__(self):
        """Initialize memory limits if auto-detect is enabled"""
        if self.max_memory_MB == None and torch.cuda.is_available():
            # Auto-detect GPU memory
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                self.max_memory_MB = int(gpu_memory / (1024 * 1024) * (1 - self.memory_safety_margin))
                logger.info(f"Auto-detected GPU memory: {self.max_memory_MB}MB")
            except Exception as e:
                logger.warning(f"Failed to auto-detect GPU memory: {e}")
                self.max_memory_MB = 12000  # Default to 12GB
                
        if self.max_cpu_memory_MB == 0:
            # Auto-detect CPU memory
            try:
                cpu_memory = psutil.virtual_memory().total
                self.max_cpu_memory_MB = int(cpu_memory / (1024 * 1024) * (1 - self.memory_safety_margin))
                logger.info(f"Auto-detected CPU memory: {self.max_cpu_memory_MB}MB")
            except Exception as e:
                logger.warning(f"Failed to auto-detect CPU memory: {e}")
                self.max_cpu_memory_MB = 16000  # Default to 16GB
                
        # Create offload directory if needed
        if self.use_disk_offloading and not os.path.exists(self.offload_dir):
            try:
                os.makedirs(self.offload_dir, exist_ok=True)
                logger.info(f"Created offload directory: {self.offload_dir}")
            except Exception as e:
                logger.error(f"Failed to create offload directory: {e}")
                self.use_disk_offloading = False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate memory configuration"""
        valid = True
        errors = []
        
        # Check for conflicting settings
        if self.use_4bit_quantization and self.use_8bit_quantization:
            valid = False
            errors.append("Cannot use both 4-bit and 8-bit quantization simultaneously")
            
        if self.use_flash_attention and self.use_xformers:
            valid = False
            errors.append("Cannot use both FlashAttention and xFormers simultaneously")
            
        # Check for resource availability
        if self.use_flash_attention and not self._is_flash_attention_available():
            valid = False
            errors.append("FlashAttention is enabled but not available in this environment")
            
        if self.use_xformers and not self._is_xformers_available():
            valid = False
            errors.append("xFormers is enabled but not available in this environment")
            
        # Check for reasonable chunk sizes
        if self.enable_chunking:
            if self.chunk_size <= 0:
                valid = False
                errors.append(f"Chunk size must be positive, got {self.chunk_size}")
                
            if self.overlap_size >= self.chunk_size:
                valid = False
                errors.append(f"Overlap size ({self.overlap_size}) must be less than chunk size ({self.chunk_size})")
                
        # Check for disk space if using disk offloading
        if self.use_disk_offloading:
            try:
                free_disk_space = psutil.disk_usage(self.offload_dir).free
                if free_disk_space < 10 * 1024 * 1024 * 1024:  # 10GB
                    valid = False
                    errors.append(f"Insufficient disk space for offloading: {free_disk_space / (1024**3):.2f}GB available")
            except Exception as e:
                logger.warning(f"Failed to check disk space: {e}")
                
        return valid, errors
    
    def _is_flash_attention_available(self) -> bool:
        """Check if FlashAttention is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
            
    def _is_xformers_available(self) -> bool:
        """Check if xFormers is available"""
        try:
            import xformers
            return True
        except ImportError:
            return False
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory usage status"""
        status = {
            "limits": {
                "gpu_memory_MB": self.max_memory_MB,
                "cpu_memory_MB": self.max_cpu_memory_MB,
            },
            "current_usage": {}
        }
        
        # Get GPU memory usage
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                    allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                    status["current_usage"][f"gpu_{i}_reserved_MB"] = reserved
                    status["current_usage"][f"gpu_{i}_allocated_MB"] = allocated
            except Exception as e:
                logger.warning(f"Failed to get GPU memory usage: {e}")
                
        # Get CPU memory usage
        try:
            vm = psutil.virtual_memory()
            status["current_usage"]["cpu_total_MB"] = vm.total / (1024 * 1024)
            status["current_usage"]["cpu_available_MB"] = vm.available / (1024 * 1024)
            status["current_usage"]["cpu_used_MB"] = vm.used / (1024 * 1024)
            status["current_usage"]["cpu_percent"] = vm.percent
        except Exception as e:
            logger.warning(f"Failed to get CPU memory usage: {e}")
            
        return status
    
    def cleanup_memory(self, force: bool = False) -> None:
        """Clean up memory"""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.debug("Emptied CUDA cache")
            except Exception as e:
                logger.warning(f"Failed to empty CUDA cache: {e}")
                
        if force:
            try:
                import gc
                gc.collect()
                logger.debug("Forced garbage collection")
            except Exception as e:
                logger.warning(f"Failed to force garbage collection: {e}")
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MemoryConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict) 