import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    CPUOffload
)
from typing import Optional

class FSDPWrapper:
    def __init__(self, config):
        self.config = config
        self.mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
        self.cpu_offload = CPUOffload() if config.use_cpu_offload else None
        
    def wrap_model(self, model: torch.nn.Module) -> FSDP:
        """Wrap model with FSDP"""
        return FSDP(
            model,
            mixed_precision=self.mixed_precision,
            cpu_offload=self.cpu_offload,
            device_id=torch.cuda.current_device(),
            sharding_strategy=self.config.sharding_strategy
        )
        
    def prepare_dataloaders(self, *loaders):
        """Prepare dataloaders for distributed training"""
        return [self._prepare_dataloader(loader) for loader in loaders]
        
    def _prepare_dataloader(self, loader):
        sampler = torch.utils.data.DistributedSampler(loader.dataset)
        return torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=sampler,
            num_workers=loader.num_workers,
            pin_memory=True
        )
