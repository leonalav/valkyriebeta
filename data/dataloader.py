import torch
from torch.utils.data import DataLoader

class MemoryEfficientDataLoader:
    def __init__(self, config):
        self.config = config
        self.buffer_size = config.buffer_size
        self.prefetch_factor = 2
        
    def create_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=True,
            num_workers=2,  # Limited workers for memory efficiency
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        # Efficient batching with dynamic padding
        max_len = max(len(x) for x in batch)
        padded = torch.full((len(batch), max_len), self.config.pad_token_id)
        for i, x in enumerate(batch):
            padded[i, :len(x)] = torch.tensor(x)
        return padded 