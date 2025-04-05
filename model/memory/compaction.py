"""
Memory compaction mechanisms for efficient long-sequence processing.
Provides strategies to optimize memory usage during training and inference.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Union, Tuple, List

logger = logging.getLogger(__name__)

class MemoryCompactor(nn.Module):
    """
    Memory compaction module that optimizes memory usage for long sequences.
    Implements different compaction strategies to reduce memory footprint.
    """
    
    def __init__(
        self,
        hidden_size: int,
        compaction_strategy: str = "block",
        threshold: int = 8192,
        interval: int = 1000,
        reduction_ratio: float = 0.5,
        device: Optional[torch.device] = None
    ):
        """
        Initialize memory compactor.
        
        Args:
            hidden_size: Model hidden dimension size
            compaction_strategy: Strategy for memory compaction ("block", "sparse", "hybrid")
            threshold: Sequence length threshold for applying compaction
            interval: Steps between compaction operations
            reduction_ratio: Target ratio for memory reduction in sparse strategy
            device: Device to use for compaction operations
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.compaction_strategy = compaction_strategy
        self.threshold = threshold
        self.interval = interval
        self.reduction_ratio = reduction_ratio
        self.device = device
        
        # Initialize strategy-specific components
        if compaction_strategy == "block":
            self.compactor = BlockCompactor(hidden_size)
        elif compaction_strategy == "sparse":
            self.compactor = SparseCompactor(hidden_size, reduction_ratio)
        elif compaction_strategy == "hybrid":
            self.compactor = HybridCompactor(hidden_size, reduction_ratio)
        else:
            raise ValueError(f"Unknown compaction strategy: {compaction_strategy}")
            
        logger.info(f"Initialized {compaction_strategy} memory compaction with threshold {threshold}")
    
    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor], Dict[str, torch.Tensor]]) -> Any:
        """
        Apply memory compaction to inputs.
        
        Args:
            x: Input tensor, tuple, list or dict of tensors
            
        Returns:
            Compacted representation with same type as input
        """
        # Handle different input types
        if isinstance(x, torch.Tensor):
            # For single tensor
            if x.size(1) < self.threshold:  # Skip if sequence length is below threshold
                return x
            return self.compactor(x)
            
        elif isinstance(x, tuple):
            # Handle tuple of tensors
            return tuple(self.forward(item) for item in x)
            
        elif isinstance(x, list):
            # Handle list of tensors
            return [self.forward(item) for item in x]
            
        elif isinstance(x, dict):
            # Handle dictionary of tensors
            return {k: self.forward(v) for k, v in x.items()}
            
        else:
            # Return as is for unsupported types
            logger.warning(f"Unsupported input type for memory compaction: {type(x)}")
            return x


class BlockCompactor(nn.Module):
    """Compaction strategy that reduces memory by averaging blocks of tokens"""
    
    def __init__(self, hidden_size: int, block_size: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.block_size = block_size
        
        # Projection layers to restore information after pooling
        self.pre_pool = nn.Linear(hidden_size, hidden_size)
        self.post_pool = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply block compaction to input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Compacted tensor with reduced sequence length
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Handle case where sequence is not divisible by block size
        if seq_len % self.block_size != 0:
            # Pad to make divisible
            pad_len = self.block_size - (seq_len % self.block_size)
            padding = torch.zeros(batch_size, pad_len, hidden_size, device=x.device)
            x = torch.cat([x, padding], dim=1)
            # Update sequence length
            seq_len = x.shape[1]
        
        # Apply pre-pooling projection
        x = self.pre_pool(x)
        
        # Reshape to group tokens into blocks
        x = x.view(batch_size, seq_len // self.block_size, self.block_size, hidden_size)
        
        # Average tokens within each block
        x = x.mean(dim=2)  # Now [batch_size, seq_len//block_size, hidden_size]
        
        # Apply post-pooling projection to restore representational capacity
        x = self.post_pool(x)
        
        return x


class SparseCompactor(nn.Module):
    """Compaction strategy that keeps important tokens and discards others"""
    
    def __init__(self, hidden_size: int, reduction_ratio: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.reduction_ratio = reduction_ratio
        
        # Token importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sparse compaction to input by keeping most important tokens.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Compacted tensor with reduced sequence length
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute importance scores for each token
        importance = self.importance_scorer(x).squeeze(-1)  # [batch_size, seq_len]
        
        # Determine number of tokens to keep
        keep_count = max(1, int(seq_len * (1 - self.reduction_ratio)))
        
        # Process each batch item independently
        outputs = []
        for batch_idx in range(batch_size):
            # Get importance scores for this batch
            batch_importance = importance[batch_idx]
            
            # Find indices of most important tokens
            _, indices = torch.topk(batch_importance, keep_count)
            indices, _ = torch.sort(indices)  # Sort to maintain sequence order
            
            # Keep only the most important tokens
            outputs.append(x[batch_idx, indices])
        
        # Stack batch outputs
        return torch.stack(outputs)


class HybridCompactor(nn.Module):
    """
    Hybrid compaction strategy that combines block and sparse approaches.
    Uses sparse retention for information-dense regions and block pooling elsewhere.
    """
    
    def __init__(self, hidden_size: int, reduction_ratio: float = 0.5, block_size: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.reduction_ratio = reduction_ratio
        self.block_size = block_size
        
        # Create individual compactors
        self.block_compactor = BlockCompactor(hidden_size, block_size)
        self.sparse_compactor = SparseCompactor(hidden_size, reduction_ratio)
        
        # Region classifier to determine which compaction to use
        self.region_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hybrid compaction to input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Compacted tensor with reduced sequence length
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Skip if sequence is already short
        if seq_len <= self.block_size * 2:
            return x
        
        # Determine information density of sequence regions
        # Work with chunks of the sequence to avoid memory issues
        chunk_size = min(1024, seq_len)
        densities = []
        
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk = x[:, start:end]
            
            # Compute density score for each token in chunk
            chunk_density = self.region_classifier(chunk).squeeze(-1)
            densities.append(chunk_density)
        
        # Combine density scores
        density = torch.cat(densities, dim=1)  # [batch_size, seq_len]
        
        # Use sparse compaction for high density regions
        # and block compaction for low density regions
        sparse_threshold = 0.7
        sparse_mask = density > sparse_threshold
        
        # Process each batch item separately
        outputs = []
        for batch_idx in range(batch_size):
            batch_x = x[batch_idx]
            batch_mask = sparse_mask[batch_idx]
            
            # Split into high and low density regions
            high_density = batch_x[batch_mask]
            low_density = batch_x[~batch_mask]
            
            # Skip if either is empty
            if high_density.size(0) == 0:
                compacted = self.block_compactor(low_density.unsqueeze(0)).squeeze(0)
            elif low_density.size(0) == 0:
                compacted = self.sparse_compactor(high_density.unsqueeze(0)).squeeze(0)
            else:
                # Apply appropriate compaction to each region
                compacted_high = self.sparse_compactor(high_density.unsqueeze(0)).squeeze(0)
                compacted_low = self.block_compactor(low_density.unsqueeze(0)).squeeze(0)
                
                # Combine compacted regions
                compacted = torch.cat([compacted_high, compacted_low], dim=0)
            
            outputs.append(compacted)
        
        # Pad sequences to same length
        max_len = max(out.size(0) for out in outputs)
        padded_outputs = []
        
        for out in outputs:
            if out.size(0) < max_len:
                padding = torch.zeros(max_len - out.size(0), hidden_size, device=out.device)
                padded = torch.cat([out, padding], dim=0)
                padded_outputs.append(padded)
            else:
                padded_outputs.append(out)
        
        # Stack batch outputs
        return torch.stack(padded_outputs) 