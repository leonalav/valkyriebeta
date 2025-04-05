import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader, Subset

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Represents a stage in curriculum learning"""
    difficulty_range: tuple  # (min_difficulty, max_difficulty)
    epochs: int
    batch_size: Optional[int] = None
    learning_rate_multiplier: float = 1.0
    sample_ratio: float = 1.0  # What fraction of the data to include in this stage


class CurriculumScheduler:
    """
    Manages curriculum learning schedules for training.
    
    Curriculum learning gradually increases the difficulty of training examples
    throughout the training process. This class handles transitioning between
    different curriculum stages based on epochs or validation metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the curriculum scheduler.
        
        Args:
            config: Dictionary containing configuration for curriculum learning
        """
        self.config = config
        self.current_stage = 0
        self.stages = []
        self.dataset = None
        self.base_dataloader = None
        self.difficulty_fn = lambda x: 0.5  # Default constant difficulty
        
        # Extract config
        self.metric_based = config.get('metric_based', False)
        self.metric_name = config.get('metric_name', 'loss')
        self.metric_threshold = config.get('metric_threshold', 0.0)
        self.patience = config.get('patience', 1)
        self.patience_counter = 0
        
        # Set up stages if provided in config
        if 'stages' in config:
            for stage_config in config['stages']:
                self.add_stage(
                    difficulty_range=stage_config.get('difficulty_range', (0.0, 1.0)),
                    epochs=stage_config.get('epochs', 1),
                    batch_size=stage_config.get('batch_size', None),
                    learning_rate_multiplier=stage_config.get('lr_multiplier', 1.0),
                    sample_ratio=stage_config.get('sample_ratio', 1.0)
                )
    
    def add_stage(self, difficulty_range: tuple, epochs: int, 
                 batch_size: Optional[int] = None,
                 learning_rate_multiplier: float = 1.0,
                 sample_ratio: float = 1.0):
        """
        Add a new curriculum stage.
        
        Args:
            difficulty_range: Tuple of (min_difficulty, max_difficulty) for this stage
            epochs: Number of epochs to stay in this stage
            batch_size: Optional batch size for this stage
            learning_rate_multiplier: Multiplier for learning rate in this stage
            sample_ratio: Fraction of the dataset to use in this stage
        """
        stage = CurriculumStage(
            difficulty_range=difficulty_range,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate_multiplier=learning_rate_multiplier,
            sample_ratio=sample_ratio
        )
        self.stages.append(stage)
        logger.info(f"Added curriculum stage {len(self.stages)}: {stage}")
    
    def set_dataset(self, dataset: Dataset, difficulty_fn: Callable):
        """
        Set the dataset and difficulty function for curriculum learning.
        
        Args:
            dataset: The dataset to apply curriculum learning to
            difficulty_fn: Function that returns a difficulty score (0.0-1.0) for each example
        """
        self.dataset = dataset
        self.difficulty_fn = difficulty_fn
        
        # Pre-compute difficulties for all examples if dataset is not too large
        max_precompute = 100000  # Don't try to precompute for very large datasets
        if not hasattr(dataset, '__len__') or len(dataset) > max_precompute:
            logger.info("Dataset too large for precomputing difficulties")
            self.difficulties = None
        else:
            logger.info("Precomputing difficulties for dataset")
            self.difficulties = [difficulty_fn(dataset[i]) for i in range(len(dataset))]
    
    def step(self, metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Advance curriculum stage if needed.
        
        Args:
            metrics: Optional dictionary of metrics to check for stage advancement
        
        Returns:
            bool: True if stage was advanced, False otherwise
        """
        # If metric-based advancement is enabled, check metrics
        if self.metric_based and metrics:
            metric_value = metrics.get(self.metric_name, float('inf'))
            
            if self.current_stage < len(self.stages) - 1:
                # Check if metric meets threshold for advancement
                if metric_value <= self.metric_threshold:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        self.current_stage += 1
                        self.patience_counter = 0
                        logger.info(f"Advanced to curriculum stage {self.current_stage + 1} based on metrics")
                        return True
                else:
                    self.patience_counter = 0
        
        # Check for epoch-based advancement
        elif self.stages and hasattr(self, 'epoch_counter'):
            self.epoch_counter += 1
            stage = self.stages[self.current_stage]
            
            if self.epoch_counter >= stage.epochs and self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.epoch_counter = 0
                logger.info(f"Advanced to curriculum stage {self.current_stage + 1} based on epochs")
                return True
        else:
            # Initialize epoch counter if not already set
            self.epoch_counter = 0
        
        return False
    
    def get_data_loader(self) -> DataLoader:
        """
        Get a data loader for the current curriculum stage.
        
        Returns:
            DataLoader filtered according to the current curriculum stage
        """
        if not self.stages or not self.dataset:
            logger.warning("No stages or dataset defined, returning None")
            return None
        
        stage = self.stages[self.current_stage]
        min_diff, max_diff = stage.difficulty_range
        
        # If we have precomputed difficulties, use them for filtering
        if self.difficulties is not None:
            indices = [i for i, diff in enumerate(self.difficulties) 
                      if min_diff <= diff <= max_diff]
            
            # Apply sample ratio if needed
            if stage.sample_ratio < 1.0:
                num_samples = int(len(indices) * stage.sample_ratio)
                indices = indices[:num_samples]
            
            subset = Subset(self.dataset, indices)
        else:
            # For larger datasets, we'll need to filter dynamically in __getitem__
            subset = CurriculumDataset(
                self.dataset, 
                self.difficulty_fn, 
                min_diff, 
                max_diff,
                sample_ratio=stage.sample_ratio
            )
        
        # Use stage-specific batch size if provided
        batch_size = stage.batch_size if stage.batch_size else self.config.get('batch_size', 32)
        
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
    
    def get_current_lr_multiplier(self) -> float:
        """Get the learning rate multiplier for the current stage"""
        if not self.stages:
            return 1.0
        return self.stages[self.current_stage].learning_rate_multiplier
    
    def get_current_stage_info(self) -> Dict[str, Any]:
        """Get information about the current curriculum stage"""
        if not self.stages:
            return {}
        
        stage = self.stages[self.current_stage]
        return {
            'stage': self.current_stage + 1,
            'total_stages': len(self.stages),
            'difficulty_range': stage.difficulty_range,
            'epochs': stage.epochs,
            'current_epoch': getattr(self, 'epoch_counter', 0),
            'batch_size': stage.batch_size,
            'lr_multiplier': stage.learning_rate_multiplier,
            'sample_ratio': stage.sample_ratio
        }


class CurriculumDataset(Dataset):
    """Dataset wrapper that filters examples based on difficulty"""
    
    def __init__(self, dataset, difficulty_fn, min_difficulty, max_difficulty, sample_ratio=1.0):
        """
        Initialize curriculum dataset.
        
        Args:
            dataset: The base dataset
            difficulty_fn: Function to compute difficulty
            min_difficulty: Minimum difficulty to include
            max_difficulty: Maximum difficulty to include
            sample_ratio: Fraction of qualifying examples to include
        """
        self.dataset = dataset
        self.difficulty_fn = difficulty_fn
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.sample_ratio = sample_ratio
        
        # For efficiency with large datasets, we'll filter dynamically
        self.cache = {}  # Cache difficulty scores
    
    def __len__(self):
        """Estimate length based on filtering criteria and sample ratio"""
        # This is an estimation since we don't know exact counts without scanning everything
        if hasattr(self.dataset, '__len__'):
            return int(len(self.dataset) * self.sample_ratio * 
                      (self.max_difficulty - self.min_difficulty))
        return 10000  # Default arbitrary size for iterable datasets
    
    def __getitem__(self, idx):
        """Get an item that satisfies the difficulty criteria"""
        if idx in self.cache:
            return self.cache[idx]
        
        # For non-indexable datasets, this won't work well
        if not hasattr(self.dataset, '__getitem__'):
            raise NotImplementedError("Dynamic filtering not supported for iterable-only datasets")
        
        # Simple implementation: scan through dataset until finding matching item
        # Note: This is inefficient for large datasets
        attempts = 0
        max_attempts = min(10000, len(self.dataset) if hasattr(self.dataset, '__len__') else 10000)
        
        while attempts < max_attempts:
            real_idx = (idx + attempts) % len(self.dataset)
            item = self.dataset[real_idx]
            difficulty = self.difficulty_fn(item)
            
            if self.min_difficulty <= difficulty <= self.max_difficulty:
                # Apply sample ratio probabilistically
                if self.sample_ratio >= 1.0 or torch.rand(1).item() < self.sample_ratio:
                    self.cache[idx] = item
                    return item
            
            attempts += 1
        
        # If we couldn't find a matching item, return something anyway
        return self.dataset[idx] 