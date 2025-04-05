from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math

@dataclass
class CurriculumStage:
    name: str
    difficulty: int
    min_sequence_length: int
    max_sequence_length: int
    logical_operations: List[str]
    required_accuracy: float
    min_epochs: int

@dataclass 
class CurriculumConfig:
    """Configuration for curriculum learning"""
    # Difficulty progression
    initial_difficulty: float = 0.3
    final_difficulty: float = 1.0
    difficulty_steps: int = 1000
    difficulty_schedule: str = "linear"  # linear, exponential, step
    
    # Sequence length progression
    initial_seq_length: int = 1024
    target_seq_length: int = 49152  # ~48K
    seq_length_schedule: str = "exponential"
    seq_length_steps: int = 2000
    
    # Task mixing
    task_weights: Dict[str, float] = None
    dynamic_task_weights: bool = True
    task_success_threshold: float = 0.7
    
    # Sample selection
    rejection_sampling: bool = True
    sample_buffer_size: int = 10000
    prefilter_samples: bool = True
    
    # Memory optimization for long sequences
    use_length_bucketing: bool = True
    length_buckets: List[int] = None
    
    def __post_init__(self):
        # Default task weights if not provided
        if self.task_weights is None:
            self.task_weights = {
                "language_modeling": 0.7,
                "comprehension": 0.1,
                "reasoning": 0.1,
                "math": 0.05,
                "coding": 0.05
            }
        
        # Default length buckets if not provided
        if self.length_buckets is None:
            self.length_buckets = [1024, 2048, 4096, 8192, 16384, 32768, 49152]

class CurriculumScheduler:
    """
    Manages curriculum learning schedules for training.
    Dynamically adjusts difficulty and sequence length based on training progress.
    """
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.step = 0
        self.task_metrics = {task: 0.0 for task in config.task_weights.keys()}
        self.current_difficulty = config.initial_difficulty
        self.current_seq_length = config.initial_seq_length
        
        # Initialize scheduling functions
        self._init_scheduling_functions()
    
    def _init_scheduling_functions(self):
        """Initialize scheduling functions based on config"""
        # Difficulty schedule
        if self.config.difficulty_schedule == "linear":
            self.difficulty_schedule_fn = lambda step: min(
                self.config.initial_difficulty + (self.config.final_difficulty - self.config.initial_difficulty) * 
                step / self.config.difficulty_steps,
                self.config.final_difficulty
            )
        elif self.config.difficulty_schedule == "exponential":
            self.difficulty_schedule_fn = lambda step: min(
                self.config.initial_difficulty * math.pow(
                    self.config.final_difficulty / self.config.initial_difficulty,
                    step / self.config.difficulty_steps
                ),
                self.config.final_difficulty
            )
        else:  # step schedule
            self.difficulty_schedule_fn = lambda step: min(
                self.config.initial_difficulty + (self.config.final_difficulty - self.config.initial_difficulty) * 
                (step // (self.config.difficulty_steps // 5)) / 5,
                self.config.final_difficulty
            )
            
        # Sequence length schedule
        if self.config.seq_length_schedule == "linear":
            self.seq_length_schedule_fn = lambda step: min(
                self.config.initial_seq_length + (self.config.target_seq_length - self.config.initial_seq_length) * 
                step / self.config.seq_length_steps,
                self.config.target_seq_length
            )
        elif self.config.seq_length_schedule == "exponential":
            self.seq_length_schedule_fn = lambda step: min(
                self.config.initial_seq_length * math.pow(
                    self.config.target_seq_length / self.config.initial_seq_length,
                    step / self.config.seq_length_steps
                ),
                self.config.target_seq_length
            )
        else:  # step schedule
            self.seq_length_schedule_fn = lambda step: min(
                self.config.initial_seq_length * 2 ** (step // (self.config.seq_length_steps // 5)),
                self.config.target_seq_length
            )
    
    def update(self, step: int, metrics: Dict[str, float] = None):
        """
        Update the curriculum based on training step and performance metrics.
        
        Args:
            step: Current training step
            metrics: Dictionary of task metrics (loss or accuracy values)
        """
        self.step = step
        
        # Update difficulty
        self.current_difficulty = self.difficulty_schedule_fn(step)
        
        # Update sequence length
        raw_seq_length = self.seq_length_schedule_fn(step)
        
        # Find closest length bucket
        if self.config.use_length_bucketing:
            # Find closest bucket (rounding up)
            self.current_seq_length = next(
                (l for l in self.config.length_buckets if l >= raw_seq_length),
                self.config.length_buckets[-1]
            )
        else:
            self.current_seq_length = int(raw_seq_length)
        
        # Update task metrics if provided
        if metrics is not None and self.config.dynamic_task_weights:
            for task, metric in metrics.items():
                if task in self.task_metrics:
                    # Exponential moving average
                    self.task_metrics[task] = 0.9 * self.task_metrics[task] + 0.1 * metric
    
    def get_difficulty(self) -> float:
        """Get current difficulty level"""
        return self.current_difficulty
    
    def get_seq_length(self) -> int:
        """Get current sequence length"""
        return self.current_seq_length
    
    def get_task_weights(self) -> Dict[str, float]:
        """
        Get current task weights, adjusted based on performance if dynamic.
        Tasks that are performing well get lower weight, challenging tasks get higher weight.
        """
        if not self.config.dynamic_task_weights:
            return self.config.task_weights
        
        # Calculate inverse success rate (higher metric = lower success)
        inverse_success = {
            task: max(0.1, 1.0 - metric / self.config.task_success_threshold)
            for task, metric in self.task_metrics.items()
        }
        
        # Normalize to get weights
        total = sum(inverse_success.values())
        dynamic_weights = {
            task: value / total for task, value in inverse_success.items()
        }
        
        # Blend with original weights for stability
        blended_weights = {
            task: 0.7 * dynamic_weights[task] + 0.3 * self.config.task_weights[task]
            for task in self.config.task_weights
        }
        
        # Normalize final weights
        total = sum(blended_weights.values())
        return {task: weight / total for task, weight in blended_weights.items()}

class CurriculumSampler:
    """
    Sampler for curriculum learning that selects appropriate samples
    based on the current curriculum state.
    
    This is optimized for hybrid RWKV-Transformer training with long contexts.
    """
    
    def __init__(
        self, 
        curriculum_scheduler: CurriculumScheduler,
        tokenized_datasets: Dict[str, Any],
        tokenizer: Any,
        difficulty_scorer: Optional[Callable] = None,
        buffer_size: int = 10000
    ):
        self.scheduler = curriculum_scheduler
        self.datasets = tokenized_datasets
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        
        # Use provided difficulty scorer or default
        self.difficulty_scorer = difficulty_scorer or self._default_difficulty_scorer
        
        # Sample buffers for each task
        self.sample_buffers = {
            task: [] for task in curriculum_scheduler.config.task_weights.keys()
        }
        
        # Precompute difficulties if datasets not too large
        self.sample_difficulties = {}
        self._precompute_difficulties()
    
    def _precompute_difficulties(self):
        """Precompute difficulties for samples if datasets are not too large"""
        for task, dataset in self.datasets.items():
            if task not in self.scheduler.config.task_weights:
                continue
                
            # Skip if dataset is very large
            if hasattr(dataset, "__len__") and len(dataset) > self.buffer_size * 10:
                continue
                
            # Compute difficulties for a subset
            max_samples = min(len(dataset), self.buffer_size * 2)
            indices = random.sample(range(len(dataset)), max_samples)
            
            for idx in indices:
                sample = dataset[idx]
                difficulty = self.difficulty_scorer(sample, task)
                self.sample_difficulties[(task, idx)] = difficulty
    
    def _default_difficulty_scorer(self, sample, task: str) -> float:
        """
        Default difficulty scoring based on sequence complexity and length.
        
        Args:
            sample: Dataset sample
            task: Task name
            
        Returns:
            difficulty: Score between 0.0 and 1.0
        """
        # Get input text
        input_text = sample.get("text", "")
        if not input_text and "input_ids" in sample:
            if hasattr(self.tokenizer, "decode"):
                input_text = self.tokenizer.decode(sample["input_ids"])
        
        # Length-based difficulty (longer is harder)
        length = len(sample.get("input_ids", [])) / self.scheduler.config.target_seq_length
        length_score = min(1.0, length)
        
        # Complexity-based difficulty
        if input_text:
            # Vocabulary diversity
            words = input_text.split()
            unique_words = len(set(words)) if words else 0
            vocab_diversity = min(1.0, unique_words / 1000)
            
            # Sentence length
            sentences = input_text.split(".")
            avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
            sentence_complexity = min(1.0, avg_sentence_length / 25)
            
            # Special tokens frequency (code blocks, math, etc.)
            special_token_ratio = sum(1 for c in input_text if not c.isalnum()) / max(1, len(input_text))
            special_token_score = min(1.0, special_token_ratio * 5)
            
            # Combine metrics
            complexity_score = 0.4 * vocab_diversity + 0.3 * sentence_complexity + 0.3 * special_token_score
        else:
            complexity_score = 0.5  # Default if no text available
        
        # Task-specific adjustments
        if task == "reasoning":
            # Reasoning tasks are generally harder
            task_multiplier = 1.2
        elif task == "math":
            # Math tasks are generally harder
            task_multiplier = 1.3
        elif task == "coding":
            # Coding tasks are generally harder
            task_multiplier = 1.2
        else:
            task_multiplier = 1.0
        
        # Combine scores
        difficulty = 0.6 * length_score + 0.4 * complexity_score
        difficulty = min(1.0, difficulty * task_multiplier)
        
        return difficulty
    
    def get_sample_difficulty(self, sample, task: str) -> float:
        """Get difficulty for a sample"""
        # Check if precomputed
        if (task, sample.get("id", -1)) in self.sample_difficulties:
            return self.sample_difficulties[(task, sample.get("id", -1))]
        
        # Compute difficulty
        difficulty = self.difficulty_scorer(sample, task)
        
        # Cache for future use if sample has an ID
        if "id" in sample:
            self.sample_difficulties[(task, sample["id"])] = difficulty
        
        return difficulty
    
    def fill_buffer(self, task: str, count: int = None):
        """
        Fill the sample buffer for a task with appropriately difficult samples.
        
        Args:
            task: Task to fill buffer for
            count: Number of samples to add (defaults to buffer_size)
        """
        if task not in self.datasets:
            return
            
        dataset = self.datasets[task]
        count = count or self.buffer_size
        
        # Current difficulty threshold
        difficulty_threshold = self.scheduler.get_difficulty()
        
        # Get random samples and filter by difficulty
        attempts = 0
        max_attempts = count * 3
        
        while len(self.sample_buffers[task]) < count and attempts < max_attempts:
            # Get random sample
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            
            # Get difficulty
            difficulty = self.get_sample_difficulty(sample, task)
            
            # Accept if within threshold + small random variation
            variation = random.uniform(-0.1, 0.1)
            if difficulty <= difficulty_threshold + variation:
                # Add to buffer if not already there
                if idx not in [s.get("idx", -1) for s in self.sample_buffers[task]]:
                    # Add index to sample for tracking
                    sample = dict(sample)
                    sample["idx"] = idx
                    self.sample_buffers[task].append(sample)
            
            attempts += 1
    
    def get_batch(self, batch_size: int) -> List[Dict]:
        """
        Get a batch of samples according to current curriculum.
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            batch: List of samples
        """
        # Get current task weights
        task_weights = self.scheduler.get_task_weights()
        
        # Calculate number of samples per task
        task_counts = {}
        remaining = batch_size
        for task, weight in task_weights.items():
            # Allocate proportional to weight, at least 1 if weight > 0
            count = max(1, int(batch_size * weight)) if weight > 0 else 0
            task_counts[task] = min(count, remaining)
            remaining -= task_counts[task]
        
        # Allocate any remaining samples
        if remaining > 0:
            # Sort tasks by fractional part of ideal count
            fractional_parts = [
                (task, batch_size * weight - int(batch_size * weight))
                for task, weight in task_weights.items()
                if weight > 0
            ]
            fractional_parts.sort(key=lambda x: x[1], reverse=True)
            
            # Allocate remaining one by one
            for task, _ in fractional_parts:
                if remaining <= 0:
                    break
                task_counts[task] += 1
                remaining -= 1
        
        # Fill buffers as needed
        for task, count in task_counts.items():
            if len(self.sample_buffers.get(task, [])) < count:
                self.fill_buffer(task, count * 2)  # Fill with some extra
        
        # Build batch
        batch = []
        for task, count in task_counts.items():
            if task not in self.sample_buffers or not self.sample_buffers[task]:
                continue
                
            # Get samples from buffer
            task_samples = random.sample(self.sample_buffers[task], min(count, len(self.sample_buffers[task])))
            
            # Remove selected samples from buffer
            for sample in task_samples:
                self.sample_buffers[task].remove(sample)
                
            # Add to batch
            batch.extend(task_samples)
        
        # Ensure batch is right size
        if len(batch) > batch_size:
            batch = random.sample(batch, batch_size)
        
        # Apply sequence length constraint
        max_length = self.scheduler.get_seq_length()
        for i, sample in enumerate(batch):
            if "input_ids" in sample and len(sample["input_ids"]) > max_length:
                # Truncate to current curriculum sequence length
                batch[i] = dict(sample)
                batch[i]["input_ids"] = sample["input_ids"][:max_length]
                if "attention_mask" in sample:
                    batch[i]["attention_mask"] = sample["attention_mask"][:max_length]
                if "labels" in sample:
                    batch[i]["labels"] = sample["labels"][:max_length]
        
        return batch

class CurriculumDataset(Dataset):
    def __init__(self, 
                 base_dataset: Dataset, 
                 stage: CurriculumStage,
                 tokenizer: Any):
        self.base_dataset = base_dataset
        self.stage = stage
        self.tokenizer = tokenizer
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base_dataset[idx]
        
        # Apply curriculum constraints
        max_length = np.random.randint(
            self.stage.min_sequence_length,
            self.stage.max_sequence_length
        )
        
        # Truncate and pad input sequences
        inputs = self.tokenizer(
            item['text'],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Filter logical operations based on current stage
        if 'logical_tree' in item:
            item['logical_tree'] = self._filter_operations(
                item['logical_tree'],
                self.stage.logical_operations
            )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': item.get('labels', None),
            'logical_tree': item.get('logical_tree', None)
        }
    
    def _filter_operations(self, tree: Dict, allowed_ops: List[str]) -> Dict:
        """Filter logical operations based on current curriculum stage"""
        if not tree:
            return tree
            
        filtered_tree = {}
        for k, v in tree.items():
            if isinstance(v, dict):
                if v.get('operation') in allowed_ops:
                    filtered_tree[k] = self._filter_operations(v, allowed_ops)
            else:
                filtered_tree[k] = v
        return filtered_tree
