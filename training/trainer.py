import sys
import os
# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Optional, Tuple, List
import wandb
import logging
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import deepspeed
from datasets import load_from_disk, load_dataset, Features, Value, Dataset, concatenate_datasets
import pandas as pd
from transformers import AutoTokenizer
import json
from safetensors.torch import save_file
import time

from training.curriculum import CurriculumScheduler, CurriculumDataset
from data.dataset import LogicalReasoningDataset
from data.efficient_loader import MemoryEfficientDataLoader, StreamingDataset
from utils.enhanced_memory_manager import EnhancedMemoryManager
from utils.memory_profiler import MemoryProfiler
from config.memory_config import MemoryConfig
from utils.training_efficiency import TrainingEfficiencyManager
from config.training_efficiency_config import TrainingEfficiencyConfig
from data.dataloader_factory import DataLoaderFactory
from data.optimized_dataset import DatasetConfig
from data.pipeline import process_dataset

# Import the LogicalReasoningTransformer model
from model.model import LogicalReasoningTransformer  # Remove 'nanogpt.' prefix since root is in path
from config import TrainingConfig, ModelConfig  # Your training configuration
from training.model_validator import ModelValidator
from scripts.prepare_data import prepare_datasets
from config import (
    ModelConfig,
    TrainingConfig,
    MemoryConfig,
    TrainingEfficiencyConfig,
    EfficientTransformerConfig
)
from model import LogicalReasoningTransformer

class CombinedDataset:
    def __init__(self, filepaths, tokenizer, config):
        self.datasets = []
        self.tokenizer = tokenizer
        self.max_length = config.max_seq_length
        
        # Define features schema - removed default values
        features = Features({
            'text': Value('string'),
            'labels': Value('int64'),
            'metadata': Value('string'),
            'source': Value('string')
        })
        
        # Verify files exist
        for filepath in filepaths:
            if not os.path.exists(filepath):
                print(f"Warning: Dataset file not found: {filepath}")
                continue
                
            try:
                if filepath.endswith('.parquet'):
                    dataset = self._load_parquet(filepath)
                elif filepath.endswith('.jsonl'):
                    dataset = self._load_jsonl(filepath)
                elif filepath.endswith('.json'):
                    dataset = self._load_json(filepath)
                else:
                    print(f"Unsupported file format: {filepath}")
                    continue
                
                if dataset is not None:
                    dataset = dataset.cast(features)
                    self.datasets.append(dataset)
                    print(f"Loaded {len(dataset)} examples from {filepath}")
                
            except Exception as e:
                print(f"Error loading dataset {filepath}: {str(e)}")
                continue
        
        if not self.datasets:
            raise ValueError("No valid datasets were loaded")
        
        # Combine all datasets
        self.combined_dataset = concatenate_datasets(self.datasets)
        print(f"Total combined examples: {len(self.combined_dataset)}")
    
    def _load_parquet(self, filepath):
        """Load and process parquet files"""
        df = pd.read_parquet(filepath)
        
        texts = []
        metadata = []
        
        # Process different dataset types
        if 'conversation' in df.columns:  # WildChat format
            for _, row in df.iterrows():
                text = f"Instruction: {row['instruction']}\nConversation: {row['conversation']}\nOutput: {row['output']}"
                texts.append(text)
                metadata.append({'source': 'wildchat', 'type': 'conversation'})
        
        elif 'problem' in df.columns:  # Math/Problem format
            for _, row in df.iterrows():
                text = f"Problem: {row['problem']}\nSolution: {row['solution']}"
                if 'answer' in row and not pd.isna(row['answer']):
                    text += f"\nAnswer: {row['answer']}"
                texts.append(text)
                metadata.append({'source': 'math', 'type': 'problem'})
        
        return Dataset.from_dict({
            'text': texts,
            'labels': [-1] * len(texts),  # Default label
            'metadata': [json.dumps(m) for m in metadata],
            'source': [m['source'] for m in metadata]
        })
    
    def _load_jsonl(self, filepath):
        """Load and process jsonl files"""
        dataset = load_dataset('json', data_files=filepath)['train']
        
        texts = []
        metadata = []
        
        for item in dataset:
            if 'question' in item:  # Question-answer format
                text = f"Question: {item['question']}"
                if 'solution' in item:
                    text += f"\nSolution: {item['solution']}"
                source_type = item.get('source_type', 'qa')
            else:  # Default format
                text = item.get('text', '')
                source_type = item.get('source_type', 'general')
            
            texts.append(text)
            metadata.append({'source': source_type, 'original_metadata': item.get('metadata', {})})
        
        return Dataset.from_dict({
            'text': texts,
            'labels': [-1] * len(texts),
            'metadata': [json.dumps(m) for m in metadata],
            'source': [m['source'] for m in metadata]
        })
    
    def _load_json(self, filepath):
        """Load and process regular json files"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        if not data:
            return None
        
        return Dataset.from_dict({
            'text': [item.get('text', '') for item in data],
            'labels': [item.get('labels', -1) for item in data],
            'metadata': [json.dumps(item.get('metadata', {})) for item in data],
            'source': [item.get('source', 'unknown') for item in data]
        })
    
    def __len__(self):
        return len(self.combined_dataset)
    
    def __getitem__(self, idx):
        item = self.combined_dataset[idx]
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'source': item['source']
        }

class MemoryEfficientTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        memory_config: MemoryConfig,
        efficiency_config: TrainingEfficiencyConfig,
        train_data_dir: str,
        val_data_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize managers and profiler
        self.memory_manager = EnhancedMemoryManager(memory_config)
        self.efficiency_manager = TrainingEfficiencyManager(efficiency_config)
        self.memory_profiler = MemoryProfiler(memory_config)
        
        # Start monitoring and profiling
        self.memory_manager.start_monitoring()
        self.efficiency_manager.start_monitoring()
        self.memory_profiler.start_profiling(model)
        
        # Get data files
        train_files = self._get_data_files(train_data_dir)
        val_files = self._get_data_files(val_data_dir) if val_data_dir else None
        
        # Create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Setup data loaders with optimized loading
        self.train_loader = DataLoaderFactory.create_loader(
            data_files=train_files,
            tokenizer=self.tokenizer,
            batch_size=config.batch_size,
            max_seq_length=config.max_seq_length,
            num_workers=config.num_workers,
            cache_dir=os.path.join(train_data_dir, ".cache"),
            enable_memory_mapping=True,
            enable_prefetch=True,
            enable_caching=True
        )
        
        if val_files:
            self.val_loader = DataLoaderFactory.create_loader(
                data_files=val_files,
                tokenizer=self.tokenizer,
                batch_size=config.eval_batch_size,
                max_seq_length=config.max_seq_length,
                num_workers=config.num_workers,
                cache_dir=os.path.join(val_data_dir, ".cache"),
                enable_memory_mapping=True,
                enable_prefetch=True,
                enable_caching=True
            )
        else:
            self.val_loader = None
            
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup memory and training optimizations
        self._setup_optimizations()
        
        # Setup distributed training if needed
        self.setup_distributed()
        
        # Validate initialization and setup error handlers
        self.__post_init__()
        
    def __post_init__(self):
        """Validate initialization and setup error handlers"""
        if not self.model:
            raise ValueError("Model cannot be None")
            
        if not hasattr(self.model, 'config'):
            raise AttributeError("Model must have a config attribute")
            
        # Set up error handlers
        self.error_handlers = {
            RuntimeError: self._handle_runtime_error,
            TypeError: self._handle_type_error,
            torch.cuda.OutOfMemoryError: self._handle_oom_error
        }

        # Validate tensor dtypes match
        self._validate_model_dtypes()
        
    def _validate_model_dtypes(self):
        """Ensure consistent dtypes across model parameters"""
        dtypes = set()
        for param in self.model.parameters():
            dtypes.add(param.dtype)
        if len(dtypes) > 1:
            raise ValueError(f"Inconsistent parameter dtypes found: {dtypes}")

    def _handle_runtime_error(self, error: RuntimeError):
        """Handle runtime errors during training"""
        self.logger.error(f"Runtime error occurred: {error}")
        if "CUDA out of memory" in str(error):
            self._handle_oom_error(error)
        elif "expected scalar type" in str(error):
            self._handle_dtype_error(error)
        else:
            raise error

    def _handle_type_error(self, error: TypeError):
        """Handle type errors during training"""
        self.logger.error(f"Type error occurred: {error}")
        if "expected torch.Tensor" in str(error):
            self._validate_batch_types()
        raise error

    def _handle_oom_error(self, error: torch.cuda.OutOfMemoryError):
        """Handle out of memory errors"""
        self.logger.warning("Out of memory error occurred, attempting recovery...")
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reduce batch size
        self.config.batch_size = max(1, self.config.batch_size // 2)
        self.logger.info(f"Reduced batch size to {self.config.batch_size}")
        
        # Enable memory efficient settings
        self.model.gradient_checkpointing_enable()
        self.memory_manager._smart_garbage_collection()
        
        # Recreate data loader with new batch size
        self._create_dataloader()

    def _validate_batch_types(self):
        """Validate types of batch tensors"""
        sample_batch = next(iter(self.train_loader))
        for key, value in sample_batch.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Batch element {key} must be a tensor, got {type(value)}")

    def _validate_batch_shapes(self, batch):
        """Validate shapes of batch tensors"""
        expected_keys = {'input_ids', 'attention_mask', 'labels'}
        if not all(k in batch for k in expected_keys):
            raise ValueError(f"Batch missing required keys: {expected_keys - batch.keys()}")
            
        B = batch['input_ids'].shape[0]  # Batch size
        for key in expected_keys:
            if batch[key].shape[0] != B:
                raise ValueError(f"Shape mismatch in batch: {key} has shape {batch[key].shape}")

    def _get_data_files(self, data_dir: str) -> List[str]:
        """Get all data files from directory"""
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")
            
        files = []
        for ext in ['.jsonl', '.parquet']:
            files.extend(
                [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.endswith(ext)]
            )
            
        if not files:
            raise ValueError(f"No .jsonl or .parquet files found in {data_dir}")
            
        return files
        
    def _setup_optimizations(self):
        """Setup memory and training optimizations"""
        # Optimize activation memory
        self.memory_manager.optimize_activation_memory(self.model)
        
        # Optimize gradient memory
        self.memory_manager.optimize_gradient_memory(self.model, self.optimizer)
        
        # Enable gradient checkpointing if configured
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        # JIT compile model if configured
        if self.efficiency_config.use_jit_compilation:
            self.model = torch.jit.script(self.model)
            
        # Setup CUDA graphs if configured
        if self.efficiency_config.use_cuda_graphs:
            self._setup_cuda_graphs()
            
    def _setup_cuda_graphs(self):
        """Setup CUDA graphs for training"""
        if not torch.cuda.is_available():
            return
            
        # Create CUDA streams
        self.streams = [
            torch.cuda.Stream()
            for _ in range(self.efficiency_config.num_cuda_streams)
        ]
        
        # Warmup for CUDA graphs
        sample_batch = next(iter(self.train_loader))
        with torch.cuda.stream(self.streams[0]):
            self.model(**sample_batch)
            
    def _create_optimizer(self):
        """Create optimizer with efficiency optimizations"""
        optimizer_config = self.efficiency_config.get_optimizer_config()
        
        if optimizer_config['type'] == 'adamw':
            if optimizer_config['fused']:
                try:
                    from apex.optimizers import FusedAdam
                    optimizer_cls = FusedAdam
                except ImportError:
                    optimizer_cls = torch.optim.AdamW
            elif optimizer_config['8bit']:
                try:
                    from bitsandbytes.optim import AdamW8bit
                    optimizer_cls = AdamW8bit
                except ImportError:
                    optimizer_cls = torch.optim.AdamW
            else:
                optimizer_cls = torch.optim.AdamW
        else:
            optimizer_cls = torch.optim.AdamW
            
        optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.efficiency_config.initial_lr,
            weight_decay=self.config.weight_decay
        )
        
        # Validate optimizer parameters
        for param_group in optimizer.param_groups:
            if not param_group['params']:
                raise ValueError("Empty parameter group found in optimizer")
                
        return optimizer
        
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.efficiency_config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.efficiency_config.max_lr,
                epochs=self.config.num_epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=self.efficiency_config.warmup_steps / (self.config.num_epochs * len(self.train_loader))
            )
        else:
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.efficiency_config.warmup_steps
            )
            
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with optimized data loading and monitoring"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Start monitoring data pipeline
        data_metrics = {
            'loading_time': 0,
            'preprocessing_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_batch_size': 0,
            'total_samples': 0
        }
        
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()
            
            # Track data loading metrics
            data_metrics['loading_time'] += time.time() - batch_start
            data_metrics['total_samples'] += len(batch)
            data_metrics['avg_batch_size'] = (
                data_metrics['total_samples'] / (batch_idx + 1)
            )
            
            # Update cache metrics
            if hasattr(self.train_loader.dataset, 'get_cache_stats'):
                cache_stats = self.train_loader.dataset.get_cache_stats()
                data_metrics['cache_hits'] = cache_stats['hits']
                data_metrics['cache_misses'] = cache_stats['misses']
            
            # Memory tracking
            current_memory = self.memory_profiler.get_current_memory_usage()
            self.logger.debug(
                f"Batch {batch_idx} memory usage: {current_memory['gpu_used']:.2f}GB GPU, "
                f"{current_memory['cpu_used']:.2f}GB CPU"
            )
            
            # Forward pass with efficiency tracking
            loss = self.efficiency_manager.optimize_forward_backward(
                self.model,
                batch,
                self.optimizer
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f"Train Epoch: {epoch} [{batch_idx}/{len(self.train_loader)} "
                    f"({100. * batch_idx / len(self.train_loader):.0f}%)]\t"
                    f"Loss: {loss.item():.6f}"
                )
                
                # Log data pipeline metrics
                self.logger.info(
                    f"Data Pipeline Stats:\n"
                    f"- Avg Loading Time: {data_metrics['loading_time']/(batch_idx+1):.3f}s/batch\n"
                    f"- Avg Batch Size: {data_metrics['avg_batch_size']:.1f}\n"
                    f"- Cache Hit Rate: {data_metrics['cache_hits']/(data_metrics['cache_hits']+data_metrics['cache_misses'])*100:.1f}%"
                )
                
                if wandb.run:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "gpu_memory_used": current_memory['gpu_used'],
                        "cpu_memory_used": current_memory['cpu_used'],
                        "data_loading_time": data_metrics['loading_time']/(batch_idx+1),
                        "avg_batch_size": data_metrics['avg_batch_size'],
                        "cache_hit_rate": data_metrics['cache_hits']/(data_metrics['cache_hits']+data_metrics['cache_misses'])
                    })
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / num_batches
        
        # Log epoch summary
        self.logger.info(
            f"\nEpoch {epoch} Summary:\n"
            f"- Average Loss: {avg_loss:.6f}\n"
            f"- Epoch Time: {epoch_time:.2f}s\n"
            f"- Samples/second: {data_metrics['total_samples']/epoch_time:.1f}\n"
            f"- Final Cache Hit Rate: {data_metrics['cache_hits']/(data_metrics['cache_hits']+data_metrics['cache_misses'])*100:.1f}%"
        )
        
        return {
            "loss": avg_loss,
            "epoch_time": epoch_time,
            "samples_per_second": data_metrics['total_samples']/epoch_time,
            "cache_hit_rate": data_metrics['cache_hits']/(data_metrics['cache_hits']+data_metrics['cache_misses']),
            "avg_batch_size": data_metrics['avg_batch_size']
        }
        
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model with efficiency optimizations"""
        if not self.val_loader:
            return {}
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # Track memory and efficiency for evaluation
        with self.memory_manager.track_memory("evaluation"), \
             self.efficiency_manager.train_step_context():
            
            for batch in self.val_loader:
                # Clear memory before forward pass
                self.memory_manager.clear_memory()
                
                # Optimized forward pass
                with torch.cuda.amp.autocast(enabled=self.efficiency_config.use_amp):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                total_loss += loss.item()
                num_batches += 1
                
        return {"val_loss": total_loss / num_batches}
        
    def train(self) -> Dict[str, float]:
        """Full training loop with efficiency optimizations"""
        best_val_loss = float("inf")
        
        try:
            for epoch in range(self.config.num_epochs):
                # Train epoch
                train_metrics = self.train_epoch(epoch)
                
                # Evaluate
                val_metrics = self.evaluate()
                
                # Log metrics
                metrics = {**train_metrics, **val_metrics}
                logging.info(f"Epoch {epoch}: {metrics}")
                
                # Save checkpoint if best validation loss
                if val_metrics and val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt")
                    
        except Exception as e:
            logging.error(f"Training error: {str(e)}")
            # Try to recover using emergency configs
            self._handle_training_error()
            
        finally:
            # Cleanup
            self.cleanup()
            
        return metrics
        
    def _handle_training_error(self):
        """Handle training errors with emergency settings"""
        logging.info("Attempting to recover from training error...")
        
        # Get emergency configurations
        memory_emergency = self.memory_config.get_emergency_config()
        efficiency_emergency = self.efficiency_config.get_emergency_config()
        
        # Update managers with emergency settings
        self.memory_manager = EnhancedMemoryManager(memory_emergency)
        self.efficiency_manager = TrainingEfficiencyManager(efficiency_emergency)
        
        # Restart monitoring
        self.memory_manager.start_monitoring()
        self.efficiency_manager.start_monitoring()
        
        # Clear all memory
        self.memory_manager._smart_garbage_collection()
        
        # Reduce batch size to minimum
        self.train_loader.current_batch_size = memory_emergency.min_batch_size
        
    def save_checkpoint(self, filename: str):
        """Save checkpoint with optimization states"""
        with self.memory_manager.track_memory("save_checkpoint"):
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
                "memory_config": self.memory_config,
                "efficiency_config": self.efficiency_config,
                "memory_stats": self.memory_manager.get_memory_stats(),
                "efficiency_stats": self.efficiency_manager.get_training_efficiency_stats()
            }
            
            if self.efficiency_manager.scaler:
                checkpoint["scaler_state_dict"] = self.efficiency_manager.scaler.state_dict()
                
            torch.save(checkpoint, filename)
            
    def load_checkpoint(self, filename: str):
        """Load checkpoint and restore optimization states"""
        with self.memory_manager.track_memory("load_checkpoint"):
            checkpoint = torch.load(filename)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            if "scaler_state_dict" in checkpoint and self.efficiency_manager.scaler:
                self.efficiency_manager.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                
            # Update configurations
            if "memory_config" in checkpoint:
                self.memory_config.update_from_dict(checkpoint["memory_config"])
            if "efficiency_config" in checkpoint:
                self.efficiency_config.update_from_dict(checkpoint["efficiency_config"])
                
    def cleanup(self):
        """Enhanced cleanup with error handling"""
        try:
            # Clean up distributed training
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
                
            # Cleanup CUDA resources 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                for stream in self.streams:
                    stream.synchronize()
                    
            # Close data loaders
            if hasattr(self.train_loader, 'cleanup'):
                self.train_loader.cleanup()
            if hasattr(self.val_loader, 'cleanup'):  
                self.val_loader.cleanup()
                
            # Other cleanup
            self.logger.info("\n=== Final Memory Analysis ===")
            self.memory_profiler.print_memory_report()
            
            # Get final recommendations
            recommendations = self.memory_profiler.get_optimization_recommendations()
            if recommendations:
                self.logger.info("\nFinal Optimization Recommendations:")
                for rec in recommendations:
                    self.logger.info(
                        f"- {rec['component']}: {rec['recommendation']} "
                        f"(Priority: {rec['priority']})"
                    )
            
            # Save memory profile data
            memory_summary = self.memory_profiler.get_memory_summary()
            with open('memory_profile.json', 'w') as f:
                json.dump(memory_summary, f, indent=2)
                
            # Cleanup data loader caches
            DataLoaderFactory.cleanup_cache()
            
            # Cleanup resources
            self.memory_manager.cleanup()
            self.efficiency_manager.cleanup()
            self.memory_profiler.cleanup()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info("Cleanup completed. Memory profile saved to 'memory_profile.json'")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise
        
    def setup_distributed(self):
        """Setup distributed training with error handling"""
        if not self.config.distributed:
            return
            
        try:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
                
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required for distributed training")
                
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank
            )
            
        except Exception as e:
            self.logger.error(f"Failed to setup distributed training: {e}")
            raise
            
def train():
    # Prepare all data in one go
    data = prepare_datasets(
        data_root="path/to/your/datasets",
        output_root="processed_data"
    )
    
    # Get your loaders
    train_loader = data['train_loader']
    eval_loader = data['eval_loader']
    inference_loader = data.get('inference_loader')
    
    # Proceed with training...

if __name__ == "__main__":
    train()
