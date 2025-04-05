import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
from tqdm import tqdm
import wandb
import time
from dataclasses import dataclass
from safetensors.torch import save_file, save_model
from contextlib import nullcontext

from ..model.logical_nanogpt import LogicalGPT
from ..model.math_reasoning import MathKnowledgeDistiller
from ..utils.enhanced_memory_manager import EnhancedMemoryManager
from ..monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

@dataclass
class AdvancedTrainingConfig:
    """Configuration for advanced math-focused training"""
    # Base training parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_difficulty_fn: Optional[Callable] = None
    
    # Knowledge distillation
    use_distillation: bool = True
    distillation_alpha: float = 0.5
    distillation_temperature: float = 2.0
    teacher_model_paths: List[str] = None
    teacher_specializations: List[str] = None
    
    # Advanced optimization
    use_8bit_optimizer: bool = True
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    
    # Evaluation and checkpointing
    eval_steps: int = 500
    save_steps: int = 1000
    output_dir: str = "./outputs"
    log_to_wandb: bool = True
    project_name: str = "math-enhanced-llm"
    run_name: Optional[str] = None
    
    # Mathematical specialization
    math_domains: List[str] = None  # ["algebra", "geometry", "calculus", "statistics"]
    domain_sampling_weights: List[float] = None
    use_specialized_tokenizer: bool = True
    math_symbols_vocab_size: int = 512
    
    # Hardware optimization
    num_workers: int = 4
    pin_memory: bool = True
    

class MathDifficultyEstimator:
    """Estimates the difficulty of mathematical problems"""
    
    def __init__(self, metrics: Optional[Dict[str, Dict[str, float]]] = None):
        self.metrics = metrics or {}
        
    def estimate_difficulty(self, problem: Dict[str, Any]) -> float:
        """Estimate difficulty of a problem on a scale from 0 to 1"""
        # Extract features that correlate with difficulty
        text = problem.get("text", "")
        solution = problem.get("solution", "")
        problem_type = problem.get("problem_type", "")
        
        # Basic heuristics for difficulty
        difficulty_score = 0.0
        
        # Length-based features
        text_length = len(text)
        solution_length = len(solution)
        difficulty_score += min(0.3, text_length / 1000)  # Text length contribution
        difficulty_score += min(0.3, solution_length / 1000)  # Solution length contribution
        
        # Math symbol density
        math_symbols = ["∫", "∑", "∏", "√", "∂", "Δ", "∇", "∞", "≠", "≤", "≥"]
        symbol_count = sum(text.count(symbol) for symbol in math_symbols)
        difficulty_score += min(0.2, symbol_count / 10)
        
        # Domain-based difficulty
        domain_difficulty = {
            "arithmetic": 0.1,
            "algebra": 0.3,
            "geometry": 0.5,
            "calculus": 0.7,
            "statistics": 0.4,
            "number_theory": 0.6,
            "linear_algebra": 0.6
        }
        difficulty_score += domain_difficulty.get(problem_type, 0.3)
        
        # Use historical metrics if available
        if problem_type in self.metrics:
            domain_metrics = self.metrics[problem_type]
            historical_difficulty = domain_metrics.get("avg_difficulty", 0.5)
            difficulty_score = 0.7 * difficulty_score + 0.3 * historical_difficulty
            
        return min(1.0, difficulty_score)
        
    def update_metrics(self, problem: Dict[str, Any], performance_metrics: Dict[str, float]):
        """Update metrics based on model performance on a problem"""
        problem_type = problem.get("problem_type", "general")
        
        if problem_type not in self.metrics:
            self.metrics[problem_type] = {
                "count": 0,
                "avg_difficulty": 0.5,
                "avg_accuracy": 0.0,
                "avg_time": 0.0
            }
            
        # Update metrics
        domain_metrics = self.metrics[problem_type]
        count = domain_metrics["count"]
        new_count = count + 1
        
        # Update running averages
        for key in ["avg_difficulty", "avg_accuracy", "avg_time"]:
            if key == "avg_difficulty":
                # Use estimated difficulty
                value = self.estimate_difficulty(problem)
            else:
                # Use performance metrics
                metric_key = key.replace("avg_", "")
                value = performance_metrics.get(metric_key, domain_metrics.get(key, 0))
                
            domain_metrics[key] = (domain_metrics[key] * count + value) / new_count
            
        domain_metrics["count"] = new_count
        self.metrics[problem_type] = domain_metrics
        
    def save_metrics(self, path: str):
        """Save metrics to a JSON file"""
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)
            
    @classmethod
    def load_metrics(cls, path: str):
        """Load metrics from a JSON file"""
        if not os.path.exists(path):
            return cls()
            
        with open(path, "r") as f:
            metrics = json.load(f)
            
        return cls(metrics)


class MathEnhancedDataset(Dataset):
    """Dataset enhanced with mathematical reasoning examples"""
    
    def __init__(
        self, 
        base_dataset: List[Dict[str, Any]], 
        tokenizer, 
        max_length: int = 1024, 
        math_domains: List[str] = None,
        domain_sampling_weights: List[float] = None
    ):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.math_domains = math_domains or ["algebra", "geometry", "calculus", "statistics", "logic"]
        
        # Normalize domain sampling weights
        if domain_sampling_weights is None:
            domain_sampling_weights = [1.0] * len(self.math_domains)
        total_weight = sum(domain_sampling_weights)
        self.domain_weights = [w / total_weight for w in domain_sampling_weights]
        
        # Group examples by domain
        self.domain_examples = {domain: [] for domain in self.math_domains}
        for example in base_dataset:
            domain = example.get("domain", "general")
            if domain in self.domain_examples:
                self.domain_examples[domain].append(example)
            else:
                # Add to most similar domain
                self.domain_examples["algebra"].append(example)
                
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        example = self.base_dataset[idx]
        
        # Get problem text and solution
        problem_text = example.get("problem", example.get("text", ""))
        solution = example.get("solution", "")
        domain = example.get("domain", "algebra")
        
        # Prepare input with prompt
        input_text = f"Problem: {problem_text}\nSolution:"
        
        # Tokenize input and target
        inputs = self.tokenizer(
            input_text, 
            max_length=self.max_length // 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize solution separately
        solution_tokens = self.tokenizer(
            solution,
            max_length=self.max_length // 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Combine input and solution tokens for targets
        # Input tokens have target -100 (ignored), solution tokens have their token IDs
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        
        solution_ids = solution_tokens["input_ids"][0]
        solution_mask = solution_tokens["attention_mask"][0]
        
        # Create labels: -100 for input, actual ids for output
        labels = torch.cat([
            torch.full_like(input_ids, -100),
            solution_ids
        ])[:self.max_length]
        
        # Create combined input_ids and attention_mask
        full_input_ids = torch.cat([input_ids, solution_ids])[:self.max_length]
        full_attention_mask = torch.cat([attention_mask, solution_mask])[:self.max_length]
        
        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "labels": labels,
            "domain": domain,
            "problem_text": problem_text,
            "solution_text": solution,
            "difficulty": example.get("difficulty", 0.5)
        }
        
    def get_domain_batch(self, domain: str, batch_size: int):
        """Get a batch of examples from a specific domain"""
        if domain not in self.domain_examples or not self.domain_examples[domain]:
            # Fall back to any domain
            domain = list(self.domain_examples.keys())[0]
            
        # Sample examples from the domain
        examples = self.domain_examples[domain]
        indices = np.random.choice(len(examples), min(batch_size, len(examples)), replace=False)
        
        batch = [self[i] for i in indices]
        return self.collate_batch(batch)
        
    def get_balanced_batch(self, batch_size: int):
        """Get a batch with balanced representation from different domains"""
        # Determine number of examples per domain
        domain_counts = {}
        remaining = batch_size
        
        for domain, weight in zip(self.math_domains, self.domain_weights):
            count = max(1, int(batch_size * weight))
            domain_counts[domain] = min(count, remaining, len(self.domain_examples[domain]))
            remaining -= domain_counts[domain]
            
        # Redistribute remaining slots
        while remaining > 0:
            for domain in self.math_domains:
                if remaining <= 0:
                    break
                if len(self.domain_examples[domain]) > domain_counts[domain]:
                    domain_counts[domain] += 1
                    remaining -= 1
                    
        # Sample from each domain
        batch = []
        for domain, count in domain_counts.items():
            if count <= 0:
                continue
                
            examples = self.domain_examples[domain]
            indices = np.random.choice(len(examples), count, replace=False)
            for idx in indices:
                batch.append(self[examples[idx]])
                
        return self.collate_batch(batch)
        
    def collate_batch(self, batch):
        """Collate examples into a batch"""
        input_ids = torch.stack([example["input_ids"] for example in batch])
        attention_mask = torch.stack([example["attention_mask"] for example in batch])
        labels = torch.stack([example["labels"] for example in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "domains": [example["domain"] for example in batch],
            "difficulties": [example["difficulty"] for example in batch]
        }


class AdvancedTrainer:
    """Advanced trainer with support for various optimization techniques"""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 10,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        device: str = "cuda",
        output_dir: str = "output",
        mixed_precision: bool = False,
        mixed_precision_dtype: str = "float16",
        use_activation_checkpointing: bool = False,
        save_strategy: str = "epoch",
        save_steps: int = 500,
        eval_strategy: str = "epoch",
        eval_steps: int = 500,
        logging_steps: int = 100,
        save_total_limit: int = 3
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.output_dir = Path(output_dir)
        self.mixed_precision = mixed_precision
        self.mixed_precision_dtype = mixed_precision_dtype
        self.use_activation_checkpointing = use_activation_checkpointing
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.eval_strategy = eval_strategy
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.save_total_limit = save_total_limit
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup mixed precision
        self.scaler = None
        if self.mixed_precision:
            if self.mixed_precision_dtype == "float16":
                self.dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
            elif self.mixed_precision_dtype == "bfloat16":
                self.dtype = torch.bfloat16
                # bfloat16 doesn't need a scaler
            else:
                logger.warning(f"Unknown mixed precision dtype: {self.mixed_precision_dtype}, using float16")
                self.dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler()
        
        # Move model to device
        self.model.to(self.device)
        
        # Tracking variables
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.checkpoints = []
        
        # Set up activation checkpointing if enabled
        if self.use_activation_checkpointing:
            self._setup_activation_checkpointing()
    
    def _setup_activation_checkpointing(self):
        """Set up activation checkpointing for memory efficiency"""
        try:
            from torch.utils.checkpoint import checkpoint_sequential
            
            # Find transformer layers for checkpointing
            transformer_layers = []
            
            # This is a simplified approach - actual implementation would depend on model architecture
            for name, module in self.model.named_modules():
                if "transformer" in name.lower() and "layer" in name.lower():
                    transformer_layers.append(module)
            
            if transformer_layers:
                logger.info(f"Setting up activation checkpointing for {len(transformer_layers)} transformer layers")
                
                # Apply checkpointing
                # This is a simplified approach - actual implementation would be more sophisticated
                for i, layer in enumerate(transformer_layers):
                    original_forward = layer.forward
                    
                    def make_checkpointed_forward(orig_forward):
                        def checkpointed_forward(*args, **kwargs):
                            return torch.utils.checkpoint.checkpoint(orig_forward, *args, **kwargs)
                        return checkpointed_forward
                    
                    layer.forward = make_checkpointed_forward(original_forward)
                    
                logger.info("Activation checkpointing set up successfully")
            else:
                logger.warning("No suitable layers found for activation checkpointing")
        except Exception as e:
            logger.warning(f"Failed to set up activation checkpointing: {e}")
    
    def train(self):
        """Train the model"""
        logger.info("Starting training")
        
        total_train_steps = len(self.train_dataloader) * self.num_epochs
        progress_bar = tqdm(total=total_train_steps, desc="Training")
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            
            # Training metrics
            train_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass with mixed precision if enabled
                if self.mixed_precision:
                    autocast_context = torch.cuda.amp.autocast(dtype=self.dtype)
                else:
                    autocast_context = nullcontext()
                
                with autocast_context:
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask", None),
                        labels=batch.get("labels", None)
                    )
                    
                    # Get loss
                    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                    
                    # Scale loss for gradient accumulation
                    scaled_loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with mixed precision if enabled
                if self.mixed_precision and self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Update metrics
                train_loss += loss.item()
                
                # Optimizer step for gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        if self.mixed_precision and self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Optimizer step
                    if self.mixed_precision and self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Scheduler step
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Update global step
                    self.global_step += 1
                    
                    # Evaluation
                    if self.eval_strategy == "steps" and self.global_step % self.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        
                        # Log metrics
                        self.log_metrics(eval_metrics, prefix="eval")
                        
                        # Check for best model
                        if eval_metrics["eval_loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["eval_loss"]
                            self.save_checkpoint("best")
                    
                    # Saving
                    if self.save_strategy == "steps" and self.global_step % self.save_steps == 0:
                        self.save_checkpoint(f"checkpoint-{self.global_step}")
                    
                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        # Calculate average metrics
                        num_batches = batch_idx + 1
                        avg_train_loss = train_loss / num_batches
                        
                        # Log metrics
                        metrics = {
                            "train_loss": avg_train_loss,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                            "epoch": epoch + batch_idx / len(self.train_dataloader),
                            "step": self.global_step
                        }
                        self.log_metrics(metrics, prefix="train")
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # End of epoch
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            
            # Calculate average metrics for the epoch
            num_batches = len(self.train_dataloader)
            avg_train_loss = train_loss / num_batches
            
            # Log epoch metrics
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} completed in {epoch_time:.2f}s")
            logger.info(f"  Train loss: {avg_train_loss:.4f}")
            
            # Evaluation at the end of epoch
            if self.eval_strategy == "epoch":
                eval_metrics = self.evaluate()
                
                # Log metrics
                self.log_metrics(eval_metrics, prefix="eval")
                
                # Check for best model
                if eval_metrics["eval_loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics["eval_loss"]
                    self.save_checkpoint("best")
            
            # Saving at the end of epoch
            if self.save_strategy == "epoch":
                self.save_checkpoint(f"checkpoint-{epoch+1}")
        
        # End of training
        progress_bar.close()
        
        # Final evaluation
        final_metrics = self.evaluate()
        self.log_metrics(final_metrics, prefix="final")
        
        logger.info("Training completed")
        logger.info(f"Best evaluation loss: {self.best_eval_loss:.4f}")
        
        return final_metrics
    
    def evaluate(self):
        """Evaluate the model"""
        logger.info("Evaluating model")
        
        self.model.eval()
        
        # Evaluation metrics
        eval_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask", None),
                    labels=batch.get("labels", None)
                )
                
                # Get loss
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                
                # Update metrics
                eval_loss += loss.item()
                num_batches += 1
        
        # Calculate average metrics
        avg_eval_loss = eval_loss / num_batches
        
        # Create metrics dictionary
        metrics = {
            "eval_loss": avg_eval_loss,
            "step": self.global_step
        }
        
        # Log evaluation metrics
        logger.info(f"Evaluation results:")
        logger.info(f"  Eval loss: {avg_eval_loss:.4f}")
        
        # Set model back to training mode
        self.model.train()
        
        return metrics
    
    def save_checkpoint(self, name):
        """Save a model checkpoint"""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model
        model_path = checkpoint_dir / "model.safetensors"
        logger.info(f"Saving model checkpoint to {model_path}")
        
        # Save model using safetensors
        try:
            save_model(self.model, str(model_path))
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            
            # Fallback to PyTorch saving
            torch.save(self.model.state_dict(), str(checkpoint_dir / "model.pt"))
        
        # Save optimizer
        optimizer_path = checkpoint_dir / "optimizer.pt"
        torch.save(self.optimizer.state_dict(), str(optimizer_path))
        
        # Save scheduler if exists
        if self.scheduler is not None:
            scheduler_path = checkpoint_dir / "scheduler.pt"
            torch.save(self.scheduler.state_dict(), str(scheduler_path))
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            "epoch": int(self.global_step / len(self.train_dataloader))
        }
        
        training_state_path = checkpoint_dir / "training_state.pt"
        torch.save(training_state, str(training_state_path))
        
        # Add to checkpoints list
        if name != "best":
            self.checkpoints.append(name)
            
            # Remove old checkpoints if exceeding limit
            if self.save_total_limit > 0 and len(self.checkpoints) > self.save_total_limit:
                checkpoint_to_remove = self.checkpoints.pop(0)
                remove_dir = self.output_dir / checkpoint_to_remove
                
                logger.info(f"Removing old checkpoint: {remove_dir}")
                import shutil
                shutil.rmtree(str(remove_dir), ignore_errors=True)
    
    def load_checkpoint(self, checkpoint_dir):
        """Load a model checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load model
        model_path = checkpoint_dir / "model.safetensors"
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            from safetensors.torch import load_model
            load_model(self.model, str(model_path))
        else:
            # Fallback to PyTorch loading
            model_path = checkpoint_dir / "model.pt"
            if model_path.exists():
                logger.info(f"Loading model from {model_path}")
                self.model.load_state_dict(torch.load(str(model_path)))
            else:
                logger.warning(f"No model checkpoint found at {checkpoint_dir}")
                return False
        
        # Load optimizer
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            logger.info(f"Loading optimizer from {optimizer_path}")
            self.optimizer.load_state_dict(torch.load(str(optimizer_path)))
        
        # Load scheduler if exists
        if self.scheduler is not None:
            scheduler_path = checkpoint_dir / "scheduler.pt"
            if scheduler_path.exists():
                logger.info(f"Loading scheduler from {scheduler_path}")
                self.scheduler.load_state_dict(torch.load(str(scheduler_path)))
        
        # Load training state
        training_state_path = checkpoint_dir / "training_state.pt"
        if training_state_path.exists():
            logger.info(f"Loading training state from {training_state_path}")
            training_state = torch.load(str(training_state_path))
            
            self.global_step = training_state["global_step"]
            self.best_eval_loss = training_state["best_eval_loss"]
        
        return True
    
    def log_metrics(self, metrics, prefix=""):
        """Log metrics to console and wandb if available"""
        # Log to console
        metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)) and not k.startswith("_")])
        logger.info(f"{prefix.capitalize()} metrics: {metrics_str}")
        
        # Log to wandb if available
        try:
            if wandb.run is not None:
                wandb.log(metrics)
        except:
            pass 