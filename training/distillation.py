import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Union
from model.core_model import EnhancedLanguageModel
from config.architecture_config import ArchitectureConfig
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
from pathlib import Path
from safetensors.torch import save_file, save_model
from contextlib import nullcontext
import wandb

from ..config.training_config import DistillationConfig

logger = logging.getLogger(__name__)

class DistillationConfig:
    """Configuration for knowledge distillation"""
    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,  # Weight for distillation loss vs. task loss
        distill_attention: bool = True,  # Whether to distill attention patterns
        distill_hidden_states: bool = True,  # Whether to distill hidden states
        distill_reasoning_trace: bool = True,  # Whether to distill reasoning traces
        layer_mapping: Optional[Dict[int, int]] = None,  # Mapping from student to teacher layers
        reasoning_component_weights: Optional[Dict[str, float]] = None,  # Weights for different reasoning components
        use_progressive_distillation: bool = False,
        progressive_stages: List[str] = [],
        current_stage: str = "",
        distill_logits: bool = True
    ):
        self.temperature = temperature
        self.alpha = alpha
        self.distill_attention = distill_attention
        self.distill_hidden_states = distill_hidden_states
        self.distill_reasoning_trace = distill_reasoning_trace
        self.layer_mapping = layer_mapping
        self.reasoning_component_weights = reasoning_component_weights or {
            "neural_symbolic": 1.0,
            "tree_reasoning": 1.0,
            "recursive_reasoning": 1.0,
            "knowledge_reasoning": 1.0,
            "verifiable_computation": 1.0
        }
        self.use_progressive_distillation = use_progressive_distillation
        self.progressive_stages = progressive_stages
        self.current_stage = current_stage
        self.distill_logits = distill_logits

class ReasoningDistiller:
    """Distills knowledge from a teacher model to a student model with focus on reasoning capabilities"""
    
    def __init__(
        self,
        teacher_model: EnhancedLanguageModel,
        student_model: EnhancedLanguageModel,
        config: DistillationConfig
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        
        # Set teacher to evaluation mode
        self.teacher.eval()
        
        # Verify that student has the necessary components
        self._verify_student_components()
        
        logger.info(f"Initialized ReasoningDistiller with temperature={config.temperature}, alpha={config.alpha}")
        
    def _verify_student_components(self):
        """Verify that the student has the necessary components for distillation"""
        missing_components = []
        
        if self.config.distill_reasoning_trace:
            # Check for reasoning components in student
            if self.config.reasoning_component_weights.get("neural_symbolic", 0) > 0 and not hasattr(self.student, "neural_symbolic"):
                missing_components.append("neural_symbolic")
            if self.config.reasoning_component_weights.get("tree_reasoning", 0) > 0 and not hasattr(self.student, "tree_reasoner"):
                missing_components.append("tree_reasoning")
            if self.config.reasoning_component_weights.get("recursive_reasoning", 0) > 0 and not hasattr(self.student, "recursive_reasoner"):
                missing_components.append("recursive_reasoning")
            if self.config.reasoning_component_weights.get("knowledge_reasoning", 0) > 0 and not hasattr(self.student, "knowledge_reasoner"):
                missing_components.append("knowledge_reasoning")
            if self.config.reasoning_component_weights.get("verifiable_computation", 0) > 0 and not hasattr(self.student, "verifiable_computation"):
                missing_components.append("verifiable_computation")
        
        if missing_components:
            logger.warning(f"Student model is missing components for distillation: {', '.join(missing_components)}")
            logger.warning("Distillation will proceed but may not be optimal")
    
    def distillation_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss between teacher and student outputs
        
        Args:
            teacher_outputs: Outputs from teacher model
            student_outputs: Outputs from student model
            labels: Optional ground truth labels
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        # Distill output logits if available
        if "logits" in teacher_outputs and "logits" in student_outputs:
            teacher_logits = teacher_outputs["logits"] / self.config.temperature
            student_logits = student_outputs["logits"] / self.config.temperature
            
            # Compute KL divergence loss
            distill_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean"
            ) * (self.config.temperature ** 2)
            
            losses["distill_logits"] = distill_loss
            total_loss += distill_loss
        
        # Distill hidden states if enabled
        if self.config.distill_hidden_states and "hidden_states" in teacher_outputs and "hidden_states" in student_outputs:
            teacher_hidden = teacher_outputs["hidden_states"]
            student_hidden = student_outputs["hidden_states"]
            
            # MSE loss for hidden states
            hidden_loss = F.mse_loss(student_hidden, teacher_hidden)
            losses["distill_hidden"] = hidden_loss
            total_loss += hidden_loss
        
        # Distill reasoning traces if enabled
        if self.config.distill_reasoning_trace:
            teacher_trace = teacher_outputs.get("reasoning_trace", {})
            student_trace = student_outputs.get("reasoning_trace", {})
            
            # Distill each reasoning component
            for component, weight in self.config.reasoning_component_weights.items():
                if weight > 0 and component in teacher_trace and component in student_trace:
                    # Extract component outputs
                    teacher_comp = teacher_trace[component]
                    student_comp = student_trace[component]
                    
                    # Handle different component output types
                    if isinstance(teacher_comp, torch.Tensor) and isinstance(student_comp, torch.Tensor):
                        # Direct MSE loss for tensor outputs
                        comp_loss = F.mse_loss(student_comp, teacher_comp) * weight
                        losses[f"distill_{component}"] = comp_loss
                        total_loss += comp_loss
                    elif isinstance(teacher_comp, dict) and isinstance(student_comp, dict):
                        # For dictionary outputs, compute loss for each tensor
                        for key in teacher_comp:
                            if key in student_comp and isinstance(teacher_comp[key], torch.Tensor) and isinstance(student_comp[key], torch.Tensor):
                                comp_loss = F.mse_loss(student_comp[key], teacher_comp[key]) * weight
                                losses[f"distill_{component}_{key}"] = comp_loss
                                total_loss += comp_loss
        
        # Add task loss if labels are provided
        if labels is not None and "logits" in student_outputs:
            task_loss = F.cross_entropy(student_outputs["logits"], labels)
            losses["task_loss"] = task_loss
            
            # Combine task loss and distillation loss
            combined_loss = self.config.alpha * total_loss + (1 - self.config.alpha) * task_loss
            losses["combined_loss"] = combined_loss
            return losses
        
        losses["total_loss"] = total_loss
        return losses
    
    def forward_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform forward pass through both teacher and student models and compute distillation loss
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels
            
        Returns:
            Dictionary of losses
        """
        # Forward pass through teacher model (no gradient)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Forward pass through student model
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Compute distillation loss
        losses = self.distillation_loss(teacher_outputs, student_outputs, labels)
        
        return losses
    
    def create_smaller_student(
        self,
        teacher_config: ArchitectureConfig,
        size_reduction_factor: float = 0.5
    ) -> EnhancedLanguageModel:
        """
        Create a smaller student model based on the teacher's architecture
        
        Args:
            teacher_config: Configuration of the teacher model
            size_reduction_factor: Factor to reduce the model size by
            
        Returns:
            Smaller student model
        """
        # Create a new config with reduced size
        student_config = ArchitectureConfig(
            # Core architecture with reduced size
            hidden_size=int(teacher_config.hidden_size * size_reduction_factor),
            num_layers=max(1, int(teacher_config.num_layers * size_reduction_factor)),
            num_attention_heads=max(1, int(teacher_config.num_attention_heads * size_reduction_factor)),
            intermediate_size=int(teacher_config.intermediate_size * size_reduction_factor),
            
            # Keep the same reasoning components
            use_moe=teacher_config.use_moe,
            use_enhanced_memory=teacher_config.use_enhanced_memory,
            use_tree_of_thought=teacher_config.use_tree_of_thought,
            use_neural_symbolic=teacher_config.use_neural_symbolic,
            use_recursive_reasoning_transformer=teacher_config.use_recursive_reasoning_transformer,
            use_knowledge_reasoning=teacher_config.use_knowledge_reasoning,
            use_verifiable_computation=teacher_config.use_verifiable_computation,
            
            # Reduce expert counts if using MoE
            num_experts=max(1, int(teacher_config.num_experts * size_reduction_factor)) if hasattr(teacher_config, 'num_experts') else None,
            
            # Keep other parameters the same
            model_name=teacher_config.model_name
        )
        
        # Create student model
        student_model = EnhancedLanguageModel(student_config)
        
        logger.info(f"Created smaller student model with {sum(p.numel() for p in student_model.parameters())} parameters")
        logger.info(f"Teacher model has {sum(p.numel() for p in self.teacher.parameters())} parameters")
        
        return student_model 

class DistillationTrainer:
    """Trainer for knowledge distillation from a teacher model to a student model"""
    
    def __init__(
        self,
        model: nn.Module,
        teacher_model: nn.Module,
        distillation_config: DistillationConfig,
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
        self.teacher_model = teacher_model
        self.distillation_config = distillation_config
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
        
        # Move teacher model to device if it's a torch module
        if isinstance(self.teacher_model, nn.Module):
            self.teacher_model.to(self.device)
            self.teacher_model.eval()  # Put teacher in evaluation mode
        
        # Progressive distillation
        if self.distillation_config.use_progressive_distillation:
            self.progressive_stage_idx = self.distillation_config.progressive_stages.index(
                self.distillation_config.current_stage
            )
        else:
            self.progressive_stage_idx = 0
        
        # Tracking variables
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.checkpoints = []
    
    def get_teacher_outputs(self, batch):
        """Get teacher model outputs for a batch"""
        # API-based teacher
        if not isinstance(self.teacher_model, nn.Module):
            # Get input text from batch tokens
            input_ids = batch["input_ids"]
            
            # Convert input ids to text using a function (implementation depends on tokenizer)
            # This is a placeholder, actual implementation would depend on your tokenizer
            input_texts = ["<placeholder>" for _ in range(input_ids.size(0))]
            
            # Get API predictions
            return self.teacher_model.get_outputs(input_texts)
        
        # Torch module teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch.get("attention_mask", None),
                labels=batch.get("labels", None)
            )
            
            # Format teacher outputs
            return {
                "logits": teacher_outputs["logits"] if isinstance(teacher_outputs, dict) else teacher_outputs[0],
                "hidden_states": teacher_outputs.get("hidden_states", None),
                "attentions": teacher_outputs.get("attentions", None)
            }
    
    def update_progressive_stage(self, metrics):
        """Update progressive distillation stage based on metrics"""
        if not self.distillation_config.use_progressive_distillation:
            return False
            
        if self.progressive_stage_idx >= len(self.distillation_config.progressive_stages) - 1:
            # Already at the last stage
            return False
            
        # Check if we should advance to the next stage
        advance = False
        
        # Simple heuristic: advance if eval loss is below a threshold or hasn't improved for a while
        if "eval_loss" in metrics and metrics["eval_loss"] < 0.1:
            advance = True
        
        if advance:
            self.progressive_stage_idx += 1
            current_stage = self.distillation_config.progressive_stages[self.progressive_stage_idx]
            logger.info(f"Advancing to progressive distillation stage: {current_stage}")
            
            # Update distillation parameters based on stage
            if current_stage == "base":
                self.distillation_config.distill_attention = False
                self.distillation_config.distill_hidden_states = False
                self.distillation_config.distill_logits = True
                self.distillation_config.alpha = 0.3
            elif current_stage == "reasoning":
                self.distillation_config.distill_attention = True
                self.distillation_config.distill_hidden_states = True
                self.distillation_config.distill_logits = True
                self.distillation_config.alpha = 0.5
            elif current_stage == "domain_specific":
                self.distillation_config.distill_attention = True
                self.distillation_config.distill_hidden_states = True
                self.distillation_config.distill_logits = True
                self.distillation_config.alpha = 0.7
            elif current_stage == "fine_tuning":
                self.distillation_config.distill_attention = False
                self.distillation_config.distill_hidden_states = False
                self.distillation_config.distill_logits = False
                self.distillation_config.alpha = 0.1
            
            return True
        
        return False
    
    def train(self):
        """Train the model with knowledge distillation"""
        logger.info("Starting training with knowledge distillation")
        
        total_train_steps = len(self.train_dataloader) * self.num_epochs
        progress_bar = tqdm(total=total_train_steps, desc="Training")
        
        # Training loop
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            self.model.train()
            
            # Training metrics
            train_loss = 0.0
            train_student_loss = 0.0
            train_distillation_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Get teacher outputs
                teacher_outputs = self.get_teacher_outputs(batch)
                
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
                        labels=batch.get("labels", None),
                        teacher_model_outputs=teacher_outputs
                    )
                    
                    # Get losses
                    loss = outputs["loss"]
                    distillation_losses = outputs.get("distillation_losses", {})
                    
                    # Scale loss for gradient accumulation
                    scaled_loss = loss / self.gradient_accumulation_steps
                
                # Backward pass with mixed precision if enabled
                if self.mixed_precision and self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Update metrics
                train_loss += loss.item()
                train_student_loss += outputs.get("student_loss", loss).item()
                if "total_distillation_loss" in distillation_losses:
                    train_distillation_loss += distillation_losses["total_distillation_loss"].item()
                
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
                        
                        # Update progressive stage
                        self.update_progressive_stage(eval_metrics)
                        
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
                        avg_student_loss = train_student_loss / num_batches
                        avg_distillation_loss = train_distillation_loss / num_batches if train_distillation_loss > 0 else 0
                        
                        # Log metrics
                        metrics = {
                            "train_loss": avg_train_loss,
                            "train_student_loss": avg_student_loss,
                            "train_distillation_loss": avg_distillation_loss,
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
            avg_student_loss = train_student_loss / num_batches
            avg_distillation_loss = train_distillation_loss / num_batches if train_distillation_loss > 0 else 0
            
            # Log epoch metrics
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} completed in {epoch_time:.2f}s")
            logger.info(f"  Train loss: {avg_train_loss:.4f}")
            logger.info(f"  Student loss: {avg_student_loss:.4f}")
            logger.info(f"  Distillation loss: {avg_distillation_loss:.4f}")
            
            # Evaluation at the end of epoch
            if self.eval_strategy == "epoch":
                eval_metrics = self.evaluate()
                
                # Update progressive stage
                self.update_progressive_stage(eval_metrics)
                
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
        eval_student_loss = 0.0
        eval_distillation_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Get teacher outputs
                teacher_outputs = self.get_teacher_outputs(batch)
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask", None),
                    labels=batch.get("labels", None),
                    teacher_model_outputs=teacher_outputs
                )
                
                # Get losses
                loss = outputs["loss"]
                distillation_losses = outputs.get("distillation_losses", {})
                
                # Update metrics
                eval_loss += loss.item()
                eval_student_loss += outputs.get("student_loss", loss).item()
                if "total_distillation_loss" in distillation_losses:
                    eval_distillation_loss += distillation_losses["total_distillation_loss"].item()
                
                num_batches += 1
        
        # Calculate average metrics
        avg_eval_loss = eval_loss / num_batches
        avg_student_loss = eval_student_loss / num_batches
        avg_distillation_loss = eval_distillation_loss / num_batches if eval_distillation_loss > 0 else 0
        
        # Create metrics dictionary
        metrics = {
            "eval_loss": avg_eval_loss,
            "eval_student_loss": avg_student_loss,
            "eval_distillation_loss": avg_distillation_loss,
            "step": self.global_step
        }
        
        # Log evaluation metrics
        logger.info(f"Evaluation results:")
        logger.info(f"  Eval loss: {avg_eval_loss:.4f}")
        logger.info(f"  Student loss: {avg_student_loss:.4f}")
        logger.info(f"  Distillation loss: {avg_distillation_loss:.4f}")
        
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
            "progressive_stage_idx": self.progressive_stage_idx,
            "distillation_config": vars(self.distillation_config),
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
            
            if "progressive_stage_idx" in training_state:
                self.progressive_stage_idx = training_state["progressive_stage_idx"]
                
                # Update distillation config
                if "distillation_config" in training_state:
                    for k, v in training_state["distillation_config"].items():
                        if hasattr(self.distillation_config, k):
                            setattr(self.distillation_config, k, v)
        
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