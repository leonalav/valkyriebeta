import os
import sys
import math
import time
import json
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm import tqdm
import copy
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext

# Import model utilities
from model.utils import ModelUtils
from model.architecture_components import ModelRegistry, ModelFactory
from model.integration import ModelIntegrator
from model.inference import InferenceOptimizer

# Import optimization utilities
from utils.optimization import (
    OptimizationConfig,
    ModelOptimizer,
    MemoryOptimizer,
    GradientOptimizer,
    find_optimal_checkpoint_config
)

# Import numerical precision components
from model.numerical_precision import (
    NumericalPrecisionConfig,
    NumericalPrecisionModule,
    NumericallyStableOperations
)
from model.math_precision_integration import MathPrecisionManager

# Import validation components
from validators import ModelValidator, ConfigValidator, ValidationResult

logger = logging.getLogger(__name__)

class TrainingEngine:
    """
    Enhanced training engine with all advanced features from train_aio.py
    """
    
    def __init__(
        self,
        model,
        teacher_model=None,
        optimizer=None,
        lr_scheduler=None,
        training_config=None,
        tokenizer=None
    ):
        """Initialize the enhanced training engine"""
        self.model = model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_config = training_config
        self.tokenizer = tokenizer
        
        # Set default device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            
        # Set up distributed training
        self.is_distributed = False
        self.local_rank = -1
        self.world_size = 1
        
        if hasattr(training_config, 'use_distributed') and training_config.use_distributed:
            if hasattr(training_config, 'local_rank'):
                self.local_rank = training_config.local_rank
                
            if self.local_rank != -1:
                self.is_distributed = True
                self.setup_distributed(self.local_rank)
                
        # Set up mixed precision
        self.use_mixed_precision = hasattr(training_config, 'use_mixed_precision') and training_config.use_mixed_precision
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Set up gradient accumulation
        self.gradient_accumulation_steps = getattr(training_config, 'gradient_accumulation_steps', 1)
        
        # Set up knowledge distillation
        self.use_distillation = hasattr(training_config, 'use_distillation') and training_config.use_distillation
        self.distillation_alpha = getattr(training_config, 'distillation_alpha', 0.5)
        self.distillation_temperature = getattr(training_config, 'distillation_temperature', 2.0)
        
        # Set up API distillation
        self.use_api_distillation = hasattr(training_config, 'use_api_distillation') and training_config.use_api_distillation
        if self.use_api_distillation:
            from model.api_distillation import APIBasedDistillation
            self.api_distillation = APIBasedDistillation(
                hidden_size=getattr(training_config, 'hidden_size', 768),
                api_config=getattr(training_config, 'api_config', None)
            )
            
        # Set up multi-teacher distillation
        self.use_multi_teacher = hasattr(training_config, 'use_multi_teacher') and training_config.use_multi_teacher
        if self.use_multi_teacher:
            self.teacher_models = getattr(training_config, 'teacher_models', [])
            self.teacher_weights = getattr(training_config, 'teacher_weights', None)
            
        # Set up intermediate layer distillation
        self.use_intermediate_distillation = hasattr(training_config, 'use_intermediate_distillation') and training_config.use_intermediate_distillation
        
        # Set up domain-specific training
        self.use_domain_training = hasattr(training_config, 'use_domain_training') and training_config.use_domain_training
        self.domain_weights = getattr(training_config, 'domain_weights', None)
        
        # Set up meta-learning
        self.use_meta_learning = hasattr(training_config, 'use_meta_learning') and training_config.use_meta_learning
        if self.use_meta_learning:
            from model.meta_learning import MetaLearner
            self.meta_learner = MetaLearner(
                hidden_size=getattr(training_config, 'hidden_size', 768),
                num_tasks=getattr(training_config, 'num_tasks', 10)
            )
            
        # Set up MoE
        self.use_moe = hasattr(training_config, 'use_moe') and training_config.use_moe
        if self.use_moe:
            from model.moe import ExpertGating
            self.expert_gating = ExpertGating(
                hidden_size=getattr(training_config, 'hidden_size', 768),
                num_experts=getattr(training_config, 'num_experts', 8)
            )
            
        # Set up memory mechanisms
        self.use_memory = hasattr(training_config, 'use_memory') and training_config.use_memory
        if self.use_memory:
            from model.memory import EnhancedMemory, EpisodicMemory, WorkingMemory, LongTermMemory
            self.memory_components = {
                'episodic': EpisodicMemory(size=getattr(training_config, 'episodic_memory_size', 1024)),
                'working': WorkingMemory(size=getattr(training_config, 'working_memory_size', 512)),
                'long_term': LongTermMemory(size=getattr(training_config, 'long_term_memory_size', 4096))
            }
            
        # Set up RLHF
        self.use_rlhf = hasattr(training_config, 'use_rlhf') and training_config.use_rlhf
        if self.use_rlhf:
            self.rlhf_algorithm = getattr(training_config, 'rlhf_algorithm', 'ppo')
            
            # Set up PPO components
            if self.rlhf_algorithm == 'ppo':
                from model.reinforcement.ppo import PPOTrainer
                self.ppo_trainer = PPOTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    reward_model=getattr(training_config, 'reward_model', None),
                    reference_model=getattr(training_config, 'reference_model', None)
                )
                
            # Set up DPO components
            elif self.rlhf_algorithm == 'dpo':
                from model.reinforcement.dpo import DPOTrainer
                self.dpo_trainer = DPOTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    reference_model=getattr(training_config, 'reference_model', None)
                )
                
            # Set up Constitutional AI components
            elif self.rlhf_algorithm == 'constitutional':
                from model.constitutional_ai import ConstitutionalAITrainer
                self.constitutional_trainer = ConstitutionalAITrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    constitution_file=getattr(training_config, 'constitution_file', None)
                )
                
        # Set up model compilation
        if hasattr(training_config, 'use_compile') and training_config.use_compile and hasattr(torch, 'compile'):
            self.model = torch.compile(
                self.model,
                backend=getattr(training_config, 'dynamo_backend', None)
            )
            
        # Initialize tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Set up numerical precision
        self.use_numerical_precision = hasattr(training_config, 'use_numerical_precision') and training_config.use_numerical_precision
        if self.use_numerical_precision:
            self.numerical_precision_config = NumericalPrecisionConfig(
                mode=getattr(training_config, 'numerical_precision_mode', 'auto'),
                use_fp8_matmul=getattr(training_config, 'use_fp8_matmul', False),
                use_stable_embedding=getattr(training_config, 'use_stable_embedding', True)
            )
            self.numerical_precision_module = NumericalPrecisionModule(self.numerical_precision_config)
            self.math_precision_manager = MathPrecisionManager()
            
        # Set up optimization components
        self.optimization_config = OptimizationConfig(
            use_gradient_checkpointing=getattr(training_config, 'use_gradient_checkpointing', True),
            use_activation_checkpointing=getattr(training_config, 'use_activation_checkpointing', True),
            optimize_memory_use=getattr(training_config, 'optimize_memory_use', True),
            optimize_cuda_kernels=getattr(training_config, 'optimize_cuda_kernels', True)
        )
        
        self.model_optimizer = ModelOptimizer(self.optimization_config)
        self.memory_optimizer = MemoryOptimizer(self.optimization_config)
        self.gradient_optimizer = GradientOptimizer(self.optimization_config)
        
        # Set up model registry and integrator
        self.model_registry = ModelRegistry()
        self.model_integrator = ModelIntegrator()
        self.inference_optimizer = InferenceOptimizer()
        
        # Set up validators
        self.model_validator = ModelValidator()
        self.config_validator = ConfigValidator()
        
    def setup_distributed(self, local_rank):
        """
        Set up distributed training
        
        Args:
            local_rank: Local rank of the process
        """
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        self.world_size = dist.get_world_size()
        
        # Wrap model with DDP
        from torch.nn.parallel import DistributedDataParallel as DDP
        self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
        
        self.logger.info(f"Initialized distributed training with world size {self.world_size}")
        
    def train_epoch(self, train_dataloader, domain_dataloaders=None, epoch=0):
        """
        Train for one epoch
        
        Args:
            train_dataloader: DataLoader for training data
            domain_dataloaders: Optional dictionary of domain-specific dataloaders
            epoch: Current epoch number
            
        Returns:
            train_loss: Average training loss for the epoch
        """
        self.model.train()
        self.epoch = epoch
        
        total_loss = 0
        total_steps = 0
        
        # Set up progress bar
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        
        # Set up domain iterators if using domain-specific training
        domain_iterators = None
        if self.use_domain_training and domain_dataloaders is not None:
            domain_iterators = {domain: iter(dataloader) for domain, dataloader in domain_dataloaders.items()}
            
        # Training loop
        for step, batch in enumerate(train_dataloader):
            # Determine if this is an accumulation step
            is_accumulation_step = (step + 1) % self.gradient_accumulation_steps == 0
            
            # Process batch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Perform training step
            step_loss = self._training_step(batch, is_accumulation_step)
            
            # Process domain-specific batches if enabled
            if self.use_domain_training and domain_iterators is not None:
                domain_loss = self._domain_training_steps(domain_iterators, self.domain_weights, domain_dataloaders, is_accumulation_step)
                step_loss += domain_loss
                
            # Update tracking
            total_loss += step_loss
            total_steps += 1
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': step_loss})
            
            # Update global step
            if is_accumulation_step:
                self.global_step += 1
                
                # Update learning rate scheduler
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    
        # Close progress bar
        progress_bar.close()
        
        # Calculate average loss
        avg_loss = total_loss / total_steps
        
        return avg_loss
        
    def _training_step(self, batch, is_accumulation_step):
        """Enhanced training step with all advanced features"""
        # Process batch
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Set up context managers
        if self.use_mixed_precision:
            context_manager = autocast()
        else:
            context_manager = nullcontext()
            
        # Forward pass
        with context_manager:
            # Apply meta-learning if enabled
            if self.use_meta_learning:
                outputs = self.meta_learner(self.model, batch)
            else:
                outputs = self.model(**batch)
                
            # Apply numerical precision optimizations
            outputs = self._apply_numerical_precision(outputs)
                
            # Get base loss
            loss = outputs['loss'] if 'loss' in outputs else outputs.loss
            if 'aux_loss' in outputs:
                loss = loss + outputs['aux_loss']
                
            # Apply knowledge distillation if enabled
            if self.use_distillation:
                distillation_loss = 0
                
                # Standard knowledge distillation
                if self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(**batch)
                    teacher_logits = teacher_outputs['logits'] if isinstance(teacher_outputs, dict) else teacher_outputs[0]
                    distillation_loss += self._calculate_distillation_loss(outputs['logits'], teacher_logits)
                    
                # Multi-teacher distillation
                if self.use_multi_teacher:
                    for i, teacher in enumerate(self.teacher_models):
                        with torch.no_grad():
                            teacher_outputs = teacher(**batch)
                        teacher_logits = teacher_outputs['logits'] if isinstance(teacher_outputs, dict) else teacher_outputs[0]
                        weight = self.teacher_weights[i] if self.teacher_weights else 1.0 / len(self.teacher_models)
                        distillation_loss += weight * self._calculate_distillation_loss(outputs['logits'], teacher_logits)
                        
                # Intermediate layer distillation
                if self.use_intermediate_distillation:
                    intermediate_loss = self._calculate_intermediate_distillation_loss(outputs, teacher_outputs)
                    distillation_loss += intermediate_loss
                    
                # API distillation
                if self.use_api_distillation:
                    api_loss = self.api_distillation(outputs, batch)
                    distillation_loss += api_loss
                    
                # Combine losses
                loss = (1 - self.distillation_alpha) * loss + self.distillation_alpha * distillation_loss
                
            # Apply memory mechanisms if enabled
            if self.use_memory:
                memory_loss = 0
                for memory_component in self.memory_components.values():
                    memory_output = memory_component(outputs)
                    memory_loss += memory_output['loss']
                loss += memory_loss
                
            # Apply MoE if enabled
            if self.use_moe:
                moe_loss = self.expert_gating(outputs)
                loss += moe_loss
                
            # Scale loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
                
        # Backward pass
        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
            
            if not is_accumulation_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.training_config, 'max_grad_norm', 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            
            if not is_accumulation_step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.training_config, 'max_grad_norm', 1.0))
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        return loss.item()
        
    def _domain_training_steps(self, domain_iterators, domain_weights, domain_dataloaders, is_accumulation_step):
        """
        Perform training steps for domain-specific data
        
        Args:
            domain_iterators: Dictionary of domain iterators
            domain_weights: Dictionary of domain weights
            domain_dataloaders: Dictionary of domain dataloaders
            is_accumulation_step: Whether this is an accumulation step
            
        Returns:
            loss: Combined loss for domain-specific steps
        """
        total_domain_loss = 0
        
        # Process each domain
        for domain, iterator in domain_iterators.items():
            try:
                # Get batch for domain
                batch = next(iterator)
            except StopIteration:
                # Reset iterator if it's exhausted
                domain_iterators[domain] = iter(domain_dataloaders[domain])
                batch = next(domain_iterators[domain])
                
            # Process batch
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Get domain weight
            weight = domain_weights.get(domain, 1.0)
            
            # Perform training step
            domain_loss = self._training_step(batch, False)  # Don't update weights yet
            
            # Apply domain weight
            total_domain_loss += weight * domain_loss
            
        # Update weights if this is an accumulation step
        if is_accumulation_step:
            if self.use_mixed_precision:
                # Unscale gradients
                self.scaler.unscale_(self.optimizer)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.training_config, 'max_grad_norm', 1.0))
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.training_config, 'max_grad_norm', 1.0))
                
                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        return total_domain_loss
        
    def _calculate_distillation_loss(self, student_logits, teacher_logits):
        """
        Calculate knowledge distillation loss
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            
        Returns:
            loss: Distillation loss
        """
        # Apply temperature scaling
        scaled_student_logits = student_logits / self.distillation_temperature
        scaled_teacher_logits = teacher_logits / self.distillation_temperature
        
        # Calculate KL divergence
        loss_fct = nn.KLDivLoss(reduction='none')
        loss = loss_fct(
            F.log_softmax(scaled_student_logits, dim=-1),
            F.softmax(scaled_teacher_logits, dim=-1)
        )
        
        return loss.mean()
        
    def _calculate_intermediate_distillation_loss(self, student_outputs, teacher_outputs):
        """Calculate distillation loss for intermediate layers"""
        loss = 0
        
        # Get intermediate representations
        student_intermediates = student_outputs.get('hidden_states', [])
        teacher_intermediates = teacher_outputs.get('hidden_states', [])
        
        # Calculate loss for each layer
        for student_layer, teacher_layer in zip(student_intermediates, teacher_intermediates):
            layer_loss = F.mse_loss(student_layer, teacher_layer.detach())
            loss += layer_loss
            
        return loss / len(student_intermediates) if student_intermediates else 0
        
    def validate(self, val_dataloader, comprehensive=False):
        """
        Validate the model
        
        Args:
            val_dataloader: DataLoader for validation data
            comprehensive: Whether to perform comprehensive evaluation
            
        Returns:
            val_loss: Validation loss
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_steps = 0
        
        # Set up progress bar
        progress_bar = tqdm(total=len(val_dataloader), desc="Validation")
        
        # Validation loop
        with torch.no_grad():
            for batch in val_dataloader:
                # Process batch
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get inputs
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask', None)
                labels = batch.get('labels', input_ids)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs
                
                # Calculate loss
                loss_fct = CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Update tracking
                total_loss += loss.item()
                total_steps += 1
                
                # Update progress bar
                progress_bar.update(1)
                
        # Close progress bar
        progress_bar.close()
        
        # Calculate average loss
        avg_loss = total_loss / total_steps
        
        # Calculate additional metrics if comprehensive evaluation is enabled
        metrics = {'loss': avg_loss}
        if comprehensive:
            # Calculate perplexity
            metrics['perplexity'] = math.exp(avg_loss)
            
        return avg_loss, metrics
        
    def setup_optimizer(self, learning_rate, weight_decay):
        """
        Set up optimizer
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        # Create parameter groups with different learning rates if needed
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
                'weight_decay': 0.0
            }
        ]
        
        # Create optimizer
        if self.training_config.use_fused_adam and torch.cuda.is_available():
            from apex.optimizers import FusedAdam
            self.optimizer = FusedAdam(param_groups, lr=learning_rate)
        else:
            self.optimizer = AdamW(param_groups, lr=learning_rate)
            
        self.logger.info(f"Optimizer set up with learning rate {learning_rate}")
        
    def setup_lr_scheduler(self, num_epochs, steps_per_epoch):
        """
        Set up learning rate scheduler
        
        Args:
            num_epochs: Number of training epochs
            steps_per_epoch: Number of steps per epoch
        """
        # Calculate total number of training steps
        total_steps = num_epochs * steps_per_epoch
        
        # Calculate warmup steps
        warmup_steps = getattr(self.training_config, 'warmup_steps', 0)
        if hasattr(self.training_config, 'warmup_ratio') and self.training_config.warmup_ratio > 0:
            warmup_steps = int(total_steps * self.training_config.warmup_ratio)
            
        # Create scheduler
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
        
        self.logger.info(f"Learning rate scheduler set up with {warmup_steps} warmup steps")
        
    def train(self, train_dataloader, val_dataloader, domain_dataloaders=None, epochs=1, output_dir="output", experiment_name="Valkyrie"):
        """
        Train the model
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            domain_dataloaders: Optional dictionary of domain-specific dataloaders
            epochs: Number of training epochs
            output_dir: Directory to save model checkpoints
            experiment_name: Name of the experiment
            
        Returns:
            model: Trained model
            metrics: Dictionary of training metrics
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up tracking
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        # Set up learning rate scheduler if not already set up
        if self.lr_scheduler is None:
            self.setup_lr_scheduler(epochs, len(train_dataloader))
            
        # Training loop
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = self.train_epoch(train_dataloader, domain_dataloaders, epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate(val_dataloader, comprehensive=True)
            val_losses.append(val_loss)
            
            # Log progress
            self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Perplexity = {val_metrics.get('perplexity', 0):.4f}")
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Save model
                model_path = os.path.join(output_dir, f"{experiment_name}_best.pt")
                self._save_checkpoint(model_path, epoch, val_metrics)
                
                self.logger.info(f"Saved best model checkpoint to {model_path}")
                
            # Save regular checkpoint
            if (epoch + 1) % getattr(self.training_config, 'save_steps', 1) == 0:
                model_path = os.path.join(output_dir, f"{experiment_name}_epoch_{epoch}.pt")
                self._save_checkpoint(model_path, epoch, val_metrics)
                
        # Save final model
        model_path = os.path.join(output_dir, f"{experiment_name}_final.pt")
        self._save_checkpoint(model_path, epochs - 1, {'loss': val_losses[-1]})
        
        # Return metrics
        metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
        
        return self.model, metrics
        
    def train_with_rlhf(self, train_dataloader, val_dataloader, rlhf_dataloader, epochs=1, output_dir="output", experiment_name="Valkyrie_RLHF"):
        """Enhanced RLHF training with multiple algorithms"""
        if not self.use_rlhf:
            raise ValueError("RLHF training is not enabled")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training loop
        metrics = {
            'train_losses': [],
            'rlhf_losses': [],
            'val_losses': []
        }
        
        for epoch in range(epochs):
            # Standard training
            train_loss = self.train_epoch(train_dataloader, epoch=epoch)
            metrics['train_losses'].append(train_loss)
            
            # RLHF training
            if self.rlhf_algorithm == 'ppo':
                rlhf_metrics = self._train_ppo(rlhf_dataloader)
            elif self.rlhf_algorithm == 'dpo':
                rlhf_metrics = self._train_dpo(rlhf_dataloader)
            elif self.rlhf_algorithm == 'constitutional':
                rlhf_metrics = self._train_constitutional(rlhf_dataloader)
                
            metrics['rlhf_losses'].append(rlhf_metrics['loss'])
            
            # Validation
            val_loss, _ = self.validate(val_dataloader)
            metrics['val_losses'].append(val_loss)
            
            # Save checkpoint
            self._save_checkpoint(
                os.path.join(output_dir, f"{experiment_name}_epoch_{epoch}.pt"),
                epoch,
                {'train_loss': train_loss, 'rlhf_loss': rlhf_metrics['loss'], 'val_loss': val_loss}
            )
            
            # Log progress
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, RLHF Loss = {rlhf_metrics['loss']:.4f}, Val Loss = {val_loss:.4f}")
            
        return self.model, metrics
        
    def _train_ppo(self, dataloader):
        """Train with PPO algorithm"""
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="PPO Training"):
            # Get prompts and responses
            prompts = batch['prompts'].to(self.device)
            responses = batch['responses'].to(self.device)
            
            # Run PPO step
            metrics = self.ppo_trainer.step(prompts, responses)
            total_loss += metrics['loss']
            num_batches += 1
            
        return {'loss': total_loss / num_batches}
        
    def _train_dpo(self, dataloader):
        """Train with DPO algorithm"""
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="DPO Training"):
            # Get prompts and responses
            prompts = batch['prompts'].to(self.device)
            chosen = batch['chosen_responses'].to(self.device)
            rejected = batch['rejected_responses'].to(self.device)
            
            # Run DPO step
            metrics = self.dpo_trainer.step(prompts, chosen, rejected)
            total_loss += metrics['loss']
            num_batches += 1
            
        return {'loss': total_loss / num_batches}
        
    def _train_constitutional(self, dataloader):
        """Train with Constitutional AI"""
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Constitutional AI Training"):
            # Get prompts and responses
            prompts = batch['prompts'].to(self.device)
            responses = batch['responses'].to(self.device)
            
            # Run Constitutional AI step
            metrics = self.constitutional_trainer.step(prompts, responses)
            total_loss += metrics['loss']
            num_batches += 1
            
        return {'loss': total_loss / num_batches}
        
    def _save_checkpoint(self, path, epoch, metrics):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict() if not self.is_distributed else self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'metrics': metrics
        }
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
    def quantize_model(self, quantization_method='dynamic', quantization_bits=8):
        """
        Quantize the model for inference
        
        Args:
            quantization_method: Method for quantization ('dynamic', 'static', or 'aware')
            quantization_bits: Number of bits for quantization (8 or 4)
            
        Returns:
            model: Quantized model
        """
        # Make sure model is in eval mode
        self.model.eval()
        
        # Get model to quantize (unwrap DDP if needed)
        model_to_quantize = self.model.module if self.is_distributed else self.model
        
        # Perform quantization
        if quantization_method == 'dynamic':
            # Dynamic quantization
            try:
                import torch.quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model_to_quantize,
                    {nn.Linear},
                    dtype=torch.qint8 if quantization_bits == 8 else torch.qint4
                )
            except Exception as e:
                logger.error(f"Failed to apply dynamic quantization: {e}")
                return self.model
        elif quantization_method == 'static':
            # Static quantization (requires calibration)
            try:
                import torch.quantization
                
                # Prepare model for static quantization
                model_to_quantize.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model_to_quantize, inplace=True)
                
                # Calibrate (would need calibration data)
                logger.warning("Static quantization requires calibration data")
                
                # Convert to quantized model
                quantized_model = torch.quantization.convert(model_to_quantize, inplace=False)
            except Exception as e:
                logger.error(f"Failed to apply static quantization: {e}")
                return self.model
        elif quantization_method == 'aware':
            # Quantization-aware training (requires training)
            raise NotImplementedError("Quantization-aware training not implemented yet")
        else:
            raise ValueError(f"Unknown quantization method: {quantization_method}")
            
        return quantized_model
        
    def validate_setup(self) -> ValidationResult:
        """
        Validate the training setup including model, configuration, and components.
        
        Returns:
            ValidationResult: Object containing validation results
        """
        # Validate model
        model_validation = self.model_validator.validate_model(self.model)
        
        # Validate configuration
        config_validation = self.config_validator.validate_config(self.training_config)
        
        # Combine validation results
        validation_result = ValidationResult()
        validation_result.errors.extend(model_validation.errors)
        validation_result.errors.extend(config_validation.errors)
        validation_result.warnings.extend(model_validation.warnings)
        validation_result.warnings.extend(config_validation.warnings)
        
        return validation_result
        
    def optimize_training_setup(self):
        """
        Optimize the training setup for better performance
        """
        # Optimize model architecture
        self.model = self.model_optimizer.optimize(self.model)
        
        # Optimize memory usage
        self.memory_optimizer.optimize(self.model)
        
        # Optimize gradient computation
        self.gradient_optimizer.optimize(self.model)
        
        # Find optimal checkpoint configuration
        checkpoint_config = find_optimal_checkpoint_config(
            model=self.model,
            batch_size=self.training_config.batch_size,
            max_memory=getattr(self.training_config, 'max_memory', None)
        )
        
        # Apply checkpoint configuration
        self.model = self.model_optimizer.apply_checkpoint_config(self.model, checkpoint_config)
        
    def _apply_numerical_precision(self, outputs):
        """
        Apply numerical precision optimizations to model outputs
        
        Args:
            outputs: Model outputs
            
        Returns:
            outputs: Numerically optimized outputs
        """
        if not self.use_numerical_precision:
            return outputs
            
        # Apply numerical precision module
        outputs = self.numerical_precision_module(outputs)
        
        # Apply math precision manager for mathematical operations
        if isinstance(outputs, dict) and 'logits' in outputs:
            outputs['logits'] = self.math_precision_manager.process_logits(outputs['logits'])
            
        return outputs
        
    def prepare_for_inference(self, optimize_for_rwkv=False):
        """
        Prepare the model for inference by applying optimizations
        """
        self.model.eval()
        
        # Apply standard inference optimizations
        model = self.model
        
        # Additional RWKV-specific optimizations
        if optimize_for_rwkv:
            self.logger.info("Applying RWKV-specific inference optimizations")
            
            # Quantize time mixing parameters if applicable
            if hasattr(model, "quantize_time_mixing"):
                model.quantize_time_mixing()
                self.logger.info("Quantized RWKV time mixing parameters")
            
            # Enable specific inference optimizations for RWKV
            if hasattr(model, "optimize_for_inference"):
                model.optimize_for_inference()
                self.logger.info("Applied RWKV inference optimizations")
        
        # Register optimized model
        self.model_registry.register_model(
            name=getattr(self.training_config, 'experiment_name', 'valkyrie'),
            model=model,
            metadata={
                'optimization_level': 'inference',
                'timestamp': time.time()
            }
        )
        
        # Integrate optimizations
        model = self.model_integrator.integrate_optimizations(model)
        
        return model

    # Add RWKV-specific methods to TrainingEngine class
    def setup_rwkv_optimizer(self, optimizer, rwkv_lr_multiplier=1.0, att_weight_decay=0.0, ffn_weight_decay=0.0):
        """
        Configure optimizer with RWKV-specific parameter groups and learning rates
        
        Args:
            optimizer: The optimizer to configure
            rwkv_lr_multiplier: Learning rate multiplier for RWKV-specific parameters
            att_weight_decay: Weight decay for attention parameters
            ffn_weight_decay: Weight decay for feed-forward parameters
            
        Returns:
            Configured optimizer
        """
        self.logger.info(f"Setting up RWKV-specific optimizer groups with lr multiplier {rwkv_lr_multiplier}")
        
        # Group parameters by type
        time_decay_params = []
        time_first_params = []
        channel_mix_params = []
        attention_params = []
        ffn_params = []
        other_params = []
        
        # Collect RWKV-specific parameter groups
        for name, param in self.model.named_parameters():
            if 'time_decay' in name:
                time_decay_params.append(param)
            elif 'time_first' in name or 'time_mix' in name:
                time_first_params.append(param)
            elif 'channel_mix' in name:
                channel_mix_params.append(param)
            elif any(attn_name in name for attn_name in ['att', 'attention', 'wkv']):
                attention_params.append(param)
            elif any(ffn_name in name for ffn_name in ['ffn', 'feed_forward', 'mlp']):
                ffn_params.append(param)
            else:
                other_params.append(param)
        
        self.logger.info(f"Grouped RWKV parameters: {len(time_decay_params)} time decay, "
                        f"{len(time_first_params)} time first/mix, {len(channel_mix_params)} channel mix, "
                        f"{len(attention_params)} attention, {len(ffn_params)} FFN, {len(other_params)} other")
        
        # Configure optimizer with parameter groups
        param_groups = []
        
        # Time decay params need special treatment - higher learning rate, no weight decay
        if time_decay_params:
            param_groups.append({
                'params': time_decay_params, 
                'lr': optimizer.param_groups[0]['lr'] * rwkv_lr_multiplier * 1.5, 
                'weight_decay': 0.0
            })
        
        # Time first/mix params
        if time_first_params:
            param_groups.append({
                'params': time_first_params, 
                'lr': optimizer.param_groups[0]['lr'] * rwkv_lr_multiplier * 1.2, 
                'weight_decay': att_weight_decay / 2
            })
        
        # Channel mix params
        if channel_mix_params:
            param_groups.append({
                'params': channel_mix_params, 
                'lr': optimizer.param_groups[0]['lr'] * rwkv_lr_multiplier, 
                'weight_decay': ffn_weight_decay / 2
            })
        
        # Attention params
        if attention_params:
            param_groups.append({
                'params': attention_params, 
                'lr': optimizer.param_groups[0]['lr'] * rwkv_lr_multiplier, 
                'weight_decay': att_weight_decay
            })
        
        # FFN params
        if ffn_params:
            param_groups.append({
                'params': ffn_params, 
                'lr': optimizer.param_groups[0]['lr'] * rwkv_lr_multiplier, 
                'weight_decay': ffn_weight_decay
            })
        
        # Other params (keep original settings)
        if other_params:
            param_groups.append({
                'params': other_params
            })
        
        # Create new optimizer with the same algorithm but with our parameter groups
        optimizer_config = {k: v for k, v in optimizer.defaults.items() 
                          if k != 'params' and k != 'lr' and k != 'weight_decay'}
        
        optim_class = optimizer.__class__
        self.optimizer = optim_class(param_groups, **optimizer_config)
        
        self.logger.info(f"Created RWKV-optimized optimizer with {len(param_groups)} parameter groups")
        return self.optimizer

    def optimize_rwkv_training_setup(self):
        """
        Apply RWKV-specific optimizations to the training setup
        
        Returns:
            self for method chaining
        """
        self.logger.info("Applying RWKV training optimizations")
        
        # Check if model supports RWKV-specific features
        is_rwkv_model = hasattr(self.model, 'process_with_state') or hasattr(self.training_config, 'use_rwkv')
        
        if not is_rwkv_model:
            self.logger.warning("Model does not appear to be an RWKV model. Skipping RWKV-specific optimizations.")
            return self
        
        # 1. Enable gradient checkpointing for RWKV layers if supported
        if hasattr(self.model, 'enable_rwkv_checkpointing'):
            self.model.enable_rwkv_checkpointing()
            self.logger.info("Enabled gradient checkpointing for RWKV layers")
        elif hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            self.logger.info("Enabled standard gradient checkpointing")
        
        # 2. Apply custom activation checkpointing for state handling
        if 'activation_checkpointing' in self.training_config.__dict__ and self.training_config.activation_checkpointing:
            # Apply activation checkpointing to RWKV blocks if present
            if hasattr(self.model, 'blocks'):
                num_checkpointed = 0
                for i, block in enumerate(self.model.blocks):
                    if i % self.training_config.checkpoint_every_n_layers == 0:
                        if hasattr(block, 'gradient_checkpointing'):
                            block.gradient_checkpointing = True
                            num_checkpointed += 1
                
                if num_checkpointed > 0:
                    self.logger.info(f"Applied activation checkpointing to {num_checkpointed} RWKV blocks")
        
        # 3. Enable RWKV-specific mixed precision handling if needed
        if self.training_config.use_mixed_precision and hasattr(self.model, 'configure_rwkv_precision'):
            self.model.configure_rwkv_precision(dtype='float16')
            self.logger.info("Configured RWKV-specific mixed precision")
        
        # 4. Optimize recurrent state handling for training
        if hasattr(self.model, 'optimize_recurrent_training'):
            self.model.optimize_recurrent_training()
            self.logger.info("Optimized recurrent state handling for training")
        
        # 5. Apply memory optimizations for RWKV
        self._apply_rwkv_memory_optimizations()
        
        return self

    def _apply_rwkv_memory_optimizations(self):
        """Apply memory optimizations specifically for RWKV models"""
        # 1. Configure optimal chunk size based on available memory
        available_memory = getattr(self.memory_config, 'max_memory_MB', None)
        if available_memory:
            # Heuristic: adjust chunk size based on available memory
            if available_memory < 8000:  # Less than 8GB
                optimal_chunk_size = 512
            elif available_memory < 16000:  # Less than 16GB
                optimal_chunk_size = 1024
            elif available_memory < 32000:  # Less than 32GB
                optimal_chunk_size = 2048
            else:  # 32GB or more
                optimal_chunk_size = 4096
            
            # Set chunk size if model supports it
            if hasattr(self.model, 'set_chunk_size'):
                self.model.set_chunk_size(optimal_chunk_size)
                self.logger.info(f"Set optimal RWKV chunk size to {optimal_chunk_size} based on available memory")
        
        # 2. Enable memory-efficient attention for RWKV
        if hasattr(self.model, 'enable_memory_efficient_attention'):
            self.model.enable_memory_efficient_attention()
            self.logger.info("Enabled memory-efficient attention for RWKV")
        
        # 3. Apply state compression for long sequences
        if getattr(self.memory_config, 'use_gradient_checkpointing', False) and hasattr(self.model, 'enable_state_compression'):
            self.model.enable_state_compression()
            self.logger.info("Enabled RWKV state compression for memory efficiency")
        
        # 4. Apply CPU offloading for RWKV weights if specified
        if hasattr(self.training_config, "rwkv_cpu_offload_ratio") and self.training_config.rwkv_cpu_offload_ratio > 0:
            self._apply_rwkv_cpu_offloading(self.training_config.rwkv_cpu_offload_ratio)
        
        # 5. Apply RWKV-specific backward optimizations
        if hasattr(self.training_config, "rwkv_optimize_backward") and self.training_config.rwkv_optimize_backward:
            self._optimize_rwkv_backward()
        
        return True

    def _apply_rwkv_cpu_offloading(self, offload_ratio):
        """Apply CPU offloading for RWKV parameters
        
        Args:
            offload_ratio: Ratio of parameters to offload
        """
        self.logger.info(f"Applying RWKV CPU offloading with ratio {offload_ratio}")
        
        # Implementation would move select RWKV parameters to CPU
        # This is a placeholder for the actual implementation
        pass

    def _optimize_rwkv_backward(self):
        """Apply RWKV-specific backward pass optimizations"""
        self.logger.info("Applying RWKV backward pass optimizations")
        
        # Implementation would optimize backward passes for RWKV
        # This is a placeholder for the actual implementation
        pass

    def train_step_rwkv(self, batch, use_mixed_precision=False):
        """Perform a training step with RWKV-specific optimizations
        
        Args:
            batch: The batch of data
            use_mixed_precision: Whether to use mixed precision training
            
        Returns:
            Loss value for the batch
        """
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get inputs and labels
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        
        # Use RWKV state caching for chunked processing if available
        if hasattr(self.model, "process_with_state"):
            return self._train_step_rwkv_with_state(input_ids, labels, use_mixed_precision)
        
        # Otherwise fall back to standard training step
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss
            
            # Scale loss and backward
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
        
        return loss.item()

    def _train_step_rwkv_with_state(self, input_ids, labels, use_mixed_precision):
        """Train step using RWKV state caching for more efficient training
        
        Args:
            input_ids: Input token IDs
            labels: Target labels
            use_mixed_precision: Whether to use mixed precision
            
        Returns:
            Loss value
        """
        # Process sequence in chunks with state caching
        chunk_size = getattr(self.memory_config, "rwkv_chunk_size", 512)
        seq_len = input_ids.size(1)
        
        # Initialize state
        state = None
        total_loss = 0
        
        for i in range(0, seq_len, chunk_size):
            # Get chunk
            chunk_input = input_ids[:, i:i+chunk_size]
            chunk_labels = labels[:, i:i+chunk_size]
            
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs, state = self.model.process_with_state(chunk_input, state)
                    chunk_loss = outputs.loss
                
                # Scale loss and backward
                self.scaler.scale(chunk_loss).backward(retain_graph=(i + chunk_size < seq_len))
            else:
                outputs, state = self.model.process_with_state(chunk_input, state)
                chunk_loss = outputs.loss
                chunk_loss.backward(retain_graph=(i + chunk_size < seq_len))
            
            total_loss += chunk_loss.item()
        
        # Return average loss
        return total_loss * chunk_size / seq_len 