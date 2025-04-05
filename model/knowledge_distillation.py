import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeDistillationConfig:
    """Configuration for knowledge distillation and transfer learning"""
    # Teacher model parameters
    use_teacher_model: bool = True
    teacher_model_path: Optional[str] = None
    teacher_model_api: Optional[str] = None  # For API-based distillation
    
    # Distillation parameters
    temperature: float = 2.0
    alpha: float = 0.5  # Weight for distillation loss vs task loss
    
    # Selective distillation
    distill_attention: bool = True
    distill_hidden_states: bool = True
    distill_logits: bool = True
    
    # Specialized domain adaptation
    domain_adaptation_layers: int = 2
    domain_data_ratio: float = 0.3  # Ratio of domain-specific data in training
    
    # Progressive knowledge transfer
    progressive_stages: List[str] = None
    current_stage: str = "base"
    
    def __post_init__(self):
        if self.progressive_stages is None:
            self.progressive_stages = ["base", "reasoning", "domain_specific", "fine_tuning"]


class DomainAdaptationLayer(nn.Module):
    """Layer for adapting the model to specific domains like math, science, etc."""
    
    def __init__(self, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Dropout(dropout_prob)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply domain adaptation
        
        Args:
            hidden_states: Hidden states from transformer [batch_size, seq_len, hidden_size]
            
        Returns:
            Adapted hidden states [batch_size, seq_len, hidden_size]
        """
        residual = hidden_states
        adapted = self.adapter(hidden_states)
        return self.layer_norm(residual + adapted)


class KnowledgeDistillationModule(nn.Module):
    """Module for distilling knowledge from larger models"""
    
    def __init__(self, config: KnowledgeDistillationConfig, model_hidden_size: int):
        super().__init__()
        self.config = config
        
        # Domain adaptation layers
        if config.domain_adaptation_layers > 0:
            self.domain_layers = nn.ModuleList([
                DomainAdaptationLayer(model_hidden_size) 
                for _ in range(config.domain_adaptation_layers)
            ])
        else:
            self.domain_layers = None
            
        # Feature alignment for teacher-student mapping
        self.feature_alignment = nn.Linear(model_hidden_size, model_hidden_size)
        
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_hidden: Optional[torch.Tensor] = None,
        teacher_hidden: Optional[torch.Tensor] = None,
        student_attentions: Optional[List[torch.Tensor]] = None,
        teacher_attentions: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss between student and teacher
        
        Args:
            student_logits: Output logits from student model
            teacher_logits: Output logits from teacher model
            student_hidden: Hidden states from student model (optional)
            teacher_hidden: Hidden states from teacher model (optional)
            student_attentions: Attention matrices from student model (optional)
            teacher_attentions: Attention matrices from teacher model (optional)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Logit distillation with temperature scaling
        if self.config.distill_logits:
            temp = self.config.temperature
            soft_student = F.log_softmax(student_logits / temp, dim=-1)
            soft_teacher = F.softmax(teacher_logits / temp, dim=-1)
            losses["logit_loss"] = F.kl_div(
                soft_student, 
                soft_teacher, 
                reduction="batchmean"
            ) * (temp ** 2)
        
        # Hidden state distillation
        if self.config.distill_hidden_states and student_hidden is not None and teacher_hidden is not None:
            # Align student hidden states to teacher dimension if needed
            aligned_student = self.feature_alignment(student_hidden)
            losses["hidden_loss"] = F.mse_loss(aligned_student, teacher_hidden)
        
        # Attention distillation
        if self.config.distill_attention and student_attentions and teacher_attentions:
            attn_loss = 0.0
            for student_attn, teacher_attn in zip(student_attentions, teacher_attentions):
                attn_loss += F.mse_loss(student_attn, teacher_attn)
            losses["attention_loss"] = attn_loss / len(student_attentions)
        
        # Compute weighted total loss
        total_loss = sum(losses.values())
        losses["total_distillation_loss"] = total_loss
        
        return losses
    
    def apply_domain_adaptation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply domain-specific adaptation layers
        
        Args:
            hidden_states: Hidden states from transformer
            
        Returns:
            Adapted hidden states
        """
        if self.domain_layers is None:
            return hidden_states
            
        adapted = hidden_states
        for layer in self.domain_layers:
            adapted = layer(adapted)
            
        return adapted


class ProgressiveKnowledgeTransfer:
    """Manages progressive knowledge transfer during training"""
    
    def __init__(self, config: KnowledgeDistillationConfig):
        self.config = config
        self.current_stage_idx = config.progressive_stages.index(config.current_stage)
        
    def update_stage(self, metrics: Dict[str, float]) -> bool:
        """Update training stage based on metrics
        
        Args:
            metrics: Dictionary of evaluation metrics
            
        Returns:
            True if stage was updated, False otherwise
        """
        # Simple progression strategy - could be made more sophisticated
        if self.current_stage_idx < len(self.config.progressive_stages) - 1:
            # Check if we should advance to next stage
            if self._should_advance_stage(metrics):
                self.current_stage_idx += 1
                self.config.current_stage = self.config.progressive_stages[self.current_stage_idx]
                logger.info(f"Advancing to stage: {self.config.current_stage}")
                return True
        
        return False
    
    def _should_advance_stage(self, metrics: Dict[str, float]) -> bool:
        """Determine if we should advance to the next stage"""
        # This is a simple heuristic - could be made more sophisticated
        if "eval_loss" in metrics and metrics["eval_loss"] < 0.1:
            return True
        
        if "accuracy" in metrics and metrics["accuracy"] > 0.95:
            return True
            
        return False
    
    def get_stage_specific_config(self) -> Dict[str, Any]:
        """Get configuration specific to current stage"""
        stage = self.config.current_stage
        
        if stage == "base":
            return {
                "distill_attention": False,
                "distill_hidden_states": False,
                "distill_logits": True,
                "alpha": 0.3
            }
        elif stage == "reasoning":
            return {
                "distill_attention": True,
                "distill_hidden_states": True,
                "distill_logits": True,
                "alpha": 0.5
            }
        elif stage == "domain_specific":
            return {
                "distill_attention": True,
                "distill_hidden_states": True,
                "distill_logits": True,
                "alpha": 0.7,
                "domain_data_ratio": 0.7
            }
        elif stage == "fine_tuning":
            return {
                "distill_attention": False,
                "distill_hidden_states": False,
                "distill_logits": False,
                "alpha": 0.1,
                "domain_data_ratio": 0.9
            }
        
        return {} 