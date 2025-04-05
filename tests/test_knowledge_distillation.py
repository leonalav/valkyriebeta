import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model.knowledge_distillation import KnowledgeDistillationModule, KnowledgeDistillationConfig, DomainAdaptationLayer

class SimpleTeacherModel(nn.Module):
    """Simple teacher model for testing"""
    def __init__(self, hidden_size=128, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size*4,
                batch_first=True
            ),
            num_layers=2
        )
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Generate embeddings
        x = self.embedding(input_ids)
        
        # Create attention mask for transformer
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
            
        # Convert attention mask to a format suitable for transformer
        # This is a simplified approach
        transformer_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply transformer
        hidden_states = self.transformer(x, src_key_padding_mask=~attention_mask)
        
        # Generate logits
        logits = self.output(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Return dictionary of outputs
        outputs = {
            "logits": logits,
            "hidden_states": hidden_states,
            "attentions": None  # No attention weights in this simple model
        }
        
        if loss is not None:
            outputs["loss"] = loss
            
        return outputs

class SimpleStudentModel(nn.Module):
    """Simple student model for testing"""
    def __init__(self, hidden_size=64, vocab_size=1000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=2,
                dim_feedforward=hidden_size*2,
                batch_first=True
            ),
            num_layers=1
        )
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, labels=None, teacher_model_outputs=None):
        # Generate embeddings
        x = self.embedding(input_ids)
        
        # Create attention mask for transformer
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)
            
        # Convert attention mask to a format suitable for transformer
        # This is a simplified approach
        transformer_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply transformer
        hidden_states = self.transformer(x, src_key_padding_mask=~attention_mask)
        
        # Generate logits
        logits = self.output(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Return dictionary of outputs
        outputs = {
            "logits": logits,
            "hidden_states": hidden_states,
            "attentions": None  # No attention weights in this simple model
        }
        
        if loss is not None:
            outputs["loss"] = loss
            
        return outputs

@pytest.fixture
def distillation_config():
    """Create a distillation config for testing"""
    return KnowledgeDistillationConfig(
        use_teacher_model=True,
        teacher_model_path=None,
        teacher_model_api=None,
        temperature=2.0,
        alpha=0.5,
        distill_attention=True,
        distill_hidden_states=True,
        distill_logits=True,
        domain_adaptation_layers=1
    )

@pytest.fixture
def distillation_module(distillation_config):
    """Create a distillation module for testing"""
    return KnowledgeDistillationModule(distillation_config, hidden_size=64)

@pytest.fixture
def teacher_model():
    """Create a teacher model for testing"""
    return SimpleTeacherModel(hidden_size=128, vocab_size=1000)

@pytest.fixture
def student_model():
    """Create a student model for testing"""
    return SimpleStudentModel(hidden_size=64, vocab_size=1000)

def test_domain_adaptation_layer():
    """Test the domain adaptation layer"""
    # Setup
    batch_size = 2
    seq_len = 10
    hidden_size = 64
    
    # Create layer
    layer = DomainAdaptationLayer(hidden_size)
    
    # Create input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Apply layer
    adapted_states = layer(hidden_states)
    
    # Check output shape
    assert adapted_states.shape == hidden_states.shape, "Output shape should match input shape"
    
    # Check that the layer has parameters
    param_count = sum(p.numel() for p in layer.parameters())
    assert param_count > 0, "Layer should have parameters"

def test_knowledge_distillation_module_init(distillation_config):
    """Test initialization of knowledge distillation module"""
    # Create module
    module = KnowledgeDistillationModule(distillation_config, hidden_size=64)
    
    # Check properties
    assert module.config == distillation_config
    
    # Check that domain layers were created
    assert module.domain_layers is not None
    assert len(module.domain_layers) == distillation_config.domain_adaptation_layers
    
    # Check feature alignment layer
    assert module.feature_alignment is not None

def test_compute_distillation_loss(distillation_module):
    """Test computation of distillation loss"""
    # Setup
    batch_size = 2
    seq_len = 10
    hidden_size = 64
    vocab_size = 1000
    
    # Create dummy logits
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Create dummy hidden states
    student_hidden = torch.randn(batch_size, seq_len, hidden_size)
    teacher_hidden = torch.randn(batch_size, seq_len, hidden_size * 2)  # Teacher has larger hidden size
    
    # Create dummy attention matrices
    student_attentions = [torch.randn(batch_size, 4, seq_len, seq_len) for _ in range(2)]
    teacher_attentions = [torch.randn(batch_size, 8, seq_len, seq_len) for _ in range(4)]
    
    # Compute distillation losses
    losses = distillation_module.compute_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_hidden=student_hidden,
        teacher_hidden=teacher_hidden,
        student_attentions=student_attentions,
        teacher_attentions=teacher_attentions
    )
    
    # Check that losses were computed
    assert "logit_loss" in losses
    assert "hidden_loss" in losses
    assert "attention_loss" in losses
    assert "total_distillation_loss" in losses
    
    # Check that losses are tensors
    for loss_name, loss in losses.items():
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0, f"{loss_name} should be positive"

def test_apply_domain_adaptation(distillation_module):
    """Test application of domain adaptation"""
    # Setup
    batch_size = 2
    seq_len = 10
    hidden_size = 64
    
    # Create dummy hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Apply domain adaptation
    adapted_states = distillation_module.apply_domain_adaptation(hidden_states)
    
    # Check output shape
    assert adapted_states.shape == hidden_states.shape, "Output shape should match input shape"
    
    # Check that the output is different from the input
    assert not torch.allclose(adapted_states, hidden_states), "Adapted states should be different from input"

def test_integration_with_models(teacher_model, student_model, distillation_module):
    """Test integration of distillation module with models"""
    # Setup
    batch_size = 2
    seq_len = 10
    
    # Create dummy input
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Get teacher outputs
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    # Apply student model
    student_outputs = student_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        teacher_model_outputs=teacher_outputs
    )
    
    # Compute distillation losses
    losses = distillation_module.compute_distillation_loss(
        student_logits=student_outputs["logits"],
        teacher_logits=teacher_outputs["logits"],
        student_hidden=student_outputs["hidden_states"],
        teacher_hidden=teacher_outputs["hidden_states"]
    )
    
    # Check that losses were computed
    assert "logit_loss" in losses
    assert "hidden_loss" in losses
    assert "total_distillation_loss" in losses
    
    # Compute combined loss
    student_loss = student_outputs["loss"]
    distillation_loss = losses["total_distillation_loss"]
    
    alpha = distillation_module.config.alpha
    combined_loss = (1 - alpha) * student_loss + alpha * distillation_loss
    
    # Check that combined loss is reasonable
    assert combined_loss.item() > 0, "Combined loss should be positive"
    assert combined_loss.item() < student_loss.item() * 2, "Combined loss should be reasonable" 