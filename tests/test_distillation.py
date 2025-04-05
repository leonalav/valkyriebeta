import pytest
import torch
from ..model.api_distillation import APITeacherModel, APIKnowledgeDistillation
from ..config.distillation_config import DistillationConfig

@pytest.fixture
def config():
    return DistillationConfig(
        api_key="test_key",
        site_url="http://test.com",
        site_name="test"
    )

@pytest.fixture
def teacher_model(config):
    return APITeacherModel(
        api_key=config.api_key,
        site_url=config.site_url,
        site_name=config.site_name
    )

def test_teacher_model_init(teacher_model):
    assert teacher_model.api_key == "test_key"
    assert teacher_model.model == "deepseek/deepseek-r1:free"

def test_knowledge_distillation_forward():
    batch_size, seq_len, hidden_size = 2, 10, 768
    student_logits = torch.randn(batch_size, seq_len, hidden_size)
    input_texts = ["test input 1", "test input 2"]
    
    config = DistillationConfig(
        api_key="test_key",
        site_url="http://test.com",
        site_name="test",
        hidden_size=hidden_size
    )
    
    teacher = APITeacherModel(
        api_key=config.api_key,
        site_url=config.site_url,
        site_name=config.site_name
    )
    
    distillation = APIKnowledgeDistillation(
        config=config,
        teacher_model=teacher,
        tokenizer=None,
        hidden_size=hidden_size
    )
    
    outputs = distillation(student_logits, input_texts)
    assert "loss" in outputs
    assert "distillation_loss" in outputs
    assert "student_loss" in outputs
