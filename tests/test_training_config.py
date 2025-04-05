import torch
import pytest
from config.training_config import EnhancedTrainingConfig, ComputationalEfficiencyConfig

def test_validation_hardware_checks():
    """Test hardware capability validation"""
    config = EnhancedTrainingConfig(
        use_mixed_precision=True,
        mixed_precision_dtype="bfloat16",
        attention_implementation="flash"
    )

    # Mock unsupported bfloat16
    original_bf16_supported = torch.cuda.is_bf16_supported
    torch.cuda.is_bf16_supported = lambda: False

    with pytest.raises(ValueError):
        config.validate()

    # Restore original
    torch.cuda.is_bf16_supported = original_bf16_supported

def test_memory_validation():
    """Test memory budgeting validation"""
    mem_config = ComputationalEfficiencyConfig(
        attention_memory_limit=0.5,
        ffn_memory_limit=0.5,
        residual_memory_limit=0.5
    )

    warnings = mem_config.validate_memory(available_memory=4000)
    assert len(warnings) == 2  # Should warn about both over-allocation and low memory

def test_attention_fallback():
    """Test attention implementation fallback"""
    config = EnhancedTrainingConfig(attention_implementation="flash")

    # Mock missing flash attention
    original_has_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    torch.nn.functional.scaled_dot_product_attention = None

    warnings = config.validate()
    assert any("falling back" in w for w in warnings)
    assert config.attention_implementation == "memory_efficient"

    # Restore if needed
    if original_has_flash:
        torch.nn.functional.scaled_dot_product_attention = lambda *args, **kwargs: None
