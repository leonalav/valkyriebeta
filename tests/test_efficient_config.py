import pytest
from ..config.efficient_config import EfficientTransformerConfig

def test_config_validation():
    config = EfficientTransformerConfig()
    config.validate()  # Should not raise

def test_memory_validation():
    config = EfficientTransformerConfig(
        max_memory_gb=1.0,  # Unrealistically low
        hidden_size=8192    # Memory intensive
    )
    with pytest.raises(ValueError, match="exceeds memory constraints"):
        config.validate()

def test_attention_head_validation():
    config = EfficientTransformerConfig(
        hidden_size=2048,
        num_heads=15  # Not divisible
    )
    with pytest.raises(ValueError, match="divisible by num_heads"):
        config.validate()

def test_version_tracking():
    config = EfficientTransformerConfig()
    config_dict = config.to_dict()
    assert "versions" in config_dict
    assert "model_version" in config_dict["versions"]
