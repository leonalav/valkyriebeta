import pytest
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model.computational_efficiency import (
    ComputationalEfficiencyConfig,
    ActivationCheckpointer,
    DynamicQuantizer,
    ModelPruner,
    EfficientAttention,
    EarlyExitController,
    ConditionalComputation,
    KVCache,
    mixed_precision_context,
    AdaptiveBatchSizer,
    ModelCompiler,
    ComputationalEfficiencyOptimizer
)

class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer for testing"""
    
    def __init__(self, hidden_size=64, num_heads=4):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        
    def forward(self, x, mask=None):
        return self.layer(x, src_mask=mask)

class SimpleTransformerModel(nn.Module):
    """Simple transformer model for testing"""
    
    def __init__(self, hidden_size=64, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_size)
        self.transformer_layers = nn.ModuleList([
            SimpleTransformerLayer(hidden_size, num_heads) 
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, 1000)
        
    def forward(self, input_ids, attention_mask=None):
        # Generate embeddings
        x = self.embedding(input_ids)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
            
        # Generate output
        logits = self.output(x)
        
        return {"hidden_states": x, "logits": logits}

@pytest.fixture
def efficiency_config():
    """Create an efficiency config for testing"""
    return ComputationalEfficiencyConfig(
        use_activation_checkpointing=True,
        checkpoint_every_n_layers=1,
        use_quantization=False,  # Set to False for testing as quantization requires calibration
        use_pruning=False,  # Set to False for testing as pruning would modify the model
        use_efficient_attention=True,
        attention_implementation="memory_efficient",  # Use memory_efficient as flash requires CUDA
        use_early_exit=True,
        exit_threshold=0.9,
        min_layers=1,
        use_conditional_computation=True,
        condition_threshold=0.5,
        use_kv_caching=True,
        max_cache_length=128,
        use_mixed_precision=False,  # Set to False for testing without CUDA
        use_adaptive_batch_size=True,
        min_batch_size=1,
        max_batch_size=16,
        use_torch_compile=False  # Set to False for testing as compile requires PyTorch 2.0+
    )

@pytest.fixture
def simple_model():
    """Create a simple model for testing"""
    return SimpleTransformerModel(hidden_size=64, num_heads=4, num_layers=2)

def test_efficiency_config():
    """Test ComputationalEfficiencyConfig initialization"""
    # Default initialization
    config = ComputationalEfficiencyConfig()
    
    # Check default values
    assert config.use_activation_checkpointing is True
    assert config.checkpoint_every_n_layers == 2
    assert config.use_efficient_attention is True
    assert config.attention_implementation == "flash"
    assert config.use_early_exit is True
    assert config.use_mixed_precision is True
    assert config.mixed_precision_dtype == "float16"
    
    # Custom initialization
    custom_config = ComputationalEfficiencyConfig(
        use_activation_checkpointing=False,
        use_efficient_attention=False,
        use_early_exit=False,
        use_mixed_precision=False
    )
    
    # Check custom values
    assert custom_config.use_activation_checkpointing is False
    assert custom_config.use_efficient_attention is False
    assert custom_config.use_early_exit is False
    assert custom_config.use_mixed_precision is False

def test_activation_checkpointer(simple_model, efficiency_config):
    """Test ActivationCheckpointer"""
    # Apply activation checkpointing
    ActivationCheckpointer.apply_checkpointing(simple_model, efficiency_config)
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass should work with checkpointing
    outputs = simple_model(input_ids)
    
    # Check outputs
    assert "hidden_states" in outputs
    assert "logits" in outputs
    assert outputs["hidden_states"].shape == (batch_size, seq_len, 64)
    assert outputs["logits"].shape == (batch_size, seq_len, 1000)

def test_efficient_attention():
    """Test EfficientAttention implementation"""
    # Create efficient attention module
    hidden_size = 64
    num_heads = 4
    attention = EfficientAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        implementation="memory_efficient"  # Use memory_efficient as flash requires CUDA
    )
    
    # Create dummy input
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Convert attention mask to the format expected by the model
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    
    # Forward pass
    output, past_key_value = attention(
        hidden_states=hidden_states,
        attention_mask=extended_attention_mask
    )
    
    # Check output shape
    assert output.shape == hidden_states.shape
    assert past_key_value is None  # No past_key_value when use_cache=False
    
    # Test with KV caching
    output, past_key_value = attention(
        hidden_states=hidden_states,
        attention_mask=extended_attention_mask,
        use_cache=True
    )
    
    # Check output shape
    assert output.shape == hidden_states.shape
    assert past_key_value is not None
    assert len(past_key_value) == 2  # k and v
    assert past_key_value[0].shape[1] == seq_len  # k has sequence dimension

def test_early_exit_controller(efficiency_config):
    """Test EarlyExitController"""
    # Create early exit controller
    controller = EarlyExitController(efficiency_config)
    
    # Test with low confidence (below threshold)
    layer_idx = 2
    confidence = torch.tensor([0.8])  # Below threshold of 0.9
    
    assert not controller.should_exit(layer_idx, confidence)
    
    # Test with high confidence (above threshold)
    confidence = torch.tensor([0.95])  # Above threshold of 0.9
    
    assert controller.should_exit(layer_idx, confidence)
    
    # Test with low layer index (below min_layers)
    layer_idx = 0  # Below min_layers of 1
    confidence = torch.tensor([0.95])  # Above threshold
    
    assert not controller.should_exit(layer_idx, confidence)

def test_conditional_computation(efficiency_config):
    """Test ConditionalComputation"""
    # Create conditional computation controller
    controller = ConditionalComputation(efficiency_config)
    
    # Test with low importance (below threshold)
    batch_size = 2
    seq_len = 10
    importance = torch.ones(batch_size, seq_len) * 0.4  # Below threshold of 0.5
    
    compute_mask = controller.should_compute(importance)
    
    assert compute_mask.shape == importance.shape
    assert not compute_mask.any()  # All values should be False
    
    # Test with high importance (above threshold)
    importance = torch.ones(batch_size, seq_len) * 0.6  # Above threshold of 0.5
    
    compute_mask = controller.should_compute(importance)
    
    assert compute_mask.shape == importance.shape
    assert compute_mask.all()  # All values should be True
    
    # Test with mixed importance
    importance = torch.tensor([
        [0.4, 0.6],
        [0.6, 0.4]
    ])
    
    compute_mask = controller.should_compute(importance)
    
    assert compute_mask.shape == importance.shape
    assert compute_mask[0, 0] == False
    assert compute_mask[0, 1] == True
    assert compute_mask[1, 0] == True
    assert compute_mask[1, 1] == False

def test_kv_cache(efficiency_config):
    """Test KVCache"""
    # Create KV cache
    cache = KVCache(efficiency_config)
    
    # Create dummy key and value tensors
    batch_size = 2
    seq_len = 10
    num_heads = 4
    head_dim = 16
    hidden_size = num_heads * head_dim
    
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Test updating cache
    layer_idx = 0
    new_k, new_v = cache.update(layer_idx, k, v)
    
    # First update should return the same values
    assert torch.allclose(new_k, k)
    assert torch.allclose(new_v, v)
    
    # Create new tensors for next token
    next_seq_len = 1
    next_k = torch.randn(batch_size, num_heads, next_seq_len, head_dim)
    next_v = torch.randn(batch_size, num_heads, next_seq_len, head_dim)
    
    # Test updating cache with new tokens
    updated_k, updated_v = cache.update(layer_idx, next_k, next_v)
    
    # Updated tensors should concatenate the cache and new tensors
    assert updated_k.shape == (batch_size, num_heads, seq_len + next_seq_len, head_dim)
    assert updated_v.shape == (batch_size, num_heads, seq_len + next_seq_len, head_dim)
    
    # Check that the first part of the updated tensors matches the original cache
    assert torch.allclose(updated_k[:, :, :seq_len, :], k)
    assert torch.allclose(updated_v[:, :, :seq_len, :], v)
    
    # Check that the last part of the updated tensors matches the new tensors
    assert torch.allclose(updated_k[:, :, -next_seq_len:, :], next_k)
    assert torch.allclose(updated_v[:, :, -next_seq_len:, :], next_v)
    
    # Test resetting cache
    cache.reset()
    
    # After reset, updating should return the original tensors
    reset_k, reset_v = cache.update(layer_idx, k, v)
    
    assert torch.allclose(reset_k, k)
    assert torch.allclose(reset_v, v)

def test_adaptive_batch_sizer(efficiency_config):
    """Test AdaptiveBatchSizer"""
    # Create adaptive batch sizer
    batch_sizer = AdaptiveBatchSizer(efficiency_config)
    
    # Test initial batch size
    assert batch_sizer.get_batch_size() == efficiency_config.max_batch_size
    
    # Test updating after OOM
    updated = batch_sizer.update_after_oom()
    
    assert updated is True
    assert batch_sizer.get_batch_size() == efficiency_config.max_batch_size // 2
    
    # Test updating after multiple OOMs
    for _ in range(3):
        batch_sizer.update_after_oom()
    
    assert batch_sizer.get_batch_size() == efficiency_config.min_batch_size
    
    # Test updating after reaching minimum
    updated = batch_sizer.update_after_oom()
    
    assert updated is False
    assert batch_sizer.get_batch_size() == efficiency_config.min_batch_size

def test_computational_efficiency_optimizer(simple_model, efficiency_config):
    """Test ComputationalEfficiencyOptimizer"""
    # Create optimizer
    optimizer = ComputationalEfficiencyOptimizer(efficiency_config)
    
    # Optimize model
    optimized_model = optimizer.optimize_model(simple_model)
    
    # Model should be the same instance but modified
    assert optimized_model is simple_model
    
    # Create other components
    kv_cache = optimizer.create_kv_cache()
    early_exit_controller = optimizer.create_early_exit_controller()
    conditional_computation = optimizer.create_conditional_computation()
    adaptive_batch_sizer = optimizer.create_adaptive_batch_sizer()
    
    # Check that components were created
    assert isinstance(kv_cache, KVCache)
    assert isinstance(early_exit_controller, EarlyExitController)
    assert isinstance(conditional_computation, ConditionalComputation)
    assert isinstance(adaptive_batch_sizer, AdaptiveBatchSizer)
    
    # Create efficient attention
    efficient_attention = optimizer.create_efficient_attention(
        hidden_size=64,
        num_heads=4
    )
    
    # Check that efficient attention was created
    assert isinstance(efficient_attention, EfficientAttention)
    assert efficient_attention.implementation == efficiency_config.attention_implementation 