import pytest
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model.adaptive_reasoning import (
    AdaptiveReasoningConfig,
    ComplexityEstimator,
    ComponentSelector,
    AdaptiveReasoningController
)

@pytest.fixture
def adaptive_config():
    """Create an adaptive reasoning config for testing"""
    return AdaptiveReasoningConfig(
        low_complexity_threshold=0.3,
        medium_complexity_threshold=0.7,
        max_computation_budget=1.0,
        min_computation_budget=0.2,
        use_early_exit=True,
        early_exit_threshold=0.9
    )

def test_adaptive_reasoning_config():
    """Test AdaptiveReasoningConfig initialization"""
    # Default initialization
    config = AdaptiveReasoningConfig()

    # Check default values
    assert config.low_complexity_threshold == 0.3
    assert config.medium_complexity_threshold == 0.7
    assert config.max_computation_budget == 1.0
    assert config.min_computation_budget == 0.2
    assert config.use_early_exit is True
    assert config.early_exit_threshold == 0.9

    # Check component costs and importance were initialized
    assert config.component_costs is not None
    assert config.component_importance is not None
    assert "moe" in config.component_costs
    assert "tree_reasoning" in config.component_importance

    # Custom initialization
    custom_config = AdaptiveReasoningConfig(
        low_complexity_threshold=0.2,
        medium_complexity_threshold=0.6,
        max_computation_budget=0.8,
        min_computation_budget=0.1,
        use_early_exit=False
    )

    # Check custom values
    assert custom_config.low_complexity_threshold == 0.2
    assert custom_config.medium_complexity_threshold == 0.6
    assert custom_config.max_computation_budget == 0.8
    assert custom_config.min_computation_budget == 0.1
    assert custom_config.use_early_exit is False

def test_complexity_estimator():
    """Test ComplexityEstimator"""
    # Create complexity estimator
    hidden_size = 64
    estimator = ComplexityEstimator(hidden_size)

    # Create dummy input
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Estimate complexity
    complexity, features = estimator(hidden_states)

    # Check shapes
    assert complexity.shape == (batch_size, 1)
    assert features.shape == (batch_size, 5)

    # Check complexity is between 0 and 1
    assert torch.all(complexity >= 0)
    assert torch.all(complexity <= 1)

def test_component_selector(adaptive_config):
    """Test ComponentSelector"""
    # Create component selector
    hidden_size = 64
    selector = ComponentSelector(adaptive_config, hidden_size)

    # Create dummy input
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Select components with full computation budget
    component_selection = selector(hidden_states, available_compute=1.0)

    # Check that component selection is a dictionary
    assert isinstance(component_selection, dict)

    # Check component keys
    expected_keys = {
        "moe", "memory_layer", "tree_reasoning", "neural_symbolic",
        "recursive_reasoning", "knowledge_reasoning", "verifiable_computation"
    }
    assert set(component_selection.keys()) == expected_keys

    # With full computation budget, most components should be selected
    assert sum(component_selection.values()) >= 4  # At least 4 components selected

    # Select components with limited computation budget
    component_selection_limited = selector(hidden_states, available_compute=0.3)

    # With limited budget, fewer components should be selected
    assert sum(component_selection_limited.values()) <= sum(component_selection.values())

def test_component_selector_complexity_levels(adaptive_config):
    """Test ComponentSelector with different complexity levels"""
    # Create complexity estimator for mocking
    hidden_size = 64
    estimator = ComplexityEstimator(hidden_size)

    # Create component selector
    selector = ComponentSelector(adaptive_config, hidden_size)

    # Create dummy input for low complexity
    batch_size = 2
    seq_len = 10
    hidden_states_low = torch.randn(batch_size, seq_len, hidden_size)

    # Mock complexity estimation for low complexity
    selector.complexity_estimator = lambda x: (torch.tensor([[0.2]]), torch.randn(batch_size, 5))

    # Select components for low complexity
    component_selection_low = selector(hidden_states_low)

    # Low complexity should use fewer components
    assert sum(component_selection_low.values()) <= 3  # At most 3 components for low complexity

    # Mock complexity estimation for medium complexity
    selector.complexity_estimator = lambda x: (torch.tensor([[0.5]]), torch.randn(batch_size, 5))

    # Select components for medium complexity
    component_selection_medium = selector(hidden_states_low)

    # Medium complexity should use more components than low
    assert sum(component_selection_medium.values()) > sum(component_selection_low.values())

    # Mock complexity estimation for high complexity
    selector.complexity_estimator = lambda x: (torch.tensor([[0.9]]), torch.randn(batch_size, 5))

    # Select components for high complexity
    component_selection_high = selector(hidden_states_low)

    # High complexity should use more components than medium
    assert sum(component_selection_high.values()) >= sum(component_selection_medium.values())

def test_adaptive_reasoning_controller(adaptive_config):
    """Test AdaptiveReasoningController"""
    # Create controller
    hidden_size = 64
    controller = AdaptiveReasoningController(adaptive_config, hidden_size)

    # Create dummy input
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # Select components with controller
    component_selection = controller.select_components(hidden_states)

    # Check that component selection is a dictionary
    assert isinstance(component_selection, dict)

    # Check component keys
    expected_keys = {
        "moe", "memory_layer", "tree_reasoning", "neural_symbolic",
        "recursive_reasoning", "knowledge_reasoning", "verifiable_computation"
    }
    assert set(component_selection.keys()) == expected_keys

    # Test early exit
    # Default implementation should not exit early
    should_exit = controller.should_exit_early(hidden_states)

    # Should be a boolean
    assert isinstance(should_exit, bool)

def test_create_adaptive_config():
    """Test create_adaptive_config function"""
    from model.adaptive_reasoning import create_adaptive_config

    # Create mock model_config
    class MockModelConfig:
        def __init__(self):
            self.hidden_size = 64
            self.num_layers = 12
            self.num_attention_heads = 8

    model_config = MockModelConfig()

    # Create adaptive config from model config
    adaptive_config = create_adaptive_config(model_config)

    # Check that config was created
    assert isinstance(adaptive_config, AdaptiveReasoningConfig)

    # Check default values
    assert adaptive_config.low_complexity_threshold == 0.3
    assert adaptive_config.medium_complexity_threshold == 0.7