import pytest
import torch
import numpy as np
from typing import Dict, Any

from model import LogicalReasoningTransformer
from config import (
    ModelConfig,
    TrainingConfig,
    MemoryConfig,
    TrainingEfficiencyConfig,
    EfficientTransformerConfig
)
from .test_framework import TestFramework, TestConfig

@pytest.fixture
def test_configs():
    """Create test configurations"""
    model_config = ModelConfig()
    training_config = TrainingConfig()
    memory_config = MemoryConfig()
    efficiency_config = TrainingEfficiencyConfig()
    
    # Create unified config
    config = EfficientTransformerConfig.from_configs(
        model_config=model_config,
        training_config=training_config,
        memory_config=memory_config,
        efficiency_config=efficiency_config
    )
    
    return {
        'model_config': config,
        'test_config': TestConfig(),
        'memory_config': memory_config,
        'efficiency_config': efficiency_config
    }

@pytest.fixture
def model(test_configs):
    """Create model instance"""
    model = LogicalReasoningTransformer(test_configs['model_config'])
    return model

@pytest.fixture
def test_framework(model, test_configs):
    """Create test framework instance"""
    framework = TestFramework(
        model=model,
        test_config=test_configs['test_config'],
        memory_config=test_configs['memory_config'],
        efficiency_config=test_configs['efficiency_config']
    )
    yield framework
    framework.cleanup()

def test_memory_leaks(test_framework):
    """Test for memory leaks"""
    results = test_framework.run_memory_tests()
    
    # Check for significant memory leaks
    assert len(results['leaks']) == 0, f"Memory leaks detected: {results['leaks']}"
    
    # Check memory growth
    memory_growth = results['current_memory'] - results['peak_memory']
    assert memory_growth < test_framework.test_config.memory_threshold_mb, \
        f"Excessive memory growth: {memory_growth}MB"

def test_performance(test_framework):
    """Test performance metrics"""
    results = test_framework.run_performance_tests()
    
    # Check latency
    assert results['avg_latency'] < test_framework.test_config.latency_threshold_ms, \
        f"Average latency ({results['avg_latency']}ms) exceeds threshold"
    
    # Check throughput
    assert results['avg_throughput'] > test_framework.test_config.throughput_threshold, \
        f"Average throughput ({results['avg_throughput']} samples/sec) below threshold"

def test_training_stability(test_framework):
    """Test training stability"""
    results = test_framework.run_stability_tests()
    
    # Check loss variance
    assert results['loss_variance'] < test_framework.test_config.loss_variance_threshold, \
        f"Loss variance ({results['loss_variance']}) exceeds threshold"
    
    # Check gradient norms
    assert results['grad_norm_mean'] < test_framework.test_config.grad_norm_threshold, \
        f"Average gradient norm ({results['grad_norm_mean']}) exceeds threshold"
    
    # Check loss isn't increasing
    assert results['loss_trend'] <= 0, \
        f"Loss is trending upward with slope {results['loss_trend']}"

def test_resource_utilization(test_framework):
    """Test resource utilization"""
    results = test_framework.run_resource_tests()
    
    # Check GPU memory usage
    if results['gpu_memory_usage'] is not None:
        assert results['gpu_memory_usage'] < test_framework.test_config.max_gpu_memory_usage, \
            f"GPU memory usage ({results['gpu_memory_usage']*100}%) exceeds threshold"
    
    # Check CPU usage
    assert results['cpu_usage'] < test_framework.test_config.max_cpu_usage, \
        f"CPU usage ({results['cpu_usage']*100}%) exceeds threshold"

def test_numerical_stability(test_framework):
    """Test numerical stability"""
    results = test_framework.run_numerical_tests()
    
    # Check numerical stability across dtypes
    for dtype, stats in results.items():
        if dtype != 'float32':  # Compare with base dtype
            assert stats['is_close'], \
                f"Numerical instability detected with {dtype}: max_diff={stats['max_diff']}"

def test_all(test_framework):
    """Run all tests"""
    results = test_framework.run_all_tests()
    
    # Verify all test categories passed
    assert all(category in results for category in [
        'memory', 'performance', 'stability', 'resources', 'numerical'
    ]), "Not all test categories were run"
    
    # Log detailed results
    for category, category_results in results.items():
        print(f"\n{category.upper()} TEST RESULTS:")
        for metric, value in category_results.items():
            print(f"{metric}: {value}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 