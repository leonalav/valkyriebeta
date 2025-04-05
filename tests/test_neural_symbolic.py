import unittest
import torch
import gc
import tracemalloc
import pytest
import psutil
import time
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional

# Import the correct modules based on project structure
from model.neural_symbolic_reasoner import (
    NeuralSymbolicConfig,
    NeuralSymbolicIntegration,
    SymbolicReasoningLayer
)

try:
    from utils.enhanced_memory_manager import EnhancedMemoryManager
except ImportError:
    # Fallback if module not available
    class EnhancedMemoryManager:
        def __init__(self, **kwargs):
            pass
        def release_memory(self, **kwargs):
            pass
        def get_memory_stats(self, **kwargs):
            return {}

class TestNeuralSymbolic(unittest.TestCase):
    def setUp(self):
        self.config = NeuralSymbolicConfig(
            hidden_size=64,
            num_attention_heads=4,
            dropout=0.1,
            # These fields no longer exist in the current implementation
            # symbol_vocabulary_size=32,
            # num_symbolic_layers=2,
            # max_symbolic_steps=3,
            # abstraction_levels=2,
            # composition_depth=1,
            # Memory management configurations
            rule_cache_size=20,
            rule_initialization="random",
            use_rule_composition=True,
            use_rule_specialization=True
        )
        self.batch_size = 2
        self.seq_len = 8
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)

        # Create memory manager for tests
        self.memory_manager = EnhancedMemoryManager(
            target_gpu_utilization=0.8,
            min_batch_size=1,
            max_batch_size=16,
            enable_monitoring=False
        )

    def tearDown(self):
        # Clean up after tests
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_initialization(self):
        """Test initialization of neural symbolic model"""
        model = NeuralSymbolicIntegration(self.config)

        # Verify model has expected attributes
        self.assertTrue(hasattr(model, 'symbolic_layer'))
        self.assertTrue(hasattr(model.symbolic_layer, 'rule_embeddings'))

        # Check rule embeddings shape
        expected_shape = (self.config.num_rules, self.config.rule_embedding_size)
        self.assertEqual(model.symbolic_layer.rule_embeddings.shape, expected_shape)

        # Check memory management attributes
        self.assertTrue(hasattr(model.symbolic_layer, '_cleanup_caches'))
        self.assertTrue(hasattr(model.symbolic_layer, 'inference_steps_since_cleanup'))

    def test_forward_pass(self):
        """Test basic forward pass"""
        model = NeuralSymbolicIntegration(self.config)

        # Run forward pass
        outputs = model(self.hidden_states)

        # Check output is a dictionary
        self.assertIsInstance(outputs, dict)

        # Check hidden states
        self.assertIn('hidden_states', outputs)
        self.assertEqual(outputs['hidden_states'].shape, self.hidden_states.shape)

    def test_cache_management(self):
        """Test cache management in symbolic reasoning"""
        # Create symbolic reasoning layer directly for more focused testing
        symbolic_layer = SymbolicReasoningLayer(self.config)

        # Check cache attributes
        self.assertTrue(hasattr(symbolic_layer, 'specialized_rules'))
        self.assertTrue(hasattr(symbolic_layer, 'specialized_rules_usage'))
        self.assertTrue(hasattr(symbolic_layer, 'max_specialized_cache_size'))

        # Fill the cache
        for i in range(symbolic_layer.max_specialized_cache_size * 2):
            key = f"context_{i}"
            symbolic_layer.specialized_rules[key] = torch.randn(1, self.config.hidden_size)
            symbolic_layer.specialized_rules_usage[key] = float(i % 10)

        # Verify cache cleanup works
        symbolic_layer._cleanup_caches()

        # Check that cache size is reduced
        self.assertLessEqual(len(symbolic_layer.specialized_rules), symbolic_layer.max_specialized_cache_size)

    def test_memory_tracking(self):
        """Test memory tracking for neural symbolic reasoning"""
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        model = NeuralSymbolicIntegration(self.config)
        model = model.cuda()

        # Convert inputs to CUDA
        hidden_states = self.hidden_states.cuda()

        # Track memory before
        torch.cuda.empty_cache()
        gc.collect()
        memory_before = torch.cuda.memory_allocated()

        # Run forward pass multiple times
        for _ in range(3):
            outputs = model(hidden_states)
            del outputs

        # Force cleanup
        model.symbolic_layer._cleanup_caches()
        torch.cuda.empty_cache()
        gc.collect()

        # Track memory after
        memory_after = torch.cuda.memory_allocated()

        # Check memory growth is bounded
        memory_growth = memory_after - memory_before
        # Allow 5MB growth during test
        max_allowed_growth = 5 * 1024 * 1024
        self.assertLessEqual(memory_growth, max_allowed_growth,
                           f"Memory growth too high: {memory_growth / (1024 * 1024):.2f} MB")

    def test_temporary_tensors(self):
        """Test tracking and cleanup of temporary tensors"""
        symbolic_layer = SymbolicReasoningLayer(self.config)

        # Verify temporary tensors list exists
        self.assertTrue(hasattr(symbolic_layer, 'temporary_tensors'))

        # Run forward pass
        hidden_states = self.hidden_states
        outputs = symbolic_layer(hidden_states)

        # Run cleanup
        symbolic_layer._cleanup_caches()

        # Verify list is empty after cleanup
        self.assertEqual(len(symbolic_layer.temporary_tensors), 0)

    def test_stress_memory_management(self):
        """Stress test memory management with many forward passes"""
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory stress test")

        # Create a larger model for more memory usage
        config = NeuralSymbolicConfig(
            hidden_size=128,
            num_attention_heads=4,
            dropout=0.1,
            rule_cache_size=50,  # Larger cache for stress test
            rule_initialization="random",
            use_rule_composition=True,
            use_rule_specialization=True,
            inference_steps=3
        )

        model = NeuralSymbolicIntegration(config)
        model = model.cuda()

        # Create inputs
        batch_size, seq_len = 4, 16

        # Track starting memory
        torch.cuda.empty_cache()
        gc.collect()
        start_memory = torch.cuda.memory_allocated()

        # Log initial state
        print(f"Starting memory: {start_memory / (1024 * 1024):.2f} MB")

        # Run many forward passes with different inputs
        num_iterations = 20
        memory_usage = []

        for i in range(num_iterations):
            # Create different inputs each time
            hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device='cuda')

            # Forward pass
            outputs = model(hidden_states)

            # Track memory
            current_memory = torch.cuda.memory_allocated()
            memory_usage.append(current_memory)

            # Log every 5 iterations
            if (i + 1) % 5 == 0:
                print(f"Iteration {i+1}, Memory: {current_memory / (1024 * 1024):.2f} MB")

            # Delete to ensure proper cleanup
            del outputs, hidden_states

            # Force cleanup every 5 iterations
            if (i + 1) % 5 == 0:
                # Call custom cleanup
                if hasattr(model.symbolic_layer, '_cleanup_caches'):
                    model.symbolic_layer._cleanup_caches()

                # Force system cleanup
                gc.collect()
                torch.cuda.empty_cache()

        # Check final memory usage
        end_memory = torch.cuda.memory_allocated()
        print(f"Final memory: {end_memory / (1024 * 1024):.2f} MB")

        # Compute memory stability
        max_memory = max(memory_usage)
        memory_growth = end_memory - start_memory

        print(f"Memory growth: {memory_growth / (1024 * 1024):.2f} MB")
        print(f"Max memory used: {max_memory / (1024 * 1024):.2f} MB")

        # Check memory stability - allowing some growth but not excessive
        # 20MB is a reasonable threshold for this test
        max_allowed_growth = 20 * 1024 * 1024
        self.assertLess(memory_growth, max_allowed_growth,
                      f"Memory growth too high: {memory_growth / (1024 * 1024):.2f} MB")

    def test_rule_management(self):
        """Test rule management functionality including export/import"""
        import tempfile
        import os

        # Create model with small rule set
        config = NeuralSymbolicConfig(
            hidden_size=64,
            num_attention_heads=4,
            dropout=0.1,
            num_rules=10,
            rule_cache_size=5
        )

        model = NeuralSymbolicIntegration(config)

        # Get initial rule embeddings
        original_rules = model.symbolic_layer.rule_embeddings.clone()

        # Create a temporary directory for export/import testing
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Export rules
            export_path = os.path.join(tmp_dir, "test_rules.pt")
            success = model.export_rules(export_path)
            self.assertTrue(success, "Failed to export rules")
            self.assertTrue(os.path.exists(export_path), "Export file not created")

            # Modify rule embeddings
            with torch.no_grad():
                model.symbolic_layer.rule_embeddings.data = torch.randn_like(model.symbolic_layer.rule_embeddings)

            # Verify rules are different
            self.assertFalse(torch.allclose(model.symbolic_layer.rule_embeddings, original_rules),
                          "Rules should be different after modification")

            # Import rules
            success = model.import_rules(export_path)
            self.assertTrue(success, "Failed to import rules")

            # Verify rules are restored
            self.assertTrue(torch.allclose(model.symbolic_layer.rule_embeddings, original_rules),
                         "Rules should be restored after import")

    def test_rule_caching_consistency(self):
        """Test that rule caching is consistent across multiple runs"""
        model = NeuralSymbolicIntegration(self.config)

        # Create contexts and track specializations
        batch_size, seq_len = 3, 4

        # Run once to initialize
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        _ = model(hidden_states)

        # Track initial specializations count
        initial_specialized_rules = len(model.symbolic_layer.specialized_rules)

        # Get symbolic layer and check initial cache state
        symbolic_layer = model.symbolic_layer
        self.assertTrue(hasattr(symbolic_layer, 'specialized_rules'))
        self.assertTrue(hasattr(symbolic_layer, 'specialized_rules_usage'))

        # Run with same input multiple times
        same_input = torch.randn(batch_size, seq_len, self.config.hidden_size)
        for _ in range(5):
            _ = model(same_input)

        # Cache should have grown but been controlled
        self.assertGreaterEqual(len(symbolic_layer.specialized_rules), initial_specialized_rules)

        # Run cleanup and check cache size is maintained
        symbolic_layer._cleanup_caches()
        self.assertLessEqual(len(symbolic_layer.specialized_rules), symbolic_layer.max_specialized_cache_size)

        # Run with different inputs
        for _ in range(30):  # Deliberately exceed cache size
            different_input = torch.randn(batch_size, seq_len, self.config.hidden_size)
            _ = model(different_input)

        # Cache may exceed limit during active use, which is expected
        print(f"Cache size before cleanup: {len(symbolic_layer.specialized_rules)}")
        print(f"Max cache size: {symbolic_layer.max_specialized_cache_size}")

        # But after explicit cleanup, it should be within limits
        symbolic_layer._cleanup_caches()

        print(f"Cache size after cleanup: {len(symbolic_layer.specialized_rules)}")
        self.assertLessEqual(len(symbolic_layer.specialized_rules), symbolic_layer.max_specialized_cache_size,
                           "Cache size should be limited after cleanup")

if __name__ == '__main__':
    unittest.main()