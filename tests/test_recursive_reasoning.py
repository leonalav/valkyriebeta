import unittest
import torch
from model.recursive_reasoning import (
    RecursiveReasoningConfig,
    RecursiveAttention,
    RecursiveMemory,
    RecursiveRouter,
    RecursiveVerifier,
    RecursiveComposer,
    RecursiveReasoningTransformer,
    RecursiveReasoningModule
)

class TestRecursiveReasoning(unittest.TestCase):
    def setUp(self):
        self.config = RecursiveReasoningConfig(
            hidden_size=64,
            num_heads=4,
            dropout=0.1,
            max_recursion_depth=3,
            use_adaptive_depth=True,
            early_stopping_threshold=0.9,
            use_recursive_attention=True,
            use_recursive_memory=True,
            memory_size=16,
            use_recursive_gating=True,
            use_recursive_routing=True,
            num_reasoning_experts=2,
            use_recursive_verification=True,
            use_recursive_composition=True,
            composition_depth=1
        )
        self.batch_size = 2
        self.seq_len = 8
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)

    def test_recursive_attention(self):
        """Test the recursive attention component"""
        attention = RecursiveAttention(self.config)

        # Test without past states
        output, attn_weights = attention(self.hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))
        self.assertEqual(attn_weights.shape, (self.batch_size, self.config.num_heads, self.seq_len, self.seq_len))

        # Test with past states
        past_states = [torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)]
        output, attn_weights = attention(self.hidden_states, past_states, recursion_level=1)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

    def test_recursive_memory(self):
        """Test the recursive memory component"""
        memory = RecursiveMemory(self.config)

        output, updated_memory = memory(self.hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))
        self.assertEqual(updated_memory.shape, (self.batch_size, self.config.memory_size, self.config.hidden_size))

    def test_recursive_router(self):
        """Test the recursive router component"""
        router = RecursiveRouter(self.config)

        output, routing_weights = router(self.hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))
        self.assertEqual(routing_weights.shape, (self.batch_size, self.seq_len, self.config.num_reasoning_experts))

    def test_recursive_verifier(self):
        """Test the recursive verifier component"""
        verifier = RecursiveVerifier(self.config)

        output, verification_scores = verifier(self.hidden_states, self.hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))
        self.assertEqual(verification_scores.shape, (self.batch_size, self.seq_len, 1))

    def test_recursive_composer(self):
        """Test the recursive composer component"""
        composer = RecursiveComposer(self.config)

        # Test without past states
        output = composer(self.hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

        # Test with past states
        past_states = [torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)]
        output = composer(self.hidden_states, past_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

    def test_recursive_reasoning_transformer(self):
        """Test the recursive reasoning transformer"""
        transformer = RecursiveReasoningTransformer(self.config)

        output, recursive_info = transformer(self.hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

        # Check recursive info
        self.assertIn('recursive_states', recursive_info)
        if self.config.use_recursive_routing:
            self.assertIn('routing_weights', recursive_info)
        if self.config.use_recursive_verification:
            self.assertIn('verification_scores', recursive_info)

    def test_recursive_reasoning_module(self):
        """Test the complete recursive reasoning module"""
        module = RecursiveReasoningModule(self.config)

        output, reasoning_info = module(self.hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

        # Check that the reasoning info contains expected keys
        if self.config.use_recursive_routing:
            self.assertIn('routing_weights', reasoning_info)
        if self.config.use_recursive_verification:
            self.assertIn('verification_scores', reasoning_info)
        self.assertIn('recursive_states', reasoning_info)

if __name__ == '__main__':
    unittest.main()