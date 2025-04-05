import unittest
import torch
from model.verifiable_computation import (
    VerifiableComputationConfig,
    ComputationUnit,
    SelfVerifier,
    CrossVerifier,
    VerificationRouter,
    VerificationMemory,
    VerificationComposer,
    VerifiableComputation,
    VerifiableComputationModule
)

class TestVerifiableComputation(unittest.TestCase):
    def setUp(self):
        self.config = VerifiableComputationConfig(
            hidden_size=64,
            num_heads=4,
            dropout=0.1,
            num_computation_units=3,
            verification_threshold=0.7,
            use_cross_verification=True,
            use_self_verification=True,
            use_verification_routing=True,
            num_verification_experts=2,
            use_verification_feedback=True,
            feedback_iterations=1,
            use_uncertainty_estimation=True,
            use_verification_memory=True,
            memory_size=16,
            use_verification_composition=True,
            composition_depth=1
        )
        self.batch_size = 2
        self.seq_len = 8
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)

    def test_computation_unit(self):
        """Test the computation unit component"""
        for unit_idx in range(self.config.num_computation_units):
            unit = ComputationUnit(self.config, unit_idx)

            output, uncertainty = unit(self.hidden_states)

            # Check output shapes
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))
            if self.config.use_uncertainty_estimation:
                self.assertEqual(uncertainty.shape, (self.batch_size, self.seq_len, 1))
            else:
                self.assertIsNone(uncertainty)

    def test_self_verifier(self):
        """Test the self verifier component"""
        verifier = SelfVerifier(self.config)

        verification_scores = verifier(self.hidden_states)

        # Check output shape
        self.assertEqual(verification_scores.shape, (self.batch_size, self.seq_len, 1))

    def test_cross_verifier(self):
        """Test the cross verifier component"""
        verifier = CrossVerifier(self.config)

        # Create mock computation results
        computation_results = [
            torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
            for _ in range(self.config.num_computation_units)
        ]

        verification_matrix = verifier(computation_results)

        # Check output shape
        self.assertEqual(verification_matrix.shape, (self.batch_size, self.seq_len, self.config.num_computation_units, self.config.num_computation_units))

    def test_verification_router(self):
        """Test the verification router component"""
        router = VerificationRouter(self.config)

        output, routing_weights = router(self.hidden_states)

        # Check output shapes
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))
        self.assertEqual(routing_weights.shape, (self.batch_size, self.seq_len, self.config.num_verification_experts))

    def test_verification_memory(self):
        """Test the verification memory component"""
        memory = VerificationMemory(self.config)

        # Create mock verification scores
        verification_scores = torch.ones(self.batch_size, self.seq_len, 1)

        output, updated_memory = memory(self.hidden_states, verification_scores)

        # Check output shapes
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))
        self.assertEqual(updated_memory.shape, (self.batch_size, self.config.memory_size, self.config.hidden_size))

    def test_verification_composer(self):
        """Test the verification composer component"""
        composer = VerificationComposer(self.config)

        # Create mock verification scores
        verification_scores = torch.ones(self.batch_size, self.seq_len, 1)

        output = composer(self.hidden_states, verification_scores)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

    def test_verifiable_computation(self):
        """Test the verifiable computation component"""
        computation = VerifiableComputation(self.config)

        output, verification_info = computation(self.hidden_states)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

        # Check verification info
        if self.config.use_self_verification:
            self.assertIn('self_verification_scores', verification_info)
        if self.config.use_cross_verification:
            self.assertIn('cross_verification_matrix', verification_info)
        if self.config.use_verification_routing:
            self.assertIn('routing_weights', verification_info)
        if self.config.use_uncertainty_estimation:
            self.assertIn('uncertainties', verification_info)

    def test_verifiable_computation_module(self):
        """Test the complete verifiable computation module"""
        module = VerifiableComputationModule(self.config)

        output, verification_info = module(self.hidden_states)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

        # Check verification info
        if self.config.use_self_verification:
            self.assertIn('self_verification_scores', verification_info)
        if self.config.use_cross_verification:
            self.assertIn('cross_verification_matrix', verification_info)
        if self.config.use_verification_routing:
            self.assertIn('routing_weights', verification_info)
        if self.config.use_uncertainty_estimation:
            self.assertIn('uncertainties', verification_info)

if __name__ == '__main__':
    unittest.main()