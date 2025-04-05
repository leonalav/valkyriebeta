import unittest
import torch
from model.logical_layers import TreeLSTMCell, MemoryAugmentedNetwork, LogicalReasoningModule
from config.model_config import ModelConfig

class TestLogicalReasoning(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.tree_lstm = TreeLSTMCell(self.config.hidden_size, self.config.hidden_size)
        self.memory_network = MemoryAugmentedNetwork(self.config)
        self.reasoning_module = LogicalReasoningModule(self.config)

    def test_tree_lstm(self):
        """Test TreeLSTM operations"""
        input_tensor = torch.randn(1, self.config.hidden_size)
        child_states = [
            (torch.randn(1, self.config.hidden_size), torch.randn(1, self.config.hidden_size))
            for _ in range(2)
        ]

        h_new, c_new = self.tree_lstm(input_tensor, child_states)

        self.assertEqual(h_new.shape, (1, self.config.hidden_size))
        self.assertEqual(c_new.shape, (1, self.config.hidden_size))

    def test_memory_network(self):
        """Test Memory Network operations"""
        batch_size = 2
        seq_length = 16
        hidden_states = torch.randn(batch_size, seq_length, self.config.hidden_size)

        output = self.memory_network(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)

    def test_logical_module(self):
        """Test complete logical reasoning module"""
        batch_size = 2
        seq_length = 16
        hidden_states = torch.randn(batch_size, seq_length, self.config.hidden_size)

        # Create sample tree structure
        tree_structure = [
            [],  # Root node
            [0],  # Child of root
            [0]   # Another child of root
        ]

        output = self.reasoning_module(hidden_states, tree_structure)
        self.assertEqual(output.shape, hidden_states.shape)

    def test_logical_consistency(self):
        """Test logical consistency of outputs"""
        def create_logical_input(operation: str) -> torch.Tensor:
            # Create input tensor encoding logical operations
            input_tensor = torch.zeros(1, self.config.hidden_size)
            if operation == "AND":
                input_tensor[0, 0] = 1  # Encode AND operation
            elif operation == "OR":
                input_tensor[0, 1] = 1  # Encode OR operation
            return input_tensor

        # Test AND operation
        and_input = create_logical_input("AND")
        child_states_true = [
            (torch.ones(1, self.config.hidden_size), torch.ones(1, self.config.hidden_size))
            for _ in range(2)
        ]

        h_new, _ = self.tree_lstm(and_input, child_states_true)
        self.assertTrue(torch.all(h_new > 0))  # AND of true values should be true

        # Test OR operation
        or_input = create_logical_input("OR")
        child_states_mixed = [
            (torch.ones(1, self.config.hidden_size), torch.ones(1, self.config.hidden_size)),
            (torch.zeros(1, self.config.hidden_size), torch.zeros(1, self.config.hidden_size))
        ]

        h_new, _ = self.tree_lstm(or_input, child_states_mixed)
        self.assertTrue(torch.any(h_new > 0))  # OR with one true should be true