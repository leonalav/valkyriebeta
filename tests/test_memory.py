import unittest
import torch
import torch.nn as nn
from utils.optimization import MemoryOptimizer, GradientOptimizer
from config.model_config import ModelConfig
from model.model import LogicalReasoningTransformer

class TestMemoryOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ModelConfig()
        cls.config.use_quantization = True
        cls.config.quantization_bits = 8
        cls.model = LogicalReasoningTransformer(cls.config)
        cls.memory_optimizer = MemoryOptimizer(cls.model, cls.config)

    def test_quantization(self):
        """Test model quantization"""
        initial_size = sum(p.numel() * p.element_size() for p in self.model.parameters())

        self.memory_optimizer.quantize_model(quantization_bits=8)

        final_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        self.assertLess(final_size, initial_size)

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing"""
        self.memory_optimizer.optimize_memory()

        # Verify gradient checkpointing is enabled
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.assertTrue(self.model.is_gradient_checkpointing_enabled())

    def test_dynamic_padding(self):
        """Test dynamic padding functionality"""
        batch = {
            'input_ids': [
                torch.tensor([1, 2, 3]),
                torch.tensor([1, 2, 3, 4, 5])
            ],
            'attention_mask': [
                torch.tensor([1, 1, 1]),
                torch.tensor([1, 1, 1, 1, 1])
            ]
        }

        padded_batch = self.memory_optimizer.apply_dynamic_padding(batch)

        self.assertEqual(padded_batch['input_ids'].shape[1], 5)
        self.assertEqual(padded_batch['attention_mask'].shape[1], 5)

class TestGradientOptimization(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.model = LogicalReasoningTransformer(self.config)
        self.gradient_optimizer = GradientOptimizer(self.model, self.config)

    def test_gradient_accumulation(self):
        """Test gradient accumulation"""
        input_ids = torch.randint(0, self.config.vocab_size, (2, 16))
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs.mean()

        self.gradient_optimizer.optimize_backward(loss)

        # Check if gradients are accumulated correctly
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

    def test_parameter_grouping(self):
        """Test parameter grouping for optimization"""
        grouped_params = self.gradient_optimizer.get_grouped_parameters(
            self.model,
            weight_decay=0.01
        )

        self.assertEqual(len(grouped_params), 2)  # Should have decay and no-decay groups
        self.assertTrue(all('weight_decay' in group for group in grouped_params))

if __name__ == '__main__':
    unittest.main()