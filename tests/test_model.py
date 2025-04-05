import unittest
import torch
import torch.nn as nn
from model.model import LogicalReasoningTransformer
from model.attention import FlashAttention
from model.crnn import CRNN
from config.model_config import ModelConfig
import numpy as np
import pytest


class TestModelArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ModelConfig()
        cls.config.vocab_size = 1000
        cls.config.max_seq_length = 128
        cls.config.hidden_size = 256
        cls.config.num_layers = 4
        cls.config.num_heads = 8

    def setUp(self):
        self.model = LogicalReasoningTransformer(self.config)

    def test_model_initialization(self):
        """Test if model initializes correctly"""
        self.assertIsInstance(self.model, nn.Module)
        self.assertEqual(self.model.token_embeddings.num_embeddings, self.config.vocab_size)
        self.assertEqual(self.model.token_embeddings.embedding_dim, self.config.hidden_size)

    def test_forward_pass(self):
        """Test forward pass with dummy input"""
        batch_size = 4
        seq_length = 32
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Check output shape
        expected_shape = (batch_size, seq_length, self.config.vocab_size)
        self.assertEqual(outputs.shape, expected_shape)

    def test_attention_mechanism(self):
        """Test attention mechanism"""
        attention = FlashAttention(self.config)
        hidden_states = torch.randn(2, 16, self.config.hidden_size)
        attention_mask = torch.ones(2, 16)

        output = attention(hidden_states, attention_mask)
        self.assertEqual(output.shape, hidden_states.shape)

    def test_crnn_component(self):
        """Test CRNN component"""
        crnn = CRNN(self.config)
        input_tensor = torch.randn(2, 16, self.config.hidden_size)

        output = crnn(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)

    def test_model_parameters(self):
        """Test if all parameters are trainable"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertGreater(trainable_params, 0)

    def test_gradient_flow(self):
        """Test gradient flow through the model"""
        input_ids = torch.randint(0, self.config.vocab_size, (2, 16))
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(0, self.config.vocab_size, (2, 16))

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, self.config.vocab_size), labels.view(-1))
        loss.backward()

        # Check if gradients are computed
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

class TestModelBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ModelConfig()
        cls.model = LogicalReasoningTransformer(cls.config)

    def test_logical_operations(self):
        """Test basic logical operations"""
        # Example input encoding logical operations
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Dummy logical sequence
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertIsNotNone(outputs)

    def test_memory_efficiency(self):
        """Test memory usage during forward pass"""
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        input_ids = torch.randint(0, self.config.vocab_size, (1, 512))
        attention_mask = torch.ones_like(input_ids)

        _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_usage = final_memory - initial_memory

        # Check if memory usage is within reasonable limits
        if torch.cuda.is_available():
            self.assertLess(memory_usage, 1e9)  # Less than 1GB

def test_model_forward():
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        max_seq_length=512
    )

    model = LogicalReasoningTransformer(config)
    batch_size = 2
    seq_length = 10

    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    outputs = model(input_ids, attention_mask=attention_mask)
    assert outputs.shape == (batch_size, seq_length, config.hidden_size)