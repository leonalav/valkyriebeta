import unittest
import torch
import os
from model.core_model import EnhancedLanguageModel
from config.architecture_config import ArchitectureConfig

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Ensure knowledge graph directory exists
        os.makedirs('data/knowledge_graphs', exist_ok=True)
        
        # Create a minimal configuration for testing
        self.config = ArchitectureConfig(
            # Core architecture
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            
            # Memory efficiency
            use_flash_attention=False,
            use_memory_efficient_linear=False,
            use_gradient_checkpointing=False,
            
            # Reasoning components
            use_tree_reasoning=True,
            max_reasoning_depth=2,
            
            # Expert routing
            use_moe=True,
            num_experts=2,
            num_experts_per_token=1,
            
            # Memory bank
            use_enhanced_memory=True,
            memory_size=16,
            use_hierarchical_memory=True,
            num_memory_hierarchies=2,
            
            # Tree of Thought reasoning
            use_tree_of_thought=True,
            max_tree_depth=2,
            branching_factor=2,
            
            # Neural-Symbolic Integration
            use_neural_symbolic=True,
            symbol_vocabulary_size=32,
            num_symbolic_layers=1,
            
            # Recursive Reasoning
            use_recursive_reasoning_transformer=True,
            max_recursion_depth=2,
            
            # Multi-Hop Knowledge Reasoning
            use_knowledge_reasoning=True,
            max_hops=2,
            max_knowledge_items=5,
            
            # Verifiable Computation
            use_verifiable_computation=True,
            num_computation_units=2,
            
            # Model name
            model_name="test-model"
        )
        
        # Create a small vocabulary size for testing
        self.vocab_size = 100
        self.batch_size = 2
        self.seq_len = 8
        
        # Create input tensors
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.attention_mask = torch.ones(self.batch_size, self.seq_len)
        
    def test_model_initialization(self):
        """Test that the model initializes with all components"""
        model = EnhancedLanguageModel(self.config)
        
        # Check that all reasoning components are initialized
        self.assertIsNotNone(model.moe)
        self.assertIsNotNone(model.memory_layer)
        self.assertIsNotNone(model.tree_reasoner)
        self.assertIsNotNone(model.neural_symbolic)
        self.assertIsNotNone(model.recursive_reasoner)
        self.assertIsNotNone(model.knowledge_reasoner)
        self.assertIsNotNone(model.verifiable_computation)
        
    def test_forward_pass(self):
        """Test a complete forward pass through the model"""
        model = EnhancedLanguageModel(self.config)
        
        # Run forward pass
        outputs = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask
        )
        
        # Check that outputs contain expected keys
        self.assertIn('hidden_states', outputs)
        self.assertIn('output', outputs)
        self.assertIn('reasoning_trace', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['hidden_states'].shape, (self.batch_size, self.seq_len, self.config.hidden_size))
        self.assertEqual(outputs['output'].shape, (self.batch_size, self.seq_len, self.config.hidden_size))
        
        # Check reasoning trace
        reasoning_trace = outputs['reasoning_trace']
        
        # MoE should be in the trace
        for key in reasoning_trace.keys():
            if key.startswith('moe_'):
                moe_found = True
                break
        else:
            moe_found = False
        self.assertTrue(moe_found, "MoE trace not found")
        
        # Check for tree reasoning trace
        self.assertIn('tree_reasoning', reasoning_trace)
        
        # Check for neural symbolic trace
        self.assertIn('neural_symbolic', reasoning_trace)
        
        # Check for recursive reasoning trace
        self.assertIn('recursive_reasoning', reasoning_trace)
        
        # Check for knowledge reasoning trace
        self.assertIn('knowledge_reasoning', reasoning_trace)
        
        # Check for verifiable computation trace
        self.assertIn('verifiable_computation', reasoning_trace)
        
    def test_selective_components(self):
        """Test that components can be selectively enabled/disabled"""
        # Create a config with only some components enabled
        selective_config = ArchitectureConfig(
            # Core architecture
            hidden_size=64,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            
            # Disable most components
            use_moe=False,
            use_enhanced_memory=False,
            use_tree_of_thought=False,
            use_neural_symbolic=True,  # Keep this one enabled
            use_recursive_reasoning_transformer=False,
            use_knowledge_reasoning=False,
            use_verifiable_computation=False,
            
            # Model name
            model_name="test-model"
        )
        
        model = EnhancedLanguageModel(selective_config)
        
        # Check that only the neural symbolic component is initialized
        self.assertIsNone(model.moe)
        self.assertIsNone(model.memory_layer)
        self.assertIsNone(model.tree_reasoner)
        self.assertIsNotNone(model.neural_symbolic)
        self.assertIsNone(model.recursive_reasoner)
        self.assertIsNone(model.knowledge_reasoner)
        self.assertIsNone(model.verifiable_computation)
        
        # Run forward pass
        outputs = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask
        )
        
        # Check that only neural symbolic trace is present
        reasoning_trace = outputs['reasoning_trace']
        self.assertIn('neural_symbolic', reasoning_trace)
        self.assertNotIn('tree_reasoning', reasoning_trace)
        self.assertNotIn('recursive_reasoning', reasoning_trace)
        self.assertNotIn('knowledge_reasoning', reasoning_trace)
        self.assertNotIn('verifiable_computation', reasoning_trace)
        
    def test_memory_reset(self):
        """Test that memory can be reset"""
        model = EnhancedLanguageModel(self.config)
        
        # Run forward pass
        _ = model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask
        )
        
        # Reset memory
        model.reset_memory()
        
        # Check that past states are reset
        self.assertIsNone(model.past_states)

if __name__ == '__main__':
    unittest.main() 