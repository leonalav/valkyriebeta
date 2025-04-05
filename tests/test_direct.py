"""
Direct test for ValkyrieLLM.
"""

import unittest
import sys
import torch

class TestValkyrieLLM(unittest.TestCase):
    def test_import(self):
        """Test that we can import the package"""
        import valkyrie_llm
        self.assertTrue(hasattr(valkyrie_llm, "__version__"))
        
    def test_model_import(self):
        """Test that we can import from the model module"""
        from valkyrie_llm.model.reasoning import ChainOfThoughtReasoner
        reasoner = ChainOfThoughtReasoner()
        self.assertIsNotNone(reasoner)
        
    def test_core_model(self):
        """Test that we can import and instantiate the core model"""
        from valkyrie_llm.model.core_model import CoreModel
        model = CoreModel(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4)
        self.assertIsNotNone(model)
        
if __name__ == "__main__":
    unittest.main() 