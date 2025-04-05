"""Test that all major components can be imported correctly."""
import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestImports(unittest.TestCase):
    def test_core_imports(self):
        """Test that core components can be imported."""
        # Test model imports
        from model import train, reasoning
        self.assertTrue(hasattr(train, 'train_model'))
        self.assertTrue(hasattr(reasoning, 'ChainOfThoughtReasoner'))

        # Test data imports
        from data import collect_data, efficient_loader
        self.assertTrue(hasattr(collect_data, 'DS_TO_SELECTION'))
        self.assertTrue(hasattr(efficient_loader, 'MemoryEfficientDataLoader'))

        # Test config imports
        from config import model_config
        self.assertTrue(hasattr(model_config, 'ModelConfig'))

if __name__ == '__main__':
    unittest.main()
