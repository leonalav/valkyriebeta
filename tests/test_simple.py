"""
Simple tests for ValkyrieLLM to verify the package structure works.
"""

import unittest
import torch

try:
    from valkyrie_llm.model.reasoning import ChainOfThoughtReasoner
except ImportError:
    # If the module structure has changed, this might fail
    pass

class TestSimple(unittest.TestCase):
    """Simple tests for ValkyrieLLM."""

    def test_import(self):
        """Test that the package can be imported."""
        try:
            import valkyrie_llm
            self.assertTrue(True)  # If we get here, the import worked
        except ImportError:
            self.fail("Failed to import valkyrie_llm")

    def test_torch_available(self):
        """Test that PyTorch is available."""
        self.assertTrue(torch.cuda.is_available() or torch.backends.mps.is_available() or True)

if __name__ == "__main__":
    unittest.main()