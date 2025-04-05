"""
Test script to verify that all imports are working correctly.
"""
import sys
import os
from pathlib import Path
import random
import argparse
import traceback
from dataclasses import dataclass, field, asdict

# Add the current directory to the path
sys.path.append('.')

# Test imports
try:
    from data.tokenizer import Tokenizer, EnhancedTokenizer
    print("✓ Successfully imported Tokenizer and EnhancedTokenizer")
except ImportError as e:
    print(f"✗ Failed to import Tokenizer and EnhancedTokenizer: {e}")

# Test asdict
try:
    @dataclass
    class TestClass:
        name: str = "test"
        value: int = 42
    
    test_instance = TestClass()
    test_dict = asdict(test_instance)
    print(f"✓ Successfully used asdict: {test_dict}")
except Exception as e:
    print(f"✗ Failed to use asdict: {e}")

# Test random
try:
    random_number = random.randint(1, 100)
    print(f"✓ Successfully used random: {random_number}")
except Exception as e:
    print(f"✗ Failed to use random: {e}")

# Test argparse
try:
    parser = argparse.ArgumentParser(description="Test script")
    parser.add_argument("--test", type=str, default="test", help="Test argument")
    print("✓ Successfully used argparse")
except Exception as e:
    print(f"✗ Failed to use argparse: {e}")

# Test Path
try:
    current_path = Path.cwd()
    print(f"✓ Successfully used Path: {current_path}")
except Exception as e:
    print(f"✗ Failed to use Path: {e}")

# Test traceback
try:
    def test_function():
        try:
            1/0
        except Exception:
            tb = traceback.format_exc()
            return tb
    
    tb = test_function()
    print("✓ Successfully used traceback")
except Exception as e:
    print(f"✗ Failed to use traceback: {e}")

print("\nAll tests completed!") 