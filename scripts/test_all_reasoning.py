#!/usr/bin/env python
"""
Comprehensive test script for all reasoning components.
This script tests all reasoning components individually and together.
"""

import os
import sys
import torch
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.architecture_config import ArchitectureConfig
from model.core_model import EnhancedLanguageModel
from tokenizer.tokenizer import Tokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_component(component_name, config_override=None):
    """Test a specific reasoning component"""
    logger.info(f"Testing {component_name}...")
    
    # Create base config with all components disabled
    base_config = ArchitectureConfig(
        # Core architecture (small for testing)
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        
        # Disable all reasoning components
        use_moe=False,
        use_enhanced_memory=False,
        use_tree_of_thought=False,
        use_neural_symbolic=False,
        use_recursive_reasoning_transformer=False,
        use_knowledge_reasoning=False,
        use_verifiable_computation=False
    )
    
    # Apply component-specific override
    if config_override:
        for key, value in config_override.items():
            setattr(base_config, key, value)
    
    # Initialize model
    model = EnhancedLanguageModel(base_config)
    
    # Create dummy input
    tokenizer = Tokenizer()
    input_text = "Testing the reasoning capabilities of this model."
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.ones_like(input_ids)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Check if component is in reasoning trace
    reasoning_trace = outputs['reasoning_trace']
    component_found = False
    
    if component_name == "moe":
        component_found = any(key.startswith('moe_') for key in reasoning_trace.keys())
    elif component_name == "memory":
        component_found = 'memory_layer' in reasoning_trace
    elif component_name == "tree":
        component_found = 'tree_reasoning' in reasoning_trace
    elif component_name == "neural_symbolic":
        component_found = 'neural_symbolic' in reasoning_trace
    elif component_name == "recursive":
        component_found = 'recursive_reasoning' in reasoning_trace
    elif component_name == "knowledge":
        component_found = 'knowledge_reasoning' in reasoning_trace
    elif component_name == "verifiable":
        component_found = 'verifiable_computation' in reasoning_trace
    
    if component_found:
        logger.info(f"✓ {component_name} component is working correctly")
        return True
    else:
        logger.error(f"✗ {component_name} component is not working")
        return False

def test_all_components():
    """Test all reasoning components individually"""
    results = {}
    
    # Test Mixture of Experts
    results["moe"] = test_component("moe", {"use_moe": True, "num_experts": 2})
    
    # Test Enhanced Memory
    results["memory"] = test_component("memory", {"use_enhanced_memory": True, "memory_size": 16})
    
    # Test Tree-of-Thought Reasoning
    results["tree"] = test_component("tree", {"use_tree_of_thought": True, "max_tree_depth": 2})
    
    # Test Neural-Symbolic Integration
    results["neural_symbolic"] = test_component("neural_symbolic", {"use_neural_symbolic": True})
    
    # Test Recursive Reasoning
    results["recursive"] = test_component("recursive", {"use_recursive_reasoning_transformer": True})
    
    # Test Knowledge Reasoning
    results["knowledge"] = test_component("knowledge", {"use_knowledge_reasoning": True})
    
    # Test Verifiable Computation
    results["verifiable"] = test_component("verifiable", {"use_verifiable_computation": True})
    
    # Print summary
    logger.info("\n--- Test Results Summary ---")
    for component, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} - {component}")
    
    # Return True if all tests passed
    return all(results.values())

def test_integration():
    """Test all reasoning components together"""
    logger.info("Testing integration of all reasoning components...")
    
    # Create config with all components enabled
    config = ArchitectureConfig(
        # Core architecture (small for testing)
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        
        # Enable all reasoning components
        use_moe=True,
        num_experts=2,
        use_enhanced_memory=True,
        memory_size=16,
        use_tree_of_thought=True,
        max_tree_depth=2,
        use_neural_symbolic=True,
        use_recursive_reasoning_transformer=True,
        use_knowledge_reasoning=True,
        use_verifiable_computation=True
    )
    
    # Initialize model
    model = EnhancedLanguageModel(config)
    
    # Create dummy input
    tokenizer = Tokenizer()
    input_text = "Testing the integration of all reasoning components."
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_ids])
    attention_mask = torch.ones_like(input_ids)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Check if all components are in reasoning trace
    reasoning_trace = outputs['reasoning_trace']
    
    # Check for each component
    moe_found = any(key.startswith('moe_') for key in reasoning_trace.keys())
    memory_found = 'memory_layer' in reasoning_trace
    tree_found = 'tree_reasoning' in reasoning_trace
    neural_symbolic_found = 'neural_symbolic' in reasoning_trace
    recursive_found = 'recursive_reasoning' in reasoning_trace
    knowledge_found = 'knowledge_reasoning' in reasoning_trace
    verifiable_found = 'verifiable_computation' in reasoning_trace
    
    # Print results
    logger.info("\n--- Integration Test Results ---")
    logger.info(f"{'✓' if moe_found else '✗'} Mixture of Experts")
    logger.info(f"{'✓' if memory_found else '✗'} Enhanced Memory")
    logger.info(f"{'✓' if tree_found else '✗'} Tree-of-Thought Reasoning")
    logger.info(f"{'✓' if neural_symbolic_found else '✗'} Neural-Symbolic Integration")
    logger.info(f"{'✓' if recursive_found else '✗'} Recursive Reasoning")
    logger.info(f"{'✓' if knowledge_found else '✗'} Knowledge Reasoning")
    logger.info(f"{'✓' if verifiable_found else '✗'} Verifiable Computation")
    
    # Return True if all components are found
    all_found = (moe_found and memory_found and tree_found and neural_symbolic_found and 
                recursive_found and knowledge_found and verifiable_found)
    
    if all_found:
        logger.info("✓ Integration test PASSED - All components are working together")
        return True
    else:
        logger.error("✗ Integration test FAILED - Some components are not working together")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test reasoning components")
    parser.add_argument("--component", type=str, choices=["all", "integration", "moe", "memory", "tree", 
                        "neural_symbolic", "recursive", "knowledge", "verifiable"],
                        default="all", help="Component to test")
    args = parser.parse_args()
    
    # Ensure knowledge graph directory exists
    os.makedirs('data/knowledge_graphs', exist_ok=True)
    
    if args.component == "all":
        success = test_all_components()
    elif args.component == "integration":
        success = test_integration()
    else:
        # Test specific component
        config_override = {
            "moe": {"use_moe": True, "num_experts": 2},
            "memory": {"use_enhanced_memory": True, "memory_size": 16},
            "tree": {"use_tree_of_thought": True, "max_tree_depth": 2},
            "neural_symbolic": {"use_neural_symbolic": True},
            "recursive": {"use_recursive_reasoning_transformer": True},
            "knowledge": {"use_knowledge_reasoning": True},
            "verifiable": {"use_verifiable_computation": True}
        }
        success = test_component(args.component, config_override[args.component])
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 