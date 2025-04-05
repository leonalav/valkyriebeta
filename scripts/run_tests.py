import os
import sys
import pytest
import json
import logging
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tests.test_framework import TestFramework, TestConfig
from model import LogicalReasoningTransformer
from config import (
    ModelConfig,
    TrainingConfig,
    MemoryConfig,
    TrainingEfficiencyConfig,
    EfficientTransformerConfig
)

def setup_logging(log_dir: str = "test_logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_run_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def save_test_results(results: Dict[str, Any], output_dir: str = "test_results"):
    """Save test results to file"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"test_results_{timestamp}.json")
    
    # Convert non-serializable types to strings
    def serialize(obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return str(obj)
    
    with open(result_file, 'w') as f:
        json.dump(results, f, default=serialize, indent=2)
        
    return result_file

def run_all_tests(configs: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run all tests and return results"""
    logger = logging.getLogger(__name__)
    
    # Create configs if not provided
    if configs is None:
        model_config = ModelConfig()
        training_config = TrainingConfig()
        memory_config = MemoryConfig()
        efficiency_config = TrainingEfficiencyConfig()
        
        config = EfficientTransformerConfig.from_configs(
            model_config=model_config,
            training_config=training_config,
            memory_config=memory_config,
            efficiency_config=efficiency_config
        )
        
        configs = {
            'model_config': config,
            'test_config': TestConfig(),
            'memory_config': memory_config,
            'efficiency_config': efficiency_config
        }
    
    # Create model
    logger.info("Initializing model...")
    model = LogicalReasoningTransformer(configs['model_config'])
    
    # Create test framework
    logger.info("Setting up test framework...")
    framework = TestFramework(
        model=model,
        test_config=configs['test_config'],
        memory_config=configs['memory_config'],
        efficiency_config=configs['efficiency_config']
    )
    
    try:
        # Run all tests
        logger.info("Starting test suite...")
        results = framework.run_all_tests()
        
        # Add test metadata
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'device': str(next(model.parameters()).device),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'model_config': configs['model_config'].to_dict(),
            'test_config': vars(configs['test_config'])
        }
        
        logger.info("All tests completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise
        
    finally:
        # Cleanup
        framework.cleanup()

def main():
    """Main entry point"""
    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting test run...")
    
    try:
        # Run tests
        results = run_all_tests()
        
        # Save results
        result_file = save_test_results(results)
        
        logger.info(f"Test results saved to: {result_file}")
        logger.info(f"Test logs saved to: {log_file}")
        
        # Print summary
        print("\nTEST SUMMARY:")
        for category, category_results in results.items():
            if category != 'metadata':
                print(f"\n{category.upper()}:")
                for metric, value in category_results.items():
                    print(f"  {metric}: {value}")
                    
        return 0
        
    except Exception as e:
        logger.error(f"Test run failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 