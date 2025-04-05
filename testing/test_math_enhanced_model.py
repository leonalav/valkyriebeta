import os
import sys
import torch
import json
import argparse
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.nanogpt import GPTConfig
from model.logical_nanogpt import LogicalGPT
from model.math_reasoning import DynamicReasoningRouter, MathReasoningConfig
from training.advanced_trainer import AdvancedTrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("math_model_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("math_model_test")

# Sample math problems for testing
SAMPLE_PROBLEMS = [
    {
        "id": "algebra-1",
        "problem": "Solve for x: 3x + 7 = 22",
        "solution": "3x + 7 = 22\n3x = 15\nx = 5",
        "answer": "5",
        "domain": "algebra"
    },
    {
        "id": "calculus-1",
        "problem": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
        "solution": "f'(x) = 3x^2 + 4x - 5",
        "answer": "3x^2 + 4x - 5",
        "domain": "calculus"
    },
    {
        "id": "geometry-1",
        "problem": "A rectangle has a length of 8 units and a width of 6 units. What is its area?",
        "solution": "Area = length × width\nArea = 8 × 6\nArea = 48 square units",
        "answer": "48",
        "domain": "geometry"
    },
    {
        "id": "statistics-1",
        "problem": "The mean of 5 numbers is 12. If 4 of the numbers are 10, 11, 13, and 14, what is the fifth number?",
        "solution": "Sum of all numbers = 5 × 12 = 60\nSum of 4 known numbers = 10 + 11 + 13 + 14 = 48\nFifth number = 60 - 48 = 12",
        "answer": "12",
        "domain": "statistics"
    },
    {
        "id": "logic-1",
        "problem": "If all A are B, and all B are C, what can we conclude?",
        "solution": "Using syllogistic reasoning:\nIf all A are B, and all B are C, then all A are C.",
        "answer": "All A are C",
        "domain": "logic"
    }
]

def load_test_cases(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load test cases from file or use default samples
    
    Args:
        file_path: Path to JSON file with test cases
        
    Returns:
        List of test case dictionaries
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    logger.info("Using built-in sample problems for testing")
    return SAMPLE_PROBLEMS

def create_model(
    vocab_size: int = 50257,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    checkpoint_path: Optional[str] = None
) -> LogicalGPT:
    """Create and initialize the enhanced LogicalGPT model
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden size dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        checkpoint_path: Optional path to load weights from
        
    Returns:
        Initialized LogicalGPT model
    """
    # Create GPT config
    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        n_embd=hidden_size,
        n_layer=num_layers,
        n_head=num_heads,
        block_size=1024
    )
    
    # Create model
    model = LogicalGPT(gpt_config)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
    return model

def evaluate_problem(
    model: LogicalGPT,
    problem: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate model on a single problem
    
    Args:
        model: LogicalGPT model
        problem: Problem dictionary
        device: Device to run inference on
        
    Returns:
        Dictionary with evaluation results
    """
    # Prepare input
    problem_text = problem["problem"]
    prompt = f"Problem: {problem_text}\nSolution:"
    
    # Tokenize input (using a dummy tokenizer for this example)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(device)  # Dummy input_ids
    
    # Record start time
    start_time = time.time()
    
    # Generate solution
    try:
        # Use the model to generate a solution
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                use_math_reasoning=True,
                problem_type=problem.get("domain", "general")
            )
            
            # Get verification info (if available)
            math_reasoning_info = outputs.get("math_reasoning_info", {})
            is_valid = math_reasoning_info.get("verification_info", {}).get("is_valid", None)
            
            # Get reasoning path info (if available)
            routing_info = math_reasoning_info.get("routing_info", {})
            path_weights = routing_info.get("path_weights", {})
            
            # For demonstration purposes (actual implementation would use the real output)
            generated_text = "This is a placeholder for the model's generated solution."
            
    except Exception as e:
        logger.error(f"Error evaluating problem {problem['id']}: {str(e)}")
        return {
            "id": problem["id"],
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }
        
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Verify solution
    expected_answer = problem.get("answer", "")
    is_correct = expected_answer in generated_text
    
    return {
        "id": problem["id"],
        "domain": problem.get("domain", "general"),
        "problem": problem_text,
        "expected_solution": problem.get("solution", ""),
        "generated_solution": generated_text,
        "success": True,
        "is_correct": is_correct,
        "is_valid": is_valid,
        "elapsed_time": elapsed_time,
        "path_weights": path_weights
    }

def run_evaluation(
    model: LogicalGPT,
    test_cases: List[Dict[str, Any]],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate model on test cases
    
    Args:
        model: LogicalGPT model
        test_cases: List of test case dictionaries
        output_file: Optional path to save detailed results
        
    Returns:
        Dictionary with aggregated results
    """
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Initialize results
    all_results = []
    aggregated_results = {
        "total_cases": len(test_cases),
        "successful_runs": 0,
        "correct_answers": 0,
        "valid_reasoning": 0,
        "total_time": 0.0,
        "domain_metrics": {}
    }
    
    # Initialize domain metrics
    domains = set(case.get("domain", "general") for case in test_cases)
    for domain in domains:
        aggregated_results["domain_metrics"][domain] = {
            "total": 0,
            "successful": 0,
            "correct": 0,
            "valid": 0,
            "avg_time": 0.0
        }
    
    # Evaluate each test case
    logger.info(f"Evaluating {len(test_cases)} test cases")
    for i, test_case in enumerate(test_cases):
        logger.info(f"Evaluating test case {i+1}/{len(test_cases)}: {test_case['id']}")
        
        # Run evaluation
        result = evaluate_problem(model, test_case, device)
        all_results.append(result)
        
        # Update aggregated metrics
        domain = result.get("domain", "general")
        if result["success"]:
            aggregated_results["successful_runs"] += 1
            aggregated_results["domain_metrics"][domain]["successful"] += 1
            
            if result.get("is_correct", False):
                aggregated_results["correct_answers"] += 1
                aggregated_results["domain_metrics"][domain]["correct"] += 1
                
            if result.get("is_valid", False):
                aggregated_results["valid_reasoning"] += 1
                aggregated_results["domain_metrics"][domain]["valid"] += 1
                
        aggregated_results["total_time"] += result["elapsed_time"]
        aggregated_results["domain_metrics"][domain]["total"] += 1
        aggregated_results["domain_metrics"][domain]["avg_time"] += result["elapsed_time"]
    
    # Calculate averages
    for domain in aggregated_results["domain_metrics"]:
        domain_metrics = aggregated_results["domain_metrics"][domain]
        if domain_metrics["total"] > 0:
            domain_metrics["avg_time"] /= domain_metrics["total"]
    
    # Add summary metrics
    aggregated_results["success_rate"] = aggregated_results["successful_runs"] / aggregated_results["total_cases"]
    aggregated_results["accuracy"] = aggregated_results["correct_answers"] / aggregated_results["total_cases"]
    aggregated_results["validation_rate"] = aggregated_results["valid_reasoning"] / aggregated_results["total_cases"]
    aggregated_results["avg_time_per_problem"] = aggregated_results["total_time"] / aggregated_results["total_cases"]
    
    # Save detailed results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                "aggregated_results": aggregated_results,
                "detailed_results": all_results
            }, f, indent=2)
        logger.info(f"Detailed results saved to {output_file}")
    
    return aggregated_results

def test_math_domains(model: LogicalGPT):
    """Test specific math domain routing capabilities"""
    # Create a dummy input tensor
    input_tensor = torch.randn(1, 10, model.config.n_embd)
    
    # Check if math_reasoner exists
    if not hasattr(model, "math_reasoner") or not hasattr(model.math_reasoner, "reasoning_router"):
        logger.warning("Model does not have math reasoning capabilities")
        return
        
    # Test the routing mechanism
    reasoning_router = model.math_reasoner.reasoning_router
    output, info = reasoning_router(input_tensor)
    
    # Log the results
    logger.info("Testing math domain routing:")
    logger.info(f"Output tensor shape: {output.shape}")
    logger.info(f"Path weights: {info['path_weights']}")
    
    # Test each domain individually
    test_domains = ["algebraic", "geometric", "statistical", "logical", "arithmetic"]
    for domain in test_domains:
        if domain in info["path_weights"]:
            logger.info(f"Domain {domain} weight: {info['path_weights'][domain]:.4f}")
            
def test_symbolic_processing(model: LogicalGPT):
    """Test symbolic math processing capabilities"""
    # Create a dummy input tensor
    input_tensor = torch.randn(1, 10, model.config.n_embd)
    
    # Check if math_reasoner exists
    if not hasattr(model, "math_reasoner") or not hasattr(model.math_reasoner, "symbolic_transformer"):
        logger.warning("Model does not have symbolic math processing capabilities")
        return
        
    # Test the symbolic transformer
    symbolic_transformer = model.math_reasoner.symbolic_transformer
    output = symbolic_transformer(input_tensor)
    
    # Log the results
    logger.info("Testing symbolic math processing:")
    logger.info(f"Input tensor shape: {input_tensor.shape}")
    logger.info(f"Output tensor shape: {output.shape}")
    logger.info(f"Output mean: {output.mean().item():.4f}")
    logger.info(f"Output std: {output.std().item():.4f}")

def main():
    parser = argparse.ArgumentParser(description="Test the enhanced mathematical reasoning model")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--test_cases", type=str, help="Path to test cases JSON file")
    parser.add_argument("--output", type=str, default="math_eval_results.json", 
                       help="Path to save evaluation results")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    
    args = parser.parse_args()
    
    # Load test cases
    test_cases = load_test_cases(args.test_cases)
    
    # Create model
    model = create_model(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        checkpoint_path=args.checkpoint
    )
    
    # Run component tests
    test_math_domains(model)
    test_symbolic_processing(model)
    
    # Run evaluation
    results = run_evaluation(model, test_cases, args.output)
    
    # Print summary results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total test cases: {results['total_cases']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Validation rate: {results['validation_rate']:.2%}")
    print(f"Average time per problem: {results['avg_time_per_problem']:.4f}s")
    print("\nResults by domain:")
    
    for domain, metrics in results["domain_metrics"].items():
        if metrics["total"] > 0:
            print(f"  {domain.capitalize()}:")
            print(f"    Total problems: {metrics['total']}")
            print(f"    Accuracy: {metrics['correct'] / metrics['total']:.2%}")
            if "valid" in metrics:
                print(f"    Valid reasoning: {metrics['valid'] / metrics['total']:.2%}")
            print(f"    Avg time: {metrics['avg_time']:.4f}s")
            
    print("="*50)

if __name__ == "__main__":
    main() 