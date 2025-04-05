import torch
import torch.nn as nn
import logging
import argparse
import sys
import os
from time import time
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration import IntegrationManager
from valkyrie_llm import ValkyrieLLM
from adaptive_reasoning import ReasoningStrategy
from computational_efficiency import ComputeTrackerModule

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description="Demo of Valkyrie's integrated reasoning capabilities")
    parser.add_argument("--model_path", type=str, default="valkyrie-7b", help="Path to pretrained model")
    parser.add_argument("--enable_meta_reasoning", action="store_true", help="Enable meta-reasoning optimizer")
    parser.add_argument("--enable_prompt_augmentation", action="store_true", help="Enable self-reflective prompt augmentation")
    parser.add_argument("--enable_strategy_memory", action="store_true", help="Enable strategy sequence memory")
    parser.add_argument("--enable_compute_tracker", action="store_true", help="Enable compute tracker")
    parser.add_argument("--enable_all", action="store_true", help="Enable all advanced reasoning components")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for logs")
    return parser

def setup_model(args):
    """Set up model with specified components."""
    # Initialize integration manager
    integration_manager = IntegrationManager()
    
    # Configure active components
    config = {
        "enable_meta_reasoning": args.enable_meta_reasoning or args.enable_all,
        "enable_prompt_augmentation": args.enable_prompt_augmentation or args.enable_all,
        "enable_strategy_memory": args.enable_strategy_memory or args.enable_all,
        "enable_compute_tracker": args.enable_compute_tracker or args.enable_all,
        "enable_advanced_reasoning_integration": args.enable_all,
        
        # Default configuration options
        "hidden_size": 768,
        "strategy_embedding_size": 128,
        "num_strategy_types": 10,
        "max_memory_size": 1000,
        "confidence_threshold": 0.7,
        "max_retries": 3,
        "compute_budget": 1.0
    }
    
    # Apply configuration
    integration_manager.configure(config)
    
    # Load the base model
    logger.info(f"Loading base model from {args.model_path}")
    try:
        # Load tokenizer first (assumes HuggingFace-compatible model)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
        # Try to load model with HuggingFace first
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            logger.warning(f"Failed to load with AutoModelForCausalLM: {e}")
            # Fallback to custom ValkyrieLLM loader
            model = ValkyrieLLM.from_pretrained(args.model_path)
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
    
    # Initialize model with components
    enhanced_model = integration_manager.initialize_model(model, tokenizer)
    
    return enhanced_model, tokenizer, integration_manager

def run_demo(model, tokenizer, integration_manager, args):
    """Run the demonstration with various reasoning examples."""
    # Define example problems with different reasoning requirements
    examples = [
        {
            "name": "Math problem",
            "prompt": "If a store sells apples for $0.50 each and oranges for $0.75 each, and I buy 6 apples and 4 oranges, how much will I spend in total?",
            "task_type": "arithmetic",
            "expected_strategy": ReasoningStrategy.STEP_BY_STEP
        },
        {
            "name": "Logical reasoning",
            "prompt": "All birds can fly. Penguins are birds. Can penguins fly? Explain your reasoning step by step.",
            "task_type": "logical",
            "expected_strategy": ReasoningStrategy.CHAIN_OF_THOUGHT
        },
        {
            "name": "Code debugging",
            "prompt": "Debug this code:\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    else:\n        fib = fibonacci(n-1)\n        fib.append(fib[n-2] + fib[n-3])\n        return fib\n```",
            "task_type": "coding",
            "expected_strategy": ReasoningStrategy.VERIFICATION
        },
        {
            "name": "Abstract reasoning",
            "prompt": "Consider the sequence: 2, 6, 12, 20, 30, ... What's the next number and why?",
            "task_type": "pattern",
            "expected_strategy": ReasoningStrategy.COMPARATIVE
        },
        {
            "name": "Symbolic reasoning",
            "prompt": "If we define the operation ⊕ such that a ⊕ b = a² + b², what is the value of (2 ⊕ 3) ⊕ 4?",
            "task_type": "symbolic",
            "expected_strategy": ReasoningStrategy.SYMBOLIC
        }
    ]
    
    # Run each example
    results = []
    compute_stats = {}
    
    for i, example in enumerate(examples):
        logger.info(f"\n\n===== Running example {i+1}: {example['name']} =====")
        print(f"\nExample {i+1}: {example['name']}\nPrompt: {example['prompt']}")
        
        # Get start time
        start_time = time()
        
        # Encode input
        inputs = tokenizer(example["prompt"], return_tensors="pt").to(model.device)
        
        # Generate with task type for compute tracking
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            prompt=example["prompt"],
            task_type=example["task_type"],
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Calculate elapsed time
        elapsed_time = time() - start_time
        
        # Decode output
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Print the result
        print(f"\nResponse:\n{output_text}\n")
        print(f"Generation took {elapsed_time:.2f} seconds")
        
        # Get compute stats if available
        if hasattr(integration_manager, "compute_tracker"):
            stats = integration_manager.compute_tracker.get_session_stats()
            compute_stats[example["name"]] = stats
            
            # Print relevant stats
            if stats:
                tokens = stats.get("total_tokens", 0)
                time_spent = stats.get("total_time", 0)
                print(f"Tokens used: {tokens}, Compute time: {time_spent:.2f}s")
                
                # Print used strategy if available
                used_strategies = stats.get("strategy_usage", {})
                if used_strategies:
                    strategies = ", ".join([f"{s} ({c} times)" for s, c in used_strategies.items() if c > 0])
                    print(f"Strategies used: {strategies}")
        
        # Get the used strategy from meta_reasoning if available
        used_strategy = "Unknown"
        if hasattr(integration_manager, "meta_reasoning"):
            # Get the selected strategy (this is just a placeholder, as we'd need to extract from meta_reasoning)
            used_strategy = example["expected_strategy"].name  # Simplified for demo
        
        # Store results
        results.append({
            "name": example["name"],
            "prompt": example["prompt"],
            "response": output_text,
            "time": elapsed_time,
            "expected_strategy": example["expected_strategy"].name,
            "used_strategy": used_strategy
        })
    
    # Print final statistics
    print("\n===== Performance Summary =====")
    
    for i, result in enumerate(results):
        print(f"\nExample {i+1}: {result['name']}")
        print(f"Time: {result['time']:.2f}s")
        print(f"Expected strategy: {result['expected_strategy']}")
        print(f"Used strategy: {result['used_strategy']}")
    
    # Print compute stats if available
    if compute_stats:
        print("\n===== Compute Statistics =====")
        total_tokens = sum(stats.get("total_tokens", 0) for stats in compute_stats.values())
        total_time = sum(stats.get("total_time", 0) for stats in compute_stats.values())
        print(f"Total tokens used: {total_tokens}")
        print(f"Total compute time: {total_time:.2f}s")
        print(f"Tokens per second: {total_tokens/total_time:.2f}" if total_time > 0 else "N/A")
    
    return results

def main():
    """Main entry point for the demo."""
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()
    
    # Set up directories
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set up model
    model, tokenizer, integration_manager = setup_model(args)
    
    # Run demo
    print("\n===== Starting Valkyrie Integrated Reasoning Demo =====")
    print("This demo showcases the advanced reasoning capabilities of the Valkyrie model")
    print(f"Enabled components:")
    print(f"  - Meta-reasoning optimizer: {args.enable_meta_reasoning or args.enable_all}")
    print(f"  - Self-reflective prompt augmentation: {args.enable_prompt_augmentation or args.enable_all}")
    print(f"  - Strategy sequence memory: {args.enable_strategy_memory or args.enable_all}")
    print(f"  - Compute tracker: {args.enable_compute_tracker or args.enable_all}")
    print(f"  - Full advanced reasoning integration: {args.enable_all}")
    
    results = run_demo(model, tokenizer, integration_manager, args)
    
    # Save results
    import json
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.log_dir, f"results_{timestamp}.json")
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Cleanup and save memory state if available
    if hasattr(integration_manager, "strategy_memory"):
        memory_path = os.path.join(args.log_dir, f"memory_{timestamp}.pkl")
        integration_manager.strategy_memory.save_state(memory_path)
        print(f"Memory state saved to {memory_path}")
    
    if hasattr(integration_manager, "compute_tracker"):
        stats_path = os.path.join(args.log_dir, f"compute_stats_{timestamp}.json")
        integration_manager.compute_tracker.save_stats(stats_path)
        print(f"Compute statistics saved to {stats_path}")
    
    print("\n===== Demo Completed =====")

if __name__ == "__main__":
    main() 
 