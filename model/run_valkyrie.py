#!/usr/bin/env python
"""
Valkyrie Enhanced Reasoning Model Runner

This script provides a simple way to run the Valkyrie model with all
enhanced reasoning capabilities.
"""

import os
import sys
import argparse
import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('valkyrie.log')
    ]
)
logger = logging.getLogger("valkyrie")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Valkyrie model with enhanced reasoning")
    
    # Model parameters
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="valkyrie-7b", 
        help="Path to pretrained model or model identifier"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=2048, 
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        choices=["bf16", "fp16", "fp32"], 
        default="fp16", 
        help="Precision for model execution"
    )
    
    # Enhanced reasoning components
    parser.add_argument(
        "--enable_components", 
        type=str, 
        nargs="+", 
        default=["meta_reasoning", "prompt_augmentation", "strategy_memory", "compute_tracker"],
        help="Enable specific reasoning components"
    )
    parser.add_argument(
        "--disable_components", 
        type=str, 
        nargs="+", 
        default=[],
        help="Disable specific reasoning components"
    )
    parser.add_argument(
        "--enable_all", 
        action="store_true", 
        help="Enable all enhanced reasoning components"
    )
    
    # Input/output parameters
    parser.add_argument(
        "--input_file", 
        type=str, 
        help="Input file with prompts (one per line)"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="valkyrie_output.txt", 
        help="Output file for generated responses"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Run in interactive mode"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="Nucleus sampling probability"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=512, 
        help="Maximum number of tokens to generate"
    )
    
    # Advanced options
    parser.add_argument(
        "--save_memory", 
        type=str, 
        help="Path to save memory state after execution"
    )
    parser.add_argument(
        "--load_memory", 
        type=str, 
        help="Path to load memory state before execution"
    )
    parser.add_argument(
        "--compute_budget", 
        type=float, 
        default=1.0, 
        help="Computational budget (0.0-1.0)"
    )
    parser.add_argument(
        "--device", 
        type=str,
        default="auto", 
        help="Device to run on (cuda, cpu, mps, or auto)"
    )
    
    return parser.parse_args()

def determine_device(device_arg):
    """Determine the device to use based on argument and availability."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg

def setup_model(args):
    """Set up and initialize the model with enhanced reasoning components."""
    from integration import IntegrationManager
    from valkyrie_llm import ValkyrieLLM
    
    logger.info(f"Setting up model from {args.model_path}")
    
    # Determine device
    device = determine_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create the integration manager for enhanced reasoning
    integration_manager = IntegrationManager()
    
    # Determine which components to enable
    components_to_enable = set(args.enable_components)
    
    if args.enable_all:
        # Enable all components including integration
        components_to_enable = {
            "meta_reasoning", 
            "prompt_augmentation", 
            "strategy_memory", 
            "compute_tracker", 
            "targeted_finetuning",
            "advanced_reasoning_integration"
        }
    
    # Remove disabled components
    for component in args.disable_components:
        if component in components_to_enable:
            components_to_enable.remove(component)
    
    # Create configuration
    config = {
        # Enable components
        "enable_meta_reasoning": "meta_reasoning" in components_to_enable,
        "enable_prompt_augmentation": "prompt_augmentation" in components_to_enable,
        "enable_strategy_memory": "strategy_memory" in components_to_enable,
        "enable_compute_tracker": "compute_tracker" in components_to_enable,
        "enable_targeted_finetuning": "targeted_finetuning" in components_to_enable,
        "enable_advanced_reasoning_integration": "advanced_reasoning_integration" in components_to_enable,
        
        # Component-specific settings
        "compute_budget": args.compute_budget,
        "max_memory_size": 1000,
        "confidence_threshold": 0.7,
        "max_retries": 3,
    }
    
    # Apply configuration
    integration_manager.configure(config)
    
    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise e
    
    # Set torch dtype based on precision
    if args.precision == "bf16":
        dtype = torch.bfloat16
    elif args.precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Load the base model
    try:
        from transformers import AutoModelForCausalLM
        
        logger.info(f"Loading model with {args.precision} precision")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map=device if device != "mps" else None  # device_map not supported on MPS
        )
        
        if device == "mps":
            model = model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model with AutoModelForCausalLM: {e}")
        try:
            # Fallback to custom ValkyrieLLM loader
            model = ValkyrieLLM.from_pretrained(args.model_path)
            model = model.to(device)
        except Exception as e2:
            logger.error(f"Failed to load model with ValkyrieLLM: {e2}")
            raise e2
    
    # Initialize model with components
    enhanced_model = integration_manager.initialize_model(model, tokenizer)
    
    # Load memory state if specified
    if args.load_memory and hasattr(integration_manager, "strategy_memory"):
        try:
            logger.info(f"Loading memory state from {args.load_memory}")
            integration_manager.strategy_memory.load_state(args.load_memory)
        except Exception as e:
            logger.warning(f"Failed to load memory state: {e}")
    
    return enhanced_model, tokenizer, integration_manager

def process_prompt(model, tokenizer, integration_manager, prompt, args, task_type=None):
    """Process a single prompt and return the generated output."""
    logger.info(f"Processing prompt: {prompt[:50]}...")
    
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with enhanced capabilities
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        prompt=prompt,  # Pass raw prompt for advanced processing
        task_type=task_type,  # Pass task type if known
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True
    )
    
    # Decode output
    output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Get compute stats if available
    stats = None
    if hasattr(integration_manager, "compute_tracker"):
        stats = integration_manager.compute_tracker.get_session_stats()
    
    return output_text, stats

def interactive_mode(model, tokenizer, integration_manager, args):
    """Run the model in interactive mode."""
    print("\n===== Valkyrie Enhanced Reasoning Model =====")
    print("Type 'exit' to quit, 'clear' to clear conversation, or your message to chat.\n")
    
    history = []
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for special commands
            if user_input.lower() == "exit":
                break
            
            if user_input.lower() == "clear":
                history = []
                print("Conversation history cleared.")
                continue
            
            # Process the input
            response, stats = process_prompt(
                model, tokenizer, integration_manager, user_input, args
            )
            
            # Print the response
            print(f"\nValkyrie: {response}")
            
            # Print stats if available
            if stats:
                print("\nGeneration stats:")
                if "total_tokens" in stats:
                    print(f"  Tokens: {stats['total_tokens']}")
                if "total_time" in stats:
                    print(f"  Time: {stats['total_time']:.2f}s")
                if "strategy_usage" in stats and stats["strategy_usage"]:
                    strategies = ", ".join([f"{s}" for s, c in stats["strategy_usage"].items() if c > 0])
                    print(f"  Strategies: {strategies}")
            
            # Update history
            history.append({"user": user_input, "response": response})
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"\nAn error occurred: {e}")
    
    # Save memory if requested
    if args.save_memory and hasattr(integration_manager, "strategy_memory"):
        try:
            logger.info(f"Saving memory state to {args.save_memory}")
            integration_manager.strategy_memory.save_state(args.save_memory)
            print(f"Memory state saved to {args.save_memory}")
        except Exception as e:
            logger.error(f"Failed to save memory state: {e}")
            print(f"Failed to save memory state: {e}")
    
    print("\nThank you for using Valkyrie Enhanced Reasoning Model!")

def batch_mode(model, tokenizer, integration_manager, args):
    """Run the model in batch mode using input file."""
    if not args.input_file:
        logger.error("Input file must be specified for batch mode")
        return
    
    # Read input prompts
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return
    
    logger.info(f"Processing {len(prompts)} prompts in batch mode")
    
    # Open output file
    try:
        with open(args.output_file, 'w', encoding='utf-8') as out_f:
            # Process each prompt
            for i, prompt in enumerate(prompts):
                logger.info(f"Processing prompt {i+1}/{len(prompts)}")
                
                # Generate response
                response, stats = process_prompt(
                    model, tokenizer, integration_manager, prompt, args
                )
                
                # Write to output file
                out_f.write(f"Prompt {i+1}: {prompt}\n\n")
                out_f.write(f"Response: {response}\n\n")
                
                if stats:
                    out_f.write(f"Stats: {stats}\n\n")
                
                out_f.write("-" * 50 + "\n\n")
                
                # Print progress
                print(f"Processed prompt {i+1}/{len(prompts)}")
                
    except Exception as e:
        logger.error(f"Error in batch mode: {e}")
        return
    
    logger.info(f"Batch processing completed. Output written to {args.output_file}")
    print(f"Batch processing completed. Output written to {args.output_file}")
    
    # Save memory if requested
    if args.save_memory and hasattr(integration_manager, "strategy_memory"):
        try:
            logger.info(f"Saving memory state to {args.save_memory}")
            integration_manager.strategy_memory.save_state(args.save_memory)
            print(f"Memory state saved to {args.save_memory}")
        except Exception as e:
            logger.error(f"Failed to save memory state: {e}")

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Display the banner
    print("\n" + "=" * 60)
    print("             Valkyrie Enhanced Reasoning Model             ")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Enabled components: {', '.join(args.enable_components)}")
    if args.enable_all:
        print("All components enabled")
    print("=" * 60 + "\n")
    
    try:
        # Set up the model
        model, tokenizer, integration_manager = setup_model(args)
        
        # Run in appropriate mode
        if args.interactive:
            interactive_mode(model, tokenizer, integration_manager, args)
        else:
            batch_mode(model, tokenizer, integration_manager, args)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 