#!/usr/bin/env python
"""
Domain Data Bridge

This module connects existing data loaders from collect_data.py and additionalcollect.py
with the enhanced training pipeline's domain-specific data management.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

import numpy as np
import torch
from tqdm import tqdm

# Import existing data loaders
try:
    from data.collect_data import (
        load_math, load_numinamath, load_olympic_arena, load_theoremqa,
        load_scieval, load_olympiad_bench, load_jeebench, load_agieval,
        load_statsqual, load_gpqa_extended, load_xword, load_usaco,
        load_quant, load_livecodebench, select_examples_omni_math,
        select_examples_scieval, decontaminate_train_data
    )
except ImportError:
    logging.warning("Could not import all functions from data.collect_data. Some loaders may not be available.")

try:
    from data.additionalcollect import load_curated_thoughts, load_wildchat
except ImportError:
    logging.warning("Could not import functions from data.additionalcollect. Some loaders may not be available.")

# Import the domain-specific data manager and config
from data.domain_specific_data import DomainDataConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Domain mapping for categorization
DOMAIN_MAPPINGS = {
    "math": [
        "load_math", "load_numinamath", "load_olympic_arena", "load_theoremqa",
        "load_olympiad_bench", "load_jeebench", "load_statsqual"
    ],
    "science": ["load_scieval", "load_agieval"],
    "coding": ["load_livecodebench", "load_usaco"],
    "logic": ["load_quant", "load_gpqa_extended"],
    "general": ["load_wildchat"],
    "reasoning": ["load_curated_thoughts", "load_xword"]
}

# Reverse mapping to find domain from loader name
LOADER_TO_DOMAIN = {}
for domain, loaders in DOMAIN_MAPPINGS.items():
    for loader in loaders:
        LOADER_TO_DOMAIN[loader] = domain

# Loader function mappings
LOADER_FUNCTIONS = {
    "load_math": load_math if 'load_math' in globals() else None,
    "load_numinamath": load_numinamath if 'load_numinamath' in globals() else None,
    "load_olympic_arena": load_olympic_arena if 'load_olympic_arena' in globals() else None,
    "load_theoremqa": load_theoremqa if 'load_theoremqa' in globals() else None,
    "load_scieval": load_scieval if 'load_scieval' in globals() else None,
    "load_olympiad_bench": load_olympiad_bench if 'load_olympiad_bench' in globals() else None,
    "load_jeebench": load_jeebench if 'load_jeebench' in globals() else None,
    "load_agieval": load_agieval if 'load_agieval' in globals() else None,
    "load_statsqual": load_statsqual if 'load_statsqual' in globals() else None,
    "load_gpqa_extended": load_gpqa_extended if 'load_gpqa_extended' in globals() else None,
    "load_xword": load_xword if 'load_xword' in globals() else None,
    "load_usaco": load_usaco if 'load_usaco' in globals() else None,
    "load_quant": load_quant if 'load_quant' in globals() else None,
    "load_livecodebench": load_livecodebench if 'load_livecodebench' in globals() else None,
    "load_curated_thoughts": load_curated_thoughts if 'load_curated_thoughts' in globals() else None,
    "load_wildchat": load_wildchat if 'load_wildchat' in globals() else None,
}

def get_available_loaders() -> Dict[str, List[str]]:
    """
    Get a dictionary of available loaders grouped by domain
    
    Returns:
        Dict[str, List[str]]: A dictionary with domains as keys and lists of available loader names as values
    """
    available_loaders = {}
    
    for domain, loaders in DOMAIN_MAPPINGS.items():
        available_loaders[domain] = []
        for loader in loaders:
            if LOADER_FUNCTIONS.get(loader) is not None:
                available_loaders[domain].append(loader)
                
    return available_loaders

def format_example_for_domain(example: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """
    Format an example from the existing loaders for the domain-specific data manager
    
    Args:
        example: The example from the loader
        domain: The domain this example belongs to
        
    Returns:
        Dict[str, Any]: A formatted example for the domain-specific data manager
    """
    formatted = {}
    
    # Common fields
    formatted["domain"] = domain
    
    # Get text from relevant fields depending on the domain
    if "question" in example and "solution" in example:
        # For math, science, logic domains with question-solution format
        formatted["text"] = f"Question: {example['question']}\n\nSolution: {example['solution']}"
    elif "input" in example and "output" in example:
        # For coding domains with input-output format
        formatted["text"] = f"Input:\n{example['input']}\n\nOutput:\n{example['output']}"
    elif "prompt" in example and "response" in example:
        # For conversational domains
        formatted["text"] = f"User: {example['prompt']}\n\nAssistant: {example['response']}"
    elif "text" in example:
        # Direct text field
        formatted["text"] = example["text"]
    else:
        # Fallback: serialize the whole example
        formatted["text"] = json.dumps(example)
    
    # Include metadata if available
    if "metadata" in example:
        formatted["metadata"] = example["metadata"]
    else:
        formatted["metadata"] = {}
    
    # Store original example fields
    for key, value in example.items():
        if key not in formatted:
            formatted[f"original_{key}"] = value
    
    return formatted

def split_dataset(
    examples: List[Dict[str, Any]], 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split a dataset into train, validation, and test sets
    
    Args:
        examples: List of examples to split
        train_ratio: Ratio of examples to use for training
        val_ratio: Ratio of examples to use for validation
        test_ratio: Ratio of examples to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary with "train", "validation", and "test" splits
    """
    np.random.seed(seed)
    
    # Check ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        logger.warning(f"Split ratios sum to {total_ratio}, not 1.0. Normalizing...")
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        test_ratio = test_ratio / total_ratio
    
    # Shuffle examples
    indices = np.random.permutation(len(examples))
    
    # Calculate split indices
    train_idx = int(len(examples) * train_ratio)
    val_idx = train_idx + int(len(examples) * val_ratio)
    
    # Split the dataset
    train_examples = [examples[i] for i in indices[:train_idx]]
    val_examples = [examples[i] for i in indices[train_idx:val_idx]]
    test_examples = [examples[i] for i in indices[val_idx:]]
    
    return {
        "train": train_examples,
        "validation": val_examples,
        "test": test_examples
    }

def save_jsonl(examples: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save examples to a JSONL file
    
    Args:
        examples: List of examples to save
        file_path: Path to save the JSONL file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
            
    logger.info(f"Saved {len(examples)} examples to {file_path}")

def extract_domain_vocab(
    examples: List[Dict[str, Any]], 
    vocab_size: int = 1000, 
    min_freq: int = 2
) -> List[str]:
    """
    Extract domain-specific vocabulary from examples
    
    Args:
        examples: List of examples to extract vocabulary from
        vocab_size: Maximum size of the vocabulary
        min_freq: Minimum frequency for a token to be included
        
    Returns:
        List[str]: List of domain-specific vocabulary tokens
    """
    from collections import Counter
    import re
    
    # Extract all text
    all_text = " ".join([example.get("text", "") for example in examples])
    
    # Simple tokenization by splitting on whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', all_text.lower())
    
    # Count token frequencies
    token_counts = Counter(tokens)
    
    # Filter by minimum frequency
    token_counts = {token: count for token, count in token_counts.items() if count >= min_freq}
    
    # Sort by frequency (descending)
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Return top tokens up to vocab_size
    return [token for token, _ in sorted_tokens[:vocab_size]]

def process_domain_data(
    loader_name: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_examples: Optional[int] = None,
    extract_vocab: bool = True,
    vocab_size: int = 1000,
    seed: int = 42,
    loader_kwargs: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process data from a loader for a specific domain
    
    Args:
        loader_name: Name of the loader function to use
        output_dir: Directory to save processed data
        train_ratio: Ratio of examples to use for training
        val_ratio: Ratio of examples to use for validation
        test_ratio: Ratio of examples to use for testing
        max_examples: Maximum number of examples to use (None for all)
        extract_vocab: Whether to extract domain-specific vocabulary
        vocab_size: Maximum size of the vocabulary
        seed: Random seed for reproducibility
        loader_kwargs: Additional arguments to pass to the loader function
        
    Returns:
        Dict[str, Any]: Statistics about the processed data
    """
    # Get domain from loader name
    domain = LOADER_TO_DOMAIN.get(loader_name, "general")
    
    # Get loader function
    loader_fn = LOADER_FUNCTIONS.get(loader_name)
    if loader_fn is None:
        raise ValueError(f"Loader function {loader_name} not available")
    
    logger.info(f"Loading data using {loader_name} for domain '{domain}'")
    
    # Call loader function with kwargs if provided
    kwargs = loader_kwargs or {}
    examples = loader_fn(**kwargs)
    
    if not examples:
        logger.warning(f"No examples returned by {loader_name}")
        return {"total_examples": 0}
    
    logger.info(f"Loaded {len(examples)} examples from {loader_name}")
    
    # Limit number of examples if specified
    if max_examples is not None and len(examples) > max_examples:
        logger.info(f"Limiting to {max_examples} examples")
        np.random.seed(seed)
        indices = np.random.choice(len(examples), max_examples, replace=False)
        examples = [examples[i] for i in indices]
    
    # Format examples for domain-specific data manager
    formatted_examples = []
    for example in tqdm(examples, desc=f"Formatting {domain} examples"):
        formatted = format_example_for_domain(example, domain)
        formatted_examples.append(formatted)
    
    # Split into train, validation, and test sets
    splits = split_dataset(
        formatted_examples, 
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Create output directory for this domain
    domain_dir = os.path.join(output_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)
    
    # Save splits to JSONL files
    for split_name, split_examples in splits.items():
        save_jsonl(
            split_examples,
            os.path.join(domain_dir, f"{split_name}.jsonl")
        )
    
    # Extract and save domain-specific vocabulary if requested
    vocab = []
    if extract_vocab:
        vocab = extract_domain_vocab(
            formatted_examples,
            vocab_size=vocab_size,
            min_freq=2
        )
        
        vocab_path = os.path.join(domain_dir, "vocab.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2)
            
        logger.info(f"Saved {len(vocab)} vocabulary items to {vocab_path}")
    
    # Return statistics
    stats = {
        "total_examples": len(formatted_examples),
        "train_examples": len(splits["train"]),
        "validation_examples": len(splits["validation"]),
        "test_examples": len(splits["test"]),
        "vocabulary_size": len(vocab)
    }
    
    return stats

def create_domain_config(
    output_dir: str,
    domains: List[str],
    domain_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Create a domain configuration file for the DomainDataManager
    
    Args:
        output_dir: Directory where domain data is stored
        domains: List of domains to include in the configuration
        domain_weights: Optional dictionary of domain weights (defaults to equal weights)
        
    Returns:
        Dict[str, Any]: Domain configuration dictionary
    """
    # Default to equal weights if not provided
    if domain_weights is None:
        domain_weights = {domain: 1.0 for domain in domains}
    else:
        # Ensure all domains have weights
        for domain in domains:
            if domain not in domain_weights:
                domain_weights[domain] = 1.0
    
    # Create configuration
    config = {
        "domains": domains,
        "domain_weights": domain_weights,
        "data_paths": {domain: os.path.join(output_dir, domain) for domain in domains},
        "vocab_files": {domain: os.path.join(output_dir, domain, "vocab.json") for domain in domains},
        "use_domain_specific_vocab": True,
        "curriculum_strategy": None  # Can be set to 'linear' or 'exponential'
    }
    
    # Save configuration
    config_path = os.path.join(output_dir, "domain_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
        
    logger.info(f"Saved domain configuration to {config_path}")
    
    return config

def prepare_all_domains(
    config_file: Optional[str] = None,
    output_dir: str = "data",
    loaders: Optional[Dict[str, Dict[str, Any]]] = None,
    domain_weights: Optional[Dict[str, float]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_examples_per_domain: Optional[int] = None,
    extract_vocab: bool = True,
    vocab_size: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Prepare all domain-specific datasets using existing loaders
    
    Args:
        config_file: Optional path to configuration file
        output_dir: Directory to save processed data
        loaders: Dictionary mapping loader names to their kwargs
        domain_weights: Optional dictionary of domain weights
        train_ratio: Ratio of examples to use for training
        val_ratio: Ratio of examples to use for validation
        test_ratio: Ratio of examples to use for testing
        max_examples_per_domain: Maximum number of examples per domain
        extract_vocab: Whether to extract domain-specific vocabulary
        vocab_size: Maximum size of the vocabulary
        seed: Random seed for reproducibility
        
    Returns:
        Dict[str, Any]: Statistics about the prepared datasets
    """
    # Load configuration from file if provided
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        loaders = config.get("loaders", {})
        domain_weights = config.get("domain_weights", {})
        train_ratio = config.get("train_ratio", train_ratio)
        val_ratio = config.get("val_ratio", val_ratio)
        test_ratio = config.get("test_ratio", test_ratio)
        max_examples_per_domain = config.get("max_examples_per_domain", max_examples_per_domain)
        extract_vocab = config.get("extract_vocab", extract_vocab)
        vocab_size = config.get("vocab_size", vocab_size)
        seed = config.get("seed", seed)
    
    # Use all available loaders if none specified
    if loaders is None:
        loaders = {}
        for domain, loader_names in get_available_loaders().items():
            for loader_name in loader_names:
                loaders[loader_name] = {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each loader
    stats = {"domains": {}}
    processed_domains = set()
    
    for loader_name, loader_kwargs in loaders.items():
        # Get domain from loader name
        domain = LOADER_TO_DOMAIN.get(loader_name, "general")
        
        try:
            # Process domain data
            domain_stats = process_domain_data(
                loader_name=loader_name,
                output_dir=output_dir,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                max_examples=max_examples_per_domain,
                extract_vocab=extract_vocab,
                vocab_size=vocab_size,
                seed=seed,
                loader_kwargs=loader_kwargs
            )
            
            # Store statistics
            if domain not in stats["domains"]:
                stats["domains"][domain] = domain_stats
            else:
                # Update existing domain stats
                for key, value in domain_stats.items():
                    if key in stats["domains"][domain]:
                        if isinstance(value, int):
                            stats["domains"][domain][key] += value
                    else:
                        stats["domains"][domain][key] = value
            
            processed_domains.add(domain)
            
        except Exception as e:
            logger.error(f"Error processing {loader_name}: {str(e)}")
    
    # Create domain configuration
    domains = list(processed_domains)
    config = create_domain_config(
        output_dir=output_dir,
        domains=domains,
        domain_weights=domain_weights
    )
    
    # Save metadata
    metadata = {
        "domains": stats["domains"],
        "config": config,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "max_examples_per_domain": max_examples_per_domain,
        "vocab_size": vocab_size,
        "seed": seed
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Log summary
    logger.info("Domain dataset preparation complete:")
    for domain, domain_stats in stats["domains"].items():
        logger.info(f"  {domain}: {domain_stats['total_examples']} examples")
    
    return stats

def load_domain_data_for_training(data_dir: str, tokenizer: Any, max_length: int = 1024, seed: int = 42) -> Any:
    """
    Load domain-specific data for training using DomainDataManager
    
    Args:
        data_dir: Directory containing prepared domain data
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
        seed: Random seed for reproducibility
        
    Returns:
        DomainDataManager: Domain data manager for training
    """
    from data.domain_specific_data import DomainDataManager
    
    # Load domain configuration
    config_path = os.path.join(data_dir, "domain_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Domain configuration not found at {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
        
    # Create domain configuration
    domain_config = DomainDataConfig(**config_dict)
    
    # Create domain data manager
    domain_manager = DomainDataManager(
        config=domain_config,
        tokenizer=tokenizer,
        max_length=max_length,
        seed=seed
    )
    
    return domain_manager

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare domain-specific datasets using existing loaders")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save processed data")
    parser.add_argument("--list_loaders", action="store_true", help="List available loaders and exit")
    args = parser.parse_args()
    
    if args.list_loaders:
        print("Available loaders by domain:")
        for domain, loaders in get_available_loaders().items():
            print(f"  {domain}:")
            for loader in loaders:
                print(f"    - {loader}")
        exit()
    
    prepare_all_domains(
        config_file=args.config,
        output_dir=args.output_dir
    ) 