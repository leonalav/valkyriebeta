#!/usr/bin/env python3
"""
Synthetic Data Generator for RLHF Training

This module provides functionality to generate synthetic data for training
RLHF models with enhanced reasoning capabilities.
"""

import os
import json
import logging
import random
import re
import math
import time
from typing import List, Dict, Tuple, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
from tqdm import tqdm
import hashlib
from pathlib import Path

try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError(
        "Transformers library is required to use the synthetic data generator. "
        "Please install it with 'pip install transformers'."
    )

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    # General settings
    num_samples: int = 1000
    output_dir: str = "synthetic_data"
    seed: int = 42
    cache_data: bool = True
    
    # Model settings
    generator_model_name: Optional[str] = None
    max_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # Component weights
    components: List[str] = field(default_factory=lambda: ["math", "logical", "causal", "nlu", "constitutional"])
    component_weights: Dict[str, float] = field(default_factory=lambda: {
        "math": 1.0,
        "logical": 1.0,
        "causal": 1.0,
        "nlu": 1.0,
        "constitutional": 1.0
    })
    
    # Data generation settings
    use_multi_step_reasoning: bool = True
    reasoning_depth: int = 3  # Number of reasoning steps
    include_explanations: bool = True
    create_preference_pairs: bool = True
    preference_gap_minimum: float = 0.2  # Minimum gap in quality for preference pairs
    
    # Diversity settings
    diversity_penalty: float = 1.5
    use_diverse_topics: bool = True
    topic_coverage_minimum: int = 5  # Minimum number of topics to cover
    
    # Filtering settings
    filter_low_quality: bool = True
    filter_nsfw: bool = True
    quality_threshold: float = 0.6  # Threshold for filtering low-quality examples

    # Format settings
    include_metadata: bool = True
    format: str = "jsonl"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to a file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Synthetic data configuration saved to {path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SyntheticDataConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'SyntheticDataConfig':
        """Load configuration from a file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        logger.info(f"Synthetic data configuration loaded from {path}")
        return cls.from_dict(config_dict)

# Template pools for different reasoning components
MATH_TEMPLATES = [
    "Solve the following {complexity} math problem step by step: {problem}",
    "Calculate {expression} and show all your work.",
    "Find the {property} of {object} given that {conditions}.",
    "Solve for {variable} in the equation: {equation}",
    "A {scenario} involves {entities}. {question}",
    "What is the value of {expression} when {conditions}?",
    "Compute the {operation} of {values} and explain each step.",
    "In a {scenario}, if {conditions}, what is {question}?",
    "If {initial_conditions}, and {action_happens}, what is {quantity}?",
    "Given a {object} with {properties}, determine {target}."
]

LOGICAL_TEMPLATES = [
    "Evaluate whether the following argument is valid: {argument}",
    "Identify the logical fallacy in the following statement: {statement}",
    "Given the premises: {premises}, is the conclusion '{conclusion}' valid?",
    "Which of the following statements must be true based on: {premises}?",
    "If {hypothesis}, then {consequence}. {condition} occurred. What can we conclude?",
    "All {category_a} are {category_b}. {instance} is {category}. What follows?",
    "Analyze the following argument for logical consistency: {argument}",
    "If we know that {fact_1} and {fact_2}, can we conclude {conclusion}?",
    "Given these facts: {facts}, what is the most logical inference about {topic}?",
    "Consider this syllogism: {syllogism}. Is it valid? Why or why not?"
]

CAUSAL_TEMPLATES = [
    "What caused {event} in the following scenario: {scenario}",
    "If {intervention} had not occurred, what would have happened to {outcome}?",
    "In the scenario: {scenario}, what is the causal relationship between {variable_a} and {variable_b}?",
    "How does {factor} influence {outcome} in the following situation: {situation}?",
    "Given that {event_a} happened, which of the following could be a cause: {options}?",
    "Explain the causal chain connecting {initial_event} to {final_outcome}.",
    "What would be the effect on {variable} if we intervene to change {intervention}?",
    "If we observe {observation}, what can we infer about the cause of {phenomenon}?",
    "Consider the counterfactual: If {alternative} had happened instead, would {outcome} still occur?",
    "Identify the mediating factors between {cause} and {effect} in this case: {case}"
]

NLU_TEMPLATES = [
    "What does the author imply in the following passage: {passage}",
    "Identify the main message in this text: {text}",
    "What is the sentiment expressed in this statement: {statement}?",
    "In the dialogue: {dialogue}, what does {speaker} really mean when saying '{utterance}'?",
    "Given the context: {context}, what is the meaning of '{phrase}'?",
    "What is the tone of the following message: {message}?",
    "Identify the rhetorical devices used in: {text}",
    "Resolve the ambiguity in the sentence: {sentence}",
    "What assumptions underlie the following argument: {argument}?",
    "In this narrative: {narrative}, what is implied but not explicitly stated about {topic}?"
]

CONSTITUTIONAL_TEMPLATES = [
    "How can this response be improved to be more helpful, harmless, and honest: {response}",
    "Given this request: {request}, provide a response that is both helpful and ethical.",
    "Evaluate the following response for factual accuracy: {response}",
    "Rewrite this response to remove harmful content while preserving helpful information: {response}",
    "How could this answer be made more balanced and objective: {answer}",
    "Identify any unsafe instructions in the following request: {request}",
    "Reframe this controversial question to be more balanced: {question}",
    "What ethical considerations should be taken into account when responding to: {query}?",
    "Improve this response to be more transparent about its limitations: {response}",
    "How can this explanation be made more accessible while remaining accurate: {explanation}?"
]

# Data pools for filling templates
MATH_ENTITIES = {
    "operations": ["addition", "subtraction", "multiplication", "division", "exponentiation", "logarithm", "integration", "differentiation"],
    "complexity": ["elementary", "basic", "intermediate", "advanced", "complex", "graduate-level"],
    "scenarios": ["bank transaction", "shopping trip", "investment", "mortgage", "recipe", "sports game", "construction project", "road trip"],
    "objects": ["triangle", "circle", "rectangle", "cylinder", "polynomial", "function", "vector", "matrix"],
    "properties": ["area", "perimeter", "volume", "derivative", "integral", "eigenvalue", "determinant", "root"],
    "expressions": ["3x^2 + 2x - 5", "ln(x+1) - ln(x-1)", "cos(2x) + sin(x)", "(x^2 + y^2) / (x - y)"]
}

LOGICAL_ENTITIES = {
    "fallacies": ["ad hominem", "straw man", "appeal to authority", "circular reasoning", "false dichotomy", "slippery slope", "hasty generalization"],
    "categories": ["mammals", "birds", "vehicles", "publications", "emotions", "theories", "inventions", "laws"],
    "syllogism_formats": ["All A are B. C is A. Therefore C is B.", "No A are B. Some C are A. Therefore some C are not B."],
    "conditions": ["raining", "sunny", "cold", "hot", "expensive", "affordable", "legal", "illegal", "proven", "disproven"]
}

CAUSAL_ENTITIES = {
    "events": ["economic recession", "climate change", "technological innovation", "policy implementation", "market crash", "disease outbreak"],
    "factors": ["government regulation", "public opinion", "technological advancement", "resource availability", "economic incentives"],
    "interventions": ["tax cut", "interest rate change", "new law", "public campaign", "product launch", "scientific discovery"],
    "mediators": ["public perception", "media coverage", "market competition", "resource allocation", "political support"]
}

NLU_ENTITIES = {
    "tones": ["sarcastic", "sincere", "ironic", "humorous", "formal", "informal", "persuasive", "informative"],
    "rhetorical_devices": ["metaphor", "simile", "hyperbole", "understatement", "rhetorical question", "personification"],
    "ambiguities": ["lexical ambiguity", "syntactic ambiguity", "semantic ambiguity", "pragmatic ambiguity"],
    "implications": ["suggestion", "presupposition", "entailment", "conventional implicature", "conversational implicature"]
}

CONSTITUTIONAL_ENTITIES = {
    "ethical_principles": ["autonomy", "beneficence", "non-maleficence", "justice", "transparency", "privacy"],
    "content_concerns": ["misinformation", "bias", "stereotyping", "harmful advice", "privacy violation", "manipulation"],
    "improvements": ["add nuance", "provide evidence", "acknowledge uncertainty", "consider diverse perspectives", "add disclaimers"]
}

class SyntheticDataGenerator:
    """Generator for synthetic reasoning data."""
    
    def __init__(
        self,
        config: SyntheticDataConfig,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None
    ):
        self.config = config
        self.set_seed(config.seed)
        
        # Initialize model and tokenizer if not provided
        if model is None or tokenizer is None:
            if config.generator_model_name:
                logger.info(f"Loading model and tokenizer: {config.generator_model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(config.generator_model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    config.generator_model_name, 
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = self.model.to(device)
            else:
                logger.warning("No generator model provided or specified. Template-based generation only.")
                self.model = None
                self.tokenizer = None
        else:
            self.model = model
            self.tokenizer = tokenizer
        
        # Set up component templates
        self.templates = {
            "math": MATH_TEMPLATES,
            "logical": LOGICAL_TEMPLATES,
            "causal": CAUSAL_TEMPLATES,
            "nlu": NLU_TEMPLATES,
            "constitutional": CONSTITUTIONAL_TEMPLATES
        }
        
        self.entities = {
            "math": MATH_ENTITIES,
            "logical": LOGICAL_ENTITIES,
            "causal": CAUSAL_ENTITIES,
            "nlu": NLU_ENTITIES,
            "constitutional": CONSTITUTIONAL_ENTITIES
        }
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Cache for generated data
        self.data_cache = {}
    
    def set_seed(self, seed: int) -> None:
        """Set seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _fill_template(self, template: str, component: str) -> str:
        """Fill a template with appropriate entities for a component."""
        # Extract placeholders using regex
        placeholders = re.findall(r'\{([^}]+)\}', template)
        filled_template = template
        
        # Track used values to avoid repetition
        used_values = set()
        
        for placeholder in placeholders:
            # Check if placeholder is in component entities
            if placeholder in self.entities[component]:
                # Get possible values
                values = self.entities[component][placeholder]
                
                # Filter out already used values
                available_values = [v for v in values if v not in used_values]
                
                # If no available values, reset and use all values
                if not available_values:
                    available_values = values
                
                # Choose a random value
                value = random.choice(available_values)
                used_values.add(value)
                
                # Replace in template
                filled_template = filled_template.replace(f"{{{placeholder}}}", value)
            else:
                # For placeholders not in our entity lists, use a placeholder value
                # This makes templates more flexible but might require model completion
                filled_template = filled_template.replace(f"{{{placeholder}}}", f"[{placeholder.upper()}]")
        
        return filled_template
    
    def generate_prompt(self, component: str) -> str:
        """Generate a prompt for a specific reasoning component."""
        if component not in self.templates:
            raise ValueError(f"Unknown component: {component}. Available components: {list(self.templates.keys())}")
        
        # Select a random template
        template = random.choice(self.templates[component])
        
        # Fill the template
        prompt = self._fill_template(template, component)
        
        return prompt
    
    def generate_completion(self, prompt: str) -> str:
        """Generate a completion for a prompt using the model."""
        if self.model is None or self.tokenizer is None:
            # If no model, return placeholder
            return "[MODEL COMPLETION WOULD GO HERE]"
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode and clean up
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = completion[len(prompt):].strip()  # Remove the prompt from the completion
        
        return completion
    
    def _create_chosen_rejected_pair(self, prompt: str) -> Tuple[str, str]:
        """Generate a chosen (better) and rejected (worse) completion pair for a prompt."""
        # Generate a high-quality (chosen) completion
        chosen = self.generate_completion(prompt)
        
        # Generate a lower-quality (rejected) completion
        # For simplicity, we'll generate another and assume it's different quality
        rejected = self.generate_completion(prompt)
        
        # Ensure they're actually different
        attempts = 0
        while chosen == rejected and attempts < 3:
            rejected = self.generate_completion(prompt)
            attempts += 1
        
        return chosen, rejected
    
    def generate_preference_pair(self, component: str) -> Dict[str, Any]:
        """Generate a preference pair for a specific component."""
        prompt = self.generate_prompt(component)
        chosen, rejected = self._create_chosen_rejected_pair(prompt)
        
        return {
            "component": component,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "generation_timestamp": time.time(),
                "component_type": component,
                "model": self.config.generator_model_name if self.config.generator_model_name else "template-based"
            }
        }
    
    def generate_dataset(self) -> List[Dict[str, Any]]:
        """Generate a full synthetic dataset based on the configuration."""
        logger.info(f"Generating {self.config.num_samples} synthetic examples...")
        
        # Check if cache exists and should be used
        cache_path = os.path.join(self.config.output_dir, f"synthetic_cache_{self.config.seed}.json")
        if self.config.cache_data and os.path.exists(cache_path):
            logger.info(f"Loading cached data from {cache_path}")
            with open(cache_path, "r") as f:
                return json.load(f)
        
        # Normalize component weights
        total_weight = sum(self.config.component_weights.values())
        normalized_weights = {k: v / total_weight for k, v in self.config.component_weights.items()}
        
        # Only use components that are in the config
        active_components = [c for c in self.config.components if c in self.templates]
        component_probs = [normalized_weights.get(c, 0) for c in active_components]
        
        # Generate examples
        examples = []
        for _ in tqdm(range(self.config.num_samples)):
            # Choose a component based on weights
            component = random.choices(active_components, weights=component_probs, k=1)[0]
            
            # Generate example
            if self.config.create_preference_pairs:
                example = self.generate_preference_pair(component)
            else:
                prompt = self.generate_prompt(component)
                completion = self.generate_completion(prompt)
                example = {
                    "component": component,
                    "prompt": prompt,
                    "completion": completion,
                    "metadata": {
                        "generation_timestamp": time.time(),
                        "component_type": component,
                        "model": self.config.generator_model_name if self.config.generator_model_name else "template-based"
                    }
                }
            
            examples.append(example)
        
        # Cache the generated data
        if self.config.cache_data:
            with open(cache_path, "w") as f:
                json.dump(examples, f, indent=2)
            logger.info(f"Cached generated data to {cache_path}")
        
        return examples
    
    def save_dataset(self, examples: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """Save the dataset to disk."""
        if filename is None:
            filename = f"synthetic_data_{int(time.time())}.{self.config.format}"
        
        output_path = os.path.join(self.config.output_dir, filename)
        
        if self.config.format == "jsonl":
            with open(output_path, "w") as f:
                for example in examples:
                    f.write(json.dumps(example) + "\n")
        else:  # Default to json
            with open(output_path, "w") as f:
                json.dump(examples, f, indent=2)
        
        logger.info(f"Saved {len(examples)} examples to {output_path}")
        return output_path
    
    def generate_and_save(self) -> str:
        """Generate and save a dataset in one step."""
        examples = self.generate_dataset()
        return self.save_dataset(examples)

def create_synthetic_dataset(
    config: Optional[Union[Dict[str, Any], SyntheticDataConfig]] = None,
    model_path: Optional[str] = None,
    num_samples: int = 1000,
    components: Optional[List[str]] = None,
    output_dir: str = "synthetic_data"
) -> str:
    """
    Convenience function to create a synthetic dataset with minimal configuration.
    
    Args:
        config: Optional configuration dictionary or SyntheticDataConfig
        model_path: Path to the model for generation
        num_samples: Number of samples to generate
        components: List of reasoning components to generate examples for
        output_dir: Directory to save the generated data
    
    Returns:
        Path to the saved dataset
    """
    if config is None:
        config = SyntheticDataConfig()
        
        # Update with provided args
        config.num_samples = num_samples
        config.output_dir = output_dir
        
        if model_path:
            config.generator_model_name = model_path
        
        if components:
            config.components = components
    elif isinstance(config, dict):
        config = SyntheticDataConfig.from_dict(config)
    
    generator = SyntheticDataGenerator(config)
    return generator.generate_and_save()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic data for RLHF training")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--model_path", type=str, help="Path to the model for generation")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--components", type=str, nargs="+", default=["math", "logical", "causal", "nlu", "constitutional"], 
                        help="Reasoning components to generate examples for")
    parser.add_argument("--output_dir", type=str, default="synthetic_data", help="Directory to save the generated data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--format", type=str, default="jsonl", choices=["json", "jsonl"], help="Output format")
    
    args = parser.parse_args()
    
    if args.config:
        config = SyntheticDataConfig.from_file(args.config)
    else:
        config = SyntheticDataConfig(
            generator_model_name=args.model_path,
            num_samples=args.num_samples,
            components=args.components,
            output_dir=args.output_dir,
            seed=args.seed,
            format=args.format
        )
    
    generator = SyntheticDataGenerator(config)
    output_path = generator.generate_and_save()
    print(f"Generated data saved to: {output_path}") 