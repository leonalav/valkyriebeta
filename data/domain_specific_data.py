import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import os
import json
import random
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import re

logger = logging.getLogger(__name__)

@dataclass
class DomainDataConfig:
    """Configuration for domain-specific data handling"""
    # Domains to include
    domains: List[str] = None
    
    # Weighting for each domain (higher = more samples)
    domain_weights: Dict[str, float] = None
    
    # Paths to domain-specific datasets
    domain_data_paths: Dict[str, str] = None
    
    # Whether to use domain-specific tokenization
    use_domain_tokenization: bool = False
    
    # Domain-specific vocabulary augmentation
    augment_vocabulary: bool = True
    domain_vocab_files: Dict[str, str] = None
    
    # Curriculum learning parameters
    use_curriculum: bool = True
    curriculum_difficulty_fn: Optional[Callable] = None
    
    # Data mixing strategy
    mixing_strategy: str = "proportional"  # Options: proportional, equal, curriculum
    
    # Data augmentation for low-resource domains
    augment_low_resource: bool = True
    min_domain_examples: int = 1000
    
    def __post_init__(self):
        if self.domains is None:
            self.domains = ["general", "math", "science", "logic", "coding"]
            
        if self.domain_weights is None:
            self.domain_weights = {
                "general": 1.0,
                "math": 1.5,
                "science": 1.2,
                "logic": 1.3,
                "coding": 1.4
            }
            
        if self.domain_data_paths is None:
            self.domain_data_paths = {
                domain: f"data/{domain}" for domain in self.domains
            }
            
        if self.domain_vocab_files is None:
            self.domain_vocab_files = {
                domain: f"data/{domain}/vocab.json" for domain in self.domains
            }


class DomainSpecificDataset(Dataset):
    """Dataset for domain-specific data"""
    
    def __init__(
        self, 
        domain: str,
        tokenizer,
        data_path: str,
        max_length: int = 1024,
        preprocessing_fn: Optional[Callable] = None
    ):
        super().__init__()
        self.domain = domain
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_length = max_length
        self.preprocessing_fn = preprocessing_fn
        
        self.examples = self._load_data()
        logger.info(f"Loaded {len(self.examples)} examples for domain {domain}")
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from disk"""
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path {self.data_path} does not exist for domain {self.domain}")
            return []
            
        examples = []
        
        # Handle different file formats
        if os.path.isdir(self.data_path):
            # Load all files in directory
            for filename in os.listdir(self.data_path):
                if filename.endswith(".json") or filename.endswith(".jsonl"):
                    file_path = os.path.join(self.data_path, filename)
                    examples.extend(self._load_file(file_path))
        else:
            # Load single file
            examples = self._load_file(self.data_path)
            
        # Apply preprocessing if provided
        if self.preprocessing_fn is not None:
            examples = [self.preprocessing_fn(ex) for ex in examples]
            
        return examples
        
    def _load_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from a single file"""
        examples = []
        
        try:
            if file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        examples = data
                    elif isinstance(data, dict) and "examples" in data:
                        examples = data["examples"]
            elif file_path.endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    examples = [json.loads(line) for line in f if line.strip()]
            else:
                logger.warning(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            
        return examples
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Handle different data formats
        if isinstance(example, dict):
            if "text" in example:
                text = example["text"]
            elif "input" in example and "output" in example:
                text = example["input"] + example.get("output", "")
            elif "prompt" in example and "completion" in example:
                text = example["prompt"] + example["completion"]
            else:
                # Try to find text field with regex
                text_fields = [v for k, v in example.items() 
                              if isinstance(v, str) and len(v) > 20 
                              and re.search(r"[A-Za-z]{10,}", v)]
                text = text_fields[0] if text_fields else str(example)
        else:
            text = str(example)
            
        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Convert to expected format
        item = {
            "input_ids": encodings.input_ids[0],
            "attention_mask": encodings.attention_mask[0],
            "domain": self.domain
        }
        
        # Add labels for autoregressive training
        item["labels"] = item["input_ids"].clone()
        
        return item


class DomainSpecificVocabAugmenter:
    """Augments tokenizer vocabulary with domain-specific tokens"""
    
    def __init__(
        self, 
        tokenizer,
        config: DomainDataConfig
    ):
        self.tokenizer = tokenizer
        self.config = config
        
    def augment_vocabulary(self) -> int:
        """Augment tokenizer vocabulary with domain-specific tokens
        
        Returns:
            Number of tokens added
        """
        if not self.config.augment_vocabulary:
            return 0
            
        total_added = 0
        
        for domain, vocab_file in self.config.domain_vocab_files.items():
            if not os.path.exists(vocab_file):
                logger.warning(f"Vocabulary file {vocab_file} for domain {domain} does not exist")
                continue
                
            try:
                with open(vocab_file, "r", encoding="utf-8") as f:
                    domain_vocab = json.load(f)
                    
                # Extract tokens
                if isinstance(domain_vocab, list):
                    tokens = domain_vocab
                elif isinstance(domain_vocab, dict):
                    tokens = list(domain_vocab.keys())
                else:
                    logger.warning(f"Unexpected vocabulary format in {vocab_file}")
                    continue
                    
                # Add tokens to tokenizer
                num_added = self.tokenizer.add_tokens(tokens)
                total_added += num_added
                logger.info(f"Added {num_added} tokens from domain {domain}")
                
            except Exception as e:
                logger.error(f"Error augmenting vocabulary from {vocab_file}: {e}")
                
        return total_added


class DomainDataManager:
    """Manages domain-specific datasets and creates mixed data loaders"""
    
    def __init__(
        self,
        config: DomainDataConfig,
        tokenizer,
        max_length: int = 1024,
        seed: int = 42
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Augment vocabulary if needed
        if config.augment_vocabulary:
            augmenter = DomainSpecificVocabAugmenter(tokenizer, config)
            num_added = augmenter.augment_vocabulary()
            logger.info(f"Added {num_added} domain-specific tokens to vocabulary")
            
        # Load datasets
        self.domain_datasets = self._load_domain_datasets()
        
    def _load_domain_datasets(self) -> Dict[str, Dataset]:
        """Load datasets for each domain"""
        datasets = {}
        
        for domain in self.config.domains:
            data_path = self.config.domain_data_paths.get(domain)
            if not data_path:
                logger.warning(f"No data path specified for domain {domain}")
                continue
                
            dataset = DomainSpecificDataset(
                domain=domain,
                tokenizer=self.tokenizer,
                data_path=data_path,
                max_length=self.max_length
            )
            
            if len(dataset) > 0:
                datasets[domain] = dataset
                
        return datasets
        
    def _augment_low_resource_domains(self):
        """Augment low-resource domains with synthetic examples"""
        if not self.config.augment_low_resource:
            return
            
        for domain, dataset in self.domain_datasets.items():
            if len(dataset) < self.config.min_domain_examples:
                logger.info(f"Domain {domain} has only {len(dataset)} examples, augmenting...")
                # Implement data augmentation techniques here
                # This could include back-translation, paraphrasing, etc.
                # For now, we'll just duplicate existing examples
                
                # Calculate how many examples to generate
                num_to_generate = self.config.min_domain_examples - len(dataset)
                
                # Simple duplication for now
                # In a real implementation, you would use more sophisticated techniques
                dataset.examples.extend(random.choices(dataset.examples, k=num_to_generate))
                logger.info(f"Augmented domain {domain} to {len(dataset)} examples")
                
    def create_mixed_dataloader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """Create a dataloader that mixes data from all domains
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for data loading
            
        Returns:
            DataLoader with mixed domain data
        """
        # Augment low-resource domains if needed
        self._augment_low_resource_domains()
        
        if self.config.mixing_strategy == "equal":
            return self._create_equal_mixed_dataloader(batch_size, shuffle, num_workers)
        elif self.config.mixing_strategy == "curriculum":
            return self._create_curriculum_dataloader(batch_size, shuffle, num_workers)
        else:  # Default to proportional
            return self._create_proportional_mixed_dataloader(batch_size, shuffle, num_workers)
            
    def _create_proportional_mixed_dataloader(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int
    ) -> DataLoader:
        """Create a dataloader with proportional mixing based on domain weights"""
        # Calculate weighted dataset sizes
        total_examples = sum(len(dataset) for dataset in self.domain_datasets.values())
        
        # Create weighted concatenated dataset
        weighted_datasets = []
        
        for domain, dataset in self.domain_datasets.items():
            weight = self.config.domain_weights.get(domain, 1.0)
            
            # Calculate how many examples to include based on weight
            # We use sampling with replacement to handle weights > 1.0
            num_samples = int(len(dataset) * weight)
            
            if num_samples > len(dataset):
                # Need to oversample
                indices = np.random.choice(len(dataset), num_samples, replace=True)
                weighted_dataset = Subset(dataset, indices)
            else:
                # Can use a subset
                indices = np.random.choice(len(dataset), num_samples, replace=False)
                weighted_dataset = Subset(dataset, indices)
                
            weighted_datasets.append(weighted_dataset)
            
        # Concatenate all datasets
        mixed_dataset = ConcatDataset(weighted_datasets)
        
        # Create dataloader
        return DataLoader(
            mixed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
    def _create_equal_mixed_dataloader(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int
    ) -> DataLoader:
        """Create a dataloader with equal representation from each domain"""
        # Find minimum dataset size
        min_size = min(len(dataset) for dataset in self.domain_datasets.values())
        
        # Create equal-sized subsets
        equal_datasets = []
        
        for domain, dataset in self.domain_datasets.items():
            if len(dataset) > min_size:
                # Randomly sample min_size examples
                indices = np.random.choice(len(dataset), min_size, replace=False)
                equal_dataset = Subset(dataset, indices)
            else:
                # Use the entire dataset
                equal_dataset = dataset
                
            equal_datasets.append(equal_dataset)
            
        # Concatenate all datasets
        mixed_dataset = ConcatDataset(equal_datasets)
        
        # Create dataloader
        return DataLoader(
            mixed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
    def _create_curriculum_dataloader(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int
    ) -> DataLoader:
        """Create a curriculum dataloader that orders examples by difficulty"""
        if not self.config.use_curriculum:
            logger.warning("Curriculum requested but use_curriculum is False, using proportional mixing")
            return self._create_proportional_mixed_dataloader(batch_size, shuffle, num_workers)
            
        if self.config.curriculum_difficulty_fn is None:
            logger.warning("No curriculum difficulty function provided, using default")
            difficulty_fn = lambda x: len(x["input_ids"])  # Simple length-based difficulty
        else:
            difficulty_fn = self.config.curriculum_difficulty_fn
            
        # Collect all examples and compute difficulty
        all_examples = []
        
        for domain, dataset in self.domain_datasets.items():
            for i in range(len(dataset)):
                example = dataset[i]
                difficulty = difficulty_fn(example)
                all_examples.append((example, difficulty, i, domain))
                
        # Sort by difficulty
        all_examples.sort(key=lambda x: x[1])
        
        # Create custom dataset that returns examples in curriculum order
        class CurriculumDataset(Dataset):
            def __init__(self, examples):
                self.examples = examples
                
            def __len__(self):
                return len(self.examples)
                
            def __getitem__(self, idx):
                return self.examples[idx][0]  # Return just the example
                
        curriculum_dataset = CurriculumDataset([ex[0] for ex in all_examples])
        
        # Create dataloader (note: shuffle should be False for curriculum)
        return DataLoader(
            curriculum_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle curriculum
            num_workers=num_workers
        )
        
    def get_domain_specific_dataloader(
        self,
        domain: str,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> Optional[DataLoader]:
        """Get a dataloader for a specific domain
        
        Args:
            domain: Domain name
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for data loading
            
        Returns:
            DataLoader for the specified domain, or None if domain not found
        """
        if domain not in self.domain_datasets:
            logger.warning(f"Domain {domain} not found in available datasets")
            return None
            
        return DataLoader(
            self.domain_datasets[domain],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        ) 