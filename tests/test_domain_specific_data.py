import pytest
import torch
import sys
import os
from pathlib import Path
import json
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.domain_specific_data import (
    DomainDataConfig, 
    DomainSpecificDataset, 
    DomainSpecificVocabAugmenter,
    DomainDataManager
)

class SimpleTokenizer:
    """Simple tokenizer for testing"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        
    def __call__(self, text, max_length=None, padding=None, truncation=None, return_tensors=None):
        """Tokenize text"""
        # This is a very simple tokenizer that just assigns random token IDs
        if isinstance(text, list):
            batch_size = len(text)
        else:
            batch_size = 1
            text = [text]
            
        if max_length is None:
            max_length = 10
            
        # Generate random token IDs
        input_ids = torch.randint(0, self.vocab_size, (batch_size, max_length))
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Return as tensors if requested
        if return_tensors == "pt":
            return SimpleEncodings(input_ids, attention_mask)
        else:
            return {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist()}
    
    def add_tokens(self, tokens):
        """Add tokens to vocabulary"""
        # Just return the number of tokens that were added
        return len(tokens)
        
class SimpleEncodings:
    """Simple encodings container for testing"""
    
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with domain-specific data for testing"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create domain directories
    domains = ["general", "math", "science"]
    for domain in domains:
        domain_dir = os.path.join(temp_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        
        # Create sample data file
        data_file = os.path.join(domain_dir, "data.json")
        examples = [
            {"text": f"This is a {domain} example {i}"} for i in range(10)
        ]
        with open(data_file, "w") as f:
            json.dump(examples, f)
        
        # Create sample vocabulary file
        vocab_file = os.path.join(domain_dir, "vocab.json")
        vocabulary = [f"{domain}_token_{i}" for i in range(5)]
        with open(vocab_file, "w") as f:
            json.dump(vocabulary, f)
    
    yield temp_dir
    
    # Clean up
    shutil.rmtree(temp_dir)

@pytest.fixture
def domain_config(temp_data_dir):
    """Create a domain config for testing"""
    domains = ["general", "math", "science"]
    
    domain_data_paths = {
        domain: os.path.join(temp_data_dir, domain, "data.json") for domain in domains
    }
    
    domain_vocab_files = {
        domain: os.path.join(temp_data_dir, domain, "vocab.json") for domain in domains
    }
    
    return DomainDataConfig(
        domains=domains,
        domain_weights={"general": 1.0, "math": 1.5, "science": 1.2},
        domain_data_paths=domain_data_paths,
        augment_vocabulary=True,
        domain_vocab_files=domain_vocab_files,
        use_curriculum=True,
        mixing_strategy="proportional"
    )

@pytest.fixture
def tokenizer():
    """Create a tokenizer for testing"""
    return SimpleTokenizer(vocab_size=1000)

def test_domain_data_config():
    """Test DomainDataConfig initialization"""
    # Default initialization
    config = DomainDataConfig()
    
    # Check default values
    assert config.domains == ["general", "math", "science", "logic", "coding"]
    assert config.domain_weights["math"] == 1.5
    assert config.domain_data_paths["general"] == "data/general"
    assert config.domain_vocab_files["science"] == "data/science/vocab.json"
    
    # Custom initialization
    custom_config = DomainDataConfig(
        domains=["custom1", "custom2"],
        domain_weights={"custom1": 1.0, "custom2": 2.0},
        domain_data_paths={"custom1": "path1", "custom2": "path2"},
        domain_vocab_files={"custom1": "vocab1", "custom2": "vocab2"}
    )
    
    # Check custom values
    assert custom_config.domains == ["custom1", "custom2"]
    assert custom_config.domain_weights["custom2"] == 2.0
    assert custom_config.domain_data_paths["custom1"] == "path1"
    assert custom_config.domain_vocab_files["custom2"] == "vocab2"

def test_domain_specific_dataset(temp_data_dir, tokenizer):
    """Test DomainSpecificDataset initialization and usage"""
    # Create dataset
    domain = "math"
    data_path = os.path.join(temp_data_dir, domain, "data.json")
    
    dataset = DomainSpecificDataset(
        domain=domain,
        tokenizer=tokenizer,
        data_path=data_path,
        max_length=20
    )
    
    # Check dataset properties
    assert dataset.domain == domain
    assert dataset.data_path == data_path
    assert len(dataset) == 10  # 10 examples were created
    
    # Check dataset item
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert "domain" in item
    assert item["domain"] == domain
    assert item["input_ids"].shape[0] == 20  # max_length
    assert torch.all(item["labels"] == item["input_ids"])  # Labels should be same as input_ids

def test_domain_specific_vocab_augmenter(domain_config, tokenizer):
    """Test DomainSpecificVocabAugmenter"""
    # Create augmenter
    augmenter = DomainSpecificVocabAugmenter(tokenizer, domain_config)
    
    # Augment vocabulary
    num_added = augmenter.augment_vocabulary()
    
    # Check that tokens were added
    assert num_added == 15  # 3 domains * 5 tokens each

def test_domain_data_manager(domain_config, tokenizer):
    """Test DomainDataManager initialization and usage"""
    # Create manager
    manager = DomainDataManager(
        config=domain_config,
        tokenizer=tokenizer,
        max_length=20,
        seed=42
    )
    
    # Check manager properties
    assert len(manager.domain_datasets) == 3  # 3 domains were created
    assert set(manager.domain_datasets.keys()) == set(["general", "math", "science"])
    
    # Create mixed dataloader
    batch_size = 4
    train_loader = manager.create_mixed_dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )
    
    # Check dataloader
    assert train_loader is not None
    
    # Get domain-specific dataloader
    math_loader = manager.get_domain_specific_dataloader(
        domain="math",
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Use 0 for testing
    )
    
    # Check domain-specific dataloader
    assert math_loader is not None
    
    # Check dataloaders work
    for batch in train_loader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "domain" in batch
        assert batch["input_ids"].shape[0] <= batch_size
        break
        
    for batch in math_loader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "domain" in batch
        assert batch["input_ids"].shape[0] <= batch_size
        assert all(d == "math" for d in batch["domain"])
        break

def test_equal_mixed_dataloader(domain_config, tokenizer):
    """Test equal mixed dataloader creation"""
    # Create manager
    manager = DomainDataManager(
        config=domain_config,
        tokenizer=tokenizer,
        max_length=20,
        seed=42
    )
    
    # Set mixing strategy to equal
    manager.config.mixing_strategy = "equal"
    
    # Create mixed dataloader
    batch_size = 4
    equal_loader = manager._create_equal_mixed_dataloader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )
    
    # Check dataloader
    assert equal_loader is not None
    
    # Check dataloader works
    for batch in equal_loader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "domain" in batch
        assert batch["input_ids"].shape[0] <= batch_size
        break

def test_curriculum_dataloader(domain_config, tokenizer):
    """Test curriculum dataloader creation"""
    # Create manager
    manager = DomainDataManager(
        config=domain_config,
        tokenizer=tokenizer,
        max_length=20,
        seed=42
    )
    
    # Set mixing strategy to curriculum
    manager.config.mixing_strategy = "curriculum"
    
    # Create mixed dataloader
    batch_size = 4
    curriculum_loader = manager._create_curriculum_dataloader(
        batch_size=batch_size,
        shuffle=False,  # Should be false for curriculum
        num_workers=0  # Use 0 for testing
    )
    
    # Check dataloader
    assert curriculum_loader is not None
    
    # Check dataloader works
    for batch in curriculum_loader:
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "domain" in batch
        assert batch["input_ids"].shape[0] <= batch_size
        break

def test_augment_low_resource_domains(domain_config, tokenizer, temp_data_dir):
    """Test augmentation of low-resource domains"""
    # Create a low-resource domain
    low_domain = "low_resource"
    low_domain_dir = os.path.join(temp_data_dir, low_domain)
    os.makedirs(low_domain_dir, exist_ok=True)
    
    # Create sample data file with very few examples
    data_file = os.path.join(low_domain_dir, "data.json")
    examples = [
        {"text": f"This is a {low_domain} example {i}"} for i in range(3)
    ]
    with open(data_file, "w") as f:
        json.dump(examples, f)
    
    # Create sample vocabulary file
    vocab_file = os.path.join(low_domain_dir, "vocab.json")
    vocabulary = [f"{low_domain}_token_{i}" for i in range(5)]
    with open(vocab_file, "w") as f:
        json.dump(vocabulary, f)
    
    # Update domain config
    domains = domain_config.domains + [low_domain]
    domain_data_paths = domain_config.domain_data_paths.copy()
    domain_data_paths[low_domain] = data_file
    domain_vocab_files = domain_config.domain_vocab_files.copy()
    domain_vocab_files[low_domain] = vocab_file
    domain_weights = domain_config.domain_weights.copy()
    domain_weights[low_domain] = 1.0
    
    # Create updated config
    updated_config = DomainDataConfig(
        domains=domains,
        domain_weights=domain_weights,
        domain_data_paths=domain_data_paths,
        domain_vocab_files=domain_vocab_files,
        augment_low_resource=True,
        min_domain_examples=5  # Set to 5 to trigger augmentation
    )
    
    # Create manager
    manager = DomainDataManager(
        config=updated_config,
        tokenizer=tokenizer,
        max_length=20,
        seed=42
    )
    
    # Check that the low-resource domain exists and has few examples
    assert low_domain in manager.domain_datasets
    assert len(manager.domain_datasets[low_domain]) == 3  # 3 examples initially
    
    # Augment low-resource domains
    manager._augment_low_resource_domains()
    
    # Check that the low-resource domain now has more examples
    assert len(manager.domain_datasets[low_domain]) >= 5  # At least min_domain_examples 