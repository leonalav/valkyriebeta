import torch
from torch.utils.data import Dataset
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

class RLHFDataset(Dataset):
    """
    Dataset for Reinforcement Learning from Human Feedback (RLHF).
    This dataset handles prompts, chosen responses, and rejected responses for RLHF training.
    """
    
    def __init__(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: Optional[List[str]] = None,
        tokenizer = None,
        max_length: int = 1024,
        prompt_max_length: int = 512,
        response_max_length: int = 512
    ):
        """
        Initialize RLHF dataset.
        
        Args:
            prompts: List of prompts
            chosen_responses: List of chosen (preferred) responses
            rejected_responses: Optional list of rejected responses (required for DPO)
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum total sequence length
            prompt_max_length: Maximum prompt length
            response_max_length: Maximum response length
        """
        self.prompts = prompts
        self.chosen_responses = chosen_responses
        self.rejected_responses = rejected_responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_max_length = prompt_max_length
        self.response_max_length = response_max_length
        
        # Validate inputs
        assert len(prompts) == len(chosen_responses), "Number of prompts and chosen responses must match"
        if rejected_responses is not None:
            assert len(prompts) == len(rejected_responses), "Number of prompts and rejected responses must match"
            
        # Pre-tokenize if tokenizer is provided
        if tokenizer is not None:
            self.tokenized_prompts = self._tokenize_texts(prompts, prompt_max_length)
            self.tokenized_chosen = self._tokenize_texts(chosen_responses, response_max_length)
            if rejected_responses is not None:
                self.tokenized_rejected = self._tokenize_texts(rejected_responses, response_max_length)
            else:
                self.tokenized_rejected = None
        else:
            self.tokenized_prompts = None
            self.tokenized_chosen = None
            self.tokenized_rejected = None
    
    def _tokenize_texts(self, texts: List[str], max_length: int) -> List[torch.Tensor]:
        """Tokenize a list of texts"""
        tokenized = []
        for text in texts:
            tokens = self.tokenizer.encode(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            tokenized.append(tokens.squeeze(0))
        return tokenized
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset.
        
        Returns:
            Dictionary containing:
                - prompt: Original prompt text
                - chosen: Chosen response text
                - rejected: Rejected response text (if available)
                - prompt_input_ids: Tokenized prompt (if tokenizer was provided)
                - chosen_input_ids: Tokenized chosen response (if tokenizer was provided)
                - rejected_input_ids: Tokenized rejected response (if tokenizer was provided)
        """
        item = {
            "prompt": self.prompts[idx],
            "chosen": self.chosen_responses[idx],
        }
        
        if self.rejected_responses is not None:
            item["rejected"] = self.rejected_responses[idx]
        
        if self.tokenized_prompts is not None:
            item["prompt_input_ids"] = self.tokenized_prompts[idx]
            item["chosen_input_ids"] = self.tokenized_chosen[idx]
            if self.tokenized_rejected is not None:
                item["rejected_input_ids"] = self.tokenized_rejected[idx]
        
        return item
    
    @classmethod
    def from_json(cls, json_file: str, tokenizer=None, **kwargs):
        """
        Create dataset from a JSON file.
        
        The JSON file should contain a list of dictionaries with the following keys:
            - prompt: The prompt text
            - chosen: The chosen response text
            - rejected: The rejected response text (optional)
            
        Args:
            json_file: Path to JSON file
            tokenizer: Tokenizer to use
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            RLHFDataset instance
        """
        import json
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompts = [item["prompt"] for item in data]
        chosen_responses = [item["chosen"] for item in data]
        
        if "rejected" in data[0]:
            rejected_responses = [item["rejected"] for item in data]
        else:
            rejected_responses = None
        
        return cls(
            prompts=prompts,
            chosen_responses=chosen_responses,
            rejected_responses=rejected_responses,
            tokenizer=tokenizer,
            **kwargs
        )

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load data from file or directory"""
        examples = []
        
        if os.path.isfile(data_path):
            # Load from single file
            examples = self._load_file(data_path)
        elif os.path.isdir(data_path):
            # Load from directory
            for filename in os.listdir(data_path):
                if filename.endswith('.json') or filename.endswith('.jsonl'):
                    file_path = os.path.join(data_path, filename)
                    file_examples = self._load_file(file_path)
                    examples.extend(file_examples)
        else:
            raise ValueError(f"Data path {data_path} is neither a file nor a directory")
            
        return examples
    
    def _load_file(self, file_path: str) -> List[Dict]:
        """Load data from a single file"""
        examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                # Load as JSON
                data = json.load(f)
                if isinstance(data, list):
                    examples = data
                else:
                    examples = [data]
            elif file_path.endswith('.jsonl'):
                # Load as JSONL
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        examples.append(example)
        
        # Validate examples
        valid_examples = []
        for example in examples:
            if 'prompt' in example and 'chosen' in example:
                if not self.include_rejected or 'rejected' in example:
                    valid_examples.append(example)
        
        return valid_examples 