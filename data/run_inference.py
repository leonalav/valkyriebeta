import os
import sys
import torch
from typing import Optional, Sequence, List, Dict, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, HfArgumentParser
from tqdm import tqdm
from dataclasses import dataclass, field
import time
import json
from torch.utils.data import Dataset as TorchDataset, DataLoader

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.io_utils import question_hash, jdump, jload
from model.reasoning import ChainOfThoughtReasoner  # Your custom model

@dataclass
class InferenceConfig:
    model_name: str = field(default="model", metadata={'help': 'Path to model checkpoint'})
    shard_index: int = field(default=0, metadata={'help': 'Shard index'})
    batch_size: int = field(default=32, metadata={'help': 'Batch size'})
    max_length: int = field(default=512, metadata={'help': 'Max sequence length'})
    device: str = field(default='cuda', metadata={'help': 'Device to run inference on'})

class InferenceDataset(TorchDataset):
    """Dataset class for inference"""
    def __init__(self, questions, tokenizer, max_length):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        inputs = self.tokenizer(
            question,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'question': question
        }

def load_questions():
    """Load questions from the collected dataset"""
    dataset = load_dataset("simplescaling/s50K")
    questions = dataset['train']['question']
    print(f"Loaded {len(questions)} questions")
    return questions

def shard_questions(questions, chunk_size=10_000):
    """Split questions into shards and save them"""
    shards = []
    for i in range(0, len(questions), chunk_size):
        shard = questions[i:i + chunk_size]
        shard_path = f"results/inference/shard_{i//chunk_size}_input.json"
        jdump(shard, shard_path)
        shards.append(shard)
    return shards

def run_inference(model_path: str, questions: List[str], config: InferenceConfig) -> List[str]:
    """Run inference using custom model"""
    # Initialize model and tokenizer
    model = ChainOfThoughtReasoner.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(config.device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = InferenceDataset(questions, tokenizer, config.max_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            # Move batch to device
            batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate outputs
            outputs = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=config.max_length,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Decode outputs
            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded_outputs)
    
    return results

def assemble_output(results: List[str], model_name: str):
    """Save inference results"""
    output_dir = f"results/inference/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output dictionary
    output = {}
    for question, result in zip(load_questions(), results):
        output[question_hash(question)] = result
    
    # Save results
    output_path = os.path.join(output_dir, "inference_output.json")
    jdump(output, output_path)
    print(f"Saved results to {output_path}")

def main():
    # Parse arguments
    parser = HfArgumentParser(InferenceConfig)
    config = parser.parse_args_into_dataclasses()[0]
    
    # Load and shard questions
    print("Loading and sharding questions...")
    questions = load_questions()
    shards = shard_questions(questions)
    
    # Run inference
    print("Running inference...")
    all_results = []
    for shard_idx, shard in enumerate(shards):
        results = run_inference(
            model_path=config.model_name,
            questions=shard,
            config=config
        )
        all_results.extend(results)
    
    # Save results
    print("Saving results...")
    assemble_output(all_results, config.model_name)
    
    print("Done!")

if __name__ == "__main__":
    main() 