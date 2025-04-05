import argparse
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer
import json
import sys
from typing import List, Optional

from .distillation_pipeline import DistillationPipeline
from ..model.api_distillation import APITeacherModel
from config.distillation_config import DistillationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_input_texts(file_path: str) -> List[str]:
    """Load input texts from file"""
    if file_path.endswith('.json') or file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item['text'] if isinstance(item, dict) else item for item in data]
        return [data['text'] if isinstance(data, dict) else data]
    else:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

def main(args):
    # Load configuration
    with open(args.config_file, 'r') as f:
        config_dict = json.load(f)
    config = DistillationConfig.from_dict(config_dict)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Initialize teacher model
    teacher_model = APITeacherModel(
        api_key=config.api_key,
        site_url=config.site_url,
        site_name=config.site_name,
        model=config.teacher_model,
        temperature=config.temperature
    )
    
    # Initialize pipeline
    pipeline = DistillationPipeline(
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        config=config
    )
    
    # Load and process input texts
    input_texts = load_input_texts(args.input_file)
    logger.info(f"Loaded {len(input_texts)} input texts")
    
    # Process examples
    examples = pipeline.process_examples(
        input_texts=input_texts,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    logger.info(f"Processed {len(examples)} examples")
    
    # Create and save dataset
    output_file = Path(args.output_dir) / "distillation_dataset.json"
    dataset = pipeline.create_dataset(examples, output_file=output_file)
    logger.info(f"Created dataset with {len(dataset)} examples")
    
    # Create data loader
    data_loader = pipeline.create_data_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    logger.info("Created data loader")
    
    # Save dataset info
    info_file = Path(args.output_dir) / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump({
            'num_examples': len(dataset),
            'max_length': args.max_length or config.max_seq_length,
            'tokenizer': args.model_name_or_path,
            'teacher_model': config.teacher_model
        }, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input file containing texts')
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Path to pretrained model/tokenizer')
    parser.add_argument('--output_dir', type=str, default='processed_data/distillation',
                        help='Output directory for processed data')
    parser.add_argument('--cache_dir', type=str, default='.cache/distillation',
                        help='Cache directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    main(args)
