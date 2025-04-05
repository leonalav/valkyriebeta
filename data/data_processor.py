import os
from typing import List, Dict, Optional, Union, Any
import logging
from pathlib import Path
import torch
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import datasets
from functools import partial
import json
import time

from .preprocessor import LogicalDataPreprocessor, LogicalExample
from .tokenization import LogicalTokenizer
from .dataloader_factory import DataLoaderFactory
from .monitors import ResourceMonitor, ProcessingMonitor

class UnifiedDataProcessor:
    """Unified data processing pipeline for HuggingFace datasets"""
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        cache_dir: str = ".cache",
        max_seq_length: int = 2048,
        dataset_dir: str = "datasets"
    ):
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.dataset_dir = dataset_dir
        self.max_seq_length = max_seq_length
        
        # Initialize components
        self.logical_tokenizer = LogicalTokenizer(tokenizer)
        self.preprocessor = LogicalDataPreprocessor(self.logical_tokenizer, None)
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Add monitoring
        self.resource_monitor = ResourceMonitor()
        self.processing_monitor = ProcessingMonitor()
        
    def load_curated_thoughts(self) -> Dict[str, datasets.Dataset]:
        """Load CuratedThoughts dataset with all configurations"""
        configs = [
            'OpenR1-Math-220k-default',
            'OpenThoughts-114k-math-default',
            'OpenThoughts-114k-metadata'
        ]
        
        datasets_dict = {}
        for config in tqdm(configs, desc="Loading CuratedThoughts"):
            try:
                ds_dict = datasets.load_dataset(
                    "bethgelab/CuratedThoughts",
                    config,
                    trust_remote_code=True,
                    cache_dir=self.dataset_dir
                )
                # Get the first split
                ds = list(ds_dict.values())[0]
                
                # Process the dataset
                ds = ds.map(lambda x: {
                    "problem": x.pop("problem", ""),
                    "solution": x.pop("solution", ""),
                    "generations": x.pop("generations", ""),
                    "answer": x.pop("answer", ""),
                    "source": f"bethgelab/CuratedThoughts/{config}",
                    "metadata": str(x)
                })
                
                datasets_dict[config] = ds
                
            except Exception as e:
                self.logger.warning(f"Error loading {config}: {e}")
                continue
                
        return datasets_dict
    
    def load_wildchat(self) -> Dict[str, datasets.Dataset]:
        """Load WildChat dataset with language splitting"""
        try:
            ds_dict = datasets.load_dataset(
                "bigstupidhats/wildchat_conversations",
                trust_remote_code=True,
                cache_dir=self.dataset_dir
            )
            # Get the first split
            ds = list(ds_dict.values())[0]
            
            def is_english(text):
                ascii_chars = sum(1 for c in text if ord(c) < 128)
                return ascii_chars / len(text) > 0.5 if text else True
            
            def process_conversation(example):
                conversation = str(example.get('conversation', ''))
                instruction = str(example.get('instruction', ''))
                output = str(example.get('output', ''))
                
                text_to_check = ' '.join([conversation, instruction, output])
                language = "english" if is_english(text_to_check) else "russian"
                
                return {
                    "conversation": conversation,
                    "instruction": instruction,
                    "output": output,
                    "language": language,
                    "source": "bigstupidhats/wildchat_conversations",
                    "metadata": str(example)
                }
            
            ds = ds.map(process_conversation)
            
            # Split into two datasets based on language
            ds_en = ds.filter(lambda x: x['language'] == 'english')
            ds_ru = ds.filter(lambda x: x['language'] == 'russian')
            
            return {"english": ds_en, "russian": ds_ru}
            
        except Exception as e:
            self.logger.error(f"Error loading WildChat: {e}")
            return {}
    
    def load_generic(self, name: str, split: str, question_field: str = "question", 
                    solution_field: str = "solution", cot_type: str = "math", 
                    version_tag: Optional[str] = None) -> Optional[datasets.Dataset]:
        """Load a generic dataset with specified fields"""
        try:
            ds_dict = datasets.load_dataset(name, split=split, cache_dir=self.dataset_dir)
            if isinstance(ds_dict, dict):
                ds = list(ds_dict.values())[0]
            else:
                ds = ds_dict
                
            # Map to standard format
            ds = ds.map(lambda x: {
                "problem": str(x.get(question_field, "")),
                "solution": str(x.get(solution_field, "")),
                "type": cot_type,
                "source": f"{name}/{version_tag if version_tag else split}",
                "metadata": str(x)
            })
            
            return ds
            
        except Exception as e:
            self.logger.warning(f"Error loading {name}: {e}")
            return None
    
    def load_math(self) -> Optional[datasets.Dataset]:
        """Load math dataset"""
        try:
            ds = self.load_generic(
                name="competition_math",
                split="train",
                question_field="problem",
                solution_field="solution",
                cot_type="math"
            )
            return {"math": ds} if ds else None
        except Exception as e:
            self.logger.warning(f"Error loading math dataset: {e}")
            return None

    def load_numinamath(self) -> Optional[datasets.Dataset]:
        """Load NuminaMath dataset"""
        try:
            ds = self.load_generic(
                name="numina/numina-math",
                split="train",
                question_field="question",
                solution_field="solution",
                cot_type="math"
            )
            return {"numinamath": ds} if ds else None
        except Exception as e:
            self.logger.warning(f"Error loading NuminaMath dataset: {e}")
            return None

    def load_olympic_arena(self) -> Optional[datasets.Dataset]:
        """Load Olympic Arena dataset"""
        try:
            ds = self.load_generic(
                name="olympicarena/olympicarena",
                split="train",
                question_field="question",
                solution_field="solution",
                cot_type="math"
            )
            return {"olympic_arena": ds} if ds else None
        except Exception as e:
            self.logger.warning(f"Error loading Olympic Arena dataset: {e}")
            return None

    def load_theoremqa(self) -> Optional[datasets.Dataset]:
        """Load TheoremQA dataset"""
        try:
            ds = self.load_generic(
                name="wellecks/theorem-proofs",
                split="train",
                question_field="question",
                solution_field="proof",
                cot_type="theorem"
            )
            return {"theoremqa": ds} if ds else None
        except Exception as e:
            self.logger.warning(f"Error loading TheoremQA dataset: {e}")
            return None

    def load_scieval(self) -> Optional[datasets.Dataset]:
        """Load SciEval dataset"""
        try:
            ds = datasets.load_dataset("tau/scieval", split="train", cache_dir=self.dataset_dir)
            
            def clean_question(x):
                return {
                    "problem": x["question"],
                    "solution": x["solution"],
                    "type": "science",
                    "source": "tau/scieval",
                    "metadata": str({
                        "subject": x.get("subject", ""),
                        "difficulty": x.get("difficulty", ""),
                        "grade": x.get("grade", "")
                    })
                }
            
            ds = ds.map(clean_question)
            return {"scieval": ds}
        except Exception as e:
            self.logger.warning(f"Error loading SciEval dataset: {e}")
            return None

    def load_olympiad_bench(self) -> Optional[datasets.Dataset]:
        """Load Olympiad Bench dataset"""
        try:
            ds = datasets.load_dataset("olympiad-bench/olympiad-bench", split="train", cache_dir=self.dataset_dir)
            ds = ds.map(lambda x: {
                "problem": x["question"],
                "solution": x["solution"],
                "type": "olympiad",
                "source": "olympiad-bench",
                "metadata": str({
                    "subject": x.get("subject", ""),
                    "difficulty": x.get("difficulty", ""),
                    "year": x.get("year", "")
                })
            })
            return {"olympiad_bench": ds}
        except Exception as e:
            self.logger.warning(f"Error loading Olympiad Bench dataset: {e}")
            return None

    def load_jeebench(self) -> Optional[datasets.Dataset]:
        """Load JEEBench dataset"""
        try:
            ds = self.load_generic(
                name="jee-bench/jee-bench",
                split="train",
                question_field="question",
                solution_field="solution",
                cot_type="jee"
            )
            return {"jeebench": ds} if ds else None
        except Exception as e:
            self.logger.warning(f"Error loading JEEBench dataset: {e}")
            return None

    def load_agieval(self) -> Optional[datasets.Dataset]:
        """Load AGIEval dataset"""
        try:
            ds = datasets.load_dataset("microsoft/agieval", split="train", cache_dir=self.dataset_dir)
            ds = ds.map(lambda x: {
                "problem": x["question"],
                "solution": x["solution"],
                "type": "agi",
                "source": "microsoft/agieval",
                "metadata": str({
                    "category": x.get("category", ""),
                    "subcategory": x.get("subcategory", "")
                })
            })
            return {"agieval": ds}
        except Exception as e:
            self.logger.warning(f"Error loading AGIEval dataset: {e}")
            return None

    def load_statsqual(self) -> Optional[datasets.Dataset]:
        """Load StatsQual dataset"""
        try:
            ds = self.load_generic(
                name="statsqual/statsqual",
                split="train",
                question_field="question",
                solution_field="solution",
                cot_type="stats"
            )
            return {"statsqual": ds} if ds else None
        except Exception as e:
            self.logger.warning(f"Error loading StatsQual dataset: {e}")
            return None

    def load_gpqa_extended(self) -> Optional[datasets.Dataset]:
        """Load GPQA Extended dataset"""
        try:
            ds = self.load_generic(
                name="gpqa/gpqa-extended",
                split="train",
                question_field="question",
                solution_field="solution",
                cot_type="physics"
            )
            return {"gpqa_extended": ds} if ds else None
        except Exception as e:
            self.logger.warning(f"Error loading GPQA Extended dataset: {e}")
            return None

    def load_xword(self) -> Optional[datasets.Dataset]:
        """Load XWord dataset"""
        try:
            ds = self.load_generic(
                name="xword/xword",
                split="train",
                question_field="question",
                solution_field="solution",
                cot_type="crossword"
            )
            return {"xword": ds} if ds else None
        except Exception as e:
            self.logger.warning(f"Error loading XWord dataset: {e}")
            return None

    def load_usaco(self) -> Optional[datasets.Dataset]:
        """Load USACO dataset"""
        try:
            ds = self.load_generic(
                name="usaco/usaco",
                split="train",
                question_field="problem",
                solution_field="solution",
                cot_type="programming"
            )
            return {"usaco": ds} if ds else None
        except Exception as e:
            self.logger.warning(f"Error loading USACO dataset: {e}")
            return None

    def load_quant(self) -> Optional[datasets.Dataset]:
        """Load Quant dataset"""
        try:
            ds = self.load_generic(
                name="quant/quant",
                split="train",
                question_field="question",
                solution_field="solution",
                cot_type="quantitative"
            )
            return {"quant": ds} if ds else None
        except Exception as e:
            self.logger.warning(f"Error loading Quant dataset: {e}")
            return None

    def load_livecodebench(self) -> Optional[datasets.Dataset]:
        """Load LiveCodeBench dataset"""
        try:
            ds = self.load_generic(
                name="livecodebench/livecodebench",
                split="train",
                question_field="problem",
                solution_field="solution",
                cot_type="coding"
            )
            return {"livecodebench": ds} if ds else None
        except Exception as e:
            self.logger.warning(f"Error loading LiveCodeBench dataset: {e}")
            return None

    def process_datasets(
        self,
        output_dir: str,
        batch_size: int = 32,
        selected_datasets: Optional[List[str]] = None,
        save_format: str = "both"  # "parquet", "jsonl", or "both"
    ) -> Dict[str, Any]:
        """Process selected or all available datasets
        
        Args:
            output_dir: Directory to save processed datasets
            batch_size: Batch size for data loaders
            selected_datasets: List of dataset names to process, or None for all
            save_format: Format to save datasets in ("parquet", "jsonl", or "both")
        """
        all_datasets = {}
        
        # Define all available dataset loaders
        dataset_loaders = {
            "curated_thoughts": self.load_curated_thoughts,
            "wildchat": self.load_wildchat,
            "math": self.load_math,
            "numinamath": self.load_numinamath,
            "olympic_arena": self.load_olympic_arena,
            "theoremqa": self.load_theoremqa,
            "scieval": self.load_scieval,
            "olympiad_bench": self.load_olympiad_bench,
            "jeebench": self.load_jeebench,
            "agieval": self.load_agieval,
            "statsqual": self.load_statsqual,
            "gpqa_extended": self.load_gpqa_extended,
            "xword": self.load_xword,
            "usaco": self.load_usaco,
            "quant": self.load_quant,
            "livecodebench": self.load_livecodebench
        }

        # Load selected or all datasets
        for name, loader in dataset_loaders.items():
            if not selected_datasets or name in selected_datasets:
                try:
                    result = loader()
                    if result:
                        all_datasets.update(result)
                except Exception as e:
                    self.logger.error(f"Error loading dataset {name}: {e}")
                    continue
        
        # Process and save datasets
        processed_datasets = {}
        for name, ds in tqdm(all_datasets.items(), desc="Processing datasets"):
            try:
                # Create output directory for this dataset
                ds_output_dir = os.path.join(output_dir, name)
                os.makedirs(ds_output_dir, exist_ok=True)
                
                # Save in specified format(s)
                if save_format in ["parquet", "both"]:
                    parquet_path = os.path.join(ds_output_dir, f"{name}.parquet")
                    ds.to_parquet(parquet_path)
                    self.logger.info(f"Saved {name} to {parquet_path}")
                
                if save_format in ["jsonl", "both"]:
                    jsonl_path = os.path.join(ds_output_dir, f"{name}.jsonl")
                    ds.to_json(jsonl_path, orient="records", lines=True)
                    self.logger.info(f"Saved {name} to {jsonl_path}")
                
                # Save dataset info
                info_path = os.path.join(ds_output_dir, "dataset_info.json")
                with open(info_path, 'w') as f:
                    json.dump({
                        "name": name,
                        "num_examples": len(ds),
                        "features": ds.features,
                        "column_names": ds.column_names,
                        "source": ds[0]["source"] if len(ds) > 0 else None
                    }, f, indent=2)
                
                # Create data loader based on format
                if save_format == "parquet":
                    data_files = [parquet_path]
                elif save_format == "jsonl":
                    data_files = [jsonl_path]
                else:  # "both" - prefer parquet for efficiency
                    data_files = [parquet_path]
                
                data_loader = DataLoaderFactory.create_loader(
                    data_files=data_files,
                    tokenizer=self.tokenizer,
                    batch_size=batch_size,
                    max_seq_length=self.max_seq_length,
                    cache_dir=self.cache_dir
                )
                
                processed_datasets[name] = {
                    'loader': data_loader,
                    'num_examples': len(ds),
                    'output_dir': ds_output_dir,
                    'parquet_path': parquet_path if save_format in ["parquet", "both"] else None,
                    'jsonl_path': jsonl_path if save_format in ["jsonl", "both"] else None
                }
                
            except Exception as e:
                self.logger.error(f"Error processing dataset {name}: {e}")
                continue
        
        return processed_datasets
    
    def cleanup(self):
        """Cleanup resources"""
        DataLoaderFactory.cleanup_cache(self.cache_dir)
        
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        return {
            'cache_size_mb': DataLoaderFactory.get_cache_size(self.cache_dir),
            'num_cached_files': len(list(Path(self.cache_dir).glob("**/*")))
        }
    
    def process_data_files(self,
                          data_files: List[str],
                          output_dir: str,
                          batch_size: int,
                          **kwargs):
        """Process data files with monitoring"""
        # Start monitoring
        self.processing_monitor.reset()
        
        result = {}
        for file in data_files:
            # Monitor resource usage
            metrics = self.resource_monitor.get_metrics()
            self.logger.info(f"Resource usage while processing {file}:")
            self.logger.info(f"CPU: {metrics.cpu_percent}%, Memory: {metrics.memory_percent}%")
            
            start_time = time.time()
            tokenization_start = time.time()
            
            # Process file
            # ...existing processing code...
            
            # Update metrics
            tokenization_time = time.time() - tokenization_start
            batch_time = time.time() - start_time
            self.processing_monitor.update(
                batch_size=batch_size,
                batch_time=batch_time,
                tokenization_time=tokenization_time,
                preprocessing_time=batch_time - tokenization_time
            )
        
        # Log final metrics
        metrics = self.processing_monitor.get_metrics()
        self.logger.info(f"Processing complete:")
        self.logger.info(f"Samples processed: {metrics.samples_processed}")
        self.logger.info(f"Samples/second: {metrics.samples_per_second:.2f}")
        
        return result