"""
Valkyrie LLM - Advanced Language Model with Enhanced Reasoning Capabilities

This is the main package for ValkyrieLLM, providing access to model components,
training utilities, and inference capabilities.
"""

__version__ = "0.1.0"

# Import key components for easier access
from valkyrie_llm.model.core_model import CoreModel as BaseModel
from valkyrie_llm.model.reasoning import ChainOfThoughtReasoner
from valkyrie_llm.training.training_engine import TrainingEngine
from valkyrie_llm.data.fineweb import FineWebDataset, setup_fineweb_dataloader
from valkyrie_llm.utils.tpu_utils import setup_tpu_strategy, is_tpu_available

__all__ = [
    "BaseModel",
    "ChainOfThoughtReasoner",
    "TrainingEngine",
    "FineWebDataset",
    "setup_fineweb_dataloader",
    "setup_tpu_strategy",
    "is_tpu_available",
] 