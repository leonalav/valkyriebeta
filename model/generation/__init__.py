# Generation module initialization
"""
This module contains text generation components for the model.

Includes:
- Standard beam search
- Logical beam search with reasoning
- Speculative decoding
- RAG-enhanced generation
"""

from .beam_search import BeamSearchGenerator
from .logical_beam_search import LogicalBeamSearch
from .speculative import SpeculativeGenerator, SpeculativeConfig
from .rag_generator import EnhancedRAGGenerator
from .sampling import SamplingGenerator

__version__ = "0.2.0"
__all__ = [
    'BeamSearchGenerator',
    'LogicalBeamSearch', 
    'SpeculativeGenerator',
    'SpeculativeConfig',
    'EnhancedRAGGenerator',
    'SamplingGenerator'
]
