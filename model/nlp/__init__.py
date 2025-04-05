# NLP module initialization
"""
This module contains NLP-related components for the model.

Components:
- EnhancedTokenizer: Neural tokenizer module for embedding generation
- TokenizerProcessor: Text tokenization processor for converting text to IDs
- TokenizerAdapter: Bridge between text processing and neural embeddings
- TextClassifier: Component for text classification tasks
- SemanticParser: Component for semantic parsing and understanding
"""

from model.nlp.enhanced_tokenizer import EnhancedTokenizer
from model.nlp.tokenization import EnhancedTokenizer as TokenizerProcessor
from model.nlp.tokenizer_adapter import TokenizerAdapter
from model.nlp.text_classifier import TextClassifier
from model.nlp.semantic_parser import SemanticParser
from model.nlp.classification import ClassificationHead

__all__ = [
    'EnhancedTokenizer',
    'TokenizerProcessor',
    'TokenizerAdapter',
    'TextClassifier',
    'SemanticParser',
    'ClassificationHead',
]

# Version
__version__ = '0.2.0' 