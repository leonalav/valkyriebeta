"""
Graph utility functions for extracting graph structures from text.

This module provides utilities for converting text into graph representations
that can be used by Graph Neural Networks (GNNs).
"""

import re
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Common English stopwords
STOPWORDS = set([
    'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 
    'about', 'as', 'of', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'but', 'if', 'or', 'because',
    'not', 'no', 'this', 'that', 'these', 'those', 'they', 'them', 'their',
    'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'some', 'such',
    'other', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can',
    'will', 'just', 'should', 'now'
])

class TextToGraphExtractor:
    """
    Class for extracting graph structures from text data.
    
    This extractor identifies entities in text and creates
    graph representations suitable for GNN processing.
    """
    
    def __init__(self, 
                 graph_min_entities=3, 
                 graph_max_entities=30,
                 create_reverse_edges=True,
                 extract_attributes=False,
                 use_coreference=False):
        """
        Initialize the text to graph extractor.
        
        Args:
            graph_min_entities: Minimum number of entities required to form a graph
            graph_max_entities: Maximum number of entities to include in a graph
            create_reverse_edges: Whether to create reverse edges for each edge
            extract_attributes: Whether to extract attributes for entities
            use_coreference: Whether to perform coreference resolution (requires additional dependencies)
        """
        self.graph_min_entities = graph_min_entities
        self.graph_max_entities = graph_max_entities
        self.create_reverse_edges = create_reverse_edges
        self.extract_attributes = extract_attributes
        self.use_coreference = use_coreference
        self.stopwords = STOPWORDS
        
        # Try to import advanced NLP tools if available
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.has_spacy = True
            logger.info("Using spaCy for enhanced graph extraction")
        except ImportError:
            self.has_spacy = False
            logger.warning("spaCy not available, using simple regex-based extraction")
            
        # Try to import graph libraries if available
        try:
            import torch
            import torch_geometric
            self.has_torch_geometric = True
            logger.info("torch_geometric available for advanced graph processing")
        except ImportError:
            self.has_torch_geometric = False
    
    def extract_graph(self, text):
        """
        Extract a graph representation from text.
        
        Args:
            text: The input text to extract a graph from
            
        Returns:
            A dictionary with graph data including nodes, edges, and features,
            or None if extraction fails or text doesn't contain enough entities
        """
        if not text or len(text) < 20:
            return None
            
        # Use spaCy for extraction if available
        if self.has_spacy:
            return self._extract_graph_with_spacy(text)
        else:
            return self._extract_graph_with_regex(text)
    
    def _extract_graph_with_regex(self, text):
        """
        Extract graph using simple regex approach.
        
        Args:
            text: The input text
            
        Returns:
            A graph dictionary or None
        """
        # Extract potential entities (capitalized words, except at sentence start)
        words = re.findall(r'\b[A-Za-z]+\b', text)
        entities = [w for w in words if len(w) > 1 and w[0].isupper() and w.lower() not in self.stopwords]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = [e for e in entities if not (e in seen or seen.add(e))]
        entities = unique_entities[:self.graph_max_entities]
        
        # Check if we have enough entities
        if len(entities) < self.graph_min_entities:
            return None
            
        # Create nodes and edges
        nodes = {entity: i for i, entity in enumerate(entities)}
        edges = []
        
        # Create edges between entities that appear close to each other
        sentences = text.split('.')
        for sentence in sentences:
            sentence_entities = [e for e in entities if e in sentence]
            for i, e1 in enumerate(sentence_entities):
                for e2 in sentence_entities[i+1:]:
                    edges.append((nodes[e1], nodes[e2]))
                    # Add reverse edge if enabled
                    if self.create_reverse_edges:
                        edges.append((nodes[e2], nodes[e1]))
        
        # Create a simple graph data structure
        return {
            'nodes': list(nodes.keys()),
            'node_indices': list(range(len(nodes))),
            'edges': edges,
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }
    
    def _extract_graph_with_spacy(self, text):
        """
        Extract graph using spaCy for better entity recognition.
        
        Args:
            text: The input text
            
        Returns:
            A graph dictionary or None
        """
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Extract named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # If not enough named entities, fall back to noun phrases
        if len(entities) < self.graph_min_entities:
            noun_chunks = [chunk.text for chunk in doc.noun_chunks 
                          if len(chunk.text) > 3 and chunk.text.lower() not in self.stopwords]
            entities = [(chunk, "NOUN_CHUNK") for chunk in noun_chunks[:self.graph_max_entities]]
        
        # Limit to max entities
        entities = entities[:self.graph_max_entities]
        
        # Check if we have enough entities
        if len(entities) < self.graph_min_entities:
            return None
        
        # Create nodes with entity types
        nodes = {}
        node_types = {}
        for i, (entity, entity_type) in enumerate(entities):
            nodes[entity] = i
            node_types[entity] = entity_type
        
        # Create edges based on syntactic dependencies and co-occurrence
        edges = []
        edge_types = []
        
        # Co-occurrence in sentences
        for sent in doc.sents:
            sent_text = sent.text
            sent_entities = [e for e, _ in entities if e in sent_text]
            for i, e1 in enumerate(sent_entities):
                for e2 in sent_entities[i+1:]:
                    edges.append((nodes[e1], nodes[e2]))
                    edge_types.append("CO_OCCURS_WITH")
                    if self.create_reverse_edges:
                        edges.append((nodes[e2], nodes[e1]))
                        edge_types.append("CO_OCCURS_WITH")
        
        # Syntactic dependencies
        for token in doc:
            head = token.head.text
            dependent = token.text
            relation = token.dep_
            
            if head in nodes and dependent in nodes and head != dependent:
                edges.append((nodes[head], nodes[dependent]))
                edge_types.append(relation)
                if self.create_reverse_edges:
                    edges.append((nodes[dependent], nodes[head]))
                    edge_types.append(f"INVERSE_{relation}")
        
        # Create node features if we have torch_geometric
        node_features = None
        if self.has_torch_geometric and self.extract_attributes:
            try:
                import torch
                import numpy as np
                
                # Convert entity types to one-hot features
                entity_types = list(set(node_types.values()))
                type_to_idx = {t: i for i, t in enumerate(entity_types)}
                
                node_features = torch.zeros(len(nodes), len(entity_types))
                for entity, node_idx in nodes.items():
                    type_idx = type_to_idx[node_types[entity]]
                    node_features[node_idx, type_idx] = 1.0
            except Exception as e:
                logger.warning(f"Error creating node features: {e}")
                node_features = None
        
        # Create the graph data structure
        graph_data = {
            'nodes': list(nodes.keys()),
            'node_indices': list(range(len(nodes))),
            'node_types': [node_types[node] for node in nodes.keys()],
            'edges': edges,
            'edge_types': edge_types if len(edge_types) == len(edges) else None,
            'num_nodes': len(nodes),
            'num_edges': len(edges)
        }
        
        if node_features is not None:
            graph_data['node_features'] = node_features
            
        return graph_data 