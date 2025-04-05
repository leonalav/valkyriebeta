import torch
import torch.nn.functional as F
from typing import List, Dict, Union, Optional
import numpy as np
from dataclasses import dataclass
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeEntry:
    """Single knowledge entry with metadata"""
    content: str
    embedding: Optional[torch.Tensor] = None
    metadata: Optional[Dict] = None
    source: Optional[str] = None
    importance: float = 1.0

class KnowledgeBankManager:
    """Manager for RAG knowledge banks"""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        max_entries: int = 100000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.embedding_dim = embedding_dim
        self.max_entries = max_entries
        self.device = device
        self.entries: List[KnowledgeEntry] = []
        self.embeddings: Optional[torch.Tensor] = None
        
    def add_entries(self, entries: List[KnowledgeEntry]) -> None:
        """Add new entries to the knowledge bank"""
        # Filter entries without embeddings
        valid_entries = [e for e in entries if e.embedding is not None]
        
        if not valid_entries:
            logger.warning("No valid entries provided")
            return
            
        # Convert embeddings to tensor
        new_embeddings = torch.stack([e.embedding for e in valid_entries])
        
        # Initialize or update embeddings tensor
        if self.embeddings is None:
            self.embeddings = new_embeddings.to(self.device)
        else:
            self.embeddings = torch.cat([self.embeddings, new_embeddings.to(self.device)], dim=0)
            
        # Maintain max size limit
        if len(self.entries) + len(valid_entries) > self.max_entries:
            logger.warning(f"Knowledge bank exceeding max size {self.max_entries}, removing older entries")
            # Keep most important entries based on importance score
            all_entries = self.entries + valid_entries
            all_entries.sort(key=lambda x: x.importance, reverse=True)
            self.entries = all_entries[:self.max_entries]
            self.embeddings = self.embeddings[-self.max_entries:]
        else:
            self.entries.extend(valid_entries)
    
    def preprocess_knowledge(self, texts: List[str], embedder: callable) -> List[KnowledgeEntry]:
        """Preprocess text into knowledge entries with embeddings"""
        entries = []
        
        for text in texts:
            # Get embedding from provided embedder function
            with torch.no_grad():
                embedding = embedder(text)
                
            # Create entry
            entry = KnowledgeEntry(
                content=text,
                embedding=embedding,
                metadata={"length": len(text)}
            )
            entries.append(entry)
            
        return entries
    
    def save(self, path: Union[str, Path]) -> None:
        """Save knowledge bank to disk"""
        path = Path(path)
        
        # Save embeddings
        torch.save(self.embeddings, path.with_suffix(".pt"))
        
        # Save entries metadata
        entries_data = []
        for entry in self.entries:
            entry_dict = {
                "content": entry.content,
                "metadata": entry.metadata,
                "source": entry.source,
                "importance": entry.importance
            }
            entries_data.append(entry_dict)
            
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(entries_data, f)
    
    def load(self, path: Union[str, Path]) -> None:
        """Load knowledge bank from disk"""
        path = Path(path)
        
        # Load embeddings
        self.embeddings = torch.load(path.with_suffix(".pt")).to(self.device)
        
        # Load entries metadata
        with open(path.with_suffix(".json"), "r") as f:
            entries_data = json.load(f)
            
        self.entries = []
        for entry_dict in entries_data:
            entry = KnowledgeEntry(
                content=entry_dict["content"],
                embedding=None,  # Embeddings loaded separately
                metadata=entry_dict["metadata"],
                source=entry_dict["source"],
                importance=entry_dict["importance"]
            )
            self.entries.append(entry)
    
    def search(
        self, 
        query_embedding: torch.Tensor,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict]:
        """Search knowledge bank with query embedding"""
        if self.embeddings is None or len(self.entries) == 0:
            return []
            
        # Compute similarity scores
        scores = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.embeddings,
            dim=1
        )
        
        # Get top-k entries
        top_scores, top_indices = torch.topk(scores, min(top_k, len(self.entries)))
        
        # Filter by minimum score
        mask = top_scores >= min_score
        top_scores = top_scores[mask]
        top_indices = top_indices[mask]
        
        # Prepare results
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            entry = self.entries[idx]
            results.append({
                "content": entry.content,
                "score": score,
                "metadata": entry.metadata,
                "source": entry.source
            })
            
        return results
    
    def update_importance(self, indices: List[int], scores: List[float]) -> None:
        """Update importance scores for entries"""
        for idx, score in zip(indices, scores):
            if 0 <= idx < len(self.entries):
                self.entries[idx].importance = max(self.entries[idx].importance, score)
                
    def cleanup(self) -> None:
        """Remove least important entries when near capacity"""
        if len(self.entries) > self.max_entries * 0.9:  # 90% threshold
            logger.info("Cleaning up knowledge bank...")
            
            # Sort by importance
            entry_scores = [(i, e.importance) for i, e in enumerate(self.entries)]
            entry_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top entries
            keep_indices = [i for i, _ in entry_scores[:self.max_entries]]
            keep_indices.sort()  # Maintain original order
            
            self.entries = [self.entries[i] for i in keep_indices]
            self.embeddings = self.embeddings[keep_indices]
            
            logger.info(f"Cleaned up knowledge bank. New size: {len(self.entries)}")