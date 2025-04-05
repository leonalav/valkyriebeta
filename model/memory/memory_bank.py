import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class StrategySequenceMemory(nn.Module):
    """
    Memory bank that stores successful sequences of reasoning strategies for similar tasks.
    
    This memory component:
    1. Stores successful reasoning paths (sequences of reasoning strategies)
    2. Clusters tasks by similarity for efficient retrieval
    3. Tracks performance of different reasoning paths
    4. Enables transfer of successful reasoning approaches across similar problems
    """
    
    def __init__(
        self,
        hidden_size=768,
        memory_size=1000,
        strategy_embedding_size=128,
        num_strategy_types=10,
        success_threshold=0.7,
        task_clustering_threshold=0.85
    ):
        """
        Initialize the strategy sequence memory.
        
        Args:
            hidden_size: Size of hidden states from model
            memory_size: Maximum number of sequences to store
            strategy_embedding_size: Size of strategy embeddings
            num_strategy_types: Number of different strategy types
            success_threshold: Threshold for storing a sequence
            task_clustering_threshold: Similarity threshold for clustering tasks
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.strategy_embedding_size = strategy_embedding_size
        self.num_strategy_types = num_strategy_types
        self.success_threshold = success_threshold
        self.task_clustering_threshold = task_clustering_threshold
        
        # Task encoder for similarity computation
        self.task_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, strategy_embedding_size)
        )
        
        # Strategy type embeddings (for different reasoning strategy types)
        self.strategy_embeddings = nn.Parameter(
            torch.randn(num_strategy_types, strategy_embedding_size)
        )
        
        # Memory data structures
        self.task_embeddings = nn.Parameter(
            torch.zeros(memory_size, strategy_embedding_size),
            requires_grad=False
        )
        
        # For each memory slot, store sequence of strategy indices
        # Implementation note: Using Python lists since tensors can't store variable-length sequences easily
        self.strategy_sequences = [[] for _ in range(memory_size)]
        
        # Track success rate for each stored sequence
        self.success_rates = nn.Parameter(
            torch.zeros(memory_size),
            requires_grad=False
        )
        
        # Track usage count for each stored sequence
        self.usage_counts = nn.Parameter(
            torch.zeros(memory_size),
            requires_grad=False
        )
        
        # Track when each sequence was last accessed (for LRU replacement)
        self.last_accessed = nn.Parameter(
            torch.zeros(memory_size),
            requires_grad=False
        )
        
        # Counter for tracking access times
        self.access_counter = 0
        
        # Track number of occupied memory slots
        self.num_occupied = 0
        
        # For each memory slot, store original task text for debugging
        self.task_texts = ["" for _ in range(memory_size)]
        
        # Task clusters for efficient retrieval
        self.cluster_centroids = nn.Parameter(
            torch.zeros(0, strategy_embedding_size),
            requires_grad=False
        )
        self.cluster_assignments = nn.Parameter(
            torch.zeros(memory_size, dtype=torch.long),
            requires_grad=False
        )
        self.num_clusters = 0
        self.min_cluster_samples = 5
    
    def encode_task(self, hidden_states):
        """
        Encode a task into a vector representation.
        
        Args:
            hidden_states: Hidden states from the model [batch_size, seq_len, hidden_size]
            
        Returns:
            task_embedding: Task embedding [batch_size, strategy_embedding_size]
        """
        # Use the first token's hidden state as task representation
        task_hidden = hidden_states[:, 0]
        
        # Encode the task
        task_embedding = self.task_encoder(task_hidden)
        
        # Normalize for cosine similarity
        task_embedding = F.normalize(task_embedding, p=2, dim=-1)
        
        return task_embedding
    
    def store_sequence(self, hidden_states, strategy_sequence, success_rate, task_text=""):
        """
        Store a successful reasoning strategy sequence.
        
        Args:
            hidden_states: Hidden states from the model
            strategy_sequence: List of strategy indices
            success_rate: Success rate of this sequence
            task_text: Optional task text for debugging
            
        Returns:
            stored: Whether the sequence was stored
        """
        # Only store successful sequences
        if success_rate < self.success_threshold:
            return False
        
        # Encode the task
        task_embedding = self.encode_task(hidden_states)
        
        # Check if we already have this task or a very similar one
        similarities = F.cosine_similarity(
            task_embedding.unsqueeze(1),
            self.task_embeddings[:self.num_occupied].unsqueeze(0),
            dim=2
        )
        
        max_similarity, max_idx = torch.max(similarities, dim=1)
        
        # If we have a very similar task, update its sequence if this one is more successful
        if max_similarity > self.task_clustering_threshold and self.num_occupied > 0:
            idx = max_idx.item()
            if success_rate > self.success_rates[idx].item():
                self.strategy_sequences[idx] = strategy_sequence
                self.success_rates[idx] = success_rate
                self.usage_counts[idx] = 0
                self.task_texts[idx] = task_text
                self.last_accessed[idx] = self.access_counter
                self.access_counter += 1
            return True
        
        # Otherwise, find a slot to store this new sequence
        if self.num_occupied < self.memory_size:
            # We have empty slots
            idx = self.num_occupied
            self.num_occupied += 1
        else:
            # Memory is full, replace least recently used item
            idx = torch.argmin(self.last_accessed).item()
        
        # Store the new sequence
        self.task_embeddings[idx] = task_embedding.squeeze(0)
        self.strategy_sequences[idx] = strategy_sequence
        self.success_rates[idx] = success_rate
        self.usage_counts[idx] = 0
        self.task_texts[idx] = task_text
        self.last_accessed[idx] = self.access_counter
        self.access_counter += 1
        
        # Update clusters if needed
        if self.num_occupied >= self.min_cluster_samples:
            self._update_clusters()
        
        return True
    
    def retrieve_sequence(self, hidden_states):
        """
        Retrieve the most relevant strategy sequence for a task.
        
        Args:
            hidden_states: Hidden states from the model
            
        Returns:
            strategy_sequence: Retrieved strategy sequence or None
            similarity: Similarity score of the retrieved sequence
        """
        if self.num_occupied == 0:
            return None, 0.0
        
        # Encode the task
        task_embedding = self.encode_task(hidden_states)
        
        # Compute similarities
        similarities = F.cosine_similarity(
            task_embedding.unsqueeze(1),
            self.task_embeddings[:self.num_occupied].unsqueeze(0),
            dim=2
        )
        
        # Get most similar task
        max_similarity, max_idx = torch.max(similarities, dim=1)
        
        # Only retrieve if similarity is high enough
        if max_similarity > self.task_clustering_threshold:
            idx = max_idx.item()
            sequence = self.strategy_sequences[idx]
            
            # Update usage statistics
            self.usage_counts[idx] += 1
            self.last_accessed[idx] = self.access_counter
            self.access_counter += 1
            
            return sequence, max_similarity.item()
        
        return None, max_similarity.item()
    
    def get_cluster_sequences(self, hidden_states):
        """
        Get strategy sequences for the cluster this task belongs to.
        
        Args:
            hidden_states: Hidden states from the model
            
        Returns:
            cluster_sequences: List of (sequence, success_rate) pairs
        """
        if self.num_clusters == 0 or self.num_occupied < self.min_cluster_samples:
            return []
        
        # Encode the task
        task_embedding = self.encode_task(hidden_states)
        
        # Find the closest cluster
        cluster_similarities = F.cosine_similarity(
            task_embedding.unsqueeze(1),
            self.cluster_centroids.unsqueeze(0),
            dim=2
        )
        
        closest_cluster = torch.argmax(cluster_similarities).item()
        
        # Get all sequences in this cluster
        cluster_mask = (self.cluster_assignments[:self.num_occupied] == closest_cluster)
        cluster_indices = torch.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            return []
        
        # Get sequences sorted by success rate
        cluster_sequences = []
        for idx in cluster_indices:
            idx = idx.item()
            cluster_sequences.append((
                self.strategy_sequences[idx],
                self.success_rates[idx].item()
            ))
        
        # Sort by success rate
        cluster_sequences.sort(key=lambda x: x[1], reverse=True)
        
        return cluster_sequences
    
    def get_top_sequences(self, k=5):
        """
        Get the top-k most successful sequences.
        
        Args:
            k: Number of sequences to retrieve
            
        Returns:
            top_sequences: List of (sequence, success_rate, task_text) tuples
        """
        if self.num_occupied == 0:
            return []
        
        # Get indices sorted by success rate
        sorted_indices = torch.argsort(self.success_rates[:self.num_occupied], descending=True)
        
        # Get top-k sequences
        k = min(k, self.num_occupied)
        top_sequences = []
        for i in range(k):
            idx = sorted_indices[i].item()
            top_sequences.append((
                self.strategy_sequences[idx],
                self.success_rates[idx].item(),
                self.task_texts[idx]
            ))
        
        return top_sequences
    
    def _update_clusters(self):
        """Update clusters based on stored task embeddings"""
        # Simple k-means clustering
        k = min(self.num_occupied // 5, 20)  # Adaptive number of clusters
        k = max(k, 1)  # At least one cluster
        
        # Only cluster if we have enough samples
        if self.num_occupied < self.min_cluster_samples:
            return
        
        # Use k-means clustering
        try:
            from sklearn.cluster import KMeans
            
            # Convert embeddings to numpy
            embeddings = self.task_embeddings[:self.num_occupied].detach().cpu().numpy()
            
            # Perform clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_assignments = kmeans.fit_predict(embeddings)
            cluster_centroids = kmeans.cluster_centers_
            
            # Update parameters
            self.cluster_assignments[:self.num_occupied] = torch.tensor(
                cluster_assignments, dtype=torch.long
            )
            self.cluster_centroids = torch.tensor(
                cluster_centroids, dtype=torch.float32
            )
            self.num_clusters = k
            
        except ImportError:
            # Fallback to simple clustering
            embeddings = self.task_embeddings[:self.num_occupied]
            
            # Initialize random centroids
            indices = torch.randperm(self.num_occupied)[:k]
            centroids = embeddings[indices]
            
            # Assign each point to nearest centroid
            assignments = torch.zeros(self.num_occupied, dtype=torch.long)
            for i in range(self.num_occupied):
                similarities = F.cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    centroids,
                    dim=1
                )
                assignments[i] = torch.argmax(similarities).item()
            
            # Update centroids
            new_centroids = torch.zeros(k, self.strategy_embedding_size)
            for j in range(k):
                cluster_mask = (assignments == j)
                if cluster_mask.sum() > 0:
                    new_centroids[j] = embeddings[cluster_mask].mean(dim=0)
                else:
                    new_centroids[j] = centroids[j]
            
            # Update parameters
            self.cluster_assignments[:self.num_occupied] = assignments
            self.cluster_centroids = new_centroids
            self.num_clusters = k
    
    def get_statistics(self):
        """
        Get memory usage statistics.
        
        Returns:
            stats: Dictionary of statistics
        """
        stats = {
            "num_occupied": self.num_occupied,
            "memory_size": self.memory_size,
            "num_clusters": self.num_clusters,
            "average_success_rate": self.success_rates[:self.num_occupied].mean().item() if self.num_occupied > 0 else 0,
            "average_usage_count": self.usage_counts[:self.num_occupied].mean().item() if self.num_occupied > 0 else 0
        }
        
        return stats
    
    def save_state(self, path):
        """
        Save memory state to a file.
        
        Args:
            path: Path to save the state
        """
        state = {
            "task_embeddings": self.task_embeddings.detach().cpu(),
            "strategy_sequences": self.strategy_sequences,
            "success_rates": self.success_rates.detach().cpu(),
            "usage_counts": self.usage_counts.detach().cpu(),
            "last_accessed": self.last_accessed.detach().cpu(),
            "access_counter": self.access_counter,
            "num_occupied": self.num_occupied,
            "task_texts": self.task_texts,
            "cluster_centroids": self.cluster_centroids.detach().cpu(),
            "cluster_assignments": self.cluster_assignments.detach().cpu(),
            "num_clusters": self.num_clusters
        }
        
        torch.save(state, path)
    
    def load_state(self, path):
        """
        Load memory state from a file.
        
        Args:
            path: Path to load the state from
        """
        state = torch.load(path)
        
        self.task_embeddings = nn.Parameter(
            state["task_embeddings"],
            requires_grad=False
        )
        self.strategy_sequences = state["strategy_sequences"]
        self.success_rates = nn.Parameter(
            state["success_rates"],
            requires_grad=False
        )
        self.usage_counts = nn.Parameter(
            state["usage_counts"],
            requires_grad=False
        )
        self.last_accessed = nn.Parameter(
            state["last_accessed"],
            requires_grad=False
        )
        self.access_counter = state["access_counter"]
        self.num_occupied = state["num_occupied"]
        self.task_texts = state["task_texts"]
        self.cluster_centroids = nn.Parameter(
            state["cluster_centroids"],
            requires_grad=False
        )
        self.cluster_assignments = nn.Parameter(
            state["cluster_assignments"],
            requires_grad=False
        )
        self.num_clusters = state["num_clusters"] 