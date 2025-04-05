import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

@dataclass
class KnowledgeReasoningConfig:
    """Configuration for knowledge-enhanced reasoning"""
    hidden_size: int = 768
    knowledge_size: int = 1024  # Size of knowledge embedding dimension
    num_attention_heads: int = 8
    dropout: float = 0.1
    knowledge_dropout: float = 0.1
    intermediate_size: int = 2048
    use_entity_embeddings: bool = True
    num_entities: int = 10000  # Number of entities in knowledge base
    entity_embedding_dim: int = 768
    use_relation_embeddings: bool = True
    num_relations: int = 1000  # Number of relations in knowledge base
    relation_embedding_dim: int = 768
    use_graph_reasoning: bool = True
    max_graph_hops: int = 3  # Maximum hops in knowledge graph
    graph_hidden_size: int = 768
    use_graph_attention: bool = True
    num_graph_attention_heads: int = 4
    use_memory_bank: bool = True
    memory_size: int = 1024
    use_retrieval_mechanism: bool = True
    num_retrievals: int = 5
    use_dense_connection: bool = True
    layer_norm_eps: float = 1e-12
    
    def __post_init__(self):
        if self.entity_embedding_dim != self.hidden_size:
            # Ensure entity embedding dimension matches hidden size for easier integration
            self.entity_embedding_dim = self.hidden_size
            
        if self.relation_embedding_dim != self.hidden_size:
            # Ensure relation embedding dimension matches hidden size for easier integration
            self.relation_embedding_dim = self.hidden_size


class KnowledgeAttention(nn.Module):
    """Multi-head attention between hidden states and knowledge"""
    
    def __init__(self, config: KnowledgeReasoningConfig):
        super().__init__()
        self.config = config
        
        # Ensure hidden size is divisible by number of attention heads
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by number of attention heads {config.num_attention_heads}"
            )
            
        self.head_size = config.hidden_size // config.num_attention_heads
        
        # Projection layers
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_proj = nn.Linear(config.knowledge_size, config.hidden_size)
        self.value_proj = nn.Linear(config.knowledge_size, config.hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def split_heads(self, x: torch.Tensor, is_key_or_value: bool = False) -> torch.Tensor:
        """Split tensor into multiple attention heads"""
        batch_size, seq_len, dim = x.shape
        
        # Reshape to [batch_size, seq_len, num_heads, head_size]
        x = x.view(batch_size, seq_len, self.config.num_attention_heads, self.head_size)
        
        # Transpose to [batch_size, num_heads, seq_len, head_size]
        return x.transpose(1, 2)
    
    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge attention heads back to flat tensor"""
        batch_size, num_heads, seq_len, head_size = x.shape
        
        # Transpose to [batch_size, seq_len, num_heads, head_size]
        x = x.transpose(1, 2)
        
        # Merge to [batch_size, seq_len, hidden_size]
        return x.reshape(batch_size, seq_len, self.config.hidden_size)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        knowledge_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head attention between hidden states and knowledge
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            knowledge_states: Knowledge tensor of shape [batch_size, num_knowledge, knowledge_size]
            attention_mask: Optional mask for knowledge attention
            
        Returns:
            output: Attention output of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project queries, keys, values
        queries = self.query_proj(hidden_states)
        keys = self.key_proj(knowledge_states)
        values = self.value_proj(knowledge_states)
        
        # Split heads
        queries = self.split_heads(queries)
        keys = self.split_heads(keys)
        values = self.split_heads(values)
        
        # Scale query
        queries = queries / math.sqrt(self.head_size)
        
        # Calculate attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention score dimensions
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            
            # Apply large negative bias to masked positions
            attention_scores = attention_scores + (1.0 - expanded_mask) * -10000.0
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, values)
        
        # Merge heads
        context = self.merge_heads(context)
        
        # Apply output projection
        output = self.output_proj(context)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network for knowledge processing"""
    
    def __init__(self, config: KnowledgeReasoningConfig):
        super().__init__()
        self.config = config
        
        # Feed-forward layers
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation"""
        # First dense layer with GELU activation
        intermediate = F.gelu(self.dense_1(hidden_states))
        
        # Second dense layer
        output = self.dense_2(intermediate)
        
        # Dropout
        output = self.dropout(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + hidden_states)
        
        return output


class GraphReasoningLayer(nn.Module):
    """Layer for reasoning over knowledge graph connections"""
    
    def __init__(self, config: KnowledgeReasoningConfig):
        super().__init__()
        self.config = config
        
        # Graph attention if enabled
        if config.use_graph_attention:
            self.graph_attention = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=config.graph_hidden_size,
                    num_heads=config.num_graph_attention_heads,
                    dropout=config.knowledge_dropout,
                    batch_first=True
                )
                for _ in range(config.max_graph_hops)
            ])
        
        # Graph transformation
        self.graph_transform = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.graph_hidden_size, config.graph_hidden_size),
                nn.LayerNorm(config.graph_hidden_size),
                nn.GELU(),
                nn.Dropout(config.knowledge_dropout)
            )
            for _ in range(config.max_graph_hops)
        ])
        
        # Final graph projection
        self.graph_projection = nn.Linear(config.graph_hidden_size, config.hidden_size)
        
    def forward(
        self, 
        entity_embeddings: torch.Tensor,
        relation_indices: torch.Tensor,
        relation_embeddings: torch.Tensor,
        adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply graph reasoning over entity and relation embeddings
        
        Args:
            entity_embeddings: Entity embeddings [num_entities, entity_embedding_dim]
            relation_indices: Relation indices in adjacency matrix [num_edges, 2]
            relation_embeddings: Relation embeddings [num_relations, relation_embedding_dim]
            adjacency_matrix: Sparse adjacency matrix [num_entities, num_entities]
            
        Returns:
            output: Enhanced entity representations [num_entities, hidden_size]
        """
        # Initialize node representations
        node_states = entity_embeddings
        
        # Apply graph reasoning for each hop
        for hop in range(self.config.max_graph_hops):
            if self.config.use_graph_attention:
                # Apply attention over graph neighborhood
                attn_output, _ = self.graph_attention[hop](
                    node_states, node_states, node_states,
                    attn_mask=~adjacency_matrix.bool()
                )
                node_states = node_states + attn_output
            else:
                # Message passing using adjacency matrix
                neighborhood = torch.matmul(adjacency_matrix, node_states)
                node_states = node_states + neighborhood
            
            # Apply transformation
            node_states = self.graph_transform[hop](node_states)
        
        # Final projection
        output = self.graph_projection(node_states)
        
        return output


class KnowledgeMemoryBank(nn.Module):
    """Memory bank for storing and retrieving knowledge"""
    
    def __init__(self, config: KnowledgeReasoningConfig):
        super().__init__()
        self.config = config
        
        # Initialize memory
        self.memory = nn.Parameter(
            torch.randn(config.memory_size, config.hidden_size)
        )
        nn.init.normal_(self.memory, mean=0.0, std=0.02)
        
        # Projection layers
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.memory_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Output layers
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Retrieve relevant knowledge from memory bank
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            retrieved_knowledge: Retrieved knowledge [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project query
        query = self.query_proj(hidden_states)
        
        # Project memory
        memory = self.memory_proj(self.memory)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, memory.t())
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Retrieve from memory
        retrieved_knowledge = torch.matmul(attention_weights, self.memory)
        
        # Output projection
        output = self.output_proj(retrieved_knowledge)
        
        return output


class KnowledgeReasoningModule(nn.Module):
    """Module for reasoning with external knowledge"""
    
    def __init__(self, config: Union[Dict, KnowledgeReasoningConfig]):
        super().__init__()
        
        # Convert dict to config if needed
        if isinstance(config, dict):
            self.config = KnowledgeReasoningConfig(**config)
        else:
            self.config = config
            
        # Entity embeddings
        if self.config.use_entity_embeddings:
            self.entity_embeddings = nn.Embedding(
                num_embeddings=self.config.num_entities,
                embedding_dim=self.config.entity_embedding_dim
            )
            
        # Relation embeddings
        if self.config.use_relation_embeddings:
            self.relation_embeddings = nn.Embedding(
                num_embeddings=self.config.num_relations,
                embedding_dim=self.config.relation_embedding_dim
            )
            
        # Knowledge attention mechanism
        self.knowledge_attention = KnowledgeAttention(self.config)
        
        # Feed-forward for integrating knowledge
        self.feed_forward = FeedForward(self.config)
        
        # Graph reasoning if enabled
        if self.config.use_graph_reasoning:
            self.graph_reasoning = GraphReasoningLayer(self.config)
            
        # Memory bank if enabled
        if self.config.use_memory_bank:
            self.memory_bank = KnowledgeMemoryBank(self.config)
            
        # Layer normalization
        self.layer_norm_1 = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.layer_norm_2 = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        
        # Entity recognition layer (to identify entities in text)
        self.entity_recognition = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        
        # Knowledge integration layer
        self.knowledge_integration = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(self.config.dropout)
        
    def retrieve_knowledge(
        self, 
        hidden_states: torch.Tensor, 
        entity_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieve knowledge based on input hidden states
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            entity_ids: Optional entity IDs identified in text [batch_size, seq_len]
            
        Returns:
            knowledge_states: Retrieved knowledge [batch_size, seq_len, knowledge_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Initialize knowledge states
        if entity_ids is not None and self.config.use_entity_embeddings:
            # Use provided entity IDs to retrieve embeddings
            # Ensure entity_ids is within range
            safe_entity_ids = torch.clamp(entity_ids, 0, self.config.num_entities - 1)
            knowledge_states = self.entity_embeddings(safe_entity_ids)
        else:
            # Infer entities from hidden states using attention
            entity_attn = F.softmax(self.entity_recognition(hidden_states), dim=-1)
            
            # Create mock entity embeddings for this simplified implementation
            # In a real system, this would retrieve from a knowledge base
            mock_entities = torch.randn(
                batch_size, self.config.num_retrievals, self.config.knowledge_size, 
                device=hidden_states.device
            )
            
            # Weight entity retrievals
            knowledge_states = torch.bmm(
                entity_attn.view(batch_size * seq_len, 1, self.config.hidden_size),
                mock_entities.view(batch_size * seq_len, self.config.knowledge_size, self.config.knowledge_size)
            ).view(batch_size, seq_len, self.config.knowledge_size)
            
        return knowledge_states
    
    def apply_graph_reasoning(
        self, 
        entity_ids: torch.Tensor, 
        relation_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply graph reasoning to entities and relations
        
        Args:
            entity_ids: Entity IDs [batch_size, seq_len]
            relation_ids: Optional relation IDs [batch_size, seq_len, seq_len]
            
        Returns:
            enhanced_entities: Graph-enhanced entity representations [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len = entity_ids.shape
        
        # Get entity embeddings
        entity_embeddings = self.entity_embeddings(
            torch.clamp(entity_ids.view(-1), 0, self.config.num_entities - 1)
        ).view(batch_size, seq_len, -1)
        
        # Create mock relation and adjacency data for this simplified implementation
        # In a real system, these would come from a knowledge graph
        if relation_ids is None:
            # Create random adjacency
            adjacency = torch.rand(batch_size, seq_len, seq_len, device=entity_ids.device) > 0.7
            relation_indices = torch.nonzero(adjacency)
            
            # Get random relation IDs
            num_edges = relation_indices.shape[0]
            if num_edges > 0:
                random_relation_ids = torch.randint(
                    0, self.config.num_relations, (num_edges,), device=entity_ids.device
                )
            else:
                random_relation_ids = torch.tensor([], device=entity_ids.device, dtype=torch.long)
        else:
            # Use provided relation IDs
            adjacency = relation_ids > 0
            relation_indices = torch.nonzero(adjacency)
            random_relation_ids = relation_ids[relation_indices[:, 0], relation_indices[:, 1], relation_indices[:, 2]]
        
        # Get relation embeddings
        if len(random_relation_ids) > 0:
            relation_embeddings = self.relation_embeddings(
                torch.clamp(random_relation_ids, 0, self.config.num_relations - 1)
            )
        else:
            relation_embeddings = torch.zeros(
                0, self.config.relation_embedding_dim, device=entity_ids.device
            )
        
        # Create adjacency matrix
        adjacency_matrix = torch.zeros(
            batch_size * seq_len, batch_size * seq_len, device=entity_ids.device
        )
        
        if relation_indices.shape[0] > 0:
            # Map batch and sequence indices to global indices
            src_indices = relation_indices[:, 0] * seq_len + relation_indices[:, 1]
            dst_indices = relation_indices[:, 0] * seq_len + relation_indices[:, 2]
            
            # Set adjacency matrix values
            adjacency_matrix[src_indices, dst_indices] = 1.0
        
        # Apply graph reasoning
        enhanced_entities = self.graph_reasoning(
            entity_embeddings.view(batch_size * seq_len, -1),
            relation_indices,
            relation_embeddings,
            adjacency_matrix
        )
        
        return enhanced_entities.view(batch_size, seq_len, -1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        entity_ids: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
        use_memory: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Apply knowledge-enhanced reasoning
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            entity_ids: Optional entity IDs [batch_size, seq_len]
            relation_ids: Optional relation IDs [batch_size, seq_len, seq_len]
            use_memory: Whether to use memory bank
            
        Returns:
            Dictionary containing:
                - hidden_states: Knowledge-enhanced hidden states
                - knowledge_states: Retrieved knowledge states
                - attention_weights: Knowledge attention weights if available
        """
        # Initialize outputs dictionary
        outputs = {"hidden_states": hidden_states}
        
        # Layer normalization
        normalized_hidden = self.layer_norm_1(hidden_states)
        
        # Retrieve knowledge
        knowledge_states = self.retrieve_knowledge(normalized_hidden, entity_ids)
        outputs["knowledge_states"] = knowledge_states
        
        # Apply graph reasoning if enabled and entity IDs provided
        if self.config.use_graph_reasoning and entity_ids is not None:
            graph_enhanced = self.apply_graph_reasoning(entity_ids, relation_ids)
            knowledge_states = knowledge_states + graph_enhanced
            
        # Apply memory retrieval if enabled
        if self.config.use_memory_bank and use_memory:
            memory_retrieved = self.memory_bank(normalized_hidden)
            knowledge_states = knowledge_states + memory_retrieved
            
        # Apply knowledge attention
        knowledge_context = self.knowledge_attention(
            normalized_hidden, knowledge_states, attention_mask
        )
        
        # Residual connection
        attention_output = self.dropout(knowledge_context) + hidden_states
        
        # Layer normalization
        normalized_attention = self.layer_norm_2(attention_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(normalized_attention)
        
        # Final knowledge integration
        if self.config.use_dense_connection:
            # Concatenate original hidden states with knowledge-enhanced states
            combined = torch.cat([hidden_states, ff_output], dim=-1)
            integrated = self.knowledge_integration(combined)
            outputs["hidden_states"] = integrated
        else:
            outputs["hidden_states"] = ff_output
            
        return outputs 