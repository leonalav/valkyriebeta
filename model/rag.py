import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
import logging
import faiss
import numpy as np
from .neural_symbolic import NeuralSymbolicConfig
from .knowledge_reasoning import KnowledgeReasoningConfig

logger = logging.getLogger(__name__)

@dataclass
class EnhancedRAGConfig:
    """Configuration for enhanced RAG with neural-symbolic integration"""
    hidden_size: int = 768
    retriever_dim: int = 768
    num_attention_heads: int = 8
    dropout: float = 0.1
    
    # Neural-Symbolic integration
    use_symbolic_reasoning: bool = True
    symbolic_config: Optional[NeuralSymbolicConfig] = None
    max_reasoning_steps: int = 3
    rule_guided_retrieval: bool = True
    verify_retrievals: bool = True
    
    # Knowledge configuration
    knowledge_config: Optional[KnowledgeReasoningConfig] = None
    max_knowledge_items: int = 100
    knowledge_pruning_threshold: float = 0.1
    enable_knowledge_composition: bool = True
    composition_layers: int = 2
    
    # Memory integration
    use_memory_hierarchy: bool = True
    memory_config: Dict[str, int] = None  # Size configs for different memory types
    memory_router_heads: int = 4
    memory_update_factor: float = 0.1
    
    # Retrieval strategy
    retrieval_strategy: str = "neural_symbolic"  # neural_symbolic, dense, sparse, hybrid
    use_uncertainty_estimation: bool = True
    uncertainty_threshold: float = 0.8
    max_retrieval_uncertainty: float = 0.3
    
    # New configuration parameters
    use_approximate_search: bool = True
    index_type: str = "IVF"  # IVF, HNSW, Flat
    num_partitions: int = 100  # For IVF index
    num_probe: int = 10  # Number of clusters to probe in IVF
    similarity_metric: str = "ip"  # ip (inner product) or l2
    normalize_embeddings: bool = True
    index_temp_path: Optional[str] = None  # Path to save/load FAISS index
    
    def __post_init__(self):
        if self.symbolic_config is None:
            self.symbolic_config = NeuralSymbolicConfig(
                hidden_size=self.hidden_size,
                use_verification=True,
                uncertainty_estimation=self.use_uncertainty_estimation
            )
        
        if self.knowledge_config is None:
            self.knowledge_config = KnowledgeReasoningConfig(
                hidden_size=self.hidden_size,
                use_retrieval_mechanism=True
            )
            
        if self.memory_config is None:
            self.memory_config = {
                "episodic": 1024,
                "working": 512,
                "long_term": 2048
            }
        
        # Validate index configuration
        if self.use_approximate_search:
            if self.index_type not in ["IVF", "HNSW", "Flat"]:
                logger.warning(f"Invalid index type {self.index_type}, falling back to Flat")
                self.index_type = "Flat"

class EnhancedRAG(nn.Module):
    """Neural-symbolic RAG implementation with advanced reasoning capabilities"""
    
    def __init__(self, config: EnhancedRAGConfig):
        super().__init__()
        self.config = config
        
        # Knowledge encoders
        self.knowledge_encoders = nn.ModuleDict({
            'query': nn.Sequential(
                nn.Linear(config.hidden_size, config.retriever_dim),
                nn.LayerNorm(config.retriever_dim) if config.normalize_embeddings else nn.Identity()
            ),
            'knowledge': nn.Sequential(
                nn.Linear(config.retriever_dim, config.retriever_dim),
                nn.LayerNorm(config.retriever_dim) if config.normalize_embeddings else nn.Identity()
            ),
            'output': nn.Linear(config.retriever_dim, config.hidden_size)
        })
        
        # Initialize FAISS index if using approximate search
        self.index = None
        if config.use_approximate_search:
            self._init_faiss_index()
        
        # Neural-symbolic reasoning components
        if config.use_symbolic_reasoning:
            from .neural_symbolic import NeuralSymbolicIntegration
            self.symbolic_reasoner = NeuralSymbolicIntegration(config.symbolic_config)
            
            # Rule-guided retrieval
            if config.rule_guided_retrieval:
                self.rule_retriever = nn.Sequential(
                    nn.Linear(config.hidden_size, config.retriever_dim),
                    nn.LayerNorm(config.retriever_dim),
                    nn.GELU(),
                    nn.Linear(config.retriever_dim, config.retriever_dim)
                )
        
        # Memory hierarchy with proper gradient flow
        if config.use_memory_hierarchy:
            self.memories = nn.ParameterDict()
            self.memory_values = nn.ParameterDict()
            for mem_type, size in config.memory_config.items():
                # Keys for attention
                self.memories[mem_type] = nn.Parameter(
                    torch.randn(size, config.hidden_size)
                )
                # Separate values for storing information
                self.memory_values[mem_type] = nn.Parameter(
                    torch.randn(size, config.hidden_size)
                )
                # Initialize both
                nn.init.normal_(self.memories[mem_type], mean=0.0, std=0.02)
                nn.init.normal_(self.memory_values[mem_type], mean=0.0, std=0.02)
            
            # Memory attention and gating
            self.memory_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.memory_router_heads,
                dropout=config.dropout,
                batch_first=True
            )
            
            # Memory update gate
            self.memory_update_gate = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, 1),
                nn.Sigmoid()
            )
        
        # Knowledge composition with efficient batched processing
        if config.enable_knowledge_composition:
            self.knowledge_composer = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.hidden_size * 2, config.hidden_size),
                    nn.LayerNorm(config.hidden_size),
                    nn.GELU()
                ) for _ in range(config.composition_layers)
            ])
        
        # Enhanced fusion mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid()
        )
        
        # Output processing
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def _init_faiss_index(self):
        """Initialize FAISS index based on configuration"""
        dim = self.config.retriever_dim
        if self.config.index_type == "Flat":
            if self.config.similarity_metric == "ip":
                self.index = faiss.IndexFlatIP(dim)
            else:
                self.index = faiss.IndexFlatL2(dim)
        elif self.config.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(dim) if self.config.similarity_metric == "ip" else faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, dim, self.config.num_partitions,
                faiss.METRIC_INNER_PRODUCT if self.config.similarity_metric == "ip" else faiss.METRIC_L2
            )
        elif self.config.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors per level
            
        # Use GPU if available
        if torch.cuda.is_available():
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )
    
    def _update_faiss_index(self, knowledge_embeddings: torch.Tensor):
        """Update FAISS index with new knowledge embeddings"""
        if self.index is None or not self.config.use_approximate_search:
            return
            
        # Convert to numpy and normalize if needed
        embeddings_np = knowledge_embeddings.detach().cpu().numpy()
        if self.config.normalize_embeddings:
            faiss.normalize_L2(embeddings_np)
            
        if not self.index.is_trained:
            self.index.train(embeddings_np)
        self.index.add(embeddings_np)
    
    def _route_to_memory(
        self, 
        query_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Enhanced memory routing with attention masking"""
        batch_size = query_states.size(0)
        routing_scores = {}
        routing_values = {}
        
        for mem_type in self.memories.keys():
            # Handle attention masking
            if attention_mask is not None:
                memory_attention_mask = attention_mask.unsqueeze(-1).expand(
                    -1, -1, self.memories[mem_type].size(0)
                )
            else:
                memory_attention_mask = None
                
            # Compute attention scores
            memory_output, _ = self.memory_attention(
                query=query_states,
                key=self.memories[mem_type].unsqueeze(0).expand(batch_size, -1, -1),
                value=self.memory_values[mem_type].unsqueeze(0).expand(batch_size, -1, -1),
                key_padding_mask=~memory_attention_mask if memory_attention_mask is not None else None
            )
            
            routing_scores[mem_type] = memory_output.mean(dim=1)  # Average over sequence length
            routing_values[mem_type] = self.memory_values[mem_type]
        
        # Select memory type with highest compatibility
        selected_type = max(routing_scores.items(), key=lambda x: x[1].mean().item())[0]
        return selected_type, self.memories[selected_type], routing_values[selected_type]

    def _compose_knowledge(self, knowledge_items: torch.Tensor) -> torch.Tensor:
        """Efficient batched knowledge composition"""
        batch_size, seq_len, num_items, hidden_size = knowledge_items.shape
        
        # Only process if we have enough items
        if num_items < 2:
            return knowledge_items
            
        # Reshape to pair items efficiently
        even_items = num_items // 2 * 2
        paired = knowledge_items[:, :, :even_items].view(batch_size, seq_len, -1, 2, hidden_size)
        
        # Compose through layers
        composed = paired.view(-1, 2 * hidden_size)
        for layer in self.knowledge_composer:
            composed = layer(composed)
        
        # Reshape back
        composed = composed.view(batch_size, seq_len, -1, hidden_size)
        
        # Handle odd number of items
        if num_items > even_items:
            remainder = knowledge_items[:, :, even_items:]
            knowledge_items = torch.cat([knowledge_items[:, :, :even_items], remainder, composed], dim=2)
        else:
            knowledge_items = torch.cat([knowledge_items, composed], dim=2)
            
        return knowledge_items

    def forward(
        self,
        hidden_states: torch.Tensor,
        knowledge_bank: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with efficient retrieval and processing"""
        batch_size, seq_len = hidden_states.shape[:2]
        outputs = {}
        
        # Encode query with normalization if configured
        query_states = self.knowledge_encoders['query'](hidden_states)
        
        # Apply rule-guided retrieval if enabled
        if self.config.rule_guided_retrieval and self.config.use_symbolic_reasoning:
            rule_states = self.rule_retriever(hidden_states)
            query_states = query_states + rule_states
        
        # Route to appropriate memory with attention masking
        memory_type, memory_keys, memory_values = self._route_to_memory(query_states, attention_mask)
        
        # Encode and index knowledge if using approximate search
        knowledge_states = self.knowledge_encoders['knowledge'](knowledge_bank)
        if self.config.use_approximate_search:
            self._update_faiss_index(knowledge_states)
            
            # Use FAISS for efficient retrieval
            query_np = query_states.detach().cpu().numpy()
            if self.config.normalize_embeddings:
                faiss.normalize_L2(query_np)
            
            # Set number of probes for IVF index
            if isinstance(self.index, faiss.IndexIVFFlat):
                self.index.nprobe = self.config.num_probe
                
            # Perform search
            scores, indices = self.index.search(
                query_np.reshape(-1, self.config.retriever_dim),
                self.config.max_knowledge_items
            )
            
            # Convert back to torch
            retrieval_scores = torch.from_numpy(scores).to(hidden_states.device)
            top_indices = torch.from_numpy(indices).to(hidden_states.device)
            
            # Reshape to match batch size and sequence length
            retrieval_scores = retrieval_scores.view(batch_size, seq_len, -1)
            top_indices = top_indices.view(batch_size, seq_len, -1)
        else:
            # Standard attention-based retrieval
            retrieval_scores = torch.matmul(query_states, knowledge_states.transpose(-2, -1))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                retrieval_scores = retrieval_scores.masked_fill(
                    ~attention_mask.unsqueeze(-1),
                    float('-inf')
                )
            
            # Get top-k retrievals
            top_k = min(self.config.max_knowledge_items, knowledge_bank.size(0))
            retrieval_scores, top_indices = torch.topk(retrieval_scores, top_k, dim=-1)
        
        # Gather top-k knowledge items
        gathered_knowledge = knowledge_states[top_indices.view(-1)].view(
            batch_size, seq_len, -1, self.config.retriever_dim
        )
        
        # Apply efficient knowledge composition
        if self.config.enable_knowledge_composition:
            gathered_knowledge = self._compose_knowledge(gathered_knowledge)
        
        # Enhanced knowledge fusion with gating
        retrieved_knowledge = torch.sum(
            gathered_knowledge * F.softmax(retrieval_scores, dim=-1).unsqueeze(-1),
            dim=2
        )
        
        # Compute fusion gate
        fusion_weights = self.fusion_gate(
            torch.cat([hidden_states, retrieved_knowledge], dim=-1)
        )
        
        # Project and fuse with gating
        output = self.knowledge_encoders['output'](retrieved_knowledge)
        output = hidden_states * (1 - fusion_weights) + output * fusion_weights
        
        # Final layer norm and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        # Update memory with proper gradient flow
        if self.config.use_memory_hierarchy:
            with torch.set_grad_enabled(True):
                # Compute update gate
                update_input = torch.cat([
                    self.memory_values[memory_type],
                    retrieved_knowledge.mean(dim=1, keepdim=True).expand(-1, self.memory_values[memory_type].size(0), -1)
                ], dim=-1)
                update_gate = self.memory_update_gate(update_input)
                
                # Compute memory updates as part of computation graph
                self.memory_values[memory_type] = (
                    self.memory_values[memory_type] * (1 - update_gate) +
                    retrieved_knowledge.mean(dim=1) * update_gate
                )
        
        # Prepare outputs
        outputs.update({
            "hidden_states": output,
            "retrieval_scores": retrieval_scores,
            "knowledge_weights": F.softmax(retrieval_scores, dim=-1)
        })
        
        # Add verification scores if enabled
        if self.config.verify_retrievals:
            verification_scores = self._verify_retrieved_knowledge(gathered_knowledge, query_states)
            outputs["verification_scores"] = verification_scores
            
        return outputs
