import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
import math
from collections import defaultdict, OrderedDict
import copy
import logging
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

try:
    from model.knowledge_reasoner import KnowledgeReasoningModule
    from model.verifiable_computation import ProofGenerator, VerifiableComputation
    from model.mcts_reasoning import MCTSEnhancedTreeReasoningModule
    from evaluation.init import LogicalReasoningEvaluator
    EXTERNAL_MODULES_AVAILABLE = True
except ImportError:
    logger.warning("Some external reasoning modules are not available. Related functionality will be disabled.")
    EXTERNAL_MODULES_AVAILABLE = False

@dataclass
class NeuralSymbolicConfig:
    """Configuration for Neural Symbolic reasoning integration"""
    hidden_size: int = 768
    rule_embedding_size: int = 768
    num_rules: int = 100
    inference_steps: int = 5
    softmax_temperature: float = 1.0
    use_gradient_through_inference: bool = True
    use_attention_for_reasoning: bool = True
    num_attention_heads: int = 8
    use_rule_selection: bool = True
    max_rule_applications: int = 10
    use_symbolic_knowledge_distillation: bool = False
    dropout: float = 0.1
    use_confidence_scores: bool = True
    use_verified_reasoning: bool = True
    use_memory_for_reasoning: bool = True
    memory_size: int = 1024
    
    # New configuration parameters for enhanced rule representation
    rule_initialization: str = "random"  # Options: "random", "pretrained", "template"
    rule_templates_path: str = ""  # Path to rule templates file if using template initialization
    trainable_rules: bool = True  # Whether rules can be updated during training
    use_rule_composition: bool = False  # Whether to enable rule composition
    
    # Memory mechanism improvements
    memory_update_type: str = "momentum"  # Options: "momentum", "gated", "dnc"
    memory_update_factor: float = 0.1
    use_memory_attention: bool = True
    
    # Scalability improvements
    use_sparse_rule_selection: bool = False
    rule_cache_size: int = 50
    rule_clustering: bool = False  # Whether to cluster similar rules
    num_rule_clusters: int = 10
    
    # Knowledge distillation settings
    distillation_temperature: float = 2.0
    teacher_weight: float = 0.5
    
    # Integration with other reasoning components
    use_knowledge_reasoner: bool = False
    use_mcts_reasoning: bool = False
    use_verification: bool = False  # Integration with verifiable computation
    use_reasoning_evaluator: bool = False  # Integration with reasoning evaluator
    
    # Dynamic rule learning
    enable_rule_learning: bool = False
    rule_learning_rate: float = 0.01
    max_new_rules: int = 10
    rule_pruning_threshold: float = 0.01  # Prune rules below this usage threshold
    
    # Rule specialization and adaptation
    use_rule_specialization: bool = False
    specialization_factor: float = 0.1
    
    # Uncertainty quantification
    uncertainty_estimation: bool = False
    use_monte_carlo_dropout: bool = False
    
    def __post_init__(self):
        # Ensure dimensions are compatible
        if self.rule_embedding_size != self.hidden_size:
            self.rule_embedding_size = self.hidden_size

class SymbolicReasoningLayer(nn.Module):
    """Symbolic reasoning layer with neural guidance"""
    
    def __init__(self, config: NeuralSymbolicConfig):
        """Initialize the symbolic reasoning layer"""
        super().__init__()
        self.config = config
        
        # Rule embeddings (parameterized reasoning rules)
        self.rule_embeddings = nn.Parameter(
            torch.randn(config.num_rules, config.rule_embedding_size))
        
        # Initialize based on configuration
        if config.rule_initialization == "pretrained" and config.rule_templates_path:
            self._load_rule_templates(config.rule_templates_path)
        
        # Projection layers
        self.rule_selector = nn.Linear(config.hidden_size, config.num_rules)
        self.rule_confidence = nn.Linear(config.hidden_size, 1)
        
        # Optional attention mechanism for rule application
        self.use_attention = config.use_attention_for_reasoning
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.dropout,
                batch_first=True
            )
            
        # Optional verification component
        self.verification_layer = nn.Linear(config.hidden_size, 1)
        
        # Memory components
        if config.use_memory_for_reasoning:
            # Memory storage
            self.memory = nn.Parameter(torch.randn(
                config.memory_size, config.hidden_size))
            
            # Memory update components
            self.memory_key = nn.Linear(config.hidden_size, config.hidden_size)
            self.memory_query = nn.Linear(config.hidden_size, config.hidden_size)
            self.memory_value = nn.Linear(config.hidden_size, config.hidden_size)
            
            # Additional memory attention
            if config.use_memory_attention:
                self.memory_attention = nn.MultiheadAttention(
                    embed_dim=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    dropout=config.dropout,
                    batch_first=True
                )
                
        # Rule composition (combining rules)
        self.use_rule_composition = config.use_rule_composition
        if self.use_rule_composition:
            self.rule_composer = nn.Linear(
                2 * config.rule_embedding_size, config.rule_embedding_size)
                
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Linear(config.hidden_size, 1)
        
        # Cache for specialized rules (context-dependent rule variants)
        self.specialized_rules = {}
        self.specialized_rules_usage = {}
        self.rule_composition_cache = {}
        self.rule_composition_usage = {}
        
        # Memory efficiency parameters
        self.max_specialized_cache_size = config.rule_cache_size * 2  # Larger specialized cache
        self.max_composition_cache_size = config.rule_cache_size * 2  # Larger composition cache
        
        # Track tensors that need cleanup
        self.temporary_tensors = []
        
        # Track number of inference steps since last cleanup
        self.inference_steps_since_cleanup = 0
        self.cleanup_frequency = 10  # Clean up every 10 inference steps
        
        # For Monte Carlo dropout uncertainty estimation
        self.monte_carlo_samples = []
        self.monte_carlo_sample_count = 0
        self.mc_dropout = nn.Dropout(config.dropout) if config.use_monte_carlo_dropout else None
        
        # Candidate rules for rule learning
        if config.enable_rule_learning:
            self.candidate_rules = []
            self.rule_learning_optimizer = None
        
        # Ensure rules are trainable or not based on config
        if hasattr(self, 'rule_embeddings'):
            self.rule_embeddings.requires_grad = config.trainable_rules
            
        # Neural components for rule selection
        self.rule_scorer = nn.Linear(config.hidden_size, config.rule_embedding_size)
        
        # Rule specialization if enabled
        if config.use_rule_specialization:
            self.rule_specializer = nn.Sequential(
                nn.Linear(config.rule_embedding_size + config.hidden_size, config.rule_embedding_size),
                nn.LayerNorm(config.rule_embedding_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.rule_embedding_size, config.rule_embedding_size)
            )
            # Specialized rule cache
            self.specialized_rules = {}
            self.specialized_rules_usage = {}  # Track usage for cache management
            self.max_specialized_cache_size = config.rule_cache_size * 2  # Larger specialized cache
        
        # For sparse rule selection
        if config.use_sparse_rule_selection:
            # Rule usage statistics for pruning
            self.rule_usage_counts = torch.zeros(config.num_rules)
            # Rule cache for frequently used rules
            self.rule_cache = OrderedDict()
            
        # For rule clustering
        if config.rule_clustering:
            self.rule_clusters = None
            self.cluster_centroids = None
            self._cluster_rules()
        
        # Rule application network
        self.rule_application = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # Confidence scoring with uncertainty estimation if enabled
        if config.use_confidence_scores:
            if config.uncertainty_estimation:
                # Output both mean and variance for uncertainty
                self.confidence_scorer = nn.Linear(config.hidden_size, 2)
            else:
                self.confidence_scorer = nn.Linear(config.hidden_size, 1)
        
        # Memory module for reasoning with improved mechanisms
        if config.use_memory_for_reasoning:
            self._initialize_memory_module()
            
        # Enable dynamic rule learning if configured
        if config.enable_rule_learning:
            self.rule_learner = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.rule_embedding_size)
            )
            self.candidate_rules = []
            self.rule_importance_scores = torch.zeros(config.num_rules)
        
        # Initialize rule cache for specialized and composed rules
        self.rule_composition_cache = {}
        self.rule_composition_usage = {}  # Track usage for cache management
        self.max_composition_cache_size = config.rule_cache_size * 2  # Larger composition cache
        
        # For knowledge distillation
        if config.use_symbolic_knowledge_distillation:
            self.teacher_projection = nn.Linear(config.hidden_size, config.hidden_size)
            self.student_projection = nn.Linear(config.hidden_size, config.hidden_size)
            
        # Setup cache cleanup trigger based on size or iterations
        self.inference_steps_since_cleanup = 0
        self.cleanup_frequency = 100  # Clean up caches every N inference steps
    
    def _initialize_rule_embeddings(self):
        """Initialize rule embeddings based on configuration"""
        if self.config.rule_initialization == "random":
            self.rule_embeddings = nn.Parameter(
                torch.randn(self.config.num_rules, self.config.rule_embedding_size)
            )
            nn.init.normal_(self.rule_embeddings, mean=0.0, std=0.02)
            
        elif self.config.rule_initialization == "pretrained":
            # Load pretrained embeddings if available, otherwise fallback to random
            if os.path.exists(self.config.rule_templates_path):
                try:
                    pretrained_embeddings = torch.load(self.config.rule_templates_path)
                    if pretrained_embeddings.shape[1] != self.config.rule_embedding_size:
                        # Resize if dimensions don't match
                        pretrained_embeddings = F.interpolate(
                            pretrained_embeddings.unsqueeze(0).unsqueeze(0),
                            size=(pretrained_embeddings.shape[0], self.config.rule_embedding_size)
                        ).squeeze(0).squeeze(0)
                    
                    if pretrained_embeddings.shape[0] >= self.config.num_rules:
                        # Use first num_rules embeddings
                        self.rule_embeddings = nn.Parameter(
                            pretrained_embeddings[:self.config.num_rules]
                        )
                    else:
                        # Pad with random embeddings if needed
                        padding = torch.randn(
                            self.config.num_rules - pretrained_embeddings.shape[0],
                            self.config.rule_embedding_size
                        )
                        nn.init.normal_(padding, mean=0.0, std=0.02)
                        self.rule_embeddings = nn.Parameter(
                            torch.cat([pretrained_embeddings, padding], dim=0)
                        )
                except Exception as e:
                    logger.warning(f"Failed to load pretrained rule embeddings: {e}. Falling back to random initialization.")
                    self.rule_embeddings = nn.Parameter(
                        torch.randn(self.config.num_rules, self.config.rule_embedding_size)
                    )
                    nn.init.normal_(self.rule_embeddings, mean=0.0, std=0.02)
            else:
                logger.warning(f"Pretrained rule embeddings path not found: {self.config.rule_templates_path}. Using random initialization.")
                self.rule_embeddings = nn.Parameter(
                    torch.randn(self.config.num_rules, self.config.rule_embedding_size)
                )
                nn.init.normal_(self.rule_embeddings, mean=0.0, std=0.02)
                
        elif self.config.rule_initialization == "template":
            # Initialize from rule templates in JSON format if available
            if os.path.exists(self.config.rule_templates_path):
                try:
                    with open(self.config.rule_templates_path, 'r') as f:
                        rule_templates = json.load(f)
                    
                    # Convert rule templates to embeddings (simplified)
                    # In a real implementation, this would use a text encoder
                    # to convert rule templates to embeddings
                    rule_embeddings = torch.randn(self.config.num_rules, self.config.rule_embedding_size)
                    nn.init.normal_(rule_embeddings, mean=0.0, std=0.02)
                    
                    # Use available templates
                    num_templates = min(len(rule_templates), self.config.num_rules)
                    self.rule_templates = rule_templates[:num_templates]
                    
                    # Add placeholders for the rest
                    for i in range(num_templates, self.config.num_rules):
                        self.rule_templates.append({
                            "name": f"rule_{i}",
                            "description": f"Auto-generated rule {i}"
                        })
                    
                    self.rule_embeddings = nn.Parameter(rule_embeddings)
                    
                except Exception as e:
                    logger.warning(f"Failed to load rule templates: {e}. Falling back to random initialization.")
                    self.rule_embeddings = nn.Parameter(
                        torch.randn(self.config.num_rules, self.config.rule_embedding_size)
                    )
                    nn.init.normal_(self.rule_embeddings, mean=0.0, std=0.02)
            else:
                logger.warning(f"Rule templates path not found: {self.config.rule_templates_path}. Using random initialization.")
                self.rule_embeddings = nn.Parameter(
                    torch.randn(self.config.num_rules, self.config.rule_embedding_size)
                )
                nn.init.normal_(self.rule_embeddings, mean=0.0, std=0.02)
        else:
            # Default to random initialization
            self.rule_embeddings = nn.Parameter(
                torch.randn(self.config.num_rules, self.config.rule_embedding_size)
            )
            nn.init.normal_(self.rule_embeddings, mean=0.0, std=0.02)
    
    def _initialize_memory_module(self):
        """Initialize memory module based on configuration"""
        self.memory_key = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.memory_value = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.memory_query = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        
        # Initialize memory based on update type
        self.memory = nn.Parameter(torch.zeros(self.config.memory_size, self.config.hidden_size))
        nn.init.normal_(self.memory, mean=0.0, std=0.02)
        
        if self.config.memory_update_type == "gated":
            # Gated memory update mechanism
            self.memory_update_gate = nn.Sequential(
                nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
                nn.Sigmoid()
            )
            self.memory_candidate = nn.Sequential(
                nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
                nn.Tanh()
            )
        elif self.config.memory_update_type == "dnc":
            # DNC-inspired memory update mechanism
            self.memory_write_gate = nn.Sequential(
                nn.Linear(self.config.hidden_size, 1),
                nn.Sigmoid()
            )
            self.memory_erase_gate = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.Sigmoid()
            )
            self.memory_write_vector = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            
            # Memory usage tracking
            self.memory_usage = nn.Parameter(torch.zeros(self.config.memory_size, 1))
            
        # Additional memory attention if enabled
        if self.config.use_memory_attention:
            self.memory_attention = nn.MultiheadAttention(
                embed_dim=self.config.hidden_size,
                num_heads=4,  # Using fewer heads for memory attention
                dropout=self.config.dropout,
                batch_first=True
            )
    
    def _cluster_rules(self):
        """Cluster rule embeddings for more efficient selection"""
        with torch.no_grad():
            # Use KMeans to cluster rule embeddings
            rule_np = self.rule_embeddings.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=self.config.num_rule_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(rule_np)
            
            # Store cluster assignments
            self.rule_clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in self.rule_clusters:
                    self.rule_clusters[label] = []
                self.rule_clusters[label].append(i)
            
            # Store cluster centroids
            self.cluster_centroids = torch.tensor(
                kmeans.cluster_centers_, 
                dtype=self.rule_embeddings.dtype,
                device=self.rule_embeddings.device
            )
    
    def select_rules(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select relevant rules based on hidden states"""
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        
        # Project hidden states to rule space
        rule_query = self.rule_scorer(hidden_states)  # [batch, seq_len, rule_emb_size]
        
        # Check rule cache first if sparse selection is enabled
        if self.config.use_sparse_rule_selection and len(self.rule_cache) > 0:
            cached_rules = list(self.rule_cache.keys())
            cached_rule_embeddings = self.rule_embeddings[cached_rules]
            
            # Compute compatibility with cached rules
            cached_scores = torch.matmul(
                rule_query, 
                cached_rule_embeddings.transpose(0, 1)
            )  # [batch, seq_len, num_cached_rules]
            
            # If cached rules have high enough scores, use only them
            max_cached_scores = cached_scores.max(dim=-1)[0]  # [batch, seq_len]
            
            if (max_cached_scores > 0.7).all():  # Threshold for using cached rules
                rule_probs = F.softmax(
                    cached_scores / self.config.softmax_temperature, 
                    dim=-1
                )
                
                # Get top-k cached rules
                top_k = min(self.config.max_rule_applications, len(cached_rules))
                top_k_probs, top_k_indices_cache = torch.topk(
                    rule_probs, k=top_k, dim=-1
                )
                
                # Convert cached indices to original rule indices
                top_k_indices = torch.zeros_like(top_k_indices_cache)
                for b in range(batch_size):
                    for s in range(seq_len):
                        for k in range(top_k):
                            cache_idx = top_k_indices_cache[b, s, k].item()
                            top_k_indices[b, s, k] = cached_rules[cache_idx]
                
                # Gather corresponding rule embeddings
                selected_rules = torch.zeros(
                    batch_size, seq_len, top_k, self.config.rule_embedding_size, 
                    device=hidden_states.device
                )
                
                for b in range(batch_size):
                    for s in range(seq_len):
                        selected_rules[b, s] = self.rule_embeddings[top_k_indices[b, s]]
                
                # Update rule usage counts
                if self.training and self.config.use_sparse_rule_selection:
                    with torch.no_grad():
                        for b in range(batch_size):
                            for s in range(seq_len):
                                for k in range(top_k):
                                    rule_idx = top_k_indices[b, s, k].item()
                                    self.rule_usage_counts[rule_idx] += 1
                                    # Update cache order
                                    if rule_idx in self.rule_cache:
                                        self.rule_cache.move_to_end(rule_idx)
                
                return selected_rules, top_k_probs
        
        # Use rule clustering if enabled
        if self.config.rule_clustering and self.rule_clusters is not None:
            # First find closest cluster centroids
            centroid_scores = torch.matmul(
                rule_query, 
                self.cluster_centroids.transpose(0, 1)
            )  # [batch, seq_len, num_clusters]
            
            # Get top clusters
            num_clusters_to_use = min(3, self.config.num_rule_clusters)  # Use top-3 clusters
            _, top_cluster_indices = torch.topk(
                centroid_scores, k=num_clusters_to_use, dim=-1
            )  # [batch, seq_len, num_clusters_to_use]
            
            # Gather rules from top clusters
            candidate_rule_indices = []
            for b in range(batch_size):
                for s in range(seq_len):
                    batch_seq_candidates = []
                    for c in range(num_clusters_to_use):
                        cluster_idx = top_cluster_indices[b, s, c].item()
                        batch_seq_candidates.extend(self.rule_clusters[cluster_idx])
                    candidate_rule_indices.append(batch_seq_candidates)
            
            # Compute scores only for rules in selected clusters
            all_rule_scores = torch.zeros(
                batch_size, seq_len, self.config.num_rules,
                device=hidden_states.device
            )
            
            for b in range(batch_size):
                for s in range(seq_len):
                    candidate_indices = candidate_rule_indices[b * seq_len + s]
                    candidate_embeddings = self.rule_embeddings[candidate_indices]
                    
                    # Compute scores for candidates
                    candidate_scores = torch.matmul(
                        rule_query[b, s].unsqueeze(0),
                        candidate_embeddings.transpose(0, 1)
                    ).squeeze(0)
                    
                    # Place scores in the full score tensor
                    all_rule_scores[b, s, candidate_indices] = candidate_scores
            
            rule_scores = all_rule_scores
        else:
            # Standard rule selection - compute compatibility with all rules
            rule_scores = torch.matmul(
                rule_query, 
                self.rule_embeddings.transpose(0, 1)
            )  # [batch, seq_len, num_rules]
        
        # Apply rule specialization if enabled
        if self.config.use_rule_specialization and self.training:
            # Create context-specific rules by specializing
            # Create specialized rule key for caching
            context_hash = hidden_states.mean(dim=1).sum(dim=-1).detach()  # [batch]
            
            for b in range(batch_size):
                context_key = context_hash[b].item()
                
                if context_key not in self.specialized_rules:
                    # Create specialized rules for this context
                    top_rule_indices = rule_scores[b].mean(dim=0).topk(
                        k=min(10, self.config.num_rules), dim=-1
                    )[1]  # Top 10 rules for this batch based on average score
                    
                    specialized_rules = []
                    for rule_idx in top_rule_indices:
                        rule_emb = self.rule_embeddings[rule_idx].unsqueeze(0)
                        context_emb = hidden_states[b].mean(dim=0).unsqueeze(0)  # Average context
                        
                        # Concatenate rule and context
                        specialization_input = torch.cat([rule_emb, context_emb], dim=-1)
                        specialized_rule = self.rule_specializer(specialization_input)
                        specialized_rules.append(specialized_rule)
                    
                    # Store specialized rules in cache
                    self.specialized_rules[context_key] = torch.cat(specialized_rules, dim=0)
                    self.specialized_rules_usage[context_key] = 1  # Track usage for cache management
                
                # Use specialized rules for scoring
                specialized_rule_embeddings = self.specialized_rules[context_key]
                specialized_scores = torch.matmul(
                    rule_query[b], 
                    specialized_rule_embeddings.transpose(0, 1)
                )  # [seq_len, num_specialized]
                
                # Replace scores for top rules with specialized scores
                top_rule_indices = rule_scores[b].mean(dim=0).topk(
                    k=min(specialized_scores.shape[1], self.config.num_rules), dim=-1
                )[1]
                
                for i, rule_idx in enumerate(top_rule_indices):
                    if i < specialized_scores.shape[1]:
                        rule_scores[b, :, rule_idx] = specialized_scores[:, i]
        
        # Apply temperature and get probabilities
        rule_probs = F.softmax(
            rule_scores / self.config.softmax_temperature, 
            dim=-1
        )
        
        # Get top-k rules if specified
        if self.config.max_rule_applications < self.config.num_rules:
            top_k_probs, top_k_indices = torch.topk(
                rule_probs, 
                k=self.config.max_rule_applications, 
                dim=-1
            )
            
            # Update rule usage counts if in training mode
            if self.training and self.config.use_sparse_rule_selection:
                with torch.no_grad():
                    for b in range(batch_size):
                        for s in range(seq_len):
                            for idx in top_k_indices[b, s]:
                                rule_idx = idx.item()
                                self.rule_usage_counts[rule_idx] += 1
                                
                                # Add to cache if not present
                                if rule_idx not in self.rule_cache:
                                    if len(self.rule_cache) >= self.config.rule_cache_size:
                                        # Remove least recently used rule
                                        self.rule_cache.popitem(last=False)
                                    self.rule_cache[rule_idx] = True
                                else:
                                    # Move to end (most recently used)
                                    self.rule_cache.move_to_end(rule_idx)
            
            # Gather corresponding rule embeddings
            selected_rules = torch.zeros(
                batch_size, seq_len, self.config.max_rule_applications, self.config.rule_embedding_size, 
                device=hidden_states.device
            )
            
            for b in range(batch_size):
                for s in range(seq_len):
                    selected_rules[b, s] = self.rule_embeddings[top_k_indices[b, s]]
            
            # Apply rule composition if enabled
            if self.config.use_rule_composition and selected_rules.shape[2] > 1:
                # Compose pairs of top rules if they haven't been composed before
                for b in range(batch_size):
                    for s in range(seq_len):
                        # Try to compose the top two rules
                        if selected_rules.shape[2] >= 2:
                            rule1_idx = top_k_indices[b, s, 0].item()
                            rule2_idx = top_k_indices[b, s, 1].item()
                            
                            # Create a unique key for this rule pair
                            rule_pair_key = f"{min(rule1_idx, rule2_idx)}_{max(rule1_idx, rule2_idx)}"
                            
                            if rule_pair_key not in self.rule_composition_cache:
                                # Compose the rules
                                rule1 = self.rule_embeddings[rule1_idx].unsqueeze(0)
                                rule2 = self.rule_embeddings[rule2_idx].unsqueeze(0)
                                
                                # Concatenate and compose
                                composition_input = torch.cat([rule1, rule2], dim=-1)
                                composed_rule = self.rule_composer(composition_input)
                                
                                # Cache the composed rule
                                self.rule_composition_cache[rule_pair_key] = composed_rule
                                self.rule_composition_usage[rule_pair_key] = 1  # Track usage for cache management
                                
                            # Replace the first rule with the composed rule
                            selected_rules[b, s, 0] = self.rule_composition_cache[rule_pair_key]
            
            return selected_rules, top_k_probs
        else:
            # Use all rules
            return self.rule_embeddings.unsqueeze(0).unsqueeze(0).expand(
                hidden_states.shape[0], hidden_states.shape[1], -1, -1
            ), rule_probs

    def apply_rules(
        self, 
        hidden_states: torch.Tensor, 
        selected_rules: torch.Tensor, 
        rule_probs: torch.Tensor
    ) -> torch.Tensor:
        """Apply selected rules to current hidden states"""
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        
        # Expand hidden states to match selected rules
        expanded_hidden = hidden_states.unsqueeze(2).expand(
            -1, -1, selected_rules.shape[2], -1
        )  # [batch, seq_len, num_selected_rules, hidden_size]
        
        # Concatenate hidden states with rules
        rule_inputs = torch.cat([
            expanded_hidden, selected_rules
        ], dim=-1)  # [batch, seq_len, num_selected_rules, hidden_size*2]
        
        # Apply rule application network
        rule_output_states = self.rule_application(rule_inputs)
        
        # Weight rule applications by their probabilities
        weighted_rule_outputs = rule_output_states * rule_probs.unsqueeze(-1)
        
        # Sum over rules
        new_hidden_states = weighted_rule_outputs.sum(dim=2)
        
        # If dynamic rule learning is enabled, learn new rules during training
        if self.training and self.config.enable_rule_learning:
            # Find cases where none of the existing rules had high probability
            max_rule_prob, _ = rule_probs.max(dim=-1)  # [batch, seq_len]
            need_new_rule = max_rule_prob < 0.3  # Threshold for needing a new rule
            
            if need_new_rule.any():
                for b in range(batch_size):
                    for s in range(seq_len):
                        if need_new_rule[b, s]:
                            # Generate a new rule embedding from the current state
                            new_rule = self.rule_learner(hidden_states[b, s].unsqueeze(0))
                            
                            # Store as a candidate for addition to rule set
                            self.candidate_rules.append(new_rule.detach())
                
                # If we have collected enough candidates, add some to the rule set
                if len(self.candidate_rules) >= 100:  # Arbitrary threshold
                    # Cluster candidate rules to find representative ones
                    candidate_stack = torch.cat(self.candidate_rules, dim=0)
                    
                    # Find which existing rules are least used
                    if self.config.use_sparse_rule_selection:
                        least_used = torch.argsort(self.rule_usage_counts)[:self.config.max_new_rules]
                        
                        # Replace least used rules with new candidate rules
                        with torch.no_grad():
                            for i, rule_idx in enumerate(least_used):
                                if i < min(len(self.candidate_rules), self.config.max_new_rules):
                                    self.rule_embeddings.data[rule_idx] = self.candidate_rules[i]
                    
                    # Clear candidate rules
                    self.candidate_rules = []
                    
                    # Recluster rules if clustering is enabled
                    if self.config.rule_clustering:
                        self._cluster_rules()
        
        return new_hidden_states
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        teacher_states: Optional[torch.Tensor] = None,
        reasoning_type: str = "symbolic"
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for symbolic reasoning
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            teacher_states: Optional tensor from teacher model for knowledge distillation
            
        Returns:
            Dictionary containing:
                - hidden_states: Updated hidden states
                - confidence_scores: Optional confidence scores for reasoning
                - uncertainty_scores: Optional uncertainty scores
                - memory: Optional updated memory state
                - distillation_loss: Optional knowledge distillation loss
        """
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        
        # Initialize return dictionary
        outputs = {"hidden_states": hidden_states}
        
        # Initial hidden states
        current_states = hidden_states
        
        # Clear temporary tensors list for this forward pass
        self.temporary_tensors = []
        
        # Track inference steps for cache cleanup
        self.inference_steps_since_cleanup += 1
        if self.inference_steps_since_cleanup >= self.cleanup_frequency:
            self._cleanup_caches()
            self.inference_steps_since_cleanup = 0
        
        # Memory retrieval if enabled
        if self.config.use_memory_for_reasoning:
            memory_query = self.memory_query(hidden_states)
            memory_attn = torch.matmul(memory_query, self.memory.transpose(0, 1))
            memory_attn = F.softmax(memory_attn, dim=-1)
            retrieved_memory = torch.matmul(memory_attn, self.memory)
            
            # Apply additional memory attention if enabled
            if self.config.use_memory_attention:
                # Use retrieved memory as keys and values, current states as queries
                if attention_mask is not None:
                    # Convert attention mask to attention bias
                    extended_attention_mask = (1.0 - attention_mask.unsqueeze(1)) * -10000.0
                    memory_attn_output, _ = self.memory_attention(
                        query=current_states,
                        key=retrieved_memory,
                        value=retrieved_memory,
                        attn_mask=extended_attention_mask
                    )
                else:
                    memory_attn_output, _ = self.memory_attention(
                        query=current_states,
                        key=retrieved_memory,
                        value=retrieved_memory
                    )
                current_states = current_states + memory_attn_output
            else:
                # Simple addition
                current_states = current_states + retrieved_memory
        
        # Apply Monte Carlo dropout for uncertainty estimation if enabled
        if self.config.uncertainty_estimation and self.config.use_monte_carlo_dropout and self.training:
            # Create multiple dropout masks for ensemble
            num_samples = 5  # Number of Monte Carlo samples
            dropout_layer = nn.Dropout(p=0.1)
            
            mc_states = []
            for _ in range(num_samples):
                mc_states.append(dropout_layer(current_states))
            
            # Store original states for later
            original_states = current_states
            
            # Process each sample separately and compute variance later
            mc_outputs = []
            for sample_states in mc_states:
                # Continue with normal processing using this sample
                current_states = sample_states
        
        # Iterative reasoning steps
        reasoning_states = []  # Store states from each step for knowledge distillation
        rule_selection_contexts = []  # Store contexts for specialized rules
        for step in range(self.config.inference_steps):
            # Rule selection
            selected_rules, rule_probs = self.select_rules(current_states)
            
            # Rule application
            new_states = self.apply_rules(current_states, selected_rules, rule_probs)
            
            # Cache contexts for rule specialization to avoid memory buildup
            if self.config.use_rule_specialization and step == 0 and hasattr(self, 'specialized_rules'):
                # Store a representation of the context to avoid memory leaks
                context_hash = hidden_states.mean(dim=1).sum(dim=-1).detach()
                rule_selection_contexts.extend(context_hash.tolist())
                
                # Update usage counts for specialized rules
                for b in range(batch_size):
                    context_key = context_hash[b].item()
                    if context_key in self.specialized_rules_usage:
                        self.specialized_rules_usage[context_key] += 1
            
            # Optional attention-based reasoning
            if self.config.use_attention_for_reasoning:
                if attention_mask is not None:
                    # Convert attention mask to attention bias
                    extended_attention_mask = (1.0 - attention_mask.unsqueeze(1)) * -10000.0
                    attn_output, _ = self.attention(
                        query=new_states,
                        key=new_states,
                        value=new_states,
                        attn_mask=extended_attention_mask
                    )
                else:
                    attn_output, _ = self.attention(
                        query=new_states,
                        key=new_states,
                        value=new_states
                    )
                new_states = new_states + attn_output
            
            # Update current states
            if self.config.use_gradient_through_inference:
                current_states = new_states
            else:
                current_states = new_states.detach()
                
            # Store states for knowledge distillation
            if self.config.use_symbolic_knowledge_distillation and teacher_states is not None:
                reasoning_states.append(current_states.clone())
            
            # Clear intermediate tensors to save memory
            if not self.training:
                del selected_rules, rule_probs, new_states
                if self.config.use_attention_for_reasoning:
                    del attn_output
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Process Monte Carlo samples if enabled
        if self.config.uncertainty_estimation and self.config.use_monte_carlo_dropout and self.training:
            # Restore original processing
            current_states = original_states
            
            # Compute mean and variance across MC samples
            mc_tensor = torch.stack(mc_outputs, dim=0)
            mc_mean = mc_tensor.mean(dim=0)
            mc_var = mc_tensor.var(dim=0)
            
            # Use mean for output and store variance for uncertainty
            current_states = mc_mean
            outputs["uncertainty_scores"] = mc_var.mean(dim=-1)  # Average variance across hidden dims
            
            # Clean up MC tensor
            del mc_tensor, mc_mean, mc_var, mc_outputs, mc_states
        
        # Optional confidence scoring
        if self.config.use_confidence_scores:
            if self.config.uncertainty_estimation:
                # Output both mean and variance for uncertainty estimation
                confidence_output = self.confidence_scorer(current_states)
                confidence_mean = confidence_output[..., 0]
                confidence_var = torch.exp(confidence_output[..., 1])  # Log variance to avoid negative values
                
                confidence = torch.sigmoid(confidence_mean)
                outputs["confidence_scores"] = confidence
                outputs["uncertainty_scores"] = confidence_var
            else:
                confidence = torch.sigmoid(self.confidence_scorer(current_states))
                outputs["confidence_scores"] = confidence
        
        # Memory update if enabled
        if self.config.use_memory_for_reasoning:
            # Prepare keys and values for memory update
            memory_keys = self.memory_key(current_states)
            memory_values = self.memory_value(current_states)
            
            # Attention over current memory
            memory_update_attn = torch.matmul(memory_keys, self.memory.transpose(0, 1))
            memory_update_attn = F.softmax(memory_update_attn, dim=-1)
            
            # Different update mechanisms based on configuration
            if self.config.memory_update_type == "momentum":
                # Simple momentum update (as in original implementation)
                with torch.no_grad():
                    batch_updates = torch.matmul(
                        memory_update_attn.transpose(1, 2), 
                        memory_values
                    )
                    # Average updates across batch
                    memory_update = batch_updates.mean(dim=0)
                    # Update memory with momentum
                    self.memory.data = self.memory.data * (1 - self.config.memory_update_factor) + memory_update * self.config.memory_update_factor
                
            elif self.config.memory_update_type == "gated":
                # Gated update mechanism
                with torch.no_grad():
                    # Current memory expanded to batch shape for gate computation
                    expanded_memory = self.memory.unsqueeze(0).expand(
                        batch_size, -1, -1
                    )  # [batch, memory_size, hidden_size]
                    
                    # Compute weighted values to update memory
                    batch_updates = torch.matmul(
                        memory_update_attn.transpose(1, 2), 
                        memory_values
                    )  # [batch, memory_size, hidden_size]
                    
                    # Average updates across batch
                    memory_update = batch_updates.mean(dim=0)  # [memory_size, hidden_size]
                    
                    # Compute update gate
                    gate_input = torch.cat([
                        self.memory,  # [memory_size, hidden_size]
                        memory_update  # [memory_size, hidden_size]
                    ], dim=-1)
                    update_gate = self.memory_update_gate(gate_input)
                    
                    # Compute candidate memory values
                    candidate_input = torch.cat([
                        self.memory,  # [memory_size, hidden_size]
                        memory_update  # [memory_size, hidden_size]
                    ], dim=-1)
                    candidate_memory = self.memory_candidate(candidate_input)
                    
                    # Apply gate to update memory
                    self.memory.data = (1 - update_gate) * self.memory.data + update_gate * candidate_memory
                    
                    # Clean up intermediates
                    del expanded_memory, batch_updates, memory_update, gate_input, update_gate, candidate_input, candidate_memory
                    
            elif self.config.memory_update_type == "dnc":
                # DNC-inspired memory update
                with torch.no_grad():
                    # Compute weighted values to update memory
                    batch_updates = torch.matmul(
                        memory_update_attn.transpose(1, 2), 
                        memory_values
                    )  # [batch, memory_size, hidden_size]
                    
                    # Average updates across batch
                    memory_update = batch_updates.mean(dim=0)  # [memory_size, hidden_size]
                    
                    # Compute write gate (whether to write to memory)
                    write_gate = self.memory_write_gate(memory_update)  # [memory_size, 1]
                    
                    # Compute erase vector (what to erase from memory)
                    erase_vector = self.memory_erase_gate(memory_update)  # [memory_size, hidden_size]
                    
                    # Compute write vector (what to write to memory)
                    write_vector = self.memory_write_vector(memory_update)  # [memory_size, hidden_size]
                    
                    # Update memory: erase then write
                    self.memory.data = self.memory.data * (1 - write_gate * erase_vector)
                    self.memory.data = self.memory.data + write_gate * write_vector
                    
                    # Update memory usage
                    self.memory_usage.data = self.memory_usage.data + write_gate * 0.1
                    
                    # Decay memory usage for less recently used slots
                    self.memory_usage.data = self.memory_usage.data * 0.99
                    
                    # Clean up intermediates
                    del batch_updates, memory_update, write_gate, erase_vector, write_vector
            
            outputs["memory"] = self.memory
            
            # Clean up memory intermediates
            del memory_keys, memory_values, memory_update_attn
        
        # Knowledge distillation if enabled and teacher states are provided
        if self.config.use_symbolic_knowledge_distillation and teacher_states is not None:
            # Compute distillation loss
            teacher_proj = self.teacher_projection(teacher_states)
            
            distillation_loss = 0.0
            for step_idx, step_states in enumerate(reasoning_states):
                student_proj = self.student_projection(step_states)
                
                # Compute similarity in embedding space
                sim_teacher = torch.matmul(teacher_proj, teacher_proj.transpose(-1, -2))
                sim_student = torch.matmul(student_proj, student_proj.transpose(-1, -2))
                
                # Normalize similarities
                sim_teacher = sim_teacher / self.config.distillation_temperature
                sim_student = sim_student / self.config.distillation_temperature
                
                # Convert to probabilities
                prob_teacher = F.softmax(sim_teacher, dim=-1)
                log_prob_student = F.log_softmax(sim_student, dim=-1)
                
                # Compute KL divergence
                step_loss = F.kl_div(log_prob_student, prob_teacher, reduction='batchmean')
                distillation_loss += step_loss
                
                # Clean up step states
                del step_states, student_proj, sim_teacher, sim_student, prob_teacher, log_prob_student
            
            # Average loss across steps
            distillation_loss = distillation_loss / max(len(reasoning_states), 1)
            outputs["distillation_loss"] = distillation_loss
            
            # Clean up distillation intermediates
            del teacher_proj, reasoning_states
            
        # Final hidden states
        outputs["hidden_states"] = current_states
        
        # Clean up any remaining large tensors
        if hasattr(self, 'specialized_rules') and len(rule_selection_contexts) > 0:
            # Remove old contexts that weren't used in this batch
            if len(self.specialized_rules) > 100:  # Only clean up if we have many contexts
                used_contexts = set(rule_selection_contexts)
                all_contexts = set(self.specialized_rules.keys())
                unused_contexts = all_contexts - used_contexts
                
                # Keep some unused contexts (to avoid repeated creation)
                contexts_to_remove = list(unused_contexts)[:-20] if len(unused_contexts) > 20 else []
                
                for context_key in contexts_to_remove:
                    if context_key in self.specialized_rules:
                        del self.specialized_rules[context_key]
                    if context_key in self.specialized_rules_usage:
                        del self.specialized_rules_usage[context_key]
        
        # Force garbage collection if memory usage is high
        if not self.training and self.inference_steps_since_cleanup >= self.cleanup_frequency // 2:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Log memory usage at debug level
        if logger.isEnabledFor(logging.DEBUG) and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            logger.debug(f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
        
        return outputs

    def _cleanup_caches(self):
        """Clean up rule caches to prevent memory growth"""
        # Clean specialized rules cache if it's too large
        if hasattr(self, 'specialized_rules') and len(self.specialized_rules) > self.max_specialized_cache_size:
            # Sort by usage count and keep only the most used rules
            if hasattr(self, 'specialized_rules_usage'):
                # Remove least used items
                keys_to_remove = []
                if len(self.specialized_rules_usage) > 0:
                    sorted_keys = sorted(self.specialized_rules_usage.items(), 
                                       key=lambda x: x[1], reverse=True)[self.max_specialized_cache_size:]
                    keys_to_remove = [k for k, _ in sorted_keys]
                    
                    # Remove the least used rules
                    for key in keys_to_remove:
                        if key in self.specialized_rules:
                            del self.specialized_rules[key]
                        if key in self.specialized_rules_usage:
                            del self.specialized_rules_usage[key]
                else:
                    # If no usage stats, remove random items
                    keys = list(self.specialized_rules.keys())
                    keys_to_remove = keys[self.max_specialized_cache_size:]
                    for key in keys_to_remove:
                        del self.specialized_rules[key]
            
        # Clean composition cache if it's too large
        if hasattr(self, 'rule_composition_cache') and len(self.rule_composition_cache) > self.max_composition_cache_size:
            # Sort by usage count and keep only the most used compositions
            if hasattr(self, 'rule_composition_usage'):
                # Remove least used items
                keys_to_remove = []
                if len(self.rule_composition_usage) > 0:
                    sorted_keys = sorted(self.rule_composition_usage.items(), 
                                       key=lambda x: x[1], reverse=True)[self.max_composition_cache_size:]
                    keys_to_remove = [k for k, _ in sorted_keys]
                    
                    # Remove the least used rule compositions
                    for key in keys_to_remove:
                        if key in self.rule_composition_cache:
                            del self.rule_composition_cache[key]
                        if key in self.rule_composition_usage:
                            del self.rule_composition_usage[key]
                else:
                    # If no usage stats, remove random items
                    keys = list(self.rule_composition_cache.keys())
                    keys_to_remove = keys[self.max_composition_cache_size:]
                    for key in keys_to_remove:
                        del self.rule_composition_cache[key]
        
        # Clear any leftover tensors 
        for tensor in self.temporary_tensors:
            if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.detach_()
        self.temporary_tensors = []
        
        # Clear Monte Carlo samples if we've accumulated too many
        if hasattr(self, 'monte_carlo_samples') and len(self.monte_carlo_samples) > 20:
            # Keep only the most recent samples
            self.monte_carlo_samples = self.monte_carlo_samples[-10:]
            self.monte_carlo_sample_count = len(self.monte_carlo_samples)
        
        # Force garbage collection if memory usage is high
        if not self.training and self.inference_steps_since_cleanup >= self.cleanup_frequency // 2:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Log memory usage at debug level
        if logger.isEnabledFor(logging.DEBUG) and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            logger.debug(f"GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")

class NeuralSymbolicIntegration(nn.Module):
    """Neural Symbolic Reasoning integration module"""
    
    def __init__(self, config: Union[Dict, NeuralSymbolicConfig]):
        super().__init__()
        
        # Convert dict to config if needed
        if isinstance(config, dict):
            self.config = NeuralSymbolicConfig(**config)
        else:
            self.config = config
            
        # Core symbolic reasoning layer
        self.symbolic_layer = SymbolicReasoningLayer(self.config)
        
        # Integration layers
        self.input_projection = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.output_projection = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        
        # Layer normalization
        self.layer_norm_1 = nn.LayerNorm(self.config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(self.config.hidden_size)
        
        # Gradient stabilization
        self.gradient_stabilizer = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )
        
        # Enhanced verification if enabled
        if self.config.use_verified_reasoning:
            if self.config.use_verification:
                # More sophisticated verification using ProofGenerator
                if EXTERNAL_MODULES_AVAILABLE:
                    from model.verifiable_computation import ProofGenerator, VerifiableComputationConfig
                    verification_config = VerifiableComputationConfig(
                        hidden_size=self.config.hidden_size,
                        num_heads=self.config.num_attention_heads,
                        dropout=self.config.dropout
                    )
                    self.verification_layer = ProofGenerator(verification_config)
                else:
                    # Fallback to simpler verification if ProofGenerator not available
                    self.verification_layer = nn.Sequential(
                        nn.Linear(self.config.hidden_size, self.config.hidden_size),
                        nn.LayerNorm(self.config.hidden_size),
                        nn.GELU(),
                        nn.Dropout(self.config.dropout),
                        nn.Linear(self.config.hidden_size, 1)
                    )
            else:
                # Basic verification layer (original implementation)
                self.verification_layer = nn.Sequential(
                    nn.Linear(self.config.hidden_size, self.config.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.config.hidden_size, 1)
                )
        
        # Knowledge reasoning integration if enabled
        if self.config.use_knowledge_reasoner and EXTERNAL_MODULES_AVAILABLE:
            try:
                from model.knowledge_reasoner import KnowledgeReasoningModule, KnowledgeReasoningConfig
                knowledge_config = KnowledgeReasoningConfig(
                    hidden_size=self.config.hidden_size,
                    knowledge_size=self.config.hidden_size,
                    num_attention_heads=self.config.num_attention_heads,
                    dropout=self.config.dropout
                )
                self.knowledge_reasoner = KnowledgeReasoningModule(knowledge_config)
            except ImportError:
                logger.warning("KnowledgeReasoningModule not available. Disabling knowledge reasoner integration.")
                self.config.use_knowledge_reasoner = False
        
        # MCTS reasoning integration if enabled
        if self.config.use_mcts_reasoning and EXTERNAL_MODULES_AVAILABLE:
            try:
                from model.mcts_reasoning import MCTSEnhancedTreeReasoningModule, MCTSConfig
                mcts_config = MCTSConfig(
                    hidden_size=self.config.hidden_size,
                    dropout=self.config.dropout
                )
                self.mcts_reasoner = MCTSEnhancedTreeReasoningModule(mcts_config)
            except ImportError:
                logger.warning("MCTSEnhancedTreeReasoningModule not available. Disabling MCTS reasoning integration.")
                self.config.use_mcts_reasoning = False
        
        # Reasoning evaluator integration if enabled
        if self.config.use_reasoning_evaluator and EXTERNAL_MODULES_AVAILABLE:
            try:
                from evaluation.init import LogicalReasoningEvaluator
                self.reasoning_evaluator = LogicalReasoningEvaluator()
            except ImportError:
                logger.warning("LogicalReasoningEvaluator not available. Disabling reasoning evaluator integration.")
                self.config.use_reasoning_evaluator = False
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reasoning_type: str = "symbolic",
        entity_ids: Optional[torch.Tensor] = None,
        relation_ids: Optional[torch.Tensor] = None,
        teacher_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply neural symbolic reasoning
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            reasoning_type: Type of reasoning to apply ('symbolic', 'neural', 'integrated', 'knowledge', 'mcts')
            entity_ids: Optional entity IDs for knowledge reasoning
            relation_ids: Optional relation IDs for knowledge reasoning
            teacher_states: Optional tensor from teacher model for knowledge distillation
            
        Returns:
            Dictionary containing:
                - hidden_states: Updated hidden states
                - confidence_scores: Optional confidence scores for reasoning
                - verification_scores: Optional verification scores
                - evaluation_metrics: Optional reasoning evaluation metrics
                - distillation_loss: Optional knowledge distillation loss
        """
        # Initialize return dict
        outputs = {"hidden_states": hidden_states}
        
        # Skip if not using symbolic reasoning
        if reasoning_type not in ["symbolic", "integrated", "knowledge", "mcts"]:
            return outputs
        
        # Apply knowledge reasoning if requested
        if reasoning_type == "knowledge" and self.config.use_knowledge_reasoner:
            if hasattr(self, 'knowledge_reasoner'):
                knowledge_outputs = self.knowledge_reasoner(
                    hidden_states,
                    attention_mask=attention_mask,
                    entity_ids=entity_ids,
                    relation_ids=relation_ids
                )
                return knowledge_outputs
            else:
                logger.warning("Knowledge reasoning requested but not available. Falling back to symbolic.")
        
        # Apply MCTS reasoning if requested
        if reasoning_type == "mcts" and self.config.use_mcts_reasoning:
            if hasattr(self, 'mcts_reasoner'):
                mcts_outputs = self.mcts_reasoner(
                    hidden_states,
                    attention_mask=attention_mask,
                    reasoning_type="mcts"
                )
                return mcts_outputs
            else:
                logger.warning("MCTS reasoning requested but not available. Falling back to symbolic.")
        
        # First layer norm and projection
        norm_states = self.layer_norm_1(hidden_states)
        projected_states = self.input_projection(norm_states)
        
        # Apply symbolic reasoning
        symbolic_outputs = self.symbolic_layer(
            projected_states, 
            attention_mask=attention_mask,
            teacher_states=teacher_states
        )
        
        # Process symbolic outputs
        symbolic_states = symbolic_outputs["hidden_states"]
        
        # Apply gradient stabilization
        stabilized_states = symbolic_states + self.gradient_stabilizer(symbolic_states)
        
        # Second layer norm and projection
        final_states = self.layer_norm_2(stabilized_states)
        output_states = self.output_projection(final_states)
        
        # Residual connection
        output_states = output_states + hidden_states
        
        # Verification if enabled
        if self.config.use_verified_reasoning:
            if self.config.use_verification and hasattr(self, 'verification_layer') and \
               isinstance(self.verification_layer, nn.Module) and not isinstance(self.verification_layer, nn.Sequential):
                # Use ProofGenerator for verification
                verification_outputs = self.verification_layer(output_states)
                verification_scores = verification_outputs["verification_scores"]
                outputs["verification_scores"] = verification_scores
                
                if "proof_steps" in verification_outputs:
                    outputs["proof_steps"] = verification_outputs["proof_steps"]
            else:
                # Use basic verification layer
                verification_scores = torch.sigmoid(self.verification_layer(output_states))
                outputs["verification_scores"] = verification_scores
        
        # Pass through other outputs from symbolic layer
        for key, value in symbolic_outputs.items():
            if key != "hidden_states" and key not in outputs:
                outputs[key] = value
        
        # Reasoning evaluator if enabled
        if self.config.use_reasoning_evaluator and self.training and hasattr(self, 'reasoning_evaluator'):
            # This would require text extraction from hidden states for evaluation
            # For simplicity, we're just showing the integration point here
            # In a real implementation, you would need to:
            # 1. Extract text representations from hidden states
            # 2. Identify premises, conclusion, and steps
            # 3. Call the evaluator with these
            
            # Placeholder for evaluation metrics
            batch_size = hidden_states.shape[0]
            metrics = {
                "logical_consistency": torch.rand(batch_size),
                "argument_strength": torch.rand(batch_size),
                "conclusion_validity": torch.rand(batch_size),
                "inference_accuracy": torch.rand(batch_size)
            }
            outputs["evaluation_metrics"] = metrics
        
        # Final outputs
        outputs["hidden_states"] = output_states
        
        return outputs

    def get_rule_importance(self) -> torch.Tensor:
        """
        Get the importance of each rule based on usage statistics
        
        Returns:
            Tensor of shape [num_rules] with importance scores
        """
        if hasattr(self.symbolic_layer, 'rule_usage_counts'):
            # Normalize usage counts to get importance
            total_usage = self.symbolic_layer.rule_usage_counts.sum()
            if total_usage > 0:
                return self.symbolic_layer.rule_usage_counts / total_usage
            else:
                return torch.zeros_like(self.symbolic_layer.rule_usage_counts)
        else:
            # If usage counts not available, return uniform importance
            return torch.ones(self.config.num_rules) / self.config.num_rules
    
    def get_rule_similarity_matrix(self) -> torch.Tensor:
        """
        Compute similarity matrix between rules
        
        Returns:
            Tensor of shape [num_rules, num_rules] with cosine similarities
        """
        # Normalize rule embeddings for cosine similarity
        rule_embs = F.normalize(self.symbolic_layer.rule_embeddings, p=2, dim=1)
        return torch.matmul(rule_embs, rule_embs.transpose(0, 1))
    
    def visualize_rules(self, output_path: str = None) -> Dict:
        """
        Visualize rules and their relationships
        
        Args:
            output_path: Optional path to save visualization
            
        Returns:
            Dictionary with rule visualization data
        """
        # Get rule similarities
        rule_sim = self.get_rule_similarity_matrix().detach().cpu().numpy()
        
        # Get rule importance
        rule_importance = self.get_rule_importance().detach().cpu().numpy()
        
        # Create visualization data
        viz_data = {
            "rule_similarity": rule_sim.tolist(),
            "rule_importance": rule_importance.tolist()
        }
        
        # Add rule templates if available
        if hasattr(self.symbolic_layer, 'rule_templates'):
            viz_data["rule_templates"] = self.symbolic_layer.rule_templates
        
        # Add rule clusters if available
        if hasattr(self.symbolic_layer, 'rule_clusters'):
            cluster_data = {}
            for cluster_id, rules in self.symbolic_layer.rule_clusters.items():
                cluster_data[str(cluster_id)] = rules
            viz_data["rule_clusters"] = cluster_data
        
        # Save to file if path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(viz_data, f, indent=2)
        
        return viz_data
    
    def prune_unused_rules(self, threshold: float = 0.01) -> int:
        """
        Prune rules that are used less than the threshold
        
        Args:
            threshold: Usage threshold below which to prune rules
            
        Returns:
            Number of rules pruned
        """
        if not hasattr(self.symbolic_layer, 'rule_usage_counts'):
            return 0
        
        # Get normalized usage
        total_usage = self.symbolic_layer.rule_usage_counts.sum()
        if total_usage == 0:
            return 0
            
        normalized_usage = self.symbolic_layer.rule_usage_counts / total_usage
        
        # Identify rules to prune
        to_prune = normalized_usage < threshold
        num_to_prune = to_prune.sum().item()
        
        if num_to_prune == 0:
            return 0
        
        # Create new random rule embeddings for pruned positions
        with torch.no_grad():
            new_rules = torch.randn(
                num_to_prune, 
                self.config.rule_embedding_size,
                device=self.symbolic_layer.rule_embeddings.device
            )
            nn.init.normal_(new_rules, mean=0.0, std=0.02)
            
            # Replace pruned rules with new ones
            self.symbolic_layer.rule_embeddings[to_prune] = new_rules
            
            # Reset usage counts for pruned rules
            self.symbolic_layer.rule_usage_counts[to_prune] = 0
            
            # Recluster if clustering is enabled
            if self.config.rule_clustering:
                self.symbolic_layer._cluster_rules()
                
        return num_to_prune
    
    def export_rules(self, output_path: str) -> bool:
        """
        Export rule embeddings to a file
        
        Args:
            output_path: Path to save exported rules
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save rule embeddings
            torch.save(self.symbolic_layer.rule_embeddings.detach().cpu(), output_path)
            
            # If templates are available, save those too
            if hasattr(self.symbolic_layer, 'rule_templates'):
                import json
                template_path = output_path + '.templates.json'
                with open(template_path, 'w') as f:
                    json.dump(self.symbolic_layer.rule_templates, f, indent=2)
                    
            return True
        except Exception as e:
            logger.error(f"Failed to export rules: {e}")
            return False
    
    def import_rules(self, input_path: str) -> bool:
        """
        Import rule embeddings from a file
        
        Args:
            input_path: Path to load rules from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load rule embeddings
            imported_rules = torch.load(input_path)
            
            # Check if dimensions match
            if imported_rules.shape[1] != self.config.rule_embedding_size:
                logger.error(f"Rule embedding size mismatch: {imported_rules.shape[1]} vs {self.config.rule_embedding_size}")
                return False
                
            # Resize if number of rules doesn't match
            if imported_rules.shape[0] != self.config.num_rules:
                logger.warning(f"Number of rules mismatch: {imported_rules.shape[0]} vs {self.config.num_rules}. Resizing.")
                
                if imported_rules.shape[0] > self.config.num_rules:
                    # Take subset
                    imported_rules = imported_rules[:self.config.num_rules]
                else:
                    # Pad with random
                    padding = torch.randn(
                        self.config.num_rules - imported_rules.shape[0],
                        self.config.rule_embedding_size
                    )
                    nn.init.normal_(padding, mean=0.0, std=0.02)
                    imported_rules = torch.cat([imported_rules, padding], dim=0)
            
            # Update rule embeddings
            with torch.no_grad():
                self.symbolic_layer.rule_embeddings.copy_(
                    imported_rules.to(self.symbolic_layer.rule_embeddings.device)
                )
                
                # Reset usage counts
                if hasattr(self.symbolic_layer, 'rule_usage_counts'):
                    self.symbolic_layer.rule_usage_counts.zero_()
                    
                # Recluster if clustering is enabled
                if self.config.rule_clustering:
                    self.symbolic_layer._cluster_rules()
                    
            # Try to load templates if available
            template_path = input_path + '.templates.json'
            if os.path.exists(template_path):
                try:
                    import json
                    with open(template_path, 'r') as f:
                        self.symbolic_layer.rule_templates = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load rule templates: {e}")
                    
            return True
        except Exception as e:
            logger.error(f"Failed to import rules: {e}")
            return False 