import torch
import torch.nn as nn
import torch.nn.functional as F
from model.tree_lstm import TreeLSTM
from model.memory_bank import MemoryBank
from model.attention import SwiGLU
from typing import Optional, Dict, Union, Tuple, List, Any

class EnhancedLogicalReasingModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Tree-structured reasoning
        self.tree_lstm = TreeLSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size
        )
        
        # Reasoning components
        self.premise_encoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.hypothesis_encoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.reasoning_gate = nn.Linear(config.hidden_size * 3, 3)  # [entail, contradict, neutral]
        
        # Knowledge integration
        self.knowledge_bank = MemoryBank(
            memory_size=config.memory_size,
            hidden_size=config.hidden_size,
            num_heads=4
        )
        
        # Symbolic reasoning components
        self.rule_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            SwiGLU(config)
        )
        
    def forward(self, x, tree_structure=None, knowledge_context=None):
        # Tree-structured reasoning
        tree_encoded = self.tree_lstm(x, tree_structure)
        
        # Knowledge integration
        knowledge = self.knowledge_bank(x, knowledge_context)
        
        # Multi-hop reasoning
        premises = self.premise_encoder(x)
        hypothesis = self.hypothesis_encoder(tree_encoded)
        
        # Combine reasoning paths
        reasoning_vector = torch.cat([
            premises,
            hypothesis,
            knowledge
        ], dim=-1)
        
        # Gated reasoning path selection
        reasoning_weights = F.softmax(self.reasoning_gate(reasoning_vector), dim=-1)
        
        # Apply symbolic rules
        rule_based = self.rule_encoder(reasoning_vector)
        
        # Combine different reasoning strategies
        output = (reasoning_weights.unsqueeze(-1) * 
                 torch.stack([premises, hypothesis, rule_based], dim=1)).sum(dim=1)
        
        return output 

class LogicalReasoningLayer(nn.Module):
    """
    A neural network layer for logical reasoning.
    Implements symbolic reasoning capabilities within a neural architecture.
    Compatible with the hybrid RWKV-Transformer model.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_operations: int = 6,  # AND, OR, NOT, IMPLIES, IFF, XOR
        num_heads: int = 4,
        dropout: float = 0.1,
        use_gating: bool = True,
        use_recursive_reasoning: bool = True,
        max_recursive_depth: int = 3,
        reasoning_dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_operations = num_operations
        self.num_heads = num_heads
        self.use_gating = use_gating
        self.use_recursive_reasoning = use_recursive_reasoning
        self.max_recursive_depth = max_recursive_depth
        
        # Operation embeddings
        self.operation_embeddings = nn.Parameter(torch.randn(num_operations, hidden_size))
        
        # Logical operation neural modules
        self.logical_operations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_operations)
        ])
        
        # Operation selection
        self.operation_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_operations)
        )
        
        # For detecting logical components
        self.proposition_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Multi-head attention for proposition relationships
        self.proposition_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Recursive reasoning modules
        if use_recursive_reasoning:
            self.recursive_projector = nn.Linear(hidden_size, hidden_size)
            self.recursive_aggregator = nn.Linear(hidden_size * 2, hidden_size)
            self.recursive_depth_embeddings = nn.Parameter(
                torch.randn(max_recursive_depth, hidden_size)
            )
            self.recursive_dropout = nn.Dropout(reasoning_dropout)
        
        # Gating mechanism
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Consistency check for logical validity
        self.consistency_checker = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_reasoning_trace: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Apply logical reasoning to the input.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            return_reasoning_trace: Whether to return the reasoning trace
            
        Returns:
            Enhanced hidden states with logical reasoning applied
            Optionally, a dictionary with the reasoning trace
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Detect propositions
        proposition_scores = self.proposition_detector(hidden_states)  # [batch_size, seq_len, 1]
        
        # Apply attention over propositions, weighting by proposition scores
        attention_weights = proposition_scores * (attention_mask.unsqueeze(-1) if attention_mask is not None else 1.0)
        proposition_representation, _ = self.proposition_attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Select logical operations
        operation_logits = self.operation_selector(hidden_states)  # [batch_size, seq_len, num_operations]
        operation_weights = F.softmax(operation_logits, dim=-1)
        
        # Initialize output and reasoning trace
        enhanced_states = hidden_states.clone()
        reasoning_trace = {
            "proposition_scores": proposition_scores,
            "operation_weights": operation_weights,
            "logical_consistency": [],
            "recursive_steps": []
        } if return_reasoning_trace else None
        
        # Apply logical operations
        for op_idx in range(self.num_operations):
            # Operation embedding
            op_embedding = self.operation_embeddings[op_idx].unsqueeze(0).unsqueeze(0)
            op_embedding = op_embedding.expand(batch_size, seq_len, -1)
            
            # Combine tokens with operation
            token_pairs = torch.cat([hidden_states, op_embedding], dim=-1)
            
            # Apply operation
            op_result = self.logical_operations[op_idx](token_pairs)
            
            # Weight by operation selection probability
            op_weight = operation_weights[:, :, op_idx].unsqueeze(-1)
            enhanced_states = enhanced_states + op_weight * op_result
        
        # Apply recursive reasoning if enabled
        if self.use_recursive_reasoning:
            recursive_states = enhanced_states
            
            for depth in range(self.max_recursive_depth):
                # Depth embedding
                depth_embedding = self.recursive_depth_embeddings[depth].unsqueeze(0).unsqueeze(0)
                depth_embedding = depth_embedding.expand(batch_size, seq_len, -1)
                
                # Project current state
                projected = self.recursive_projector(recursive_states)
                
                # Apply attention to find dependencies
                attended, _ = self.proposition_attention(
                    projected, projected, projected,
                    key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
                )
                
                # Combine with depth embedding
                combined = torch.cat([attended, depth_embedding], dim=-1)
                
                # Aggregate and apply dropout
                recursive_states = self.recursive_aggregator(combined)
                recursive_states = self.recursive_dropout(recursive_states)
                
                # Add to output with diminishing contribution
                factor = 1.0 / (2 ** (depth + 1))
                enhanced_states = enhanced_states + factor * recursive_states
                
                # Track for reasoning trace
                if return_reasoning_trace:
                    reasoning_trace["recursive_steps"].append({
                        "depth": depth,
                        "contribution_factor": factor
                    })
        
        # Apply gating if enabled
        if self.use_gating:
            # Compute gate value
            gate_value = self.gate(torch.cat([hidden_states, enhanced_states], dim=-1))
            
            # Apply gate
            enhanced_states = hidden_states + gate_value * (enhanced_states - hidden_states)
        
        # Apply output layer
        enhanced_states = self.output_layer(enhanced_states)
        
        # Check logical consistency
        if return_reasoning_trace:
            consistency_scores = self.consistency_checker(enhanced_states)
            reasoning_trace["logical_consistency"] = consistency_scores
        
        if return_reasoning_trace:
            return enhanced_states, reasoning_trace
        
        return enhanced_states 