import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RecursiveReasoningConfig:
    """Configuration for recursive reasoning transformer"""
    hidden_size: int = 768
    num_heads: int = 8
    dropout: float = 0.1
    max_recursion_depth: int = 5
    use_adaptive_depth: bool = True
    early_stopping_threshold: float = 0.9
    use_intermediate_supervision: bool = True
    use_recursive_attention: bool = True
    use_recursive_memory: bool = True
    memory_size: int = 64
    use_recursive_gating: bool = True
    use_recursive_routing: bool = True
    num_reasoning_experts: int = 4
    use_recursive_verification: bool = True
    verification_threshold: float = 0.7
    use_recursive_composition: bool = True
    composition_depth: int = 2

class RecursiveAttention(nn.Module):
    """Attention mechanism that can attend to its own outputs recursively"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        # Query, key, value projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Recursive projections
        self.recursive_q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.recursive_k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.recursive_v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Gating mechanism
        self.recursive_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states, past_states=None, recursion_level=0):
        """
        Apply recursive attention
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            past_states: Optional list of previous recursive states
            recursion_level: Current recursion level
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Standard self-attention
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(context)
        
        # Apply recursive attention if enabled and past states are available
        if self.config.use_recursive_attention and past_states is not None and recursion_level > 0:
            # Project current output for recursive attention
            recursive_q = self.recursive_q_proj(output)
            recursive_q = recursive_q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Process past states
            recursive_outputs = []
            recursive_weights = []
            
            for past_state in past_states:
                # Project past state
                recursive_k = self.recursive_k_proj(past_state)
                recursive_v = self.recursive_v_proj(past_state)
                
                # Reshape for multi-head attention
                recursive_k = recursive_k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                recursive_v = recursive_v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                
                # Compute recursive attention scores
                recursive_scores = torch.matmul(recursive_q, recursive_k.transpose(-1, -2)) / math.sqrt(self.head_dim)
                
                # Apply softmax and dropout
                recursive_attn_weights = F.softmax(recursive_scores, dim=-1)
                recursive_attn_weights = self.dropout(recursive_attn_weights)
                
                # Apply attention to values
                recursive_context = torch.matmul(recursive_attn_weights, recursive_v)
                
                # Reshape
                recursive_context = recursive_context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
                
                # Store
                recursive_outputs.append(recursive_context)
                recursive_weights.append(recursive_attn_weights)
            
            # Combine recursive outputs if any
            if recursive_outputs:
                # Average recursive outputs
                recursive_output = torch.stack(recursive_outputs).mean(0)
                
                # Compute gate
                gate_input = torch.cat([output, recursive_output], dim=-1)
                gate = self.recursive_gate(gate_input)
                
                # Apply gate
                output = output * (1 - gate) + recursive_output * gate
        
        return output, attn_weights

class RecursiveMemory(nn.Module):
    """Memory module that maintains state across recursive steps"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Memory slots
        self.memory_size = config.memory_size
        self.memory = nn.Parameter(torch.randn(1, config.memory_size, config.hidden_size))
        nn.init.normal_(self.memory, mean=0.0, std=0.02)
        
        # Memory controllers
        self.read_controller = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.write_controller = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Memory update gate
        self.update_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_states, recursion_level=0):
        """
        Apply recursive memory
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            recursion_level: Current recursion level
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            updated_memory: [batch_size, memory_size, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Expand memory to batch size
        memory = self.memory.expand(batch_size, -1, -1)
        
        # Read from memory
        read_output, _ = self.read_controller(
            query=hidden_states,
            key=memory,
            value=memory
        )
        
        # Write to memory
        write_output, _ = self.write_controller(
            query=memory,
            key=hidden_states,
            value=hidden_states
        )
        
        # Update memory with gate
        gate_input = torch.cat([memory, write_output], dim=-1)
        update_gate = self.update_gate(gate_input)
        updated_memory = memory * (1 - update_gate) + write_output * update_gate
        
        # Combine read output with input
        output = hidden_states + read_output
        
        return output, updated_memory

class RecursiveRouter(nn.Module):
    """Routes inputs to different reasoning experts based on content"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Routing network
        self.router = nn.Linear(config.hidden_size, config.num_reasoning_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 2),
                nn.GELU(),
                nn.Linear(config.hidden_size * 2, config.hidden_size)
            ) for _ in range(config.num_reasoning_experts)
        ])
        
    def forward(self, hidden_states):
        """
        Route inputs to experts
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            routing_weights: [batch_size, seq_len, num_experts]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute routing weights
        routing_logits = self.router(hidden_states)
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Apply each expert
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(hidden_states)
            expert_outputs.append(expert_output)
        
        # Stack expert outputs
        stacked_outputs = torch.stack(expert_outputs, dim=-2)  # [batch_size, seq_len, num_experts, hidden_size]
        
        # Weight outputs by routing weights
        routing_weights_expanded = routing_weights.unsqueeze(-1)  # [batch_size, seq_len, num_experts, 1]
        weighted_outputs = stacked_outputs * routing_weights_expanded
        
        # Sum over experts
        output = weighted_outputs.sum(dim=-2)  # [batch_size, seq_len, hidden_size]
        
        return output, routing_weights

class RecursiveVerifier(nn.Module):
    """Verifies the correctness of recursive reasoning steps"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Verification network
        self.verifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Correction network
        self.corrector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
    def forward(self, hidden_states, original_states):
        """
        Verify and correct reasoning
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            original_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            verification_scores: [batch_size, seq_len, 1]
        """
        # Compute verification scores
        verification_scores = self.verifier(hidden_states)
        
        # Apply correction where verification score is low
        corrections = self.corrector(hidden_states)
        
        # Apply corrections based on verification scores
        output = hidden_states * verification_scores + corrections * (1 - verification_scores)
        
        return output, verification_scores

class RecursiveComposer(nn.Module):
    """Composes reasoning steps into higher-level reasoning"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Composition layers
        self.composition_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.composition_depth)
        ])
        
    def forward(self, hidden_states, past_states=None):
        """
        Compose reasoning steps
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            past_states: Optional list of previous recursive states
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # Initialize with current hidden states
        composed = hidden_states
        
        # Apply composition layers
        for layer in self.composition_layers:
            composed = layer(composed)
        
        # If past states are available, integrate them
        if past_states is not None and past_states:
            # Concatenate past states along sequence dimension
            past_concat = torch.cat(past_states, dim=1)
            
            # Apply self-attention between composed and past states
            # This is a simplified version; a more sophisticated approach would use
            # cross-attention between composed and past states
            combined = torch.cat([composed, past_concat], dim=1)
            
            # Apply composition layers to combined
            for layer in self.composition_layers:
                combined = layer(combined)
            
            # Extract the part corresponding to the original sequence length
            composed = combined[:, :hidden_states.size(1), :]
        
        return composed

class RecursiveReasoningTransformer(nn.Module):
    """Transformer that applies reasoning operations recursively"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Recursive components
        self.recursive_attention = RecursiveAttention(config)
        
        if config.use_recursive_memory:
            self.recursive_memory = RecursiveMemory(config)
        
        if config.use_recursive_routing:
            self.recursive_router = RecursiveRouter(config)
        
        if config.use_recursive_verification:
            self.recursive_verifier = RecursiveVerifier(config)
        
        if config.use_recursive_composition:
            self.recursive_composer = RecursiveComposer(config)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        
        # Layer normalization
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        self.final_norm = nn.LayerNorm(config.hidden_size)
        
        # Adaptive depth controller
        if config.use_adaptive_depth:
            self.depth_controller = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.hidden_size // 2, 1),
                nn.Sigmoid()
            )
        
    def forward(self, hidden_states, past_recursive_states=None):
        """
        Apply recursive reasoning
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            past_recursive_states: Optional list of previous recursive states
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            recursive_info: Dict containing recursive reasoning information
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Initialize recursive info
        recursive_info = {
            'recursion_depths': [],
            'verification_scores': [],
            'routing_weights': [],
            'recursive_states': []
        }
        
        # Initialize recursive states
        if past_recursive_states is None:
            past_recursive_states = []
        
        # Initialize memory if used
        if hasattr(self, 'recursive_memory'):
            memory = None
        
        # Apply recursive reasoning
        current_states = hidden_states
        for depth in range(self.config.max_recursion_depth):
            # Store current state
            recursive_info['recursive_states'].append(current_states.detach())
            
            # Check if we should stop recursion
            if hasattr(self, 'depth_controller') and depth > 0:
                # Compute stopping probability
                stopping_prob = self.depth_controller(current_states.mean(dim=1, keepdim=True))
                stopping_prob = stopping_prob.mean()
                
                # Store recursion depth
                recursive_info['recursion_depths'].append(depth)
                
                # Stop if stopping probability is high enough
                if stopping_prob > self.config.early_stopping_threshold:
                    break
            
            # Apply recursive attention
            attn_output, _ = self.recursive_attention(
                current_states, 
                past_states=past_recursive_states,
                recursion_level=depth
            )
            attn_output = self.attn_norm(attn_output + current_states)  # Residual connection
            
            # Apply recursive memory if enabled
            if hasattr(self, 'recursive_memory'):
                memory_output, memory = self.recursive_memory(attn_output, recursion_level=depth)
                attn_output = memory_output
            
            # Apply recursive routing if enabled
            if hasattr(self, 'recursive_router'):
                routing_output, routing_weights = self.recursive_router(attn_output)
                attn_output = routing_output
                recursive_info['routing_weights'].append(routing_weights.detach())
            
            # Apply feed-forward network
            ffn_output = self.ffn(attn_output)
            ffn_output = self.ffn_norm(ffn_output + attn_output)  # Residual connection
            
            # Apply recursive verification if enabled
            if hasattr(self, 'recursive_verifier'):
                verified_output, verification_scores = self.recursive_verifier(ffn_output, hidden_states)
                ffn_output = verified_output
                recursive_info['verification_scores'].append(verification_scores.detach())
            
            # Apply recursive composition if enabled
            if hasattr(self, 'recursive_composer'):
                composed_output = self.recursive_composer(ffn_output, past_states=past_recursive_states)
                ffn_output = composed_output
            
            # Update current states
            current_states = ffn_output
            
            # Add to past recursive states
            past_recursive_states.append(current_states.detach())
        
        # Final layer norm
        output = self.final_norm(current_states)
        
        return output, recursive_info

class RecursiveReasoningModule(nn.Module):
    """Main module for recursive reasoning"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Recursive reasoning transformer
        self.recursive_transformer = RecursiveReasoningTransformer(config)
        
        # Input and output projections
        self.input_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, hidden_states):
        """
        Apply recursive reasoning
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            reasoning_info: Dict containing reasoning information
        """
        # Project input
        projected_input = self.input_projection(hidden_states)
        
        # Apply recursive reasoning
        reasoned_output, recursive_info = self.recursive_transformer(projected_input)
        
        # Project output and add residual connection
        output = self.output_projection(reasoned_output)
        output = self.layer_norm(output + hidden_states)
        
        return output, recursive_info

class RecursiveReasoner(nn.Module):
    """
    Recursive reasoning module for complex reasoning tasks.
    Implements recursive thinking and reasoning over nested structures.
    """
    
    def __init__(
        self,
        config,
        hidden_size: int = 768,
        num_heads: int = 8,
        max_recursion_depth: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size if hasattr(config, 'hidden_size') else hidden_size
        self.max_recursion_depth = max_recursion_depth
        
        # Recursive step encoder
        self.step_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Recursive reasoning layers
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=num_heads,
                dim_feedforward=self.hidden_size * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(max_recursion_depth)
        ])
        
        # Recursive memory
        self.recursive_memory = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            dropout=dropout if max_recursion_depth > 1 else 0,
            batch_first=True
        )
        
        # Termination predictor
        self.termination_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        # Initialize state
        self.is_initialized = False
        
    def initialize(self):
        """Initialize recursive reasoning components"""
        if not self.is_initialized:
            # Initialize weights
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                        
            self.is_initialized = True
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Apply recursive reasoning to input hidden states
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            reasoned_states: Hidden states after recursive reasoning
            reasoning_info: Dictionary with reasoning information
        """
        # Ensure initialized
        if not self.is_initialized:
            self.initialize()
            
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        
        # Initialize recursive state
        current_state = hidden_states
        memory_state = None
        recursive_states = [current_state]
        termination_scores = []
        
        # Apply recursive reasoning
        for depth in range(self.max_recursion_depth):
            # Encode current state
            encoded_state = self.step_encoder(current_state)
            
            # Apply reasoning layer
            reasoned_state = self.reasoning_layers[depth](
                encoded_state,
                src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
            
            # Update memory
            if memory_state is None:
                # Initialize memory
                memory_output, memory_state = self.recursive_memory(reasoned_state)
            else:
                # Update memory with current state
                memory_output, memory_state = self.recursive_memory(reasoned_state, memory_state)
            
            # Combine reasoned state with memory
            current_state = reasoned_state + memory_output
            
            # Store state
            recursive_states.append(current_state)
            
            # Predict termination
            termination_score = self.termination_predictor(current_state.mean(dim=1))
            termination_scores.append(termination_score)
            
            # Check for early termination
            if self.training is False and termination_score.mean() > 0.9:
                break
        
        # Stack states and scores
        recursive_states_tensor = torch.stack(recursive_states, dim=1)  # [batch_size, depth+1, seq_len, hidden_size]
        termination_scores_tensor = torch.cat(termination_scores, dim=1)  # [batch_size, depth]
        
        # Get final state (last or weighted by termination scores)
        if self.training:
            # During training, use all states weighted by termination scores
            depth_weights = F.softmax(termination_scores_tensor, dim=1).unsqueeze(2).unsqueeze(3)
            final_state = (recursive_states_tensor[:, 1:] * depth_weights).sum(dim=1)  # Skip initial state
        else:
            # During inference, use the last state
            final_state = recursive_states_tensor[:, -1]
        
        # Combine with original input
        output = self.output_projection(
            torch.cat([hidden_states, final_state], dim=-1)
        )
        
        # Prepare reasoning info
        reasoning_info = {
            'recursive_states': recursive_states_tensor,
            'termination_scores': termination_scores_tensor,
            'memory_state': memory_state
        }
        
        return output, reasoning_info
    
    def reason(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Apply recursive reasoning and return enhanced representations
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask
            
        Returns:
            enhanced_states: Enhanced hidden states after reasoning
        """
        reasoned_states, _ = self.forward(hidden_states, attention_mask)
        return reasoned_states

class RecurrentReasoningBlock(nn.Module):
    """
    Recurrent reasoning block that performs iterative refinement of representations
    through a multi-step reasoning process.
    
    This block can be integrated with transformer models to enhance their reasoning
    capabilities for complex tasks that require step-by-step thinking.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_reasoning_steps: int = 5,
        min_reasoning_steps: int = 1,
        intermediate_size: Optional[int] = None,
        dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        activation: str = "gelu",
        early_stopping: bool = True,
        convergence_threshold: float = 0.01,
        use_residual_connection: bool = True,
        use_gating: bool = True,
        use_memory: bool = True,
        memory_size: int = 64
    ):
        """
        Initialize the recurrent reasoning block.
        
        Args:
            hidden_size: Size of the hidden states
            max_reasoning_steps: Maximum number of reasoning steps to perform
            min_reasoning_steps: Minimum number of reasoning steps to perform
            intermediate_size: Size of the intermediate representations (defaults to 4*hidden_size)
            dropout_prob: Dropout probability
            layer_norm_eps: Layer normalization epsilon
            activation: Activation function ('gelu', 'relu', or 'silu')
            early_stopping: Whether to use early stopping based on convergence
            convergence_threshold: Threshold for early stopping convergence
            use_residual_connection: Whether to use residual connections
            use_gating: Whether to use gating mechanism for updates
            use_memory: Whether to use memory for reasoning
            memory_size: Size of the memory if used
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_reasoning_steps = max_reasoning_steps
        self.min_reasoning_steps = min_reasoning_steps
        self.intermediate_size = intermediate_size or (4 * hidden_size)
        self.dropout_prob = dropout_prob
        self.early_stopping = early_stopping
        self.convergence_threshold = convergence_threshold
        self.use_residual_connection = use_residual_connection
        self.use_gating = use_gating
        self.use_memory = use_memory
        self.memory_size = memory_size
        
        # Activation function
        if activation == "gelu":
            self.activation_fn = F.gelu
        elif activation == "relu":
            self.activation_fn = F.relu
        elif activation == "silu":
            self.activation_fn = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Reasoning core network
        self.reasoning_core = nn.Sequential(
            nn.Linear(hidden_size, self.intermediate_size),
            nn.LayerNorm(self.intermediate_size, eps=layer_norm_eps),
            nn.Dropout(dropout_prob)
        )
        
        # Reasoning output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.intermediate_size, hidden_size),
            nn.LayerNorm(hidden_size, eps=layer_norm_eps),
            nn.Dropout(dropout_prob)
        )
        
        # Gating mechanism
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
        
        # Memory mechanism
        if use_memory:
            self.memory_key = nn.Linear(hidden_size, memory_size)
            self.memory_value = nn.Linear(hidden_size, memory_size)
            self.memory_query = nn.Linear(hidden_size, memory_size)
            self.memory_output = nn.Linear(memory_size, hidden_size)
            self.register_buffer("memory_keys", torch.zeros(0, memory_size))
            self.register_buffer("memory_values", torch.zeros(0, memory_size))
            self.memory_capacity = 1000  # Maximum number of memory entries
        
        # Quality assessment
        self.quality_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Statistics
        self.avg_steps = 0
        self.total_forward_calls = 0
        self.early_stops = 0
    
    def _reasoning_step(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Perform a single reasoning step.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Updated hidden states after reasoning
        """
        # Apply reasoning core
        intermediate = self.reasoning_core(hidden_states)
        intermediate = self.activation_fn(intermediate)
        
        # Apply memory if enabled
        if self.use_memory and self.memory_keys.size(0) > 0:
            # Generate query from current state
            query = self.memory_query(hidden_states)  # [batch_size, seq_len, memory_size]
            
            # Compute attention with memory
            attention_scores = torch.matmul(query, self.memory_keys.t())  # [batch_size, seq_len, memory_entries]
            attention_scores = attention_scores / math.sqrt(self.memory_size)
            attention_probs = F.softmax(attention_scores, dim=-1)
            
            # Retrieve values from memory
            memory_output = torch.matmul(attention_probs, self.memory_values)  # [batch_size, seq_len, memory_size]
            memory_output = self.memory_output(memory_output)  # [batch_size, seq_len, hidden_size]
            
            # Add memory output to intermediate representation
            intermediate = intermediate + memory_output
        
        # Apply output projection
        new_hidden_states = self.output_projection(intermediate)
        
        # Apply gating if enabled
        if self.use_gating:
            gate_input = torch.cat([hidden_states, new_hidden_states], dim=-1)
            gate = self.gate(gate_input)
            new_hidden_states = gate * new_hidden_states + (1 - gate) * hidden_states
        # Apply residual connection if enabled
        elif self.use_residual_connection:
            new_hidden_states = new_hidden_states + hidden_states
        
        return new_hidden_states
    
    def _update_memory(self, hidden_states: torch.Tensor):
        """
        Update memory with new hidden states.
        
        Args:
            hidden_states: Hidden states to store [batch_size, seq_len, hidden_size]
        """
        if not self.use_memory:
            return
        
        # Get mean representation for each sequence
        mean_hidden = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Generate keys and values
        keys = self.memory_key(mean_hidden)  # [batch_size, memory_size]
        values = self.memory_value(mean_hidden)  # [batch_size, memory_size]
        
        # Add to memory
        self.memory_keys = torch.cat([self.memory_keys, keys], dim=0)
        self.memory_values = torch.cat([self.memory_values, values], dim=0)
        
        # Limit memory size
        if self.memory_keys.size(0) > self.memory_capacity:
            self.memory_keys = self.memory_keys[-self.memory_capacity:]
            self.memory_values = self.memory_values[-self.memory_capacity:]
    
    def _assess_quality(self, previous_states: torch.Tensor, current_states: torch.Tensor) -> torch.Tensor:
        """
        Assess the quality of the current reasoning step.
        
        Args:
            previous_states: Previous hidden states
            current_states: Current hidden states
            
        Returns:
            Quality score for the current step
        """
        # Compute L2 distance between states as a measure of convergence
        state_diff = ((current_states - previous_states) ** 2).sum(dim=-1).mean(dim=-1)
        
        # Check if difference is below threshold
        converged = state_diff < self.convergence_threshold
        
        # Also get quality prediction from the network
        quality = self.quality_predictor(current_states).squeeze(-1).mean(dim=-1)
        
        return quality, converged
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reasoning_depth: Optional[int] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Apply recurrent reasoning to input hidden states.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            reasoning_depth: Override the default reasoning depth
            
        Returns:
            Tuple of (refined hidden states, number of steps taken)
        """
        batch_size, seq_len, _ = hidden_states.shape
        max_steps = reasoning_depth or self.max_reasoning_steps
        
        # Initialize with input hidden states
        current_states = hidden_states
        
        # Track number of steps for each batch item
        steps_taken = torch.zeros(batch_size, device=hidden_states.device)
        
        # Apply masked attention if mask is provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Perform iterative reasoning
        for step in range(max_steps):
            # Store previous states for convergence check
            previous_states = current_states
            
            # Apply reasoning step
            current_states = self._reasoning_step(current_states)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                current_states = current_states * attention_mask
            
            # Update steps taken
            steps_taken += 1
            
            # Check for early stopping if enabled and we've done minimum steps
            if self.early_stopping and step >= self.min_reasoning_steps - 1:
                quality, converged = self._assess_quality(previous_states, current_states)
                
                # Early stop if quality is high enough or convergence reached
                if converged.all():
                    self.early_stops += 1
                    break
            
        # Update statistics
        total_steps = steps_taken.sum().item()
        self.avg_steps = (self.avg_steps * self.total_forward_calls + total_steps) / (self.total_forward_calls + 1)
        self.total_forward_calls += 1
        
        # Update memory with final states
        self._update_memory(current_states)
        
        return current_states, steps_taken.max().int().item()
    
    def reset_memory(self):
        """Reset the reasoning memory."""
        if self.use_memory:
            self.memory_keys = torch.zeros(0, self.memory_size, device=self.memory_keys.device)
            self.memory_values = torch.zeros(0, self.memory_size, device=self.memory_values.device)
    
    def get_memory_utilization(self) -> float:
        """
        Get the current memory utilization.
        
        Returns:
            Memory utilization as a fraction of capacity
        """
        if not self.use_memory:
            return 0.0
        return self.memory_keys.size(0) / self.memory_capacity
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the reasoning process.
        
        Returns:
            Dictionary with statistics
        """
        early_stop_ratio = self.early_stops / max(1, self.total_forward_calls)
        return {
            "avg_reasoning_steps": self.avg_steps,
            "total_forward_calls": self.total_forward_calls,
            "early_stop_ratio": early_stop_ratio,
            "memory_utilization": self.get_memory_utilization()
        }
    
    def reset_stats(self):
        """Reset the reasoning statistics."""
        self.avg_steps = 0
        self.total_forward_calls = 0
        self.early_stops = 0

class IterativeReasoningLayer(nn.Module):
    """
    Layer that combines transformer attention with recurrent reasoning.
    
    This layer can be inserted into transformer models to enhance their
    reasoning capabilities with iterative refinement.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_dropout_prob: float = 0.1,
        max_reasoning_steps: int = 3,
        layer_norm_eps: float = 1e-12
    ):
        """
        Initialize the iterative reasoning layer.
        
        Args:
            hidden_size: Size of the hidden states
            num_attention_heads: Number of attention heads
            attention_head_size: Size of each attention head (defaults to hidden_size / num_attention_heads)
            intermediate_size: Size of the intermediate representations (defaults to 4*hidden_size)
            hidden_dropout_prob: Dropout probability
            max_reasoning_steps: Maximum number of reasoning steps
            layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or (hidden_size // num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.intermediate_size = intermediate_size or (4 * hidden_size)
        
        # Self-attention layer
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.attention_dropout = nn.Dropout(hidden_dropout_prob)
        self.output_projection = nn.Linear(self.all_head_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Reasoning block
        self.recurrent_reasoning = RecurrentReasoningBlock(
            hidden_size=hidden_size,
            max_reasoning_steps=max_reasoning_steps,
            intermediate_size=self.intermediate_size,
            dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps
        )
        
        # Integration layer
        self.integration_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size, eps=layer_norm_eps),
            nn.Dropout(hidden_dropout_prob)
        )
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape hidden states for attention computation.
        
        Args:
            x: Input tensor [batch_size, seq_len, all_head_size]
            
        Returns:
            Reshaped tensor [batch_size, num_attention_heads, seq_len, attention_head_size]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reasoning_depth: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass with both attention and reasoning.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len] or [batch_size, 1, seq_len, seq_len]
            reasoning_depth: Override the default reasoning depth
            
        Returns:
            Refined hidden states
        """
        # Apply self-attention
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Adjust mask dimensions if needed
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Apply mask
            attention_scores = attention_scores + attention_mask
        
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Project back to hidden size
        attention_output = self.output_projection(context_layer)
        attention_output = self.output_dropout(attention_output)
        attention_output = self.layer_norm(attention_output + hidden_states)
        
        # Apply recurrent reasoning
        seq_mask = attention_mask[:, 0, 0, :] if attention_mask is not None and attention_mask.dim() == 4 else None
        reasoning_output, _ = self.recurrent_reasoning(attention_output, seq_mask, reasoning_depth)
        
        # Integrate attention and reasoning outputs
        integrated_output = self.integration_layer(
            torch.cat([attention_output, reasoning_output], dim=-1)
        )
        
        return integrated_output

class RecursiveReasoningNetwork(nn.Module):
    """
    Network that applies recursive reasoning to enhance model capabilities.
    
    This network can be used as a standalone reasoning component or
    integrated into a transformer model to enhance its reasoning abilities.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_recursive_layers: int = 3,
        num_attention_heads: int = 8,
        intermediate_size: Optional[int] = None,
        max_reasoning_depth: int = 5,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        use_residual_connection: bool = True
    ):
        """
        Initialize the recursive reasoning network.
        
        Args:
            hidden_size: Size of the hidden states
            num_recursive_layers: Number of recursive reasoning layers
            num_attention_heads: Number of attention heads per layer
            intermediate_size: Size of the intermediate representations
            max_reasoning_depth: Maximum reasoning depth per layer
            hidden_dropout_prob: Dropout probability
            layer_norm_eps: Layer normalization epsilon
            use_residual_connection: Whether to use residual connections
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_recursive_layers = num_recursive_layers
        self.use_residual_connection = use_residual_connection
        
        # Initialize recursive reasoning layers
        self.recursive_layers = nn.ModuleList([
            IterativeReasoningLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_dropout_prob=hidden_dropout_prob,
                max_reasoning_steps=max_reasoning_depth,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_recursive_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        reasoning_depths: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Apply recursive reasoning to input hidden states.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len] or [batch_size, 1, seq_len, seq_len]
            reasoning_depths: List of reasoning depths for each layer
            
        Returns:
            Refined hidden states after recursive reasoning
        """
        # Initialize reasoning depths if not provided
        if reasoning_depths is None:
            reasoning_depths = [None] * self.num_recursive_layers
        elif len(reasoning_depths) < self.num_recursive_layers:
            reasoning_depths = reasoning_depths + [None] * (self.num_recursive_layers - len(reasoning_depths))
        
        # Apply recursive reasoning layers
        for i, layer in enumerate(self.recursive_layers):
            layer_input = hidden_states
            layer_output = layer(layer_input, attention_mask, reasoning_depths[i])
            
            # Apply residual connection if enabled
            if self.use_residual_connection:
                hidden_states = layer_output + layer_input
            else:
                hidden_states = layer_output
        
        # Apply final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states
    
    def reset_memory(self):
        """Reset memory in all reasoning layers."""
        for layer in self.recursive_layers:
            layer.recurrent_reasoning.reset_memory()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the reasoning process.
        
        Returns:
            Dictionary with statistics for all layers
        """
        stats = {}
        for i, layer in enumerate(self.recursive_layers):
            layer_stats = layer.recurrent_reasoning.get_stats()
            stats[f"layer_{i}"] = layer_stats
        return stats
    
    def reset_stats(self):
        """Reset statistics in all reasoning layers."""
        for layer in self.recursive_layers:
            layer.recurrent_reasoning.reset_stats() 