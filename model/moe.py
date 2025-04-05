import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any, Union
import math
import gc

class EnhancedMoEConfig:
    def __init__(
        self,
        hidden_size: int = 768,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        expert_capacity_factor: float = 1.25,
        router_jitter_noise: float = 0.01,
        router_z_loss_coef: float = 0.001,
        expert_dropout: float = 0.1,
        use_load_balancing: bool = True,
        use_auxiliary_loss: bool = True,
        use_expert_choice_routing: bool = False,
        use_balanced_assignment: bool = True,
        expert_hidden_size: Optional[int] = None,
        block_size: int = 512,
        entropy_threshold: float = 0.6
    ):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.expert_capacity_factor = expert_capacity_factor
        self.router_jitter_noise = router_jitter_noise
        self.router_z_loss_coef = router_z_loss_coef
        self.expert_dropout = expert_dropout
        self.use_load_balancing = use_load_balancing
        self.use_auxiliary_loss = use_auxiliary_loss
        self.use_expert_choice_routing = use_expert_choice_routing
        self.use_balanced_assignment = use_balanced_assignment
        self.expert_hidden_size = expert_hidden_size or (hidden_size * 4)  # Default to 4x hidden size
        self.block_size = block_size
        self.entropy_threshold = entropy_threshold

class ExpertModule(nn.Module):
    """Individual expert in the Mixture of Experts layer"""
    
    def __init__(self, config, expert_idx: int):
        super().__init__()
        self.expert_idx = expert_idx
        self.config = config
        
        # Expert-specific layers
        self.up_proj = nn.Linear(config.hidden_size, config.expert_hidden_size)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(config.expert_hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.expert_dropout)
        
        # Specialized reasoning components for different experts
        if expert_idx % 4 == 0:
            # Logical reasoning expert
            self.specialized = nn.Sequential(
                nn.Linear(config.expert_hidden_size, config.expert_hidden_size),
                nn.LayerNorm(config.expert_hidden_size),
                nn.GELU()
            )
        elif expert_idx % 4 == 1:
            # Mathematical reasoning expert
            self.specialized = nn.Sequential(
                nn.Linear(config.expert_hidden_size, config.expert_hidden_size // 2),
                nn.GELU(),
                nn.Linear(config.expert_hidden_size // 2, config.expert_hidden_size),
                nn.LayerNorm(config.expert_hidden_size)
            )
        elif expert_idx % 4 == 2:
            # Factual/knowledge expert
            self.specialized = nn.Sequential(
                nn.Linear(config.expert_hidden_size, config.expert_hidden_size),
                nn.Dropout(0.2),
                nn.GELU()
            )
        else:
            # General reasoning expert
            self.specialized = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.up_proj(x)
        hidden = self.act(hidden)
        hidden = self.specialized(hidden)
        hidden = self.dropout(hidden)
        output = self.down_proj(hidden)
        return output

class ExpertRouter(nn.Module):
    """Routes tokens to experts based on learned routing weights"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Initialize with small random weights
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            routing_weights: [batch_size, seq_len, num_experts_per_token, num_experts]
            aux_loss: Dictionary of auxiliary losses
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get router logits
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        
        if self.training and self.config.router_jitter_noise > 0:
            # Add noise during training for exploration
            router_logits += torch.randn_like(router_logits) * self.config.router_jitter_noise
        
        # Calculate routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Calculate auxiliary losses
        aux_loss = {}
        
        # Z-loss to prevent router from producing extremely large values
        if self.config.router_z_loss_coef > 0:
            z_loss = torch.mean(torch.square(torch.logsumexp(router_logits, dim=-1)))
            aux_loss["router_z_loss"] = z_loss * self.config.router_z_loss_coef
        
        # Load balancing loss
        if self.config.use_load_balancing:
            # Calculate the fraction of tokens routed to each expert
            expert_usage = torch.mean(routing_weights, dim=(0, 1))
            # We want a uniform distribution: 1/num_experts for each expert
            target_usage = torch.ones_like(expert_usage) / self.config.num_experts
            # Calculate the load balancing loss
            load_balancing_loss = torch.sum(target_usage * torch.log(target_usage / expert_usage))
            aux_loss["load_balancing_loss"] = load_balancing_loss * 0.01
        
        # Get top-k experts for each token
        if not self.config.use_expert_choice_routing:
            # Token-choice routing: each token selects its top-k experts
            routing_weights_k, indices = torch.topk(
                routing_weights, 
                self.config.num_experts_per_token, 
                dim=-1
            )
            # Create a mask for the selected experts
            mask = torch.zeros_like(routing_weights)
            mask.scatter_(-1, indices, 1.0)
            # Mask out non-selected experts and renormalize
            routing_weights = routing_weights * mask
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        else:
            # Expert-choice routing: each expert selects its top-k tokens
            # This is more complex and would require a custom CUDA kernel for efficiency
            # Simplified implementation for now
            pass
        
        return routing_weights, aux_loss

class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with balanced assignment and auxiliary losses"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertModule(config, i) for i in range(config.num_experts)
        ])
        
        # Create router
        self.router = ExpertRouter(config)
        
        # Layer norm before routing
        self.layer_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_loss: Dictionary of auxiliary losses
        """
        # Apply layer norm
        normalized_states = self.layer_norm(hidden_states)
        
        # Get routing weights and auxiliary losses
        routing_weights, aux_loss = self.router(normalized_states)
        
        # Initialize output tensor
        batch_size, seq_len, hidden_size = hidden_states.shape
        output = torch.zeros_like(hidden_states)
        
        # Process tokens through experts
        for expert_idx, expert in enumerate(self.experts):
            # Get the routing weight for this expert
            expert_weights = routing_weights[:, :, expert_idx].unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Only process tokens that have non-zero routing weight for this expert
            if torch.any(expert_weights > 0):
                # Process all tokens through this expert
                expert_output = expert(normalized_states)
                
                # Weight the expert output by the routing weights
                output += expert_output * expert_weights
        
        # Residual connection
        output = output + hidden_states
        
        return output, aux_loss

class ReasoningMoE(nn.Module):
    """Specialized Mixture of Experts for reasoning tasks"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create MoE layers
        self.moe_layers = nn.ModuleList([
            MixtureOfExperts(config) for _ in range(2)  # Use 2 MoE layers
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_losses: Dictionary of auxiliary losses
        """
        aux_losses = {}
        
        # Apply MoE layers
        for i, moe_layer in enumerate(self.moe_layers):
            hidden_states, layer_aux_loss = moe_layer(hidden_states)
            
            # Accumulate auxiliary losses
            for key, value in layer_aux_loss.items():
                aux_losses[f"layer_{i}_{key}"] = value
        
        # Apply final layer norm
        output = self.layer_norm(hidden_states)
        
        return output, aux_losses 

class ExpertGating(nn.Module):
    """
    Gating mechanism for Mixture of Experts (MoE) architecture.
    This allows the model to selectively route tokens to different specialized experts.
    Optimized for use with the hybrid RWKV-Transformer model.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        k: int = 2,  # Top-k experts to route to
        capacity_factor: float = 1.5,
        dropout: float = 0.1,
        use_aux_loss: bool = True,
        aux_loss_weight: float = 0.01,
        noise_factor: float = 1.0,
        routing_algorithm: str = "top_k"  # top_k, hash_based, learned_balancing
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.capacity_factor = capacity_factor
        self.use_aux_loss = use_aux_loss
        self.aux_loss_weight = aux_loss_weight
        self.noise_factor = noise_factor
        self.routing_algorithm = routing_algorithm
        
        # Expert router - determines which experts to use for each token
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_experts)
        )
        
        # Ensure parameters are properly initialized
        self.reset_parameters()
        
        # For tracking and balancing
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_routed', torch.tensor(0.))
        
        # Gate dropping mechanism for optimization
        self.gate_dropping = False
        self.gate_dropping_threshold = 0.0
    
    def reset_parameters(self):
        """Initialize router parameters"""
        for module in self.router:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def enable_gate_dropping(self, threshold: float = 0.1):
        """
        Enable gate dropping for inference optimization.
        
        Args:
            threshold: Minimum routing probability for an expert to be considered
        """
        self.gate_dropping = True
        self.gate_dropping_threshold = threshold
        return self
    
    def disable_gate_dropping(self):
        """Disable gate dropping"""
        self.gate_dropping = False
        return self
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        expert_outputs: Optional[List[torch.Tensor]] = None,
        return_loss: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route inputs to the appropriate experts and combine their outputs.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            expert_outputs: Optional list of expert outputs [num_experts, batch_size, seq_len, hidden_size]
            return_loss: Whether to return the auxiliary loss
            
        Returns:
            combined_output: Weighted combination of expert outputs
            aux_info: Dict with auxiliary information (loss, gate values, expert counts)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Calculate router logits
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        
        # Add noise during training for exploration
        if self.training and self.noise_factor > 0:
            router_noise = torch.randn_like(router_logits) * self.noise_factor
            router_logits = router_logits + router_noise
        
        # Different routing algorithms
        if self.routing_algorithm == "top_k":
            # Get routing probabilities with softmax
            router_probs = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]
            
            # Select top-k experts per token
            if self.gate_dropping and not self.training:
                # Simplified routing for inference with gate dropping
                gate_values, expert_indices = torch.max(router_probs, dim=-1)  # [batch_size, seq_len]
                
                # Apply threshold
                mask = gate_values >= self.gate_dropping_threshold
                gate_values = gate_values * mask.float()
                
                # Set shape for compatibility with other modes
                top_k_indices = expert_indices.unsqueeze(-1)  # [batch_size, seq_len, 1]
                top_k_gates = gate_values.unsqueeze(-1)  # [batch_size, seq_len, 1]
            else:
                # Full top-k routing for training
                top_k_gates, top_k_indices = torch.topk(
                    router_probs, self.k, dim=-1
                )  # Both [batch_size, seq_len, k]
            
            # Load balancing loss if needed
            if self.use_aux_loss and self.training:
                # Calculate fraction of tokens routed to each expert
                router_probs_sum = router_probs.sum(dim=[0, 1])  # [num_experts]
                router_probs_fraction = router_probs_sum / router_probs_sum.sum()
                
                # Ideal distribution would be uniform
                target_probs = torch.ones_like(router_probs_fraction) / self.num_experts
                
                # Calculate aux loss (encourage balanced expert usage)
                aux_loss = torch.sum(target_probs * torch.log(target_probs / router_probs_fraction))
            else:
                aux_loss = torch.tensor(0.0, device=hidden_states.device)
            
            # Update tracking
            if self.training:
                # Count how many tokens are routed to each expert
                with torch.no_grad():
                    self.expert_counts += router_probs.sum(dim=[0, 1]).detach()
                    self.total_routed += batch_size * seq_len
        
        elif self.routing_algorithm == "hash_based":
            # Simple hash-based routing (less computation)
            # This assigns tokens to experts based on a hash function
            token_hashes = torch.sum(hidden_states, dim=-1) % self.num_experts
            top_k_indices = token_hashes.unsqueeze(-1) % self.num_experts
            
            # Create uniform gate values
            top_k_gates = torch.ones_like(top_k_indices, dtype=torch.float) / self.k
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
        
        elif self.routing_algorithm == "learned_balancing":
            # Learned balancing with multiplicative updates
            router_probs = F.softmax(router_logits, dim=-1)
            
            # Get expert counts from previous batch
            expert_fractions = self.expert_counts / max(1.0, self.total_routed)
            
            # Adjust routing probabilities based on historical usage
            adjusted_probs = router_probs * (1.0 / (expert_fractions + 1e-5))
            adjusted_probs = F.softmax(adjusted_probs, dim=-1)
            
            # Select top-k experts with adjusted probabilities
            top_k_gates, top_k_indices = torch.topk(
                adjusted_probs, self.k, dim=-1
            )
            
            # Auxiliary loss is implicit in the adjustment
            aux_loss = torch.tensor(0.0, device=hidden_states.device)
            
            # Update tracking
            if self.training:
                with torch.no_grad():
                    self.expert_counts += adjusted_probs.sum(dim=[0, 1]).detach()
                    self.total_routed += batch_size * seq_len
        
        # Combine expert outputs
        if expert_outputs is None:
            # We just return the routing information for external expert computation
            combined_output = None
        else:
            # Ensure we have the correct number of experts
            assert len(expert_outputs) == self.num_experts, \
                f"Expected {self.num_experts} expert outputs, got {len(expert_outputs)}"
            
            # Combine expert outputs weighted by gate values
            combined_output = torch.zeros_like(hidden_states)
            
            # Reshape for efficient computation
            batch_indices = torch.arange(batch_size, device=hidden_states.device).view(-1, 1, 1).expand(-1, seq_len, self.k)
            seq_indices = torch.arange(seq_len, device=hidden_states.device).view(1, -1, 1).expand(batch_size, -1, self.k)
            
            # Gather and weight each expert's output
            for k_idx in range(self.k):
                expert_idx = top_k_indices[:, :, k_idx]
                gate_value = top_k_gates[:, :, k_idx].unsqueeze(-1)
                
                # For each expert, add its contribution
                for i, expert_output in enumerate(expert_outputs):
                    # Create mask for tokens routed to this expert
                    mask = (expert_idx == i).unsqueeze(-1).float()
                    
                    # Add weighted output (gate_value * expert_output * mask)
                    combined_output += gate_value * expert_output * mask
        
        # Return combined output and auxiliary information
        aux_info = {
            "aux_loss": aux_loss * self.aux_loss_weight if self.training else torch.tensor(0.0),
            "router_probs": router_probs if self.routing_algorithm != "hash_based" else None,
            "top_k_gates": top_k_gates,
            "top_k_indices": top_k_indices,
            "expert_counts": self.expert_counts.clone() if self.training else None
        }
        
        if return_loss:
            return combined_output, aux_info
        
        return combined_output 

class AdaptiveHierarchicalRouter(nn.Module):
    """
    Enhanced hierarchical router that dynamically adjusts routing strategy
    for extremely long sequences (32K-64K), using progressively coarser 
    routing for longer contexts.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        
        # Block sizes increase with sequence length
        self.base_block_size = getattr(config, 'block_size', 512)
        self.max_block_size = getattr(config, 'max_block_size', 2048)
        
        # Entropy threshold decreases with sequence length to reduce token-level routing
        self.base_entropy_threshold = getattr(config, 'entropy_threshold', 0.6)
        
        # Token budget - maximum percentage of tokens that can use token-level routing
        self.token_routing_budget = getattr(config, 'token_routing_budget', 0.3)
        
        # Block-level router (using smaller hidden dimension for efficiency)
        block_router_dim = max(64, self.hidden_size // 8)
        self.block_router = nn.Sequential(
            nn.Linear(self.hidden_size, block_router_dim),
            nn.LayerNorm(block_router_dim),
            nn.GELU(),
            nn.Linear(block_router_dim, self.num_experts)
        )
        
        # Token-level router (more expressive)
        self.token_router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(getattr(config, 'router_dropout', 0.1)),
            nn.Linear(self.hidden_size // 2, self.num_experts)
        )
        
        # For tracking statistics
        self.register_buffer('token_routing_ratio', torch.tensor(0.0))
        self.register_buffer('block_entropy', torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router weights"""
        for module in [self.block_router, self.token_router]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def _compute_entropy(self, probs):
        """Compute entropy of probability distribution"""
        # Add small epsilon to avoid log(0)
        probs = probs + 1e-10
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        # Normalize by log(num_experts) to get values between 0 and 1
        max_entropy = torch.log(torch.tensor(self.num_experts, dtype=torch.float, device=probs.device))
        return entropy / max_entropy
    
    def _adjust_parameters_for_length(self, seq_len):
        """Adjust routing parameters based on sequence length"""
        # Scale block size based on sequence length
        if seq_len <= 4096:
            block_size = self.base_block_size
            entropy_threshold = self.base_entropy_threshold
            token_budget = self.token_routing_budget
        elif seq_len <= 16384:
            # For medium sequences, increase block size and threshold
            block_size = min(1024, self.max_block_size)
            entropy_threshold = self.base_entropy_threshold * 1.1  # Higher threshold = fewer token-level routings
            token_budget = self.token_routing_budget * 0.7
        else:
            # For very long sequences, use larger blocks and stricter threshold
            scale_factor = min(seq_len / 16384, 4)  # Cap the scaling factor at 4
            block_size = min(int(self.base_block_size * scale_factor), self.max_block_size)
            entropy_threshold = self.base_entropy_threshold * 1.2
            token_budget = self.token_routing_budget * (1.0 / scale_factor)  # Reduce budget for ultra-long sequences
        
        return block_size, entropy_threshold, token_budget
    
    def _segment_into_blocks(self, hidden_states, block_size):
        """Split sequence into blocks for efficient processing"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Calculate number of blocks and padded sequence length
        num_blocks = (seq_len + block_size - 1) // block_size
        padded_len = num_blocks * block_size
        
        # Pad sequence if needed
        if padded_len > seq_len:
            padding = torch.zeros(
                batch_size, padded_len - seq_len, hidden_size, 
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            hidden_states = torch.cat([hidden_states, padding], dim=1)
        
        # Reshape into blocks
        blocks = hidden_states.view(batch_size, num_blocks, block_size, hidden_size)
        
        return blocks, num_blocks, seq_len
    
    def forward(self, hidden_states):
        """
        Adaptive hierarchical routing optimized for long sequences.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            routing_weights: Sparse tensor for routing [batch_size, seq_len, num_experts]
            aux_loss: Load balancing loss
            metadata: Dictionary with routing stats
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Adjust parameters based on sequence length
        block_size, entropy_threshold, token_budget = self._adjust_parameters_for_length(seq_len)
        
        # Step 1: Segment sequence into blocks
        blocks, num_blocks, original_seq_len = self._segment_into_blocks(hidden_states, block_size)
        
        # Step 2: Block-level routing (cheap)
        # Mean-pool each block to get block representation
        block_repr = torch.mean(blocks, dim=2)  # [batch_size, num_blocks, hidden_size]
        
        # Get block routing logits and probabilities
        block_logits = self.block_router(block_repr)  # [batch_size, num_blocks, num_experts]
        block_probs = F.softmax(block_logits, dim=-1)
        
        # Compute entropy for each block to identify high-entropy blocks
        block_entropy = self._compute_entropy(block_probs)  # [batch_size, num_blocks]
        
        # Track mean block entropy for monitoring
        with torch.no_grad():
            self.block_entropy = block_entropy.mean().detach()
        
        # Step 3: Identify high-entropy blocks that need token-level routing
        high_entropy_mask = block_entropy > entropy_threshold
        
        # Initialize routing weights tensor
        routing_weights = torch.zeros(
            batch_size, original_seq_len, self.num_experts,
            device=device, dtype=hidden_states.dtype
        )
        
        # Apply block-level routing to low-entropy blocks
        for b in range(batch_size):
            for i in range(num_blocks):
                # Skip if this is a high-entropy block
                if high_entropy_mask[b, i]:
                    continue
                
                # Get block expert assignment (top-1 for simplicity)
                block_expert = torch.argmax(block_probs[b, i]).item()
                block_weight = block_probs[b, i, block_expert].item()
                
                # Compute start and end indices for this block
                start_idx = i * block_size
                end_idx = min(original_seq_len, (i + 1) * block_size)
                
                # Assign all tokens in this block to the same expert
                if end_idx > start_idx:  # Ensure we're not out of bounds
                    routing_weights[b, start_idx:end_idx, block_expert] = block_weight
        
        # Step 4: Apply token-level routing with budget constraints
        # Count high-entropy blocks and tokens
        total_high_entropy_blocks = high_entropy_mask.sum().item()
        high_entropy_tokens = total_high_entropy_blocks * block_size
        
        # Calculate token routing budget (max number of tokens for token-level routing)
        max_token_routing = int(original_seq_len * token_budget)
        
        # If we exceed budget, prioritize blocks by entropy
        if high_entropy_tokens > max_token_routing and total_high_entropy_blocks > 0:
            # Flatten for easier processing
            flat_entropy = block_entropy.view(-1)
            flat_mask = high_entropy_mask.view(-1)
            
            # Get entropy values for high-entropy blocks only
            high_entropies = flat_entropy[flat_mask]
            
            # Calculate how many blocks we can process with token-level routing
            blocks_within_budget = max(1, max_token_routing // block_size)
            
            # Get threshold to limit to top blocks_within_budget blocks
            if blocks_within_budget < len(high_entropies):
                entropy_threshold_adjusted = torch.sort(high_entropies, descending=True)[0][blocks_within_budget-1].item()
                # Update high entropy mask with new threshold
                high_entropy_mask = block_entropy > entropy_threshold_adjusted
        
        tokens_routed = 0
        
        for b in range(batch_size):
            for i in range(num_blocks):
                # Only process high-entropy blocks
                if not high_entropy_mask[b, i]:
                    continue
                
                # Compute start and end indices for this block
                start_idx = i * block_size
                end_idx = min(original_seq_len, (i + 1) * block_size)
                
                # Skip if out of bounds
                if start_idx >= original_seq_len:
                    continue
                
                # Check if adding this block exceeds our token budget
                if tokens_routed + (end_idx - start_idx) > max_token_routing:
                    # If exceeding budget, do block-level routing for this block
                    block_expert = torch.argmax(block_probs[b, i]).item()
                    block_weight = block_probs[b, i, block_expert].item()
                    routing_weights[b, start_idx:end_idx, block_expert] = block_weight
                    continue
                
                # Get token embeddings for this block
                block_tokens = hidden_states[b, start_idx:end_idx]
                
                # Apply token-level routing
                token_logits = self.token_router(block_tokens)
                token_probs = F.softmax(token_logits, dim=-1)
                
                # Assign routing weights
                routing_weights[b, start_idx:end_idx] = token_probs
                tokens_routed += end_idx - start_idx
        
        # Track the ratio of tokens that received token-level routing
        with torch.no_grad():
            self.token_routing_ratio = tokens_routed / (batch_size * original_seq_len)
        
        # Calculate load balancing loss
        expert_usage = torch.mean(routing_weights, dim=(0, 1))
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        load_balance_loss = torch.sum(target_usage * torch.log(target_usage / (expert_usage + 1e-10)))
        
        # Additional metrics for token routing budget
        token_budget_metrics = {
            "token_budget_percent": token_budget * 100,
            "token_budget_utilized_percent": (tokens_routed / (batch_size * original_seq_len)) * 100,
            "adjusted_block_size": block_size
        }
        
        # Metadata for monitoring
        metadata = {
            "block_entropy": self.block_entropy.item(),
            "token_routing_ratio": self.token_routing_ratio.item(),
            "high_entropy_blocks": high_entropy_mask.float().mean().item(),
            **token_budget_metrics
        }
        
        return routing_weights, load_balance_loss, metadata

class QAwareHeterogeneousExpertModule(nn.Module):
    """
    Improved expert module with:
    1. Quantization-Aware Training (QAT)
    2. Expert-specific quantization precision (4-bit for light, 8-bit for heavy)
    3. KV cache management
    4. Gradient checkpointing support
    """
    
    def __init__(self, config, expert_idx: int):
        super().__init__()
        self.expert_idx = expert_idx
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Determine expert type based on index
        # We'll create a mix of lightweight and heavy experts
        if expert_idx % 3 == 0:
            # Heavy expert for complex patterns (math, code, etc.)
            self.expert_type = "heavy"
            self.num_layers = getattr(config, 'heavy_expert_layers', 4)
            self.use_attention = True
            self.expansion_factor = 4
            self.quant_bits = 8  # Higher precision for complex experts
        else:
            # Lightweight expert for simpler patterns
            self.expert_type = "light"
            self.num_layers = getattr(config, 'light_expert_layers', 1)
            self.use_attention = False
            self.expansion_factor = 2
            self.quant_bits = 4  # Lower precision for simpler experts
        
        # Input projection with optional QAT
        self.input_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Expert layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.hidden_size
            
            # Create appropriate layer based on expert type
            if self.expert_type == "heavy" and self.use_attention and i % 2 == 0:
                # Add attention layer for heavy experts
                self.layers.append(self._create_attention_layer())
            else:
                # Add feed-forward layer
                self.layers.append(self._create_ffn_layer(in_dim))
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # For QAT
        self.qat_enabled = False
        self.quant_state = {}
        
        # For gradient checkpointing
        self.use_gradient_checkpointing = False
        
        # For KV cache
        self.kv_cache_enabled = False
        self.kv_linear_k = None
        self.kv_linear_v = None
        
        # Initialize QAT observers and scale/zero-point params
        if getattr(config, 'use_qat', False):
            self.enable_qat()
    
    def _create_attention_layer(self):
        """Create attention layer for heavy experts with optional QAT"""
        return nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=max(1, self.hidden_size // 64),
                dropout=0.1,
                batch_first=True
            ),
            nn.Dropout(0.1)
        )
    
    def _create_ffn_layer(self, in_dim):
        """Create feed-forward layer with optional QAT"""
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim * self.expansion_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim * self.expansion_factor, self.hidden_size),
            nn.Dropout(0.1)
        )
    
    def _fake_quantize(self, x, num_bits=8):
        if not self.qat_enabled:
            return x
            
        # Get correct dtype to match AMP context
        dtype = x.dtype
        
        # Calculate quantization range (typically [-127, 127] for 8-bit)
        qmin = -(2**(num_bits-1))
        qmax = 2**(num_bits-1) - 1
        
        # Calculate scale based on x's range
        x_min = x.min().detach()
        x_max = x.max().detach()
        scale = (x_max - x_min) / (qmax - qmin) if x_max > x_min else torch.tensor(1.0, device=x.device)
        zero_point = qmin - torch.round(x_min / scale) if scale.item() != 0 else torch.tensor(0, device=x.device)
        
        # Fake quantize
        x_q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        
        # Dequantize
        x_dq = (x_q - zero_point) * scale
        
        return x_dq.to(dtype)  # Return with the original dtype
    
    def enable_qat(self):
        """Enable Quantization-Aware Training"""
        if not self.qat_enabled:
            self.qat_enabled = True
            
            # Initialize quantization state dictionary with parameters for each layer
            self.quant_state = {
                "input_proj": {"scale": None, "zero_point": None},
                "output_proj": {"scale": None, "zero_point": None},
                "layers": [{} for _ in range(len(self.layers))]
            }
            
            # Create projection matrices for KV cache if using attention
            if self.expert_type == "heavy" and self.use_attention:
                head_dim = self.hidden_size // max(1, self.hidden_size // 64)
                num_heads = max(1, self.hidden_size // 64)
                self.kv_linear_k = nn.Linear(self.hidden_size, num_heads * head_dim)
                self.kv_linear_v = nn.Linear(self.hidden_size, num_heads * head_dim)
    
    def disable_qat(self):
        """Disable Quantization-Aware Training"""
        if self.qat_enabled:
            self.qat_enabled = False
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.use_gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        self.use_gradient_checkpointing = False
    
    def init_kv_cache(self, batch_size, seq_len, device):
        """Initialize KV cache for attention layers"""
        if not self.expert_type == "heavy" or not self.use_attention:
            return None
            
        self.kv_cache_enabled = True
        kv_cache = {}
        
        # Create KV cache for attention layers
        head_dim = self.hidden_size // max(1, self.hidden_size // 64)
        num_heads = max(1, self.hidden_size // 64)
        
        # Initialize empty KV cache
        kv_cache = {
            "keys": torch.zeros(batch_size, 0, num_heads, head_dim, device=device),
            "values": torch.zeros(batch_size, 0, num_heads, head_dim, device=device),
            "last_seq_len": 0
        }
        
        return kv_cache
    
    def update_kv_cache(self, kv_cache, token_indices, token_outputs):
        """Update KV cache with new tokens"""
        if not self.kv_cache_enabled or kv_cache is None:
            return kv_cache
            
        if not self.expert_type == "heavy" or not self.use_attention:
            return kv_cache
            
        # Ensure KV projection matrices exist
        if self.kv_linear_k is None or self.kv_linear_v is None:
            head_dim = self.hidden_size // max(1, self.hidden_size // 64)
            num_heads = max(1, self.hidden_size // 64)
            self.kv_linear_k = nn.Linear(self.hidden_size, num_heads * head_dim).to(token_outputs.device)
            self.kv_linear_v = nn.Linear(self.hidden_size, num_heads * head_dim).to(token_outputs.device)
            
        # Group indices by batch
        batch_indices = token_indices[0].unique()
        
        # Process each batch separately
        updated_cache = {
            "keys": kv_cache["keys"].clone(),
            "values": kv_cache["values"].clone(),
            "last_seq_len": kv_cache["last_seq_len"]
        }
        
        # Extract batch size, num_heads, and head_dim
        batch_size = updated_cache["keys"].size(0)
        num_heads = updated_cache["keys"].size(2) if updated_cache["keys"].size(0) > 0 else \
                    max(1, self.hidden_size // 64)
        head_dim = updated_cache["keys"].size(3) if updated_cache["keys"].size(0) > 0 else \
                   self.hidden_size // num_heads
        
        # Get maximum sequence position
        max_seq_pos = token_indices[1].max().item() + 1
        
        # If cache needs to grow
        if max_seq_pos > updated_cache["last_seq_len"]:
            # Compute how many new positions we need
            new_tokens = max_seq_pos - updated_cache["last_seq_len"]
            
            # Calculate new keys and values
            if len(token_outputs) > 0:
                # Project token outputs to get keys and values
                keys = self.kv_linear_k(token_outputs).view(-1, num_heads, head_dim)
                values = self.kv_linear_v(token_outputs).view(-1, num_heads, head_dim)
                
                # Create expanded cache
                new_keys = torch.zeros(
                    batch_size, max_seq_pos, num_heads, head_dim,
                    device=token_outputs.device, dtype=token_outputs.dtype
                )
                new_values = torch.zeros(
                    batch_size, max_seq_pos, num_heads, head_dim,
                    device=token_outputs.device, dtype=token_outputs.dtype
                )
                
                # Copy existing cache if it exists
                if updated_cache["last_seq_len"] > 0:
                    new_keys[:, :updated_cache["last_seq_len"]] = updated_cache["keys"]
                    new_values[:, :updated_cache["last_seq_len"]] = updated_cache["values"]
                
                # Update cache with new token values
                for i, (b, s) in enumerate(zip(token_indices[0], token_indices[1])):
                    new_keys[b, s] = keys[i]
                    new_values[b, s] = values[i]
                
                # Update cache
                updated_cache["keys"] = new_keys
                updated_cache["values"] = new_values
                updated_cache["last_seq_len"] = max_seq_pos
        
        return updated_cache
    
    def forward(self, hidden_states, kv_cache=None, use_checkpointing=None):
        """Forward pass through the expert with QAT and KV cache support"""
        # Override checkpointing from function parameter if provided
        if use_checkpointing is not None:
            use_checkpointing = use_checkpointing
        else:
            use_checkpointing = self.use_gradient_checkpointing and self.training
            
        # Input projection with optional QAT
        if self.qat_enabled and self.training:
            x = self._fake_quantize(self.input_proj(hidden_states), self.quant_bits)
        else:
            x = self.input_proj(hidden_states)
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if self.expert_type == "heavy" and self.use_attention and i % 2 == 0:
                # Attention layer
                norm_x = layer[0](x)  # Layer norm
                attn_layer = layer[1]  # Attention
                dropout = layer[2]  # Dropout
                
                # Apply attention with KV cache if provided
                if kv_cache is not None and self.kv_cache_enabled:
                    # Use cached key/value states if provided
                    attn_out, _ = attn_layer(
                        norm_x, 
                        kv_cache["keys"] if kv_cache["keys"].size(1) > 0 else norm_x, 
                        kv_cache["values"] if kv_cache["values"].size(1) > 0 else norm_x
                    )
                else:
                    # Standard attention
                    attn_out, _ = attn_layer(norm_x, norm_x, norm_x)
                
                # Apply fake quantization if QAT is enabled
                if self.qat_enabled and self.training:
                    attn_out = self._fake_quantize(attn_out, self.quant_bits)
                
                # Apply dropout and residual connection
                x = x + dropout(attn_out)
            else:
                # Feed-forward layer with optional checkpointing
                if use_checkpointing:
                    def create_custom_forward(module, x_input):
                        def custom_forward(*inputs):
                            x_for_module = inputs[0]
                            return module(x_for_module)
                        return custom_forward
                    
                    layer_output = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer, x),
                        x,
                        use_reentrant=False
                    )
                else:
                    layer_output = layer(x)
                
                # Apply fake quantization if QAT is enabled
                if self.qat_enabled and self.training:
                    layer_output = self._fake_quantize(layer_output, self.quant_bits)
                
                # Apply residual connection
                x = x + layer_output
        
        # Output projection and normalization
        if self.qat_enabled and self.training:
            x = self._fake_quantize(self.output_proj(x), self.quant_bits)
        else:
            x = self.output_proj(x)
            
        x = self.layer_norm(x)
        
        return x

class EnhancedMemoryEfficientMoE(nn.Module):
    """
    Ultra memory-efficient Mixture of Experts for long contexts (32K-64K):
    1. Adaptive hierarchical routing that scales with sequence length
    2. Expert-specific KV caches to reduce memory usage
    3. Gradient checkpointing for memory-efficient training
    4. Quantization-aware training
    5. Improved capacity enforcement with overlap handling
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        
        # Create adaptive hierarchical router
        self.router = AdaptiveHierarchicalRouter(config)
        
        # Create heterogeneous experts
        self.experts = nn.ModuleList([
            QAwareHeterogeneousExpertModule(config, i) for i in range(config.num_experts)
        ])
        
        # Layer norm before routing
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # For sparse activation storage
        self.use_sparse_activations = getattr(config, 'use_sparse_activations', True)
        
        # Expert capacity factor with overlap handling
        self.capacity_factor = getattr(config, 'capacity_factor', 1.5)
        
        # Expert-specific KV caches
        self.expert_kv_caches = None
        self.use_expert_kv_caches = getattr(config, 'use_expert_kv_caches', True)
        
        # Gradient checkpointing
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        
        # Expert computation order optimization (process largest expert batches first)
        self.optimize_expert_computation = getattr(config, 'optimize_expert_computation', True)
        
        # Loss scaling for auxiliary losses
        self.aux_loss_scale_factor = getattr(config, 'aux_loss_scale_factor', 0.01)
        self.adaptive_aux_loss_scaling = getattr(config, 'adaptive_aux_loss_scaling', True)
        
        # For tracking computation
        self.register_buffer('total_tokens', torch.tensor(0))
        self.register_buffer('routed_tokens', torch.tensor(0))
        self.register_buffer('task_loss_history', torch.zeros(10))  # For adaptive aux loss scaling
        self.task_loss_ptr = 0
    
    def _compute_capacity_with_overlap(self, routing_weights, expert_idx):
        """
        Compute and enforce expert capacity constraints with overlap handling.
        Ensures tokens routed to multiple experts are counted properly.
        """
        batch_size, seq_len, num_experts = routing_weights.shape
        total_tokens = batch_size * seq_len
        
        # Get assignment counts per expert (accounting for tokens assigned to multiple experts)
        expert_assignment_matrix = (routing_weights > 0).float()  # [batch_size, seq_len, num_experts]
        token_assignment_counts = expert_assignment_matrix.sum(dim=2)  # [batch_size, seq_len]
        
        # Scale expert weights by assignment counts to prevent tokens from counting multiple times
        scaled_weights = routing_weights.clone()
        token_mask = token_assignment_counts > 0
        if token_mask.any():
            # Only scale where tokens are assigned to at least one expert
            token_counts_expanded = token_assignment_counts.unsqueeze(-1).expand_as(routing_weights)
            scaled_weights = torch.where(
                token_counts_expanded > 0,
                routing_weights / token_counts_expanded,
                routing_weights
            )
        
        # Count effective tokens per expert using scaled weights
        expert_counts = scaled_weights.sum(dim=(0, 1))
        
        # Compute capacity (with expert-specific adjustments for heavy/light experts)
        base_capacity = int(total_tokens * self.capacity_factor / self.num_experts)
        
        # Expert-specific capacity multiplier (heavy experts get more capacity)
        expert_type = "heavy" if expert_idx % 3 == 0 else "light"
        capacity_multiplier = 1.5 if expert_type == "heavy" else 0.8
        capacity = int(base_capacity * capacity_multiplier)
        
        # Check if expert is over capacity
        if expert_counts[expert_idx] > capacity:
            # Find the top-k tokens by routing probability
            expert_probs = routing_weights[:, :, expert_idx].reshape(-1)
            _, indices = torch.topk(expert_probs, capacity)
            
            # Create mask for tokens within capacity
            mask = torch.zeros_like(expert_probs)
            mask[indices] = 1.0
            mask = mask.reshape(routing_weights.shape[0], routing_weights.shape[1])
            
            return mask
        else:
            # All tokens are within capacity
            return (routing_weights[:, :, expert_idx] > 0).float()
    
    def init_expert_kv_caches(self, batch_size, seq_len, device):
        """Initialize expert-specific KV caches"""
        if not self.use_expert_kv_caches:
            return None
            
        # Create KV caches for each expert
        self.expert_kv_caches = {}
        for expert_idx in range(self.num_experts):
            expert = self.experts[expert_idx]
            if hasattr(expert, 'init_kv_cache'):
                # Expert class handles its own KV cache
                self.expert_kv_caches[expert_idx] = expert.init_kv_cache(batch_size, seq_len, device)
            else:
                # Default cache initialization for attention layers
                self.expert_kv_caches[expert_idx] = None
        
        return self.expert_kv_caches
    
    def update_kv_cache(self, expert_idx, token_indices, token_outputs):
        """Update KV cache for specific expert based on processed tokens"""
        if not self.use_expert_kv_caches or self.expert_kv_caches is None:
            return
            
        if expert_idx in self.expert_kv_caches:
            expert = self.experts[expert_idx]
            if hasattr(expert, 'update_kv_cache'):
                self.expert_kv_caches[expert_idx] = expert.update_kv_cache(
                    self.expert_kv_caches[expert_idx], token_indices, token_outputs
                )
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency during training"""
        self.use_gradient_checkpointing = True
        for expert in self.experts:
            if hasattr(expert, 'enable_gradient_checkpointing'):
                expert.enable_gradient_checkpointing()
        return self
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        self.use_gradient_checkpointing = False
        for expert in self.experts:
            if hasattr(expert, 'disable_gradient_checkpointing'):
                expert.disable_gradient_checkpointing()
        return self
    
    def update_task_loss_history(self, task_loss):
        """Track task loss for adaptive auxiliary loss scaling"""
        if task_loss is not None and self.adaptive_aux_loss_scaling:
            self.task_loss_history[self.task_loss_ptr] = task_loss.detach()
            self.task_loss_ptr = (self.task_loss_ptr + 1) % self.task_loss_history.size(0)
    
    def get_adaptive_aux_loss_scale(self, aux_loss=None):
        """Calculate adaptive scale for auxiliary losses based on task loss history"""
        if not self.adaptive_aux_loss_scaling:
            return self.aux_loss_scale_factor
            
        # Get mean of recent task losses
        mean_task_loss = self.task_loss_history.mean().item()
        if mean_task_loss == 0:
            return self.aux_loss_scale_factor
            
        # Scale aux loss to be proportional to task loss but not dominate it
        if aux_loss is not None:
            # Direct scaling based on specific aux loss
            aux_loss_val = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
            if aux_loss_val > 0:
                # Target ratio: aux_loss should be 5-10% of task loss
                target_ratio = 0.07  # 7%
                current_ratio = aux_loss_val / (mean_task_loss + 1e-10)
                adaptive_scale = self.aux_loss_scale_factor * (target_ratio / (current_ratio + 1e-10))
                # Clamp to reasonable range
                return max(1e-5, min(1e-1, adaptive_scale))
        
        # Default scaling based on task loss magnitude
        return self.aux_loss_scale_factor * (0.1 / (mean_task_loss + 1e-10))
    
    def forward(self, hidden_states, task_loss=None):
        """
        Memory-optimized forward pass for long sequences.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            task_loss: Optional current task loss for adaptive scaling
            
        Returns:
            output: Output tensor [batch_size, seq_len, hidden_size]
            aux_loss: Auxiliary loss for load balancing
            metadata: Dict with metadata for monitoring
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Update task loss tracker for adaptive scaling
        self.update_task_loss_history(task_loss)
        
        # Normalize input
        normalized_states = self.layer_norm(hidden_states)
        
        # Get routing weights from adaptive hierarchical router
        routing_weights, load_balance_loss, metadata = self.router(normalized_states)
        
        # Initialize output tensor
        output = torch.zeros_like(hidden_states)
        
        # Initialize expert KV caches for this forward pass
        if self.training or self.expert_kv_caches is None:
            self.init_expert_kv_caches(batch_size, seq_len, device)
        
        # Update tracking metrics
        with torch.no_grad():
            self.total_tokens += batch_size * seq_len
            self.routed_tokens += torch.sum(routing_weights > 0)
        
        # Compute number of tokens per expert for optimized processing order
        expert_token_counts = []
        for expert_idx in range(self.num_experts):
            # Compute capacity and mask with improved overlap handling
            expert_mask = self._compute_capacity_with_overlap(routing_weights, expert_idx)
            count = torch.sum(expert_mask).item()
            expert_token_counts.append((expert_idx, count))
        
        # Process experts in order of decreasing token count for better utilization
        if self.optimize_expert_computation:
            expert_token_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Process through experts with sparse activation storage
        for expert_idx, token_count in expert_token_counts:
            # Skip if no tokens routed to this expert
            if token_count == 0:
                continue
            
            # Compute capacity and mask with improved overlap handling
            expert_mask = self._compute_capacity_with_overlap(routing_weights, expert_idx)
            
            # Skip this expert if no tokens are routed to it after capacity enforcement
            if not torch.any(expert_mask):
                continue
            
            # Extract only the tokens that are routed to this expert
            if self.use_sparse_activations:
                # Find indices of tokens routed to this expert
                indices = torch.nonzero(expert_mask, as_tuple=True)
                
                if len(indices[0]) == 0:
                    continue
                
                # Extract only those token embeddings
                expert_inputs = normalized_states[indices]
                
                # Process through expert with gradient checkpointing if enabled
                if self.use_gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(inputs[0])
                        return custom_forward
                    
                    expert_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.experts[expert_idx]),
                        expert_inputs,
                        use_reentrant=False
                    )
                else:
                    expert_outputs = self.experts[expert_idx](expert_inputs)
                
                # Update KV cache for this expert
                self.update_kv_cache(expert_idx, indices, expert_outputs)
                
                # Weight by router probability
                weight_indices = (indices[0], indices[1], torch.tensor([expert_idx] * len(indices[0]), device=expert_mask.device))
                expert_weights = routing_weights[weight_indices].unsqueeze(-1)
                weighted_outputs = expert_outputs * expert_weights
                
                # Scatter back to output tensor
                for i in range(len(indices[0])):
                    b, s = indices[0][i], indices[1][i]
                    output[b, s] += weighted_outputs[i]
            else:
                # Traditional approach - process all tokens but mask out irrelevant ones
                expert_weights = routing_weights[:, :, expert_idx].unsqueeze(-1)
                if self.use_gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(inputs[0])
                        return custom_forward
                    
                    expert_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.experts[expert_idx]),
                        normalized_states,
                        use_reentrant=False
                    )
                else:
                    expert_outputs = self.experts[expert_idx](normalized_states)
                
                output += expert_outputs * expert_weights
        
        # Add residual connection
        output = output + hidden_states
        
        # Scale auxiliary loss based on task loss or fixed factor
        aux_loss_scale = self.get_adaptive_aux_loss_scale(load_balance_loss)
        scaled_load_balance_loss = load_balance_loss * aux_loss_scale
        
        # Update metadata with token utilization stats and scaling info
        metadata.update({
            "routed_token_ratio": (self.routed_tokens / self.total_tokens).item(),
            "aux_loss_scale": aux_loss_scale
        })
        
        return output, scaled_load_balance_loss, metadata

class EnhancedRWKVMoEIntegration(nn.Module):
    """
    Improved RWKV + MoE integration that better leverages RWKV's recurrent state.
    - Uses RWKV for long-range dependencies with state tracking
    - Applies attention-based routing between RWKV and MoE paths
    - Optimization for ultra-long context sequences up to 64K tokens
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # MoE component
        self.moe = EnhancedMemoryEfficientMoE(config)
        
        # Advanced gating mechanism based on content and state
        self.token_content_proj = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.rwkv_state_proj = nn.Linear(self.hidden_size, self.hidden_size // 2)
        
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.LayerNorm(self.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Improved feature fusion for better recurrent state integration
        self.fusion_layer = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        # State-aware routing layer
        self.state_aware_router = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, 2)  # 2 outputs: RWKV or MoE
        )
        
        # State compression for memory efficiency
        self.state_compressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.hidden_size // 4, self.hidden_size)
        )
        
        # State expansion for decompression
        self.state_expander = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        # For tracking system usage
        self.register_buffer('rwkv_usage', torch.tensor(0.0))
        self.register_buffer('moe_usage', torch.tensor(0.0))
        self.register_buffer('total_usage', torch.tensor(0.0))
        
        # State tracking
        self.use_state_compression = getattr(config, 'use_state_compression', False)
        self.use_recurrent_integration = getattr(config, 'use_recurrent_integration', True)
    
    def enable_state_compression(self):
        """Enable state compression for memory efficiency"""
        self.use_state_compression = True
        return self
    
    def disable_state_compression(self):
        """Disable state compression"""
        self.use_state_compression = False
        return self
    
    def compress_state(self, state):
        """Compress RWKV state for memory efficiency"""
        if not self.use_state_compression:
            return state
            
        return self.state_compressor(state)
    
    def expand_state(self, compressed_state):
        """Expand compressed RWKV state"""
        if not self.use_state_compression:
            return compressed_state
            
        return self.state_expander(compressed_state)
    
    def forward(self, hidden_states, rwkv_states=None, recurrent_state=None, task_loss=None):
        """
        Forward pass with enhanced RWKV state integration.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            rwkv_states: States from RWKV processing (optional)
            recurrent_state: Previous recurrent state from this module (optional)
            task_loss: Task loss for adaptive scaling
            
        Returns:
            output: Enhanced output tensor
            aux_loss: Auxiliary loss
            metadata: Dict with metadata for monitoring
            new_recurrent_state: Updated recurrent state for stateful processing
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Process through MoE
        moe_output, aux_loss, metadata = self.moe(hidden_states, task_loss)
        
        # If RWKV states are provided, we're in the integrated mode
        if rwkv_states is not None:
            # For ultra-long sequences, apply state-aware routing
            if self.use_recurrent_integration and seq_len > 4096:
                # Process with recurrent awareness
                state_features = self.rwkv_state_proj(rwkv_states)
                content_features = self.token_content_proj(hidden_states)
                
                # Combine state and content features for gate computation
                gate_input = torch.cat([state_features, content_features], dim=-1)
                
                # Compute gating values based on both state and content
                combined_features = torch.cat([rwkv_states, hidden_states], dim=-1)
                router_logits = self.state_aware_router(combined_features)
                router_probs = F.softmax(router_logits, dim=-1)
                
                # Determine if token should use RWKV, MoE or both
                rwkv_prob = router_probs[:, :, 0].unsqueeze(-1)
                moe_prob = router_probs[:, :, 1].unsqueeze(-1)
                
                # Content-based gate values
                gate_values = self.gate(hidden_states)
                
                # State influence factor (how much the recurrent state influences routing)
                if recurrent_state is not None:
                    state_influence = F.sigmoid(recurrent_state.mean(dim=1, keepdim=True))
                    state_influence = state_influence.unsqueeze(1).expand(-1, seq_len, -1)
                else:
                    state_influence = torch.ones((batch_size, seq_len, 1), device=device) * 0.5
                
                # Combine all factors for final weights
                rwkv_weight = rwkv_prob * (1 - gate_values) * state_influence
                moe_weight = moe_prob * gate_values * (1 - state_influence)
                
                # Normalize weights
                sum_weights = rwkv_weight + moe_weight + 1e-8
                rwkv_weight = rwkv_weight / sum_weights
                moe_weight = moe_weight / sum_weights
            else:
                # Simpler routing for shorter sequences
                gate_values = self.gate(hidden_states)
                rwkv_weight = (1 - gate_values)
                moe_weight = gate_values
            
            # Update tracking metrics
            with torch.no_grad():
                self.rwkv_usage += torch.mean(rwkv_weight).item()
                self.moe_usage += torch.mean(moe_weight).item()
                self.total_usage += 1
            
            # Combine RWKV and MoE outputs
            combined_output = (rwkv_weight * rwkv_states) + (moe_weight * moe_output)
            
            # Final fusion layer
            output = self.fusion_layer(combined_output)
            
            # Update recurrent state
            if self.use_recurrent_integration:
                # Create new recurrent state by combining current outputs and previous state
                if recurrent_state is not None:
                    # Update with exponential moving average
                    new_recurrent_state = 0.9 * recurrent_state + 0.1 * output.mean(dim=1)
                else:
                    # Initialize new state
                    new_recurrent_state = output.mean(dim=1)
                
                # Apply state compression if enabled
                if self.use_state_compression:
                    new_recurrent_state = self.compress_state(new_recurrent_state)
            else:
                new_recurrent_state = None
            
            # Update metadata
            metadata.update({
                "rwkv_usage": (self.rwkv_usage / self.total_usage).item(),
                "moe_usage": (self.moe_usage / self.total_usage).item(),
                "state_compression": self.use_state_compression,
                "recurrent_integration": self.use_recurrent_integration
            })
            
            return output, aux_loss, metadata, new_recurrent_state
        else:
            # MoE-only mode (no RWKV states provided)
            return moe_output, aux_loss, metadata, None

class MoEBenchmarker:
    """
    Utility class for benchmarking MoE models with long contexts.
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.results = {}
    
    def benchmark_memory(self, seq_lengths=[1024, 4096, 16384, 32768]):
        """Benchmark memory usage at different sequence lengths"""
        import gc
        import torch
        
        results = {}
        
        for seq_len in seq_lengths:
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
            
            # Measure starting memory
            start_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            
            # Create dummy input
            batch_size = 1
            dummy_input = torch.randn(
                batch_size, seq_len, self.config.hidden_size,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Forward pass
            with torch.no_grad():
                try:
                    _ = self.model(dummy_input)
                    
                    # Measure memory after forward pass
                    end_mem = torch.cuda.memory_allocated() / (1024 ** 3)
                    memory_used = end_mem - start_mem
                    results[seq_len] = {
                        "memory_gb": memory_used,
                        "success": True
                    }
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        results[seq_len] = {
                            "memory_gb": None,
                            "success": False,
                            "error": "OOM"
                        }
                    else:
                        results[seq_len] = {
                            "memory_gb": None,
                            "success": False,
                            "error": str(e)
                        }
            
            # Clear memory again
            gc.collect()
            torch.cuda.empty_cache()
        
        self.results["memory"] = results
        return results
    
    def benchmark_throughput(self, seq_lengths=[1024, 4096, 16384, 32768]):
        """Benchmark throughput at different sequence lengths"""
        import time
        import torch
        
        results = {}
        
        for seq_len in seq_lengths:
            # Create dummy input
            batch_size = 1
            dummy_input = torch.randn(
                batch_size, seq_len, self.config.hidden_size,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Warmup
            with torch.no_grad():
                try:
                    _ = self.model(dummy_input)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        results[seq_len] = {
                            "tokens_per_sec": None,
                            "success": False,
                            "error": "OOM"
                        }
                    else:
                        results[seq_len] = {
                            "tokens_per_sec": None,
                            "success": False,
                            "error": str(e)
                        }
                    continue
            
            # Benchmark
            num_runs = 5
            total_time = 0
            
            with torch.no_grad():
                try:
                    for _ in range(num_runs):
                        start_time = time.time()
                        _ = self.model(dummy_input)
                        torch.cuda.synchronize()
                        end_time = time.time()
                        total_time += (end_time - start_time)
                    
                    avg_time = total_time / num_runs
                    tokens_per_sec = seq_len / avg_time
                    
                    results[seq_len] = {
                        "tokens_per_sec": tokens_per_sec,
                        "time_per_token_ms": (avg_time * 1000) / seq_len,
                        "success": True
                    }
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        results[seq_len] = {
                            "tokens_per_sec": None,
                            "success": False,
                            "error": "OOM"
                        }
                    else:
                        results[seq_len] = {
                            "tokens_per_sec": None,
                            "success": False,
                            "error": str(e)
                        }
        
        self.results["throughput"] = results
        return results
    
    def print_report(self):
        """Print benchmark results in a formatted table"""
        if not self.results:
            print("No benchmark results available. Run benchmarks first.")
            return
        
        print("\n==== MoE Benchmark Report ====\n")
        
        if "memory" in self.results:
            print("Memory Usage:")
            print(f"{'Seq Length':<12} {'Memory (GB)':<15} {'Success':<10}")
            print("-" * 37)
            
            for seq_len, result in sorted(self.results["memory"].items()):
                memory = f"{result['memory_gb']:.2f}" if result["memory_gb"] is not None else "N/A"
                success = "✓" if result["success"] else "✗"
                print(f"{seq_len:<12} {memory:<15} {success:<10}")
            
            print()
        
        if "throughput" in self.results:
            print("Throughput:")
            print(f"{'Seq Length':<12} {'Tokens/Sec':<15} {'Time/Token (ms)':<18} {'Success':<10}")
            print("-" * 55)
            
            for seq_len, result in sorted(self.results["throughput"].items()):
                if result["success"]:
                    tokens_per_sec = f"{result['tokens_per_sec']:.2f}"
                    time_per_token = f"{result['time_per_token_ms']:.3f}"
                else:
                    tokens_per_sec = "N/A"
                    time_per_token = "N/A"
                
                success = "✓" if result["success"] else "✗"
                print(f"{seq_len:<12} {tokens_per_sec:<15} {time_per_token:<18} {success:<10}")
            
            print()
        
        print("==== End of Report ====\n") 

class LongContextMoEConfig:
    """Configuration for MoE models specialized for ultra-long contexts (32K-64K)"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_experts: int = 8,
        max_seq_length: int = 32768,
        block_size: int = 512,
        token_routing_budget: float = 0.3,
        use_rwkv_integration: bool = True,
        use_gradient_checkpointing: bool = True,
        use_state_compression: bool = True,
        use_quantization: bool = True,
        use_qat: bool = True,
        use_expert_kv_caches: bool = True,
        use_sparse_activations: bool = True,
        optimize_expert_computation: bool = True,
        adaptive_aux_loss_scaling: bool = True,
        aux_loss_scale_factor: float = 0.01,
        capacity_factor: float = 1.5,
        heavy_expert_layers: int = 4,
        light_expert_layers: int = 1
    ):
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.max_seq_length = max_seq_length
        self.block_size = block_size
        self.token_routing_budget = token_routing_budget
        self.use_rwkv_integration = use_rwkv_integration
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_state_compression = use_state_compression
        self.use_quantization = use_quantization
        self.use_qat = use_qat
        self.use_expert_kv_caches = use_expert_kv_caches
        self.use_sparse_activations = use_sparse_activations
        self.optimize_expert_computation = optimize_expert_computation
        self.adaptive_aux_loss_scaling = adaptive_aux_loss_scaling
        self.aux_loss_scale_factor = aux_loss_scale_factor
        self.capacity_factor = capacity_factor
        self.heavy_expert_layers = heavy_expert_layers
        self.light_expert_layers = light_expert_layers
        
        # Adjustments for ultra-long sequences
        if max_seq_length >= 32768:
            self.token_routing_budget = min(0.2, token_routing_budget)
            self.use_gradient_checkpointing = True
            
        if max_seq_length >= 65536:
            self.token_routing_budget = min(0.1, token_routing_budget)
            self.block_size = min(2048, block_size * 4)
            self.use_state_compression = True

def create_long_context_moe(config):
    """
    Create a MoE model optimized for ultra-long contexts (32K-64K).
    
    Args:
        config: Configuration object
        
    Returns:
        Long-context optimized MoE model
    """
    # Ensure config has the necessary parameters
    if not hasattr(config, 'block_size'):
        config.block_size = 512
    
    if not hasattr(config, 'use_sparse_activations'):
        config.use_sparse_activations = True
    
    # Scale block size based on target sequence length
    max_seq_len = getattr(config, 'max_seq_length', 16384)
    if max_seq_len >= 32768:  # Ultra-long context mode
        config.block_size = min(2048, config.block_size * 4)
        config.token_routing_budget = 0.15  # Reduce token routing for ultra-long contexts
        config.use_state_compression = True
        
        # For 64K contexts, further reduce the budget
        if max_seq_len >= 65536:
            config.token_routing_budget = 0.1
            config.use_gradient_checkpointing = True
    
    # Enable QAT by default for long contexts
    if not hasattr(config, 'use_qat'):
        config.use_qat = True
    
    # Create the model
    if getattr(config, 'use_rwkv_integration', False):
        model = EnhancedRWKVMoEIntegration(config)
        
        # Enable state compression for very long contexts
        if getattr(config, 'use_state_compression', False):
            model.enable_state_compression()
    else:
        model = EnhancedMemoryEfficientMoE(config)
        
        # Enable gradient checkpointing for memory efficiency
        if getattr(config, 'use_gradient_checkpointing', False):
            model.enable_gradient_checkpointing()
    
    # Apply quantization if specified
    if getattr(config, 'use_quantization', False):
        bits = getattr(config, 'quantization_bits', 8)
        # Apply per-expert quantization precision
        if hasattr(model, 'experts'):
            for i, expert in enumerate(model.experts):
                # Use 4-bit for light experts, 8-bit for heavy experts
                expert_bits = 8 if i % 3 == 0 else 4
                if hasattr(expert, 'enable_qat'):
                    expert.enable_qat()
    
    return model 

def example_long_context_moe_creation():
    """Example of creating a long-context optimized MoE model"""
    # Create configuration for 64K context
    config = LongContextMoEConfig(
        hidden_size=1024,
        num_experts=16,
        max_seq_length=65536,  # 64K context
        block_size=1024,
        token_routing_budget=0.15,
        use_rwkv_integration=True,
        use_gradient_checkpointing=True,
        use_state_compression=True,
        use_qat=True
    )
    
    # Create model
    model = create_long_context_moe(config)
    
    # Enable optimizations
    model.moe.enable_gradient_checkpointing()
    
    # For benchmarking
    benchmarker = MoEBenchmarker(model, config)
    
    # Return the model and benchmarker
    return model, benchmarker

def clear_kv_caches(self):
    """Clear expert KV caches to free memory"""
    self.expert_kv_caches = None
    # Force garbage collection
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def forward(self, hidden_states, task_loss=None):
    batch_size, seq_len, hidden_size = hidden_states.shape
    
    # Fallback for extremely long sequences
    if seq_len > self.max_safe_seq_len:
        try:
            return self._forward_impl(hidden_states, task_loss)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # Fallback to chunk processing
                return self._forward_chunked(hidden_states, task_loss)
            else:
                raise e