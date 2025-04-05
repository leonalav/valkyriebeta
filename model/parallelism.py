"""
Advanced parallelism strategies for large language models.
Provides tensor parallelism, sequence parallelism, and expert parallelism.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import math
import types

logger = logging.getLogger(__name__)

def apply_tensor_parallelism(
    model: nn.Module,
    tensor_parallel_size: int = 2,
    communication_mode: str = "allreduce"  # "allreduce", "ring", "hierarchical"
) -> nn.Module:
    """
    Apply tensor parallelism to a model, splitting operations across devices.
    
    Args:
        model: Model to parallelize
        tensor_parallel_size: Number of devices for tensor parallelism
        communication_mode: Communication strategy for parallel operations
        
    Returns:
        Parallelized model
    """
    # Check if distributed is initialized
    if not dist.is_available() or not dist.is_initialized():
        logger.warning("PyTorch distributed not initialized, tensor parallelism not applied")
        return model
    
    # Get local rank and world size
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if world_size < tensor_parallel_size:
        logger.warning(f"Requested tensor_parallel_size {tensor_parallel_size} > world_size {world_size}")
        tensor_parallel_size = world_size
    
    # Compute tensor parallel group
    tp_group = None
    tp_rank = 0
    if tensor_parallel_size > 1:
        # Create process groups for tensor parallelism
        for tp_id in range(world_size // tensor_parallel_size):
            ranks = list(range(tp_id * tensor_parallel_size, (tp_id + 1) * tensor_parallel_size))
            group = dist.new_group(ranks=ranks)
            
            # Store my group
            if local_rank in ranks:
                tp_group = group
                tp_rank = ranks.index(local_rank)
    
    if tp_group is None and tensor_parallel_size > 1:
        logger.warning("Could not create tensor parallel group, tensor parallelism not applied")
        return model
    
    # Apply tensor parallelism to model components
    _apply_tensor_parallel_to_linear_layers(model, tp_rank, tensor_parallel_size, tp_group, communication_mode)
    
    # Store tensor parallel info on model
    model.tensor_parallel_size = tensor_parallel_size
    model.tensor_parallel_rank = tp_rank
    model.tensor_parallel_group = tp_group
    
    logger.info(f"Applied tensor parallelism with size {tensor_parallel_size}, rank {tp_rank}, mode: {communication_mode}")
    
    return model


def _apply_tensor_parallel_to_linear_layers(
    model: nn.Module,
    tp_rank: int,
    tp_size: int,
    tp_group: Any,
    communication_mode: str
):
    """Apply tensor parallelism to linear layers in the model."""
    
    # Count parallelized layers
    count = 0
    
    # Parallelize different types of layers
    for name, module in model.named_modules():
        # General rule: parallelize any nn.Linear layer that's not an output layer
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            
            # For column parallelism (splitting output features)
            if "output" not in name.lower() and "head" not in name.lower():
                _apply_column_parallel(module, tp_rank, tp_size, tp_group, communication_mode)
                count += 1
            
            # For row parallelism (splitting input features)
            else:
                _apply_row_parallel(module, tp_rank, tp_size, tp_group, communication_mode)
                count += 1
    
    logger.info(f"Applied tensor parallelism to {count} layers")


def _apply_column_parallel(
    module: nn.Linear,
    tp_rank: int,
    tp_size: int,
    tp_group: Any,
    communication_mode: str
):
    """
    Apply column parallelism to a linear layer (split along output dimension).
    Each device computes a subset of the outputs.
    """
    orig_forward = module.forward
    orig_out_features = module.out_features
    
    # Adjust out_features
    module.out_features = module.out_features // tp_size
    
    # Adjust weight and bias
    with torch.no_grad():
        # Split weight along output dimension
        orig_weight = module.weight.data
        module.weight.data = orig_weight[tp_rank * module.out_features:(tp_rank + 1) * module.out_features, :]
        
        # Split bias if it exists
        if module.bias is not None:
            orig_bias = module.bias.data
            module.bias.data = orig_bias[tp_rank * module.out_features:(tp_rank + 1) * module.out_features]
    
    # Create new forward method
    def column_parallel_forward(self, x):
        # Just apply normal forward on local partition
        output = orig_forward(x)
        
        if communication_mode == "allreduce":
            # Do not perform any communication
            # Each device has a subset of the outputs
            return output
        else:
            # In future: implement ring or hierarchical communication for better scaling
            return output
    
    # Replace forward method
    module.forward = types.MethodType(column_parallel_forward, module)
    module.is_column_parallel = True
    module.orig_out_features = orig_out_features


def _apply_row_parallel(
    module: nn.Linear,
    tp_rank: int,
    tp_size: int,
    tp_group: Any,
    communication_mode: str
):
    """
    Apply row parallelism to a linear layer (split along input dimension).
    Each device computes the full output but with a subset of the inputs.
    """
    orig_forward = module.forward
    orig_in_features = module.in_features
    
    # Adjust in_features
    module.in_features = module.in_features // tp_size
    
    # Adjust weight
    with torch.no_grad():
        # Split weight along input dimension
        orig_weight = module.weight.data
        module.weight.data = orig_weight[:, tp_rank * module.in_features:(tp_rank + 1) * module.in_features]
    
    # Create new forward method
    def row_parallel_forward(self, x):
        # Apply normal forward on local partition
        output = orig_forward(x)
        
        # Gather outputs from all devices
        if communication_mode == "allreduce":
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
            return output
        else:
            # In future: implement ring or hierarchical communication for better scaling
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=tp_group)
            return output
    
    # Replace forward method
    module.forward = types.MethodType(row_parallel_forward, module)
    module.is_row_parallel = True
    module.orig_in_features = orig_in_features


def apply_sequence_parallelism(
    model: nn.Module,
    sequence_parallel_size: int = 2,
    chunk_size: int = 1024,
    overlap_communication: bool = True
) -> nn.Module:
    """
    Apply sequence parallelism to a model, processing different parts of the sequence on different devices.
    
    Args:
        model: Model to parallelize
        sequence_parallel_size: Number of devices for sequence parallelism
        chunk_size: Size of sequence chunks to process
        overlap_communication: Whether to overlap computation and communication
        
    Returns:
        Parallelized model
    """
    # Check if distributed is initialized
    if not dist.is_available() or not dist.is_initialized():
        logger.warning("PyTorch distributed not initialized, sequence parallelism not applied")
        return model
    
    # Get local rank and world size
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if world_size < sequence_parallel_size:
        logger.warning(f"Requested sequence_parallel_size {sequence_parallel_size} > world_size {world_size}")
        sequence_parallel_size = world_size
    
    # Compute sequence parallel group
    seq_group = None
    seq_rank = 0
    if sequence_parallel_size > 1:
        # Create process groups for sequence parallelism
        for seq_id in range(world_size // sequence_parallel_size):
            ranks = list(range(seq_id * sequence_parallel_size, (seq_id + 1) * sequence_parallel_size))
            group = dist.new_group(ranks=ranks)
            
            # Store my group
            if local_rank in ranks:
                seq_group = group
                seq_rank = ranks.index(local_rank)
    
    if seq_group is None and sequence_parallel_size > 1:
        logger.warning("Could not create sequence parallel group, sequence parallelism not applied")
        return model
    
    # Patch model forward method with sequence parallel version
    orig_forward = model.forward
    
    def sequence_parallel_forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Only apply to training or when input is large enough
        if not self.training or (input_ids is not None and input_ids.size(1) < chunk_size * sequence_parallel_size):
            return orig_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Apply sequence parallelism
        batch_size, seq_len = input_ids.size()
        
        # Compute chunk size dynamically if needed
        actual_chunk_size = max(1, seq_len // sequence_parallel_size)
        
        # Compute start and end indices for this rank's chunk
        start_idx = seq_rank * actual_chunk_size
        end_idx = min(seq_len, (seq_rank + 1) * actual_chunk_size)
        
        # Extract this rank's chunk
        chunk_input_ids = input_ids[:, start_idx:end_idx]
        chunk_attention_mask = attention_mask[:, start_idx:end_idx] if attention_mask is not None else None
        
        # Process chunk
        chunk_outputs = orig_forward(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask, **kwargs)
        
        # Gather and combine outputs
        if isinstance(chunk_outputs, torch.Tensor):
            # Simple case: single tensor output
            full_output_size = list(chunk_outputs.size())
            full_output_size[1] = seq_len
            
            # Create full output tensor
            full_output = torch.zeros(full_output_size, dtype=chunk_outputs.dtype, device=chunk_outputs.device)
            
            # Gather from all processes
            for i in range(sequence_parallel_size):
                # Calculate start and end for each rank
                rank_start = i * actual_chunk_size
                rank_end = min(seq_len, (i + 1) * actual_chunk_size)
                rank_size = rank_end - rank_start
                
                # Create placeholder for this rank's output
                if i == seq_rank:
                    # We already have our output
                    rank_output = chunk_outputs
                else:
                    # Create placeholder
                    rank_output_size = list(chunk_outputs.size())
                    rank_output_size[1] = rank_size
                    rank_output = torch.zeros(rank_output_size, dtype=chunk_outputs.dtype, device=chunk_outputs.device)
                
                # Broadcast from source rank to all others
                dist.broadcast(rank_output, src=i, group=seq_group)
                
                # Insert into full output
                full_output[:, rank_start:rank_end] = rank_output
            
            return full_output
        else:
            # Handle dictionary or other structured outputs
            logger.warning("Sequence parallelism with structured outputs not fully implemented")
            return chunk_outputs
    
    # Replace forward method
    model.forward = types.MethodType(sequence_parallel_forward, model)
    
    # Store sequence parallel info on model
    model.sequence_parallel_size = sequence_parallel_size
    model.sequence_parallel_rank = seq_rank
    model.sequence_parallel_group = seq_group
    model.sequence_chunk_size = chunk_size
    
    logger.info(f"Applied sequence parallelism with size {sequence_parallel_size}, rank {seq_rank}, chunk size: {chunk_size}")
    
    return model


def apply_expert_parallelism(
    model: nn.Module,
    expert_parallel_size: int = 4,
    communication_mode: str = "alltoall"  # "alltoall", "scatter_gather"
) -> nn.Module:
    """
    Apply expert parallelism to a Mixture of Experts model, distributing experts across devices.
    
    Args:
        model: Model with MoE components to parallelize
        expert_parallel_size: Number of devices for expert parallelism
        communication_mode: Communication strategy for expert routing
        
    Returns:
        Parallelized model
    """
    # Check if distributed is initialized
    if not dist.is_available() or not dist.is_initialized():
        logger.warning("PyTorch distributed not initialized, expert parallelism not applied")
        return model
    
    # Get local rank and world size
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if world_size < expert_parallel_size:
        logger.warning(f"Requested expert_parallel_size {expert_parallel_size} > world_size {world_size}")
        expert_parallel_size = world_size
    
    # Compute expert parallel group
    ep_group = None
    ep_rank = 0
    if expert_parallel_size > 1:
        # Create process groups for expert parallelism
        for ep_id in range(world_size // expert_parallel_size):
            ranks = list(range(ep_id * expert_parallel_size, (ep_id + 1) * expert_parallel_size))
            group = dist.new_group(ranks=ranks)
            
            # Store my group
            if local_rank in ranks:
                ep_group = group
                ep_rank = ranks.index(local_rank)
    
    if ep_group is None and expert_parallel_size > 1:
        logger.warning("Could not create expert parallel group, expert parallelism not applied")
        return model
    
    # Find and parallelize MoE layers
    count = 0
    
    # Find all MoE modules
    for name, module in model.named_modules():
        # Look for MoE layers by name or structure
        if any(moe_name in name.lower() for moe_name in ['moe', 'expert']):
            if hasattr(module, 'experts') and isinstance(module.experts, (list, nn.ModuleList)):
                # This is an MoE layer with a list of experts
                _parallelize_moe_layer(module, ep_rank, expert_parallel_size, ep_group, communication_mode)
                count += 1
            elif hasattr(module, 'expert_routing') or hasattr(module, 'router'):
                # This is an MoE layer with a router
                _parallelize_moe_layer(module, ep_rank, expert_parallel_size, ep_group, communication_mode)
                count += 1
    
    # Store expert parallel info on model
    model.expert_parallel_size = expert_parallel_size
    model.expert_parallel_rank = ep_rank
    model.expert_parallel_group = ep_group
    
    logger.info(f"Applied expert parallelism to {count} MoE layers with size {expert_parallel_size}, rank {ep_rank}")
    
    return model


def _parallelize_moe_layer(
    module: nn.Module,
    ep_rank: int,
    ep_size: int,
    ep_group: Any,
    communication_mode: str
):
    """
    Parallelize a Mixture of Experts layer by distributing experts across devices.
    
    Args:
        module: MoE module to parallelize
        ep_rank: Current device rank in expert parallel group
        ep_size: Size of expert parallel group
        ep_group: Expert parallel process group
        communication_mode: Communication strategy
    """
    # Handle case where experts are stored in a ModuleList
    if hasattr(module, 'experts') and isinstance(module.experts, (list, nn.ModuleList)):
        experts = module.experts
        num_experts = len(experts)
        
        # Compute experts per rank
        base_experts_per_rank = num_experts // ep_size
        remainder = num_experts % ep_size
        
        # Determine this rank's experts
        start_idx = ep_rank * base_experts_per_rank + min(ep_rank, remainder)
        experts_this_rank = base_experts_per_rank + (1 if ep_rank < remainder else 0)
        end_idx = start_idx + experts_this_rank
        
        # Keep only this rank's experts
        module.experts = nn.ModuleList(experts[start_idx:end_idx])
        module.num_local_experts = experts_this_rank
        module.num_total_experts = num_experts
        module.expert_parallel_rank = ep_rank
        module.expert_parallel_size = ep_size
        module.expert_parallel_group = ep_group
        
        # Save original forward method
        orig_forward = module.forward
        
        # Create new forward method with expert parallelism
        def expert_parallel_forward(self, hidden_states, *args, **kwargs):
            batch_size, seq_len, hidden_dim = hidden_states.size()
            
            # If training, apply expert parallelism
            if self.training:
                # Get router outputs (device local)
                if hasattr(self, 'router'):
                    router_logits = self.router(hidden_states)  # [batch, seq, num_total_experts]
                elif hasattr(self, 'expert_routing'):
                    router_logits = self.expert_routing(hidden_states)
                else:
                    # Fallback to direct attribute access
                    router_probs = getattr(self, 'router_probs', None)
                    if router_probs is None:
                        logger.warning("Could not find router in MoE layer, expert parallelism may not work correctly")
                        return orig_forward(hidden_states, *args, **kwargs)
                
                # Apply softmax to get expert weights
                router_probs = torch.softmax(router_logits, dim=-1)  # [batch, seq, num_total_experts]
                
                # For simplicity in this implementation, we'll use a basic version
                # In a full implementation, we would use all-to-all communication
                
                # Use only local experts
                local_expert_outputs = []
                for i, expert in enumerate(self.experts):
                    # Get global expert index
                    global_expert_idx = start_idx + i
                    
                    # Get tokens assigned to this expert
                    expert_weights = router_probs[:, :, global_expert_idx].unsqueeze(-1)  # [batch, seq, 1]
                    
                    # Apply expert to all tokens (inefficient but simple)
                    expert_out = expert(hidden_states)
                    
                    # Weight outputs by router probabilities
                    weighted_output = expert_out * expert_weights
                    local_expert_outputs.append(weighted_output)
                
                # Sum outputs from local experts
                local_output = sum(local_expert_outputs)
                
                # All-reduce across expert parallel group to get outputs from all experts
                if ep_size > 1:
                    dist.all_reduce(local_output, op=dist.ReduceOp.SUM, group=ep_group)
                
                return local_output
            else:
                # For inference, just use the original forward pass
                return orig_forward(hidden_states, *args, **kwargs)
        
        # Replace forward method
        module.forward = types.MethodType(expert_parallel_forward, module)
        
        logger.info(f"Parallelized MoE layer with {num_experts} experts, {experts_this_rank} on this rank")