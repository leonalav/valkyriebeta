import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
import gc
from torch.cuda import amp
import bitsandbytes as bnb
from functools import partial
from torch.utils.checkpoint import checkpoint
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    quantization_bits: int = 8
    use_gradient_checkpointing: bool = True
    use_parameter_sharing: bool = True
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    use_dynamic_memory: bool = True
    use_sparse_attention: bool = True
    pruning_threshold: float = 0.1
    activation_sparsity: bool = True

class MemoryOptimizer:
    def __init__(self, model: nn.Module, config: Any):
        self.model = model
        self.config = config
        self.scaler = amp.GradScaler()
        
    def optimize_memory(self):
        """Apply memory optimization techniques"""
        # Enable gradient checkpointing
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def quantize_model(self, quantization_bits: int = 8):
        """Quantize model weights"""
        if quantization_bits == 4:
            # 4-bit quantization using bitsandbytes
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    quantized_module = bnb.nn.Linear4Bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compute_dtype=torch.float16
                    )
                    parent_name = name.rsplit('.', 1)[0]
                    parent_module = self.model.get_submodule(parent_name)
                    setattr(parent_module, name.split('.')[-1], quantized_module)
        
        elif quantization_bits == 8:
            # 8-bit quantization using bitsandbytes
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    quantized_module = bnb.nn.Linear8Bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=True
                    )
                    parent_name = name.rsplit('.', 1)[0]
                    parent_module = self.model.get_submodule(parent_name)
                    setattr(parent_module, name.split('.')[-1], quantized_module)
    
    @staticmethod
    def apply_dynamic_padding(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply dynamic padding to batch"""
        max_length = max(len(seq) for seq in batch['input_ids'])
        padded_batch = {}
        
        for key, value in batch.items():
            if key in ['input_ids', 'attention_mask']:
                padded_batch[key] = torch.nn.utils.rnn.pad_sequence(
                    value,
                    batch_first=True,
                    padding_value=0
                )
        return padded_batch

class GradientOptimizer:
    def __init__(self, model: nn.Module, config: Any):
        self.model = model
        self.config = config
        
    def optimize_backward(self, loss: torch.Tensor):
        """Optimize backward pass"""
        # Gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Mixed precision training
        with amp.autocast():
            loss.backward()
    
    @staticmethod
    def get_grouped_parameters(model: nn.Module, weight_decay: float):
        """Get parameters grouped by weight decay"""
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

def apply_model_optimizations(model: nn.Module, config: Any) -> nn.Module:
    """Apply all optimization techniques to model"""
    memory_optimizer = MemoryOptimizer(model, config)
    
    # Apply memory optimizations
    memory_optimizer.optimize_memory()
    
    # Apply quantization if enabled
    if config.use_quantization:
        memory_optimizer.quantize_model(config.quantization_bits)
    
    # Enable model parallelism if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    return model 

class ModelOptimizer:
    def __init__(self, model: nn.Module, config: OptimizationConfig):
        self.model = model
        self.config = config
        
    def optimize(self) -> nn.Module:
        """Apply all optimization techniques to the model"""
        self.model = self.quantize_model()
        self.model = self.apply_gradient_checkpointing()
        self.model = self.apply_parameter_sharing()
        self.model = self.add_lora_layers()
        self.model = self.enable_dynamic_memory()
        self.model = self.implement_sparse_attention()
        self.model = self.apply_structured_pruning()
        self.model = self.apply_activation_sparsity()
        return self.model
    
    def quantize_model(self) -> nn.Module:
        """Quantize model weights using bitsandbytes"""
        def _quantize_layer(layer):
            if isinstance(layer, nn.Linear):
                if self.config.quantization_bits == 4:
                    return bnb.nn.Linear4bit(
                        layer.in_features,
                        layer.out_features,
                        bias=layer.bias is not None,
                        compute_dtype=torch.float16
                    )
                else:  # 8-bit quantization
                    return bnb.nn.Linear8bitLt(
                        layer.in_features,
                        layer.out_features,
                        bias=layer.bias is not None,
                        threshold=6.0
                    )
            return layer
            
        for name, module in self.model.named_children():
            if isinstance(module, nn.ModuleList):
                for i, layer in enumerate(module):
                    module[i] = _quantize_layer(layer)
            else:
                setattr(self.model, name, _quantize_layer(module))
                
        return self.model
    
    def apply_gradient_checkpointing(self) -> nn.Module:
        """Enable gradient checkpointing for memory efficiency"""
        if self.config.use_gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*args):
                    return module(*args)
                return custom_forward
            
            # Apply to transformer layers
            if hasattr(self.model, 'transformer'):
                for layer in self.model.transformer.h:
                    layer.forward = checkpoint(
                        create_custom_forward(layer.forward),
                        use_reentrant=False
                    )
        return self.model
    
    def apply_parameter_sharing(self) -> nn.Module:
        """Implement ALBERT-style parameter sharing"""
        if self.config.use_parameter_sharing:
            if hasattr(self.model, 'transformer'):
                # Group layers into clusters
                num_layers = len(self.model.transformer.h)
                num_clusters = math.ceil(num_layers / 3)  # Share parameters among 3 layers
                
                for i in range(1, num_layers):
                    if i % 3 != 0:  # Share parameters within each cluster
                        cluster_head = (i // 3) * 3
                        self.model.transformer.h[i] = self.model.transformer.h[cluster_head]
                        
        return self.model
    
    def add_lora_layers(self) -> nn.Module:
        """Add LoRA layers for efficient fine-tuning"""
        if self.config.use_lora:
            class LoRALayer(nn.Module):
                def __init__(self, layer: nn.Linear, rank: int, alpha: int):
                    super().__init__()
                    self.layer = layer
                    self.lora_A = nn.Parameter(torch.zeros(rank, layer.in_features))
                    self.lora_B = nn.Parameter(torch.zeros(layer.out_features, rank))
                    self.scale = alpha / rank
                    
                    # Initialize LoRA parameters
                    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                    nn.init.zeros_(self.lora_B)
                    
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    base_output = self.layer(x)
                    lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
                    return base_output + lora_output
            
            # Add LoRA to attention layers
            for module in self.model.modules():
                if isinstance(module, nn.Linear) and module.in_features == module.out_features:
                    parent_name = [name for name, mod in self.model.named_modules() if mod is module][0]
                    if 'attention' in parent_name:
                        setattr(module, 'forward', LoRALayer(
                            module,
                            self.config.lora_rank,
                            self.config.lora_alpha
                        ).forward)
                        
        return self.model
    
    def enable_dynamic_memory(self) -> nn.Module:
        """Implement dynamic memory allocation"""
        if self.config.use_dynamic_memory:
            class DynamicMemoryLayer(nn.Module):
                def __init__(self, hidden_size: int):
                    super().__init__()
                    self.hidden_size = hidden_size
                    self.memory_gate = nn.Linear(hidden_size, 1)
                    
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # Calculate complexity-based memory allocation
                    complexity = torch.sigmoid(self.memory_gate(x))
                    return x * complexity
                    
            # Add dynamic memory to each transformer layer
            if hasattr(self.model, 'transformer'):
                for layer in self.model.transformer.h:
                    layer.dynamic_memory = DynamicMemoryLayer(self.model.config.n_embd)
                    
        return self.model
    
    def implement_sparse_attention(self) -> nn.Module:
        """Implement block-sparse attention"""
        if self.config.use_sparse_attention:
            class BlockSparseAttention(nn.Module):
                def __init__(self, attention: nn.Module, block_size: int = 64):
                    super().__init__()
                    self.attention = attention
                    self.block_size = block_size
                    
                def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                    B, H, L, D = q.shape
                    num_blocks = L // self.block_size
                    
                    # Reshape into blocks
                    q = q.view(B, H, num_blocks, self.block_size, D)
                    k = k.view(B, H, num_blocks, self.block_size, D)
                    v = v.view(B, H, num_blocks, self.block_size, D)
                    
                    # Compute attention only within blocks and neighboring blocks
                    output = []
                    for i in range(num_blocks):
                        start_idx = max(0, i-1)
                        end_idx = min(num_blocks, i+2)
                        
                        block_q = q[:, :, i:i+1]
                        block_k = k[:, :, start_idx:end_idx]
                        block_v = v[:, :, start_idx:end_idx]
                        
                        block_output = self.attention(
                            block_q.view(B, H, self.block_size, D),
                            block_k.reshape(B, H, -1, D),
                            block_v.reshape(B, H, -1, D)
                        )
                        output.append(block_output)
                        
                    return torch.cat(output, dim=2)
                    
            # Replace attention mechanisms with sparse versions
            if hasattr(self.model, 'transformer'):
                for layer in self.model.transformer.h:
                    if hasattr(layer, 'attn'):
                        layer.attn = BlockSparseAttention(layer.attn)
                        
        return self.model
    
    def apply_structured_pruning(self) -> nn.Module:
        """Apply structured pruning"""
        if hasattr(self.model, 'transformer'):
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    # Calculate importance scores
                    importance = torch.abs(param).mean(dim=0)
                    mask = importance > self.config.pruning_threshold
                    
                    # Apply structured pruning
                    param.data[:, ~mask] = 0
                    param.register_hook(lambda grad: grad * mask)
                    
        return self.model
    
    def apply_activation_sparsity(self) -> nn.Module:
        """Apply activation sparsity techniques"""
        if self.config.activation_sparsity:
            class SparseActivation(nn.Module):
                def __init__(self, threshold: float = 6.0):
                    super().__init__()
                    self.threshold = threshold
                    
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return torch.clamp(x, min=0.0, max=self.threshold)
            
            # Replace ReLU activations with sparse versions
            for name, module in self.model.named_modules():
                if isinstance(module, nn.ReLU):
                    setattr(self.model, name, SparseActivation())
                    
        return self.model 

def find_optimal_checkpoint_config(model: nn.Module, max_memory: Optional[int] = None) -> Dict[str, Any]:
    """
    Find the optimal checkpoint configuration for a model based on available memory.
    
    Args:
        model: PyTorch model to analyze
        max_memory: Maximum memory constraint in MB (if None, uses 80% of available memory)
        
    Returns:
        Dict containing optimal checkpoint configuration
    """
    # In a real implementation, this would analyze model layers and memory requirements
    # Here we provide a simplified version that returns reasonable defaults
    
    # Count total number of transformer layers
    num_layers = 0
    for name, module in model.named_modules():
        if any(layer_type in name for layer_type in ['encoder.layer', 'decoder.layer', 'transformer.h']):
            num_layers += 1
    
    # Determine checkpoint frequency based on layer count
    if num_layers <= 6:
        checkpoint_freq = 2  # Checkpoint every 2 layers for small models
    elif num_layers <= 12:
        checkpoint_freq = 3  # Checkpoint every 3 layers for medium models
    elif num_layers <= 24:
        checkpoint_freq = 4  # Checkpoint every 4 layers for large models
    else:
        checkpoint_freq = 6  # Checkpoint every 6 layers for very large models
    
    return {
        'checkpoint_freq': checkpoint_freq,
        'checkpoint_layers': [i for i in range(0, num_layers, checkpoint_freq)],
        'memory_efficient': True,
        'use_reentrant': False,  # Default for PyTorch 2.0+
        'preserve_rng_state': True
    }

def estimate_memory_usage(model: nn.Module) -> Dict[str, float]:
    """
    Estimate memory usage for a PyTorch model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dict containing memory estimates in MB
    """
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    param_memory = param_count * 4 / (1024 * 1024)  # Convert to MB (assume float32)
    
    # Estimate activation memory (rough approximation)
    activation_memory = param_memory * 1.5  # Activations typically use ~1.5x parameter memory
    
    # Estimate optimizer memory
    optimizer_memory = param_memory * 2  # Adam-like optimizers use ~2x parameter memory
    
    # Estimate gradient memory
    gradient_memory = param_memory
    
    # Total estimate
    total_memory = param_memory + activation_memory + optimizer_memory + gradient_memory
    
    return {
        'parameters_mb': param_memory,
        'activations_mb': activation_memory,
        'optimizer_mb': optimizer_memory,
        'gradients_mb': gradient_memory,
        'total_mb': total_memory
    } 