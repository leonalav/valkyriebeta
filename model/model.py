import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import math
import torch.nn.functional as F

from .attention import MultiScaleAttention, AdaptiveSparsityAttention, EfficientAttention
from .layers import EfficientTransformerLayer, ParallelFeedForward
from .embedding import EfficientEmbeddingLayer  # Updated import path
from .lora import LoRALinear  # Updated import path
from .normalization import get_norm_layer
from .efficient_transformer import EfficientTransformer
from .memory_bank import MemoryBank
from .reasoning import LogicalAttention, LogicalReasoningLayer, KnowledgeIntegrationModule
from .training.contrastive import ContrastiveLearningModule
from .uncertainty.calibration import UncertaintyCalibrationModule
from .generation.logical_beam_search import LogicalBeamSearch
from .api_distillation import APIKnowledgeDistillation, APITeacherModel
from config.model_config import ModelConfig
class LogicalReasoningTransformer(nn.Module):
    """Main model that wraps EfficientTransformer with proper configuration handling"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embedding layers
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Output head
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ):
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
            
        # Get embeddings
        word_embeds = self.word_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        hidden_states = word_embeds + pos_embeds
        
        # Apply transformer layers
        present_key_values = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                present_key_values += (layer_outputs[1],)
                
        # Final layer norm and dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        if use_cache:
            return (hidden_states, present_key_values)
        return hidden_states

class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        # Multi-head attention
        self.attention = EfficientAttention(config)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.hidden_dropout)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ):
        # Self-attention
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            use_cache=use_cache
        )
        
        if use_cache:
            hidden_states, present_key_value = attention_outputs
        else:
            hidden_states = attention_outputs
            present_key_value = None
            
        hidden_states = residual + hidden_states
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        if use_cache:
            return (hidden_states, present_key_value)
        return hidden_states