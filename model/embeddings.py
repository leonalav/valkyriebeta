import torch
import torch.nn as nn
import math
from typing import Optional, Dict
import torch.nn.functional as F

class SharedEmbedding(nn.Module):
    """Memory-efficient embedding with weight sharing and quantization"""
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 share_with: Optional[nn.Linear] = None,
                 quantize: bool = True,
                 bits: int = 8):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.quantize = quantize
        self.bits = bits
        
        if share_with is not None:
            # Share weights with output layer
            self.weight = share_with.weight
        else:
            # Initialize new weights
            self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
            self._init_weights()
            
        if quantize:
            self.register_buffer('scale', torch.ones(1))
            self.register_buffer('zero_point', torch.zeros(1))
            
    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
                
    def _quantize_weights(self) -> torch.Tensor:
        if not self.quantize:
            return self.weight
            
        # Dynamic quantization
        qmin, qmax = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1
        
        # Update scale and zero point
        weight_min, weight_max = self.weight.min(), self.weight.max()
        self.scale[0] = (weight_max - weight_min) / (qmax - qmin)
        self.zero_point[0] = qmin - weight_min / self.scale[0]
        
        # Quantize
        quantized = torch.clamp(
            torch.round(self.weight / self.scale[0] + self.zero_point[0]),
            qmin, qmax
        )
        
        # Dequantize
        return (quantized - self.zero_point[0]) * self.scale[0]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._quantize_weights()
        return F.embedding(x, weight, self.padding_idx)

class CompactPositionalEncoding(nn.Module):
    """Memory-efficient positional encoding with interpolation"""
    def __init__(self,
                 max_seq_length: int,
                 embedding_dim: int,
                 base_seq_length: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_seq_length = max_seq_length
        self.base_seq_length = base_seq_length
        self.embedding_dim = embedding_dim
        
        # Create compact base encodings
        pe = torch.zeros(base_seq_length, embedding_dim)
        position = torch.arange(0, base_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer
        self.register_buffer('base_pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.size(1)
        
        if seq_length <= self.base_seq_length:
            # Use base encodings directly
            pe = self.base_pe[:, :seq_length]
        else:
            # Interpolate for longer sequences
            indices = torch.linspace(0, self.base_seq_length - 1, seq_length)
            pe = F.interpolate(
                self.base_pe.transpose(1, 2),
                size=seq_length,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            
        return self.dropout(x + pe)

class EfficientEmbeddingLayer(nn.Module):
    """Combined embedding layer with all optimizations"""
    def __init__(self, config):
        super().__init__()
        # Share weights between input embeddings and output layer if specified
        self.share_weights = config.tie_word_embeddings
        self.output_layer = None
        
        # Token embeddings with optional weight sharing
        self.token_embedding = SharedEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            share_with=self.output_layer if self.share_weights else None,
            quantize=config.use_quantization,
            bits=config.quantization_bits
        )
        
        # Compact positional encoding
        self.position_embedding = CompactPositionalEncoding(
            max_seq_length=config.max_seq_length,
            embedding_dim=config.hidden_size,
            base_seq_length=min(512, config.max_seq_length),
            dropout=config.embedding_dropout
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def set_output_layer(self, output_layer: nn.Linear):
        """Set output layer for weight sharing"""
        self.output_layer = output_layer
        if self.share_weights:
            self.token_embedding.weight = output_layer.weight
            
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Get token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Add positional encodings
        embeddings = self.position_embedding(embeddings)
        
        # Layer norm
        embeddings = self.layer_norm(embeddings)
        
        return embeddings 