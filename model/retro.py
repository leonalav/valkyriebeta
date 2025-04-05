import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class RetroEncoder(nn.Module):
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 projection_dim: int = 768,
                 max_length: int = 512,
                 pooling_strategy: str = "cls",
                 normalize_embeddings: bool = True,
                 use_kv_cache: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, projection_dim)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling_strategy = pooling_strategy
        self.normalize_embeddings = normalize_embeddings
        
        # Enhanced caching
        self.use_kv_cache = use_kv_cache
        if use_kv_cache:
            self.kv_cache = {}
            self.cache_size = 1000
            self.cache_hits = 0
            self.cache_misses = 0
            
    def get_cache_key(self, input_ids: torch.Tensor) -> str:
        """Generate consistent cache key from input ids"""
        return str(input_ids.cpu().numpy().tobytes())
        
    def forward(self, texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        if isinstance(texts, torch.Tensor):
            inputs = {'input_ids': texts}
        else:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.encoder.device)
            
        # Enhanced caching
        if self.use_kv_cache:
            cache_key = self.get_cache_key(inputs['input_ids'])
            if cache_key in self.kv_cache:
                self.cache_hits += 1
                return self.kv_cache[cache_key]
            self.cache_misses += 1
            
        # Get embeddings
        outputs = self.encoder(**inputs)
        
        if self.pooling_strategy == "cls":
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        elif self.pooling_strategy == "mean":
            embeddings = outputs.last_hidden_state.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
            
        # Project to common space
        embeddings = self.projection(embeddings)
        
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            
        if self.use_kv_cache and len(self.kv_cache) < self.cache_size:
            self.kv_cache[cache_key] = embeddings
            
        return embeddings

class RetroLayer(nn.Module):
    def __init__(self,
                hidden_size: int,
                retrieval_dim: int = 768,
                num_retrieved: int = 5,
                retrieval_dropout: float = 0.1,
                use_flash_attention: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.retrieval_dim = retrieval_dim
        self.num_retrieved = num_retrieved
        
        # Enhanced projection layers with residual connections
        self.query_proj = nn.Sequential(
            nn.Linear(hidden_size, retrieval_dim),
            nn.LayerNorm(retrieval_dim)
        )
        self.key_proj = nn.Sequential(
            nn.Linear(hidden_size, retrieval_dim),
            nn.LayerNorm(retrieval_dim)
        )
        self.value_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Configurable attention implementation
        self.use_flash_attention = use_flash_attention
        if use_flash_attention:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ImportError:
                self.use_flash_attention = False
                logger.warning("FlashAttention not available, falling back to standard attention")
        
        if not self.use_flash_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=retrieval_dropout,
                batch_first=True
            )
        
        # Normalization with configurable epsilon
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Memory compression for retrieved items
        self.retrieval_compressor = nn.Linear(retrieval_dim, hidden_size // 2)
        
    def forward(self, x: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        # Compress retrieved items if needed
        if retrieved.size(-1) != self.retrieval_dim:
            retrieved = self.retrieval_compressor(retrieved)
        
        # Project queries and keys
        queries = self.query_proj(x)
        keys = self.key_proj(x)
        
        # Prepare attention inputs
        value = torch.cat([x, self.value_proj(retrieved)], dim=1)
        key = torch.cat([keys, retrieved], dim=1)
        
        if self.use_flash_attention:
            # Reshape for flash attention
            batch_size, seq_len, _ = queries.shape
            queries = queries.view(batch_size, seq_len, -1, self.retrieval_dim // 4)
            key = key.view(batch_size, -1, 4, self.retrieval_dim // 4)
            value = value.view(batch_size, -1, 4, self.hidden_size // 4)
            
            attn_output = self.flash_attn_func(
                queries, key, value,
                dropout_p=0.1 if self.training else 0.0,
                causal=False
            )
            attn_output = attn_output.reshape(batch_size, seq_len, -1)
        else:
            attn_output, _ = self.attention(
                query=queries,
                key=key,
                value=value
            )
        
        return self.norm(x + attn_output)

class RetroIntegration(nn.Module):
    def __init__(self,
                hidden_size: int,
                retrieval_dim: int = 768,
                num_layers: int = 2,
                num_retrieved: int = 5,
                layer_dropout: float = 0.1,
                use_residual: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            RetroLayer(
                hidden_size=hidden_size,
                retrieval_dim=retrieval_dim,
                num_retrieved=num_retrieved,
                retrieval_dropout=layer_dropout,
                use_flash_attention=True
            ) for _ in range(num_layers)
        ])
        self.use_residual = use_residual
        self.layer_dropout = layer_dropout
        
    def forward(self, x: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            # Apply layer dropout during training
            if self.training and torch.rand(1).item() < self.layer_dropout:
                continue
                
            if self.use_residual:
                x = x + layer(x, retrieved)
            else:
                x = layer(x, retrieved)
        return x