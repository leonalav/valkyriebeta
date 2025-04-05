import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import math
import inspect

from .nanogpt import GPTConfig, Block, LayerNorm, ReasoningBlock

class GPT(nn.Module):
    """
    GPT Language Model implementation based on the nanogpt architecture.
    
    This class implements a standard GPT model that can be enhanced with
    mathematical precision improvements and reasoning capabilities.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        # Transformer components
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = self._init_blocks(config),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply weight tying between embedding and output layer
        self.transformer.wte.weight = self.lm_head.weight
        
        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")
        
    def _init_blocks(self, config: GPTConfig) -> nn.ModuleList:
        """Initialize transformer blocks with optional reasoning capabilities"""
        blocks = nn.ModuleList()
        
        # Determine which layers will use reasoning blocks
        use_reasoning = False
        reasoning_layers = set()
        
        if hasattr(config, 'use_reasoning_blocks') and config.use_reasoning_blocks:
            use_reasoning = True
            # If specific layer indices are provided, use those
            if hasattr(config, 'reasoning_block_indices') and config.reasoning_block_indices is not None:
                reasoning_layers = set(config.reasoning_block_indices)
            else:
                # Default: apply reasoning to later layers starting from 2/3 through the network
                start_idx = max(1, config.n_layer * 2 // 3)
                reasoning_layers = set(range(start_idx, config.n_layer))
        
        # Initialize reasoning configuration
        reasoning_config = None
        if use_reasoning:
            reasoning_config = {
                'use_skip_connections': True,
                'use_gating': True,
                'use_tree_attention': config.block_size <= 4096,
                'reasoning_dim': getattr(config, 'reasoning_dim', config.n_embd),
                'reasoning_dropout': getattr(config, 'reasoning_dropout', config.dropout),
                'use_tree_lstm': getattr(config, 'use_tree_lstm', False),
                'use_recursive_reasoning': getattr(config, 'use_recursive_reasoning', False)
            }
        
        # Create all blocks
        for i in range(config.n_layer):
            if i in reasoning_layers:
                print(f"Layer {i}: Using ReasoningBlock")
                blocks.append(ReasoningBlock(config, reasoning_config))
            else:
                blocks.append(Block(config))
                
        return blocks
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def get_num_params(self):
        """Return the number of parameters in the model"""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
            
    def forward(
        self, 
        input_ids: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        collect_reasoning_info: bool = False
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Get position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        
        # forward pass to gpt model
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Track reasoning information if requested
        reasoning_info = {'layer_info': []} if collect_reasoning_info else None
        
        # Apply transformer blocks
        for i, block in enumerate(self.transformer.h):
            if isinstance(block, ReasoningBlock):
                x, block_info = block(x)
                if reasoning_info is not None:
                    block_info['layer'] = i
                    reasoning_info['layer_info'].append(block_info)
            else:
                x = block(x)
            
        # Apply final layer norm
        x = self.transformer.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
            
        results = {
            'logits': logits,
            'loss': loss,
            'hidden_states': x
        }
        
        # Add reasoning info if collected
        if reasoning_info is not None:
            results['reasoning_info'] = reasoning_info
            
        return results
    
    def generate(
        self, 
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        collect_reasoning_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Generate text using the model
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            collect_reasoning_info: Whether to collect reasoning information
            
        Returns:
            Generated token IDs and optional reasoning info
        """
        generation_reasoning_info = [] if collect_reasoning_info else None
        
        for _ in range(max_new_tokens):
            # Crop input_ids to block_size if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(idx_cond, collect_reasoning_info=collect_reasoning_info)
                logits = outputs['logits']
                
                # Collect reasoning info if requested
                if collect_reasoning_info and 'reasoning_info' in outputs:
                    generation_reasoning_info.append(outputs['reasoning_info'])
                
            # Get logits for the last token and apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
                
            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
        if collect_reasoning_info:
            return input_ids, {'generation_reasoning_info': generation_reasoning_info}
        else:
            return input_ids, None 