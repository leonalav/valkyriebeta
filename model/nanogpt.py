import math
import torch
import torch.nn as nn
import inspect
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True
    # Reasoning-specific configuration
    use_reasoning_blocks: bool = False
    reasoning_block_indices: Optional[List[int]] = None  # Which layers use reasoning blocks
    reasoning_dim: Optional[int] = None  # If None, use n_embd
    reasoning_dropout: Optional[float] = None  # If None, use dropout
    use_tree_lstm: bool = False  # Use TreeLSTM for hierarchical reasoning
    tree_lstm_max_depth: int = 8
    use_adaptive_reasoning: bool = False  # Adaptively apply reasoning based on input
    use_recursive_reasoning: bool = False  # Enable recursive reasoning pathways

class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Flash Attention optimization
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                               attn_mask=None,
                                                               dropout_p=self.dropout if self.training else 0,
                                                               is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 

class ReasoningBlock(Block):
    """Extended Block with additional reasoning capabilities"""
    def __init__(self, config: GPTConfig, reasoning_config=None):
        super().__init__(config)
        
        # Default reasoning configuration if not provided
        if reasoning_config is None:
            reasoning_config = {
                'use_skip_connections': True,
                'use_gating': True,
                'use_tree_attention': config.block_size <= 4096,  # Only use for smaller contexts
                'reasoning_dim': config.n_embd,
                'reasoning_heads': max(1, config.n_head // 4),  # Use fewer heads for reasoning
                'reasoning_dropout': config.dropout
            }
            
        self.reasoning_config = reasoning_config
        
        # Reasoning gate to control information flow
        if reasoning_config.get('use_gating', True):
            self.reasoning_gate = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd // 4),
                nn.GELU(),
                nn.Linear(config.n_embd // 4, 1),
                nn.Sigmoid()
            )
            
        # Tree attention layer for hierarchical reasoning
        if reasoning_config.get('use_tree_attention', False):
            self.tree_attn = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd),
                nn.GELU(),
                nn.Linear(config.n_embd, config.n_embd)
            )
            
        # Additional reasoning projection
        self.reasoning_proj = nn.Linear(config.n_embd, reasoning_config.get('reasoning_dim', config.n_embd))
        
        # Reasoning MLP with potential for symbolic reasoning
        self.reasoning_mlp = nn.Sequential(
            nn.Linear(reasoning_config.get('reasoning_dim', config.n_embd), 
                    4 * reasoning_config.get('reasoning_dim', config.n_embd)),
            nn.GELU(),
            nn.Linear(4 * reasoning_config.get('reasoning_dim', config.n_embd), 
                    reasoning_config.get('reasoning_dim', config.n_embd)),
            nn.Dropout(reasoning_config.get('reasoning_dropout', config.dropout))
        )
        
        # Output projection
        self.reasoning_out = nn.Linear(
            reasoning_config.get('reasoning_dim', config.n_embd), 
            config.n_embd
        )
        
        # Layer norm for reasoning
        self.ln_reasoning = LayerNorm(config.n_embd, bias=config.bias)
        
    def apply_tree_reasoning(self, x):
        """Apply hierarchical tree-like reasoning to input"""
        B, T, C = x.size()
        
        # Simple approximation of tree structure using pairwise correlations
        # This can be replaced with a more sophisticated TreeLSTM or tree attention
        corr = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(C)  # [B, T, T]
        corr = torch.softmax(corr, dim=-1)
        
        # Apply tree attention (optional)
        if hasattr(self, 'tree_attn'):
            tree_x = self.tree_attn(x)
            # Hierarchical pooling based on correlation matrix
            tree_rep = torch.matmul(corr, tree_x)  # [B, T, C]
            return tree_rep
        
        return torch.matmul(corr, x)  # [B, T, C]

    def forward(self, x: torch.Tensor, reasoning_state=None) -> Tuple[torch.Tensor, Optional[Dict]]:
        # Standard transformer block processing
        residual = x
        x = self.ln_1(x)
        attn_out = self.attn(x)
        x = residual + attn_out
        
        # MLP processing
        residual = x
        x = self.ln_2(x)
        mlp_out = self.mlp(x)
        x = residual + mlp_out
        
        # Apply reasoning component
        reasoning_residual = x
        x_reasoning = self.ln_reasoning(x)
        
        # Apply gating if configured
        if hasattr(self, 'reasoning_gate'):
            gate = self.reasoning_gate(x_reasoning)
            if gate.mean() < 0.1:  # Skip reasoning if gate is mostly closed
                return x, {'gate_value': gate.mean().item(), 'used_reasoning': False}
        
        # Apply tree-based reasoning
        tree_rep = self.apply_tree_reasoning(x_reasoning)
        
        # Project to reasoning space
        reasoning_proj = self.reasoning_proj(tree_rep)
        
        # Apply reasoning MLP
        reasoning_features = self.reasoning_mlp(reasoning_proj)
        
        # Project back to model dimension
        reasoning_out = self.reasoning_out(reasoning_features)
        
        # Apply skip connection if configured
        if self.reasoning_config.get('use_skip_connections', True):
            if hasattr(self, 'reasoning_gate'):
                # Apply gated reasoning
                x = reasoning_residual + gate * reasoning_out
            else:
                # Apply standard residual
                x = reasoning_residual + reasoning_out
        else:
            x = reasoning_out
        
        # Return the output and reasoning state for monitoring
        reasoning_info = {
            'gate_value': gate.mean().item() if hasattr(self, 'reasoning_gate') else 1.0,
            'used_reasoning': True,
            'reasoning_norm': reasoning_out.norm().item()
        }
        
        return x, reasoning_info 

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = self._init_blocks(config),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless.
        # TODO: investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_blocks(self, config: GPTConfig) -> nn.ModuleList:
        """Initialize transformer blocks with optional reasoning capabilities"""
        blocks = nn.ModuleList()
        
        # determine which layers will use reasoning blocks
        use_reasoning = False
        reasoning_layers = set()
        
        if config.use_reasoning_blocks:
            use_reasoning = True
            # if specific layer indices are provided, use those
            if config.reasoning_block_indices is not None:
                reasoning_layers = set(config.reasoning_block_indices)
            else:
                # default: apply reasoning to later layers starting from 2/3 through the network
                start_idx = max(1, config.n_layer * 2 // 3)
                reasoning_layers = set(range(start_idx, config.n_layer))
        
        # initialize reasoning configuration
        reasoning_config = None
        if use_reasoning:
            reasoning_config = {
                'use_skip_connections': True,
                'use_gating': True,
                'use_tree_attention': config.block_size <= 4096,
                'reasoning_dim': config.reasoning_dim or config.n_embd,
                'reasoning_dropout': config.reasoning_dropout or config.dropout,
                'use_tree_lstm': config.use_tree_lstm,
                'use_recursive_reasoning': config.use_recursive_reasoning
            }
        
        # blocks builder!!!
        for i in range(config.n_layer):
            if i in reasoning_layers:
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

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        collect_reasoning_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Track reasoning information if requested
        reasoning_info = {'layer_info': []} if collect_reasoning_info else None
        
        # Forward through transformer blocks
        for i, block in enumerate(self.transformer.h):
            if isinstance(block, ReasoningBlock):
                x, block_info = block(x)
                if reasoning_info is not None:
                    block_info['layer'] = i
                    reasoning_info['layer_info'].append(block_info)
            else:
                x = block(x)
                
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss, reasoning_info

    def crop_block_size(self, block_size):
        """Model surgery to decrease the block size if necessary"""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # we can override the dropout rate
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']
        # block_size can also be specified at this point
        if 'block_size' in override_args:
            config_args['block_size'] = override_args['block_size']
        # add reasoning configuration
        for arg in ['use_reasoning_blocks', 'reasoning_block_indices', 'reasoning_dim', 
                     'reasoning_dropout', 'use_tree_lstm', 'tree_lstm_max_depth',
                     'use_adaptive_reasoning', 'use_recursive_reasoning']:
            if arg in override_args:
                config_args[arg] = override_args[arg]
        
        # create a from-scratch initialized GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu 