import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math

# Rotary embedding helper
# (Assuming rotary embeddings might be used in conjunction with Hyena,
# similar to how they are used with attention)
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # cos = cos[position_ids].unsqueeze(1) # deprecated
    # sin = sin[position_ids].unsqueeze(1) # deprecated
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class HyenaOperator(nn.Module):
    """
    Hyena operator inspired by the paper "Hyena Hierarchy: Towards Larger Convolutional Language Models".
    This implementation uses explicit short convolutions and gating, parameterizing
    the long convolution implicitly via projections. A full FFT implementation
    would be more efficient for very long sequences but is more complex.

    Args:
        dim (int): Dimension of the input and output.
        order (int): Order of the Hyena polynomial projection (usually 2 or 3). Controls complexity.
        filter_order (int): Order of the learned filter generating function.
        num_heads (int): Number of heads for parallel filter application.
        inner_factor (int): Expansion factor for inner projections.
        num_blocks (int): Number of Hyena blocks applied sequentially.
        filter_dropout (float): Dropout rate for the filter parameters.
        use_short_conv (bool): Whether to include a parallel short convolution branch.
        short_conv_size (int): Kernel size for the short convolution.
    """
    def __init__(self,
                 dim: int,
                 order: int = 2,
                 filter_order: int = 64, # Order of the learned filter function
                 num_heads: int = 1, # Number of heads for filter projections
                 inner_factor: int = 1,
                 num_blocks: int = 1, # Corresponds to hyena_num_blocks in config
                 filter_dropout: float = 0.0,
                 use_short_conv: bool = True,
                 short_conv_size: int = 3,
                 activation: str = "silu",
                 use_rotary_emb: bool = False, # Whether to apply rotary embeddings
                 max_seq_len: int = 2048 # Needed for rotary embeddings
                 ):
        super().__init__()
        self.dim = dim
        self.order = order
        self.filter_order = filter_order
        self.num_heads = num_heads
        self.inner_factor = inner_factor
        self.num_blocks = num_blocks
        self.use_short_conv = use_short_conv
        self.short_conv_size = short_conv_size
        self.filter_dropout = filter_dropout
        self.use_rotary_emb = use_rotary_emb
        self.max_seq_len = max_seq_len

        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"
        self.head_dim = dim // num_heads

        # Input projections (similar to Q, K, V in attention)
        self.proj_in = nn.Linear(dim, dim * (2 + order)) # Proj for x, z, v_i...

        # Filter parameterization (learnable function of position)
        # We learn parameters that generate the filter based on position
        self.filter_proj = nn.Linear(filter_order, num_heads * dim)
        self.filter_dropout_layer = nn.Dropout(filter_dropout)

        # Positional encoding/basis for filter generation
        # Using a simple sinusoidal basis here, more complex options exist
        pos_basis = torch.arange(max_seq_len).unsqueeze(1) * torch.arange(filter_order).unsqueeze(0) / filter_order
        self.register_buffer('pos_encoding_basis', torch.sin(pos_basis)) # (max_seq_len, filter_order)

        # Short convolution branch (optional)
        if use_short_conv:
            self.short_conv = nn.Conv1d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=short_conv_size,
                padding=short_conv_size // 2,
                groups=dim # Depthwise convolution
            )

        # Activation function
        if activation == "silu":
            self.activation = F.silu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = lambda x: x # Linear

        # Output projection
        self.proj_out = nn.Linear(dim, dim)

        # Rotary embeddings (optional)
        if use_rotary_emb:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=max_seq_len)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, dim)
        Returns:
            output: (batch_size, seq_len, dim)
        """
        B, L, D = x.shape
        H = self.num_heads
        HD = self.head_dim

        # --- Optional Short Convolution Branch ---
        x_short = 0.0
        if self.use_short_conv:
            x_short = self.short_conv(x.transpose(1, 2)).transpose(1, 2)

        # --- Hyena Operator Core ---
        # 1. Input Projections
        projections = self.proj_in(x) # (B, L, D * (2 + order))
        x_proj, z_proj, *v_projs = torch.split(projections, D, dim=-1)

        # Apply activation to z (gate)
        z = self.activation(z_proj)

        # Reshape projections for heads if num_heads > 1
        if H > 1:
            x_proj = x_proj.view(B, L, H, HD)
            z = z.view(B, L, H, HD)
            v_projs = [v.view(B, L, H, HD) for v in v_projs]

        # Apply rotary embeddings if enabled
        if self.use_rotary_emb:
            cos, sin = self.rotary_emb(v_projs[0], seq_len=L) # Use v0 for embedding calculation
            x_proj, _ = apply_rotary_pos_emb(x_proj, x_proj, cos, sin) # Apply to x_proj
            v_projs = [apply_rotary_pos_emb(v, v, cos, sin)[0] for v in v_projs] # Apply to all v_projs

        # 2. Generate Filter based on Position
        # Use positional basis up to sequence length L
        pos_basis = self.pos_encoding_basis[:L, :].to(x.device) # (L, filter_order)
        # Project positional basis to generate filter parameters per head
        filter_params = self.filter_proj(pos_basis) # (L, H * D)
        filter_params = self.filter_dropout_layer(filter_params)
        # Reshape filter parameters: (L, H, D) -> (H, D, L) for convolution
        learned_filter = filter_params.view(L, H, D).permute(1, 2, 0) # (H, D, L)

        # 3. Apply Implicit Long Convolution via FFT (Simplified using 1D Conv)
        # A true Hyena uses FFT convolution. Here we simulate with Conv1d for simplicity,
        # assuming the learned_filter captures the long-range dependencies implicitly.
        # This is a major simplification but aligns with the original code's conv use.
        # We apply the filter to x_proj.

        # Reshape x_proj for convolution: (B, L, H, HD) -> (B*H, HD, L)
        x_conv_input = x_proj.permute(0, 2, 3, 1).reshape(B * H, HD, L)

        # Reshape filter for grouped convolution: (H, D, L) -> (H*HD, 1, L) ? No, needs (out_channels, in_channels/groups, kernel_size)
        # We need one filter per head's dimension.
        # Let's try grouped convolution: (H*HD, 1, L) where groups = H*HD
        # This means each channel gets its own filter of length L.

        # Pad input for 'causal' convolution simulation
        # Padding = Filter Length - 1 = L - 1
        # Using padding='causal' in Conv1d requires PyTorch 1.9+
        # Manual padding:
        # x_conv_input = F.pad(x_conv_input, (L - 1, 0)) # Pad left for causality

        # Apply convolution
        # The filter `learned_filter` has shape (H, D, L). D = HD here.
        # Conv1d expects filter (out_channels, in_channels/groups, kernel_size)
        # We want (B*H, HD, L) convolved with (H, HD, L) -> (B*H, HD, L)
        # This requires careful reshaping or a custom FFT implementation.

        # --- Simplified Conv1d Approach (closer to original code, less like true Hyena) ---
        # Treat each head independently. Filter shape (H, HD, L)
        # Input shape (B*H, HD, L)
        output_conv = torch.zeros_like(x_conv_input)
        for h in range(H):
            # Filter for head h: (HD, L) -> (HD, 1, L) for conv1d groups=HD
            head_filter = learned_filter[h].unsqueeze(1) # (HD, 1, L)
            # Input for head h: (B, HD, L)
            head_input = x_conv_input.view(B, H, HD, L)[:, h, :, :] # (B, HD, L)

            # Apply convolution with padding L//2 for 'same' output size
            # This is NOT causal, unlike true Hyena FFT conv.
            conv_out_h = F.conv1d(head_input, head_filter, padding=L // 2, groups=HD)
            output_conv.view(B, H, HD, L)[:, h, :, :] = conv_out_h[:, :, :L] # Trim padding

        # Reshape output back: (B*H, HD, L) -> (B, L, H, HD)
        y = output_conv.view(B, H, HD, L).permute(0, 3, 1, 2) # (B, L, H, HD)

        # 4. Apply Polynomial Projections (Order) and Gating
        # y is the result of the long convolution (approximated)
        y = y * v_projs[0] # Order 1
        if self.order >= 2:
            y = y * v_projs[1] # Order 2
        if self.order >= 3:
             y = y * v_projs[2] # Order 3
        # ... continue for higher orders if needed

        # Apply gating
        y = y * z

        # Reshape if using heads
        if H > 1:
            y = y.reshape(B, L, D)

        # 5. Combine with Short Convolution and Project Output
        output = self.proj_out(y + x_short)

        return output


def replace_attention_with_hyena(model: nn.Module,
                                 config, # Pass the full training config
                                 layer_indices: Optional[List[int]] = None):
    """
    Replace the attention mechanism OR the FFN in specified transformer layers
    with Hyena operators.

    Args:
        model: The model to modify (e.g., ValkyrieLLM or its CoreModel).
        config: The AdvancedTrainingConfig containing Hyena parameters.
        layer_indices: List of layer indices to apply Hyena to. If None, applies to all.
    """
    if not config.use_hyena:
        return model

    logger.info(f"Applying Hyena operator to layers: {layer_indices if layer_indices else 'All'}")

    # Determine target layers
    target_layers = []
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        target_layers = model.transformer.layers
    elif hasattr(model, 'layers'): # Handle models with direct layer list
        target_layers = model.layers
    else:
        logger.warning("Could not find transformer layers in the model to replace with Hyena.")
        return model

    if layer_indices is None:
        layer_indices = list(range(len(target_layers)))

    for i, layer in enumerate(target_layers):
        if i in layer_indices:
            logger.info(f"Replacing components in layer {i} with Hyena.")
            # Create Hyena operator instance based on config
            hyena_op = HyenaOperator(
                dim=config.hidden_size,
                order=getattr(config, 'hyena_order', 2), # Use getattr for safety
                filter_order=getattr(config, 'hyena_filter_order', 64),
                num_heads=getattr(config, 'hyena_num_heads', 1),
                num_blocks=getattr(config, 'hyena_num_blocks', 1),
                filter_dropout=getattr(config, 'hyena_filter_dropout', 0.0),
                use_short_conv=getattr(config, 'hyena_use_short_conv', True),
                short_conv_size=getattr(config, 'hyena_short_conv_size', 3),
                activation=getattr(config, 'hyena_activation', 'silu'),
                use_rotary_emb=getattr(config, 'use_rotary_embeddings', False), # Link to rotary config
                max_seq_len=config.max_seq_length
            )

            # --- Strategy: Replace Attention ---
            # Find the attention module (could be named 'attn', 'self_attn', etc.)
            attn_module_name = None
            if hasattr(layer, 'self_attn'):
                attn_module_name = 'self_attn'
            elif hasattr(layer, 'attn'):
                attn_module_name = 'attn'
            elif hasattr(layer, 'attention'):
                 attn_module_name = 'attention'

            if attn_module_name:
                logger.info(f"  Replacing attention module '{attn_module_name}' with Hyena.")
                setattr(layer, attn_module_name, hyena_op)
            else:
                # --- Strategy: Replace FFN (Alternative) ---
                # If no attention module found, try replacing the feed-forward network
                ffn_module_name = None
                if hasattr(layer, 'feed_forward'):
                    ffn_module_name = 'feed_forward'
                elif hasattr(layer, 'mlp'):
                    ffn_module_name = 'mlp'

                if ffn_module_name:
                    logger.info(f"  Replacing FFN module '{ffn_module_name}' with Hyena.")
                    setattr(layer, ffn_module_name, hyena_op)
                else:
                    logger.warning(f"  Could not find attention or FFN module in layer {i} to replace with Hyena.")

    return model

# Example usage within train_llm.py setup:
# if config.use_hyena:
#     from model.hyena import replace_attention_with_hyena
#     model = replace_attention_with_hyena(model, config, config.hyena_layer_indices)

# Need logger setup if running standalone
import logging
logger = logging.getLogger(__name__)
# Basic config if run standalone
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Example Test
    class MockConfig:
        hidden_size = 64
        hyena_order = 2
        hyena_filter_order = 32
        hyena_num_heads = 4
        hyena_num_blocks = 1
        hyena_use_short_conv = True
        use_rotary_embeddings = True
        max_seq_length = 128
        use_hyena = True

    config = MockConfig()
    op = HyenaOperator(dim=config.hidden_size, order=config.hyena_order, filter_order=config.hyena_filter_order,
                       num_heads=config.hyena_num_heads, use_rotary_emb=config.use_rotary_embeddings, max_seq_len=config.max_seq_length)
    test_input = torch.randn(4, 50, 64) # B, L, D
    output = op(test_input)
    print("Input Shape:", test_input.shape)
    print("Output Shape:", output.shape)
    assert output.shape == test_input.shape
    logger.info("HyenaOperator test passed.")

    # Test replacement function (conceptual)
    class MockAttention(nn.Module):
        def forward(self, x): return x
    class MockFFN(nn.Module):
        def forward(self, x): return x
    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = MockAttention()
            self.norm1 = nn.Identity()
            self.feed_forward = MockFFN()
            self.norm2 = nn.Identity()
        def forward(self, x):
            x = x + self.self_attn(self.norm1(x))
            x = x + self.feed_forward(self.norm2(x))
            return x
    class MockModel(nn.Module):
        def __init__(self, num_layers=4):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer() for _ in range(num_layers)])
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = MockModel()
    replace_attention_with_hyena(model, config, layer_indices=[0, 2])
    print("\nModel after replacement:")
    print(model)
    assert isinstance(model.layers[0].self_attn, HyenaOperator)
    assert isinstance(model.layers[1].self_attn, MockAttention) # Not replaced
    assert isinstance(model.layers[2].self_attn, HyenaOperator)
    logger.info("Hyena replacement test passed (conceptually).")