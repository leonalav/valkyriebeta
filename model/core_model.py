import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Tuple, List, Any, Union
from contextlib import nullcontext

from .transformer import EfficientTransformerEnhanced
try:
    from .nanogpt import ReasoningBlock, LayerNorm, GPTConfig
except ImportError:
    # Define placeholders if nanogpt components aren't available
    ReasoningBlock = None
    LayerNorm = None
    GPTConfig = None

try:
    from .config_3b import Config3B
except ImportError:
    # Define a simple placeholder if the 3B config isn't available
    class Config3B:
        hidden_size: int = 4096
        num_layers: int = 32
        num_heads: int = 32
        intermediate_size: int = 16384
        vocab_size: int = 50257
        max_seq_length: int = 16384
        dropout: float = 0.1
        layer_norm_eps: float = 1e-5
        activation: str = "gelu"

logger = logging.getLogger(__name__)

class CoreModel(nn.Module):
    """
    Core model implementation for the Valkyrie LLM.
    
    This model serves as the foundation for all Valkyrie models,
    providing basic transformer functionality with optional enhancements.
    
    Supports scaling from small models (125M parameters) to large models (3B+ parameters).
    Enhanced with reasoning capabilities for improved performance on reasoning tasks.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
        config = None,
        training_config = None,
        tokenizer = None,
        use_3b_config: bool = False,
        use_reasoning_blocks: bool = False,
        reasoning_block_indices: Optional[List[int]] = None
    ):
        """
        Initialize the core model.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Size of hidden layers
            num_layers: Number of transformer layers
            num_heads: Number of attention heads in each layer
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            config: Optional model configuration
            training_config: Optional training configuration
            tokenizer: Optional tokenizer
            use_3b_config: Whether to use the 3B parameter configuration
            use_reasoning_blocks: Whether to use reasoning-enhanced blocks
            reasoning_block_indices: Which layers should use reasoning blocks
        """
        super().__init__()
        
        # If use_3b_config is True, use the 3B configuration
        if use_3b_config:
            try:
                from .config_3b import default_config_3b
                config = default_config_3b
                vocab_size = config.vocab_size
                hidden_size = config.hidden_size
                num_layers = config.num_layers
                num_heads = config.num_heads
                max_seq_length = config.max_seq_length
                dropout = config.dropout
                logger.info(f"Using 3B configuration with {hidden_size} hidden size and {num_layers} layers")
            except ImportError:
                logger.warning("Could not import 3B configuration, using provided parameters instead")
        
        # Use config values if provided
        if config is not None:
            if hasattr(config, 'vocab_size'):
                vocab_size = config.vocab_size
            if hasattr(config, 'hidden_size'):
                hidden_size = config.hidden_size
            if hasattr(config, 'num_layers'):
                num_layers = config.num_layers
            if hasattr(config, 'num_heads') or hasattr(config, 'num_attention_heads'):
                num_heads = getattr(config, 'num_heads', getattr(config, 'num_attention_heads', num_heads))
            if hasattr(config, 'max_seq_length') or hasattr(config, 'max_position_embeddings'):
                max_seq_length = getattr(config, 'max_seq_length', 
                                       getattr(config, 'max_position_embeddings', max_seq_length))
            if hasattr(config, 'dropout'):
                dropout = config.dropout
            if hasattr(config, 'use_reasoning_blocks'):
                use_reasoning_blocks = config.use_reasoning_blocks
            if hasattr(config, 'reasoning_block_indices'):
                reasoning_block_indices = config.reasoning_block_indices
        
        # Store configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.config = config
        self.training_config = training_config
        self.tokenizer = tokenizer
        self.use_reasoning_blocks = use_reasoning_blocks
        self.reasoning_block_indices = reasoning_block_indices
        
        # Create transformer backbone
        if use_reasoning_blocks and ReasoningBlock is not None:
            # Create a nanogpt compatible config for reasoning blocks
            gpt_config = GPTConfig(
                block_size=max_seq_length,
                vocab_size=vocab_size,
                n_layer=num_layers,
                n_head=num_heads,
                n_embd=hidden_size,
                dropout=dropout,
                bias=True,
                use_reasoning_blocks=True,
                reasoning_block_indices=reasoning_block_indices,
                reasoning_dim=hidden_size,
                reasoning_dropout=dropout,
                use_tree_lstm=getattr(config, 'use_tree_lstm', False),
                use_adaptive_reasoning=getattr(config, 'use_adaptive_reasoning', False),
                use_recursive_reasoning=getattr(config, 'use_recursive_reasoning', False)
            )
            
            # Create enhanced transformer with reasoning capabilities
            self.transformer = self._create_enhanced_transformer(gpt_config)
            logger.info(f"Using enhanced transformer with reasoning blocks")
        else:
            # Use standard efficient transformer
            self.transformer = EfficientTransformerEnhanced(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                max_position_embeddings=max_seq_length,
                dropout=dropout
            )
        
        # Token embedding from transformer
        self.token_embedding = self.transformer.token_embedding
        
        # Position embedding from transformer
        self.position_embedding = self.transformer.position_embedding
        
        # LM head for language modeling
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie embeddings and lm_head weights
        self.lm_head.weight = self.token_embedding.weight
        
        # Calculate approximate parameter count
        param_count = sum(p.numel() for p in self.parameters())
        param_count_b = param_count / 1_000_000_000
        
        logger.info(f"Initialized CoreModel with {num_layers} layers, {num_heads} heads")
        logger.info(f"Model has approximately {param_count:,} parameters ({param_count_b:.2f}B)")
        
        if use_reasoning_blocks:
            logger.info(f"Model uses reasoning blocks for enhanced reasoning capabilities")
            if reasoning_block_indices:
                logger.info(f"Reasoning blocks at layers: {reasoning_block_indices}")
            else:
                start_idx = max(1, num_layers * 2 // 3)
                logger.info(f"Reasoning blocks at layers: {list(range(start_idx, num_layers))}")
    
    def _create_enhanced_transformer(self, gpt_config):
        """
        Create a transformer with reasoning capabilities
        
        This method creates a custom transformer that incorporates reasoning blocks
        as specified in the configuration.
        """
        # Determine which layers will use reasoning blocks
        use_reasoning_layers = set()
        if self.reasoning_block_indices is not None:
            use_reasoning_layers = set(self.reasoning_block_indices)
        else:
            # Default: apply reasoning to later layers (2/3 onwards)
            start_idx = max(1, self.num_layers * 2 // 3)
            use_reasoning_layers = set(range(start_idx, self.num_layers))
        
        # Create reasoning configuration
        reasoning_config = {
            'use_skip_connections': True,
            'use_gating': True,
            'use_tree_attention': gpt_config.block_size <= 4096,
            'reasoning_dim': gpt_config.n_embd,
            'reasoning_heads': max(1, gpt_config.n_head // 4),
            'reasoning_dropout': gpt_config.dropout
        }
        
        # Create custom transformer
        from .transformer import TransformerBase
        
        class EnhancedReasoningTransformer(TransformerBase):
            def __init__(self, config):
                super().__init__()
                
                # Token and position embeddings
                self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
                self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
                self.embedding_dropout = nn.Dropout(config.dropout)
                
                # Create transformer blocks with reasoning capabilities
                self.layers = nn.ModuleList()
                for i in range(config.n_layer):
                    if i in use_reasoning_layers:
                        self.layers.append(ReasoningBlock(config, reasoning_config))
                    else:
                        from .nanogpt import Block
                        self.layers.append(Block(config))
                
                # Final layer normalization
                self.ln_final = LayerNorm(config.n_embd, bias=True)
                
            def forward(
                self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                output_hidden_states=False,
                output_attentions=False,
                **kwargs
            ):
                batch_size, seq_length = input_ids.size()
                device = input_ids.device
                
                # Get positional IDs if not provided
                if position_ids is None:
                    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
                    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                
                # Get embeddings
                token_emb = self.token_embedding(input_ids)
                pos_emb = self.position_embedding(position_ids)
                x = self.embedding_dropout(token_emb + pos_emb)
                
                # Track states for output if requested
                all_hidden_states = [] if output_hidden_states else None
                all_attentions = [] if output_attentions else None
                reasoning_info = []
                
                # Process through transformer layers
                for i, layer in enumerate(self.layers):
                    if output_hidden_states:
                        all_hidden_states.append(x)
                    
                    # Apply layer
                    if isinstance(layer, ReasoningBlock):
                        x, block_info = layer(x)
                        block_info['layer'] = i
                        reasoning_info.append(block_info)
                    else:
                        x = layer(x)
                        
                # Final layer norm
                x = self.ln_final(x)
                
                # Prepare output dictionary
                outputs = {
                    "hidden_states": x,
                    "all_hidden_states": all_hidden_states,
                    "attentions": all_attentions,
                    "reasoning_info": reasoning_info if reasoning_info else None
                }
                
                return outputs
                
        # Instantiate and return the enhanced transformer
        return EnhancedReasoningTransformer(gpt_config)
    
    @classmethod
    def from_3b_config(cls):
        """
        Create a 3B parameter model using the default 3B configuration.
        
        Returns:
            CoreModel: A model with 3B parameters
        """
        return cls(use_3b_config=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        collect_reasoning_info: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the CoreModel.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            labels: Labels for language modeling [batch_size, seq_len]
            output_hidden_states: Whether to output all hidden states
            output_attentions: Whether to output attention weights
            collect_reasoning_info: Whether to collect reasoning information
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of outputs
        """
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        # Get hidden states
        hidden_states = transformer_outputs["hidden_states"]
        
        # Apply lm_head to get logits
        logits = self.lm_head(hidden_states)
        
        # Initialize outputs dictionary
        outputs = {
            "logits": logits,
            "hidden_states": hidden_states
        }
        
        # Add additional outputs if available
        if output_hidden_states and "all_hidden_states" in transformer_outputs:
            outputs["all_hidden_states"] = transformer_outputs["all_hidden_states"]
            
        if output_attentions and "attentions" in transformer_outputs:
            outputs["attentions"] = transformer_outputs["attentions"]
            
        # Add reasoning info if collected and requested
        if collect_reasoning_info and "reasoning_info" in transformer_outputs:
            outputs["reasoning_info"] = transformer_outputs["reasoning_info"]
        
        # Calculate loss if labels are provided
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
            
            outputs["loss"] = loss
        
        return outputs
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.transformer, 'gradient_checkpointing_enable'):
            self.transformer.gradient_checkpointing_enable()
        else:
            # Manually enable gradient checkpointing for the transformer layers
            for module in self.modules():
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
        
        logger.info("Gradient checkpointing enabled for memory efficiency")
