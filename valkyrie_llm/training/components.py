import logging
import os
import json
from typing import Dict, List, Optional, Any, Tuple, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HybridModelConfigurator:
    """
    Configures hybrid models that combine different architectural components.
    
    Particularly focused on RWKV-Transformer hybrid architectures where some layers
    use RWKV mechanisms and others use standard Transformer attention.
    """
    
    def __init__(self, model_config):
        """
        Initialize the hybrid model configurator.
        
        Args:
            model_config: Configuration object for the model
        """
        self.model_config = model_config
        self.num_layers = getattr(model_config, 'num_layers', 12)
        self.rwkv_layer_indices = getattr(model_config, 'rwkv_layer_indices', [])
        
        # Default to 50-50 split if not specified
        if not self.rwkv_layer_indices:
            self.rwkv_layer_indices = list(range(0, self.num_layers, 2))
        
        self.transformer_layer_indices = [i for i in range(self.num_layers) 
                                         if i not in self.rwkv_layer_indices]
    
    def configure_layer_architecture(self, rwkv_layer_indices=None, pattern=None):
        """
        Configure the layer architecture of the hybrid model.
        
        Args:
            rwkv_layer_indices: Optional list of indices for RWKV layers
            pattern: Optional pattern to use ('alternating', 'block', 'deepstart', 'deepend')
        """
        if rwkv_layer_indices is not None:
            self.rwkv_layer_indices = rwkv_layer_indices
        elif pattern is not None:
            self._configure_by_pattern(pattern)
            
        # Update transformer layer indices
        self.transformer_layer_indices = [i for i in range(self.num_layers) 
                                         if i not in self.rwkv_layer_indices]
        
        logger.info(f"Configured hybrid model with {len(self.rwkv_layer_indices)} RWKV layers and "
                   f"{len(self.transformer_layer_indices)} Transformer layers")
    
    def _configure_by_pattern(self, pattern: str, rwkv_ratio: float = 0.5):
        """
        Configure RWKV layer indices based on a pattern.
        
        Args:
            pattern: Pattern to use ('alternating', 'block', 'deepstart', 'deepend')
            rwkv_ratio: Ratio of RWKV layers to total layers (0.0-1.0)
        """
        rwkv_layer_count = int(self.num_layers * rwkv_ratio)
        
        if pattern == "alternating":
            # Alternate RWKV and Transformer layers
            self.rwkv_layer_indices = list(range(0, self.num_layers, 2))[:rwkv_layer_count]
        elif pattern == "block":
            # First half RWKV, second half Transformer
            self.rwkv_layer_indices = list(range(rwkv_layer_count))
        elif pattern == "deepstart":
            # RWKV in earlier layers, Transformer in later layers
            self.rwkv_layer_indices = list(range(rwkv_layer_count))
        elif pattern == "deepend":
            # Transformer in earlier layers, RWKV in later layers
            self.rwkv_layer_indices = list(range(self.num_layers - rwkv_layer_count, self.num_layers))
        else:
            logger.warning(f"Unknown pattern {pattern}, defaulting to alternating")
            self.rwkv_layer_indices = list(range(0, self.num_layers, 2))[:rwkv_layer_count]
    
    def get_layer_distribution(self) -> Dict[str, List[int]]:
        """
        Get the distribution of layer types.
        
        Returns:
            Dictionary with RWKV and Transformer layer indices
        """
        return {
            "rwkv_layers": self.rwkv_layer_indices,
            "transformer_layers": self.transformer_layer_indices
        }
    
    def get_layer_type(self, layer_idx: int) -> str:
        """
        Get the type of a specific layer.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            String 'rwkv' or 'transformer'
        """
        return 'rwkv' if layer_idx in self.rwkv_layer_indices else 'transformer'
    
    def update_model_config(self):
        """
        Update the model config with the current layer distribution.
        
        Returns:
            Updated model config
        """
        # Modify the model_config object with current settings
        self.model_config.rwkv_layer_indices = self.rwkv_layer_indices
        
        # Add a helper function to get layer type
        def get_layer_type(layer_idx):
            return 'rwkv' if layer_idx in self.rwkv_layer_indices else 'transformer'
            
        self.model_config.get_layer_type = get_layer_type
        
        return self.model_config
    
    def export_config(self, path: str):
        """
        Export the hybrid configuration to a JSON file.
        
        Args:
            path: Path to save the configuration
        """
        config_dict = {
            "num_layers": self.num_layers,
            "rwkv_layer_indices": self.rwkv_layer_indices,
            "transformer_layer_indices": self.transformer_layer_indices,
            "rwkv_ratio": len(self.rwkv_layer_indices) / self.num_layers
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Exported hybrid model configuration to {path}")


class HybridRWKVTransformerModel(nn.Module):
    """
    Hybrid model combining RWKV and Transformer layers.
    
    This model uses a mixture of RWKV and standard Transformer layers,
    allowing it to leverage the strengths of both architectures.
    """
    
    def __init__(self, config, tokenizer=None):
        """
        Initialize the hybrid model.
        
        Args:
            config: Model configuration
            tokenizer: Optional tokenizer
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Get layer distribution
        self.rwkv_layer_indices = getattr(config, 'rwkv_layer_indices', [])
        self.num_layers = getattr(config, 'num_layers', 12)
        
        # Create embeddings
        self.embeddings = self._create_embeddings()
        
        # Create hybrid transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            if i in self.rwkv_layer_indices:
                # Create RWKV block
                from valkyrie_llm.training.layers.rwkv_layer import RWKVBlock
                self.blocks.append(RWKVBlock(config, i))
            else:
                # Create Transformer block
                from valkyrie_llm.training.layers.rwkv_layer import TransformerBlock
                self.blocks.append(TransformerBlock(config, i))
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.hidden_size)
        
        # Output projection
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie weights if specified
        if getattr(config, 'tie_word_embeddings', False):
            self.head.weight = self.embeddings.word_embeddings.weight
    
    def _create_embeddings(self):
        """Create embeddings for the model"""
        class Embeddings(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
                
                # Position embeddings if needed
                self.use_position_embeddings = getattr(config, 'use_position_embeddings', True)
                if self.use_position_embeddings:
                    self.position_embeddings = nn.Embedding(
                        config.max_seq_len, config.hidden_size
                    )
                
                self.ln = nn.LayerNorm(config.hidden_size)
                self.dropout = nn.Dropout(getattr(config, 'hidden_dropout_prob', 0.1))
            
            def forward(self, input_ids, position_ids=None):
                # Get word embeddings
                embeddings = self.word_embeddings(input_ids)
                
                # Add position embeddings if enabled
                if self.use_position_embeddings:
                    if position_ids is None:
                        position_ids = torch.arange(
                            input_ids.size(1), device=input_ids.device
                        ).unsqueeze(0).expand(input_ids.size(0), -1)
                    
                    position_embeddings = self.position_embeddings(position_ids)
                    embeddings = embeddings + position_embeddings
                
                # Apply layer norm and dropout
                embeddings = self.ln(embeddings)
                embeddings = self.dropout(embeddings)
                
                return embeddings
        
        return Embeddings(self.config)
    
    def _init_weights(self, module):
        """Initialize weights for the model"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None, 
                past_key_values=None, use_cache=False, return_dict=True, **kwargs):
        """Forward pass through the hybrid model"""
        # Get embeddings
        x = self.embeddings(input_ids, position_ids)
        
        # Initialize past states for RWKV layers if needed
        batch_size = input_ids.size(0)
        state = {}
        new_states = {} if use_cache else None
        
        # Setup attention mask for transformer blocks
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, input_ids.size(1), device=input_ids.device)
        
        # Process through blocks
        for i, block in enumerate(self.blocks):
            if i in self.rwkv_layer_indices:
                # RWKV block
                past_state = None if past_key_values is None else past_key_values.get(f"layer_{i}", None)
                x, new_state = block(x, past_state)
                if use_cache:
                    new_states[f"layer_{i}"] = new_state
            else:
                # Transformer block
                past_kv = None if past_key_values is None else past_key_values.get(f"layer_{i}", None)
                x, past_kv, _ = block(x, attention_mask, past_key_value=past_kv)
                if use_cache:
                    new_states[f"layer_{i}"] = past_kv
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Get logits
        logits = self.head(x)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        # Return appropriate output format
        if return_dict:
            return {'loss': loss, 'logits': logits, 'past_key_values': new_states}
        else:
            return (loss, logits, new_states)


class RWKVIntegrator:
    """
    Integrates RWKV components into existing models and architectures.
    
    Handles conversion, optimization, and integration of RWKV-specific
    components with other model architectures.
    """
    
    def __init__(self, model=None, model_config=None, training_config=None):
        """
        Initialize the RWKV integrator.
        
        Args:
            model: Optional model to integrate RWKV components into
            model_config: Configuration for the model
            training_config: Training configuration
        """
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        
        # If a model is provided, extract its config
        if self.model is not None and self.model_config is None:
            if hasattr(self.model, 'config'):
                self.model_config = self.model.config
    
    def convert_to_rwkv(self, transformer_model, rwkv_layer_indices=None):
        """
        Convert a transformer model to a hybrid RWKV-Transformer model.
        
        Args:
            transformer_model: The transformer model to convert
            rwkv_layer_indices: Indices of layers to convert to RWKV
            
        Returns:
            Hybrid RWKV-Transformer model
        """
        if rwkv_layer_indices is None:
            # Default to converting half the layers
            num_layers = len(transformer_model.blocks) if hasattr(transformer_model, 'blocks') else 12
            rwkv_layer_indices = list(range(0, num_layers, 2))  # Every other layer
        
        # Update config with RWKV layer indices
        config = transformer_model.config if hasattr(transformer_model, 'config') else self.model_config
        config.rwkv_layer_indices = rwkv_layer_indices
        
        # Create hybrid model
        hybrid_model = HybridRWKVTransformerModel(config, transformer_model.tokenizer)
        
        # Copy compatible weights
        self._copy_transformer_weights(transformer_model, hybrid_model)
        
        return hybrid_model
    
    def _copy_transformer_weights(self, src_model, tgt_model):
        """
        Copy compatible weights from transformer model to hybrid model.
        
        Args:
            src_model: Source transformer model
            tgt_model: Target hybrid model
        """
        # Copy embeddings
        if hasattr(src_model, 'embeddings') and hasattr(tgt_model, 'embeddings'):
            for tgt_name, tgt_param in tgt_model.embeddings.named_parameters():
                if hasattr(src_model.embeddings, tgt_name.split('.')[-1]):
                    src_param = getattr(src_model.embeddings, tgt_name.split('.')[-1]).parameters()
                    if next(src_param).shape == tgt_param.shape:
                        tgt_param.data.copy_(next(src_param))
        
        # Copy compatible transformer blocks
        for i, (src_block, tgt_block) in enumerate(zip(src_model.blocks, tgt_model.blocks)):
            if i not in tgt_model.rwkv_layer_indices:
                # Copy transformer block parameters
                for tgt_name, tgt_param in tgt_block.named_parameters():
                    for src_name, src_param in src_block.named_parameters():
                        if tgt_name.split('.')[-1] == src_name.split('.')[-1] and tgt_param.shape == src_param.shape:
                            tgt_param.data.copy_(src_param)
        
        # Copy output head if compatible
        if hasattr(src_model, 'head') and hasattr(tgt_model, 'head'):
            if src_model.head.weight.shape == tgt_model.head.weight.shape:
                tgt_model.head.weight.data.copy_(src_model.head.weight)
    
    def apply_rwkv_optimizations(self):
        """
        Apply RWKV-specific optimizations to the model.
        
        Returns:
            Optimized model
        """
        if self.model is None:
            logger.warning("No model provided for RWKV optimizations")
            return None
        
        # Apply optimizations
        if hasattr(self.model, 'rwkv_layer_indices'):
            logger.info("Applying RWKV-specific optimizations")
            
            # Set chunk size for efficient processing
            chunk_size = getattr(self.model_config, 'rwkv_chunk_size', 1024)
            if hasattr(self.model, 'chunk_size'):
                self.model.chunk_size = chunk_size
            
            # Enable state reuse for inference
            if hasattr(self.model, 'enable_state_reuse'):
                self.model.enable_state_reuse()
            
            # Apply fused kernels if available
            try:
                from valkyrie_llm.training.kernels import fuse_rwkv_kernels
                self.model = fuse_rwkv_kernels(self.model)
                logger.info("Applied fused RWKV kernels")
            except (ImportError, AttributeError):
                logger.warning("Fused RWKV kernels not available, skipping optimization")
        
        return self.model
    
    def export_rwkv_weights(self, output_path):
        """
        Export RWKV-specific weights to a separate file.
        
        Args:
            output_path: Path to save the weights
            
        Returns:
            Path to the saved weights
        """
        if self.model is None:
            logger.warning("No model provided for RWKV weight export")
            return None
        
        # Extract RWKV weights
        rwkv_state_dict = {}
        full_state_dict = self.model.state_dict()
        
        for name, param in full_state_dict.items():
            # Check if this is a RWKV layer parameter
            for idx in getattr(self.model, 'rwkv_layer_indices', []):
                if f'blocks.{idx}' in name:
                    rwkv_state_dict[name] = param.clone().detach().cpu()
        
        # Save weights
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        
        torch.save(rwkv_state_dict, output_path)
        logger.info(f"Exported RWKV weights to {output_path}")
        
        return output_path 