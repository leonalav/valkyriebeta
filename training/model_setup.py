import os
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from training.layers.hybrid_model import HybridRWKVTransformerModel

logger = logging.getLogger(__name__)

def setup_model(args, model_config, tokenizer, training_config, architecture_params=None, hybrid_configurator=None):
    """
    Set up the model for training
    
    Args:
        args: Command line arguments
        model_config: Model configuration
        tokenizer: Tokenizer
        training_config: Training configuration
        architecture_params: Optional architecture parameters
        hybrid_configurator: Optional hybrid model configuration
    
    Returns:
        Model ready for training
    """
    logger.info(f"Setting up model with {model_config.hidden_size} hidden size and {model_config.num_layers} layers")
    
    # If using RWKV architecture
    if getattr(model_config, 'use_rwkv', False):
        logger.info("Setting up RWKV-based model")
        model = setup_rwkv_model(args, model_config, tokenizer, hybrid_configurator)
    else:
        logger.info("Setting up Transformer-based model")
        model = setup_transformer_model(args, model_config, tokenizer, architecture_params)
    
    # Apply common setup steps
    if training_config.compile_model and hasattr(torch, 'compile'):
        logger.info(f"Compiling model with {training_config.dynamo_backend} backend")
        model = torch.compile(model, backend=training_config.dynamo_backend)
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model

def setup_rwkv_model(args, model_config, tokenizer, hybrid_configurator=None):
    """
    Set up RWKV or hybrid RWKV-Transformer model
    
    Args:
        args: Command line arguments
        model_config: Model configuration
        tokenizer: Tokenizer
        hybrid_configurator: Optional hybrid model configuration
    
    Returns:
        RWKV or hybrid model
    """
    # Check if we're using a hybrid model or pure RWKV
    is_hybrid = getattr(model_config, 'attention_mechanism', None) == "hybrid_rwkv_transformer"
    
    if is_hybrid and hybrid_configurator:
        logger.info("Setting up hybrid RWKV-Transformer model")
        
        # Get RWKV and transformer layer indices from configurator
        rwkv_layer_indices = hybrid_configurator.rwkv_layer_indices
        transformer_layer_indices = hybrid_configurator.transformer_layer_indices
        
        # Create hybrid model
        model = HybridRWKVTransformerModel(
            config=model_config,
            rwkv_layer_indices=rwkv_layer_indices,
            transformer_layer_indices=transformer_layer_indices
        )
        
        # Log layer allocation
        logger.info(f"Created hybrid model with {len(rwkv_layer_indices)} RWKV layers and "
                    f"{len(transformer_layer_indices)} Transformer layers")
                    
    else:
        # Import here to avoid circular import
        from training.layers.rwkv_layer import RWKVModel
        logger.info("Setting up pure RWKV model")
        
        # Create pure RWKV model
        model = RWKVModel(config=model_config)
        
        logger.info(f"Created pure RWKV model with {model_config.num_layers} layers")
    
    # Initialize model
    _initialize_rwkv_parameters(model, model_config)
    
    return model

def setup_transformer_model(args, model_config, tokenizer, architecture_params=None):
    """
    Set up standard transformer model
    
    Args:
        args: Command line arguments
        model_config: Model configuration
        tokenizer: Tokenizer
        architecture_params: Optional architecture parameters
    
    Returns:
        Transformer model
    """
    # Check for pretrained model path
    if hasattr(args, 'pretrained_model_path') and args.pretrained_model_path:
        logger.info(f"Loading pretrained model from {args.pretrained_model_path}")
        # Load pretrained model
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model_path,
            torch_dtype=torch.bfloat16 if getattr(args, 'use_bfloat16', False) else torch.float32
        )
    else:
        # Create configuration
        logger.info("Creating new transformer model from configuration")
        
        config_kwargs = {
            "vocab_size": model_config.vocab_size,
            "n_positions": model_config.max_seq_len,
            "n_embd": model_config.hidden_size,
            "n_layer": model_config.num_layers,
            "n_head": model_config.num_attention_heads,
        }
        
        # Add architecture parameters if provided
        if architecture_params:
            if 'ffn_hidden_size' in architecture_params:
                config_kwargs['n_inner'] = architecture_params['ffn_hidden_size']
            if 'activation_function' in architecture_params:
                config_kwargs['activation_function'] = architecture_params['activation_function']
            if 'attention_dropout' in architecture_params:
                config_kwargs['attn_pdrop'] = architecture_params['attention_dropout']
            if 'layer_norm_epsilon' in architecture_params:
                config_kwargs['layer_norm_epsilon'] = architecture_params['layer_norm_epsilon']
        
        # Create model config
        model_type = getattr(args, 'model_type', 'gpt2')
        config = AutoConfig.from_pretrained(model_type, **config_kwargs)
        
        # Create model
        model = AutoModelForCausalLM.from_config(config)
        
        logger.info(f"Created new {model_type} model with {model_config.num_layers} layers")
    
    return model

def _initialize_rwkv_parameters(model, config):
    """
    Initialize RWKV model parameters
    
    Args:
        model: The RWKV or hybrid model
        config: Model configuration
    """
    logger.info("Initializing RWKV parameters")
    
    # Get initialization scale
    rwkv_init_scale = getattr(config, 'rwkv_init_scale', 0.1)
    
    # Initialize embeddings with small normal
    if hasattr(model, 'emb'):
        nn.init.normal_(model.emb.weight, mean=0.0, std=0.02)
    
    # Properly initialize time decay parameters
    if hasattr(model, 'blocks'):
        for i, block_item in enumerate(model.blocks):
            # Handle hybrid model with tuples
            if isinstance(block_item, tuple):
                block_type, block = block_item
                if block_type != 'rwkv':
                    continue
            else:
                block = block_item
                if block is None:
                    continue
            
            # Initialize RWKV time decay parameters with layer-specific scaling
            decay_base = 0.9 ** (i + 1)
            
            if hasattr(block, 'att') and hasattr(block.att, 'time_decay'):
                # Initialize with exponentially decreasing values
                block.att.time_decay.data = torch.ones_like(block.att.time_decay.data) * math.log(decay_base)
                
            # Initialize time mix parameters
            if hasattr(block, 'att') and hasattr(block.att, 'time_mix_r'):
                # Set time mix parameters to sensible defaults
                time_mix_ratio = 1.0 - 0.1 * (i / max(1, config.num_layers))
                block.att.time_mix_r.data.fill_(time_mix_ratio)
                block.att.time_mix_k.data.fill_(time_mix_ratio)
                block.att.time_mix_v.data.fill_(time_mix_ratio)
            
            # Initialize output gate if present
            if hasattr(block, 'att_gate'):
                block.att_gate.data.fill_(1e-3)  # Start with mostly residual connections
            if hasattr(block, 'ffn_gate'):
                block.ffn_gate.data.fill_(1e-3)  # Start with mostly residual connections
    
    # Initialize output projection
    if hasattr(model, 'head'):
        nn.init.normal_(model.head.weight, mean=0.0, std=0.02 * rwkv_init_scale)

def setup_teacher_model(args, model_config, tokenizer):
    """
    Set up teacher model for knowledge distillation
    
    Args:
        args: Command line arguments
        model_config: Model configuration
        tokenizer: Tokenizer
    
    Returns:
        Teacher model for knowledge distillation
    """
    logger.info(f"Setting up teacher model from {args.teacher_model_path}")
    
    if not args.teacher_model_path:
        raise ValueError("Teacher model path must be provided for knowledge distillation")
    
    # Load teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path,
        torch_dtype=torch.bfloat16 if getattr(args, 'use_bfloat16', False) else torch.float32
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    
    # Set teacher model to evaluation mode
    teacher_model.eval()
    
    return teacher_model

def setup_reward_model(args, model_config, tokenizer):
    """
    Set up reward model for RLHF
    
    Args:
        args: Command line arguments
        model_config: Model configuration
        tokenizer: Tokenizer
    
    Returns:
        Reward model for RLHF
    """
    logger.info(f"Setting up reward model from {args.reward_model_path}")
    
    if not args.reward_model_path:
        raise ValueError("Reward model path must be provided for RLHF")
    
    # Load reward model
    reward_model = AutoModelForCausalLM.from_pretrained(
        args.reward_model_path,
        torch_dtype=torch.bfloat16 if getattr(args, 'use_bfloat16', False) else torch.float32
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model = reward_model.to(device)
    
    # Set reward model to evaluation mode
    reward_model.eval()
    
    return reward_model

def setup_reference_model(args, model_config, tokenizer):
    """
    Set up reference model for RLHF
    
    Args:
        args: Command line arguments
        model_config: Model configuration
        tokenizer: Tokenizer
    
    Returns:
        Reference model for RLHF
    """
    logger.info(f"Setting up reference model from {args.reference_model_path}")
    
    if not args.reference_model_path:
        raise ValueError("Reference model path must be provided for RLHF")
    
    # Load reference model
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.reference_model_path,
        torch_dtype=torch.bfloat16 if getattr(args, 'use_bfloat16', False) else torch.float32
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_model = reference_model.to(device)
    
    # Set reference model to evaluation mode
    reference_model.eval()
    
    return reference_model 