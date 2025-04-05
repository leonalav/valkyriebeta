import torch
import logging
import heapq
import math
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union, Any
from contextlib import nullcontext
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class BeamSearchConfig:
    """Configuration for BeamSearchGenerator"""
    beam_size: int = 4
    length_penalty: float = 1.0
    early_stopping: bool = True
    do_sample: bool = False
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 1.0
    num_return_sequences: int = 1
    use_logical_consistency: bool = False
    logical_reward_weight: float = 0.5
    logical_penalty_weight: float = 0.8
    consistency_threshold: float = 0.7
    uncertainty_penalty: float = 0.3
    use_mixed_precision: bool = False
    mixed_precision_dtype: torch.dtype = torch.float16

class BeamHypothesis:
    """Beam search hypothesis with score and sequence"""
    def __init__(self, sequence, score, states=None):
        self.sequence = sequence
        self.score = score
        self.states = states
        # Additional metrics for logical consistency
        self.consistency_score = 0.5  # Default neutral consistency
        self.uncertainty = 0.1  # Default low uncertainty
        self.logical_violations = 0  # No violations initially
    
    def __lt__(self, other):
        return self.score > other.score  # Inverted for max-heap

class BeamSearchGenerator:
    """
    An advanced beam search generator for text generation with logical consistency support.
    
    Features:
    - Standard beam search with customizable parameters
    - Optional logical consistency checking
    - State management for RWKV and recurrent models
    - Mixed precision support
    - Dynamic parameter adjustment
    - Comprehensive error handling
    """
    def __init__(self, model=None, tokenizer=None, config=None, **kwargs):
        """
        Initialize the beam search generator.
        
        Args:
            model: The language model to use for generation
            tokenizer: Tokenizer for the model
            config: Configuration for beam search (BeamSearchConfig or dict)
            **kwargs: Additional parameters to override config
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize config from provided config or defaults
        if config is None:
            self.config = BeamSearchConfig()
        elif isinstance(config, dict):
            self.config = BeamSearchConfig(**config)
        else:
            self.config = config
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Extract common config parameters for convenience
        self.beam_size = self.config.beam_size
        self.length_penalty = self.config.length_penalty
        self.do_sample = self.config.do_sample
        self.early_stopping = self.config.early_stopping
        
        # Check if model is RWKV type
        self.is_rwkv_model = False
        if model is not None:
            self.is_rwkv_model = hasattr(model, 'states') or hasattr(model, 'rwkv_states')
        
        # Special tokens handling
        self.eos_token_id = kwargs.get('eos_token_id', None)
        if self.eos_token_id is None and model is not None:
            if hasattr(model, 'config'):
                self.eos_token_id = getattr(model.config, 'eos_token_id', None)
            
            if self.eos_token_id is None and tokenizer is not None:
                self.eos_token_id = tokenizer.eos_token_id
                
            if self.eos_token_id is None:
                self.eos_token_id = 2  # Common default
        
        # Setup logging
        self.verbose = kwargs.get('verbose', True)
        if self.verbose:
            logger.info(f"Initialized BeamSearchGenerator with beam size {self.beam_size}")
            if self.config.use_logical_consistency:
                logger.info("Logical consistency checking enabled")
            if self.is_rwkv_model:
                logger.info("RWKV model detected, state management enabled")
    
    def _top_k_top_p_filtering(
        self, 
        logits: torch.Tensor, 
        top_k: int = 50, 
        top_p: float = 0.9, 
        filter_value: float = -float("Inf")
    ) -> torch.Tensor:
        """
        Filter logits using top-k and top-p (nucleus) filtering
        
        Args:
            logits: Logits to filter
            top_k: Keep only top k tokens with highest probability
            top_p: Keep the top tokens with cumulative probability >= top_p
            filter_value: Value to assign to filtered tokens
            
        Returns:
            Filtered logits
        """
        # Remove low probabilities using top-k
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, filter_value)
        
        # Remove low probabilities using top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter back
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, filter_value)
        
        return logits
    
    def _extract_model_states(self, model_outputs):
        """Extract states from model outputs for recurrent models"""
        states = None
        
        # Try to extract states from various possible locations
        if hasattr(model_outputs, 'states'):
            states = model_outputs.states
        elif isinstance(model_outputs, dict) and 'states' in model_outputs:
            states = model_outputs['states']
        elif self.is_rwkv_model:
            if hasattr(self.model, 'states'):
                states = self.model.states
            elif hasattr(self.model, 'rwkv_states'):
                states = self.model.rwkv_states
        
        return states
    
    def _evaluate_logical_consistency(self, model_outputs, logits=None):
        """
        Evaluate logical consistency and extract metrics from model outputs
        
        Returns:
            Tuple of (consistency score, uncertainty, violation count)
        """
        # Default values
        consistency = 0.5  # Neutral consistency
        uncertainty = 0.1  # Low uncertainty
        violations = 0
        
        # Check for explicit reasoning outputs
        if isinstance(model_outputs, dict) and 'reasoning_trace' in model_outputs:
            trace = model_outputs['reasoning_trace']
            if isinstance(trace, list) and trace:
                last_trace = trace[-1]
                if isinstance(last_trace, dict):
                    consistency = last_trace.get('confidence', consistency)
                    uncertainty = last_trace.get('uncertainty', uncertainty)
                    violations = last_trace.get('violations', violations)
                elif isinstance(last_trace, (int, float)):
                    consistency = last_trace
        
        # Try to estimate from hidden states
        elif isinstance(model_outputs, dict) and 'hidden_states' in model_outputs:
            hidden = model_outputs['hidden_states']
            if isinstance(hidden, torch.Tensor) and hidden.dim() > 1:
                # Use hidden state norm as proxy for consistency
                state_norm = torch.norm(hidden[:, -1, :], dim=-1)
                consistency = min(1.0, state_norm.item() / math.sqrt(hidden.size(-1)))
                
                # Estimate uncertainty from hidden state variance
                if hidden.size(1) > 1:
                    variance = torch.var(hidden[:, -2:, :], dim=1).mean().item()
                    uncertainty = min(1.0, variance)
        
        # Try to estimate from logits if available
        elif logits is not None:
            # Use entropy of logits as proxy for uncertainty
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            max_entropy = math.log(probs.size(-1))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else entropy
            
            uncertainty = normalized_entropy.item()
            # Higher entropy usually indicates lower consistency
            consistency = 1.0 - min(1.0, uncertainty * 2)
        
        return consistency, uncertainty, violations
        
    def generate(
        self, 
        input_ids, 
        attention_mask=None, 
        max_length=100, 
        min_length=0,
        use_cache=True,
        use_custom_implementation=True,
        **kwargs
    ):
        """
        Generate text using beam search with enhanced logical consistency.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask for input
            max_length: Maximum length of generated sequence
            min_length: Minimum length of generated sequence
            use_cache: Whether to use KV-cache for faster generation
            use_custom_implementation: Use our custom beam search implementation instead of model.generate()
            **kwargs: Additional parameters to override config
            
        Returns:
            Generated token IDs [batch_size * num_return_sequences, seq_len]
        """
        if self.verbose:
            logger.info(f"Generating with beam search (beam size: {self.beam_size})")
        
        # Mock implementation if no model
        if self.model is None:
            return self._mock_generate(input_ids, max_length)
        
        # Use model's generate() if custom implementation not requested
        if not use_custom_implementation and hasattr(self.model, 'generate'):
            try:
                # Extract generation parameters from config and kwargs
                generate_kwargs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'max_length': max_length,
                    'min_length': min_length,
                    'num_beams': self.beam_size,
                    'length_penalty': self.length_penalty,
                    'early_stopping': self.early_stopping,
                    'num_return_sequences': self.config.num_return_sequences,
                    'use_cache': use_cache,
                }
                
                # Add optional parameters if they're set
                if self.do_sample:
                    generate_kwargs.update({
                        'do_sample': True,
                        'top_k': self.config.top_k,
                        'top_p': self.config.top_p,
                        'temperature': self.config.temperature,
                    })
                
                # Override with any provided kwargs
                generate_kwargs.update(kwargs)
                
                return self.model.generate(**generate_kwargs)
            except Exception as e:
                logger.warning(f"Error using model.generate(): {e}")
                logger.info("Falling back to custom beam search implementation")
                use_custom_implementation = True
        
        # Custom beam search implementation with logical consistency
        if use_custom_implementation:
            return self._custom_beam_search_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=min_length,
                use_cache=use_cache,
                **kwargs
            )
    
    def _mock_generate(self, input_ids, max_length):
        """Generate mock output when no model is provided"""
        # Just return the input followed by some placeholder tokens
        batch_size = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
        mock_length = min(max_length, 20)  # Limit mock generation
        
        if isinstance(input_ids, torch.Tensor):
            device = input_ids.device
            # Create mock output: input_ids followed by some tokens
            output = torch.cat([
                input_ids,
                torch.randint(
                    100, 1000, 
                    (batch_size, mock_length - input_ids.shape[1]), 
                    device=device
                )
            ], dim=1)
            return output
        else:
            # Handle non-tensor inputs
            return [1, 2, 3, 4, 5]  # Mock output
    
    def _custom_beam_search_generate(
        self, 
        input_ids, 
        attention_mask=None, 
        max_length=100,
        min_length=0,
        use_cache=True,
        **kwargs
    ):
        """
        Custom beam search implementation with logical consistency checking.
        
        This implementation supports:
        - Logical consistency evaluation and rewarding
        - State management for RWKV models
        - Mixed precision for faster generation
        - Dynamic parameter adjustment
        """
        # Initialize generation parameters
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Extract config values, with kwargs overrides
        beam_size = kwargs.get('num_beams', self.beam_size)
        length_penalty = kwargs.get('length_penalty', self.length_penalty)
        do_sample = kwargs.get('do_sample', self.do_sample)
        early_stopping = kwargs.get('early_stopping', self.early_stopping)
        top_k = kwargs.get('top_k', self.config.top_k)
        top_p = kwargs.get('top_p', self.config.top_p)
        temperature = kwargs.get('temperature', self.config.temperature)
        num_return_sequences = min(kwargs.get('num_return_sequences', 
                                             self.config.num_return_sequences), 
                                  beam_size)
        
        # Logical consistency parameters
        use_logical = kwargs.get('use_logical_consistency', self.config.use_logical_consistency)
        logical_reward = kwargs.get('logical_reward_weight', self.config.logical_reward_weight)
        logical_penalty = kwargs.get('logical_penalty_weight', self.config.logical_penalty_weight)
        consistency_threshold = kwargs.get('consistency_threshold', self.config.consistency_threshold)
        uncertainty_penalty = kwargs.get('uncertainty_penalty', self.config.uncertainty_penalty)
        
        # Mixed precision settings
        use_mixed_precision = kwargs.get('use_mixed_precision', self.config.use_mixed_precision)
        mixed_precision_dtype = kwargs.get('mixed_precision_dtype', self.config.mixed_precision_dtype)
        
        # Configure mixed precision context
        if use_mixed_precision and torch.cuda.is_available():
            mp_ctx = torch.cuda.amp.autocast(dtype=mixed_precision_dtype)
        else:
            mp_ctx = nullcontext()
        
        # Initialize beams for each batch item
        beams = [[] for _ in range(batch_size)]
        for batch_idx in range(batch_size):
            # Create initial hypothesis for this batch item
            batch_input = input_ids[batch_idx:batch_idx+1]
            
            # Initialize RWKV states if needed
            states = None
            if self.is_rwkv_model:
                # Run model on prefix to get initial state
                with mp_ctx:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=batch_input,
                            attention_mask=attention_mask[batch_idx:batch_idx+1] if attention_mask is not None else None,
                            use_cache=use_cache,
                            return_dict=True,
                            **kwargs.get('model_kwargs', {})
                        )
                        states = self._extract_model_states(outputs)
            
            # Add initial hypothesis to beams
            beams[batch_idx].append(BeamHypothesis(
                sequence=batch_input,
                score=0.0,
                states=states
            ))
        
        # Track generation statistics
        generation_stats = {"logical_violations_avoided": 0}
        
        # Generate step by step
        for step in range(max_length - input_ids.size(1)):
            next_beams = [[] for _ in range(batch_size)]
            
            # Process each batch item independently
            for batch_idx in range(batch_size):
                current_beams = beams[batch_idx]
                candidates = []
                
                # Process each beam
                for beam in current_beams:
                    # Skip completed sequences
                    if beam.sequence.size(1) > 0 and beam.sequence[0, -1].item() == self.eos_token_id:
                        if beam.sequence.size(1) >= min_length:
                            candidates.append(beam)
                        continue
                    
                    # Forward pass through model with state handling
                    with mp_ctx:
                        with torch.no_grad():
                            # Prepare model inputs
                            model_inputs = {
                                'input_ids': beam.sequence,
                                'use_cache': use_cache,
                                'return_dict': True
                            }
                            
                            # Add attention mask if provided
                            if attention_mask is not None:
                                model_inputs['attention_mask'] = attention_mask[batch_idx:batch_idx+1, :beam.sequence.size(1)]
                            
                            # Handle RWKV states
                            if self.is_rwkv_model and beam.states is not None:
                                # Save original model states
                                original_states = None
                                if hasattr(self.model, 'states'):
                                    original_states = self.model.states
                                    self.model.states = beam.states
                                elif hasattr(self.model, 'rwkv_states'):
                                    original_states = self.model.rwkv_states
                                    self.model.rwkv_states = beam.states
                            
                            # Add additional model kwargs
                            model_inputs.update(kwargs.get('model_kwargs', {}))
                            
                            # Forward pass
                            outputs = self.model(**model_inputs)
                    
                    # Get next token logits
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        next_token_logits = outputs['logits'][:, -1, :] / temperature
                    else:
                        next_token_logits = outputs[:, -1, :] / temperature
                    
                    # Apply min_length constraint
                    if step < min_length - 1:
                        next_token_logits[:, self.eos_token_id] = -float('inf')
                    
                    # Apply logical consistency evaluation if enabled
                    if use_logical:
                        consistency, uncertainty, violations = self._evaluate_logical_consistency(
                            outputs, next_token_logits
                        )
                        
                        # Store metrics in beam
                        beam.consistency_score = consistency
                        beam.uncertainty = uncertainty
                        beam.logical_violations = violations if violations > beam.logical_violations else beam.logical_violations
                        
                        # Apply logical rewards/penalties
                        if consistency > consistency_threshold:
                            next_token_logits = next_token_logits + logical_reward
                        elif violations > beam.logical_violations:
                            next_token_logits = next_token_logits - logical_penalty
                            generation_stats["logical_violations_avoided"] += 1
                        
                        # Apply uncertainty penalty
                        if uncertainty > 0.3:  # High uncertainty threshold
                            next_token_logits = next_token_logits - (uncertainty * uncertainty_penalty)
                    
                    # Get next tokens with sampling or greedy selection
                    if do_sample:
                        # Apply top-k and top-p filtering
                        filtered_logits = self._top_k_top_p_filtering(
                            next_token_logits, top_k=top_k, top_p=top_p
                        )
                        
                        # Sample from the filtered distribution
                        probs = F.softmax(filtered_logits, dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=beam_size)
                        token_scores = torch.gather(probs, -1, next_tokens)
                    else:
                        # Get top-k candidates deterministically
                        token_scores, next_tokens = next_token_logits.topk(
                            min(beam_size, next_token_logits.size(-1)), dim=-1
                        )
                        token_scores = F.softmax(token_scores, dim=-1)
                    
                    # Extract model states for next step
                    next_states = self._extract_model_states(outputs)
                    
                    # Create candidates for each possible next token
                    for token_idx, (score, token) in enumerate(zip(token_scores[0], next_tokens[0])):
                        # Create new sequence with the next token
                        new_sequence = torch.cat([
                            beam.sequence, token.unsqueeze(0).unsqueeze(0)
                        ], dim=1)
                        
                        # Calculate sequence score with length penalty
                        sequence_score = beam.score + math.log(score.item() + 1e-8)
                        normalized_score = sequence_score / ((5 + step + 1) / 6) ** length_penalty
                        
                        # Create new candidate
                        new_beam = BeamHypothesis(
                            sequence=new_sequence,
                            score=normalized_score,
                            states=next_states
                        )
                        
                        # Copy logical metrics if available
                        if use_logical:
                            new_beam.consistency_score = beam.consistency_score
                            new_beam.uncertainty = beam.uncertainty
                            new_beam.logical_violations = beam.logical_violations
                        
                        candidates.append(new_beam)
                
                # Select top beams according to score
                candidates.sort()  # Uses __lt__ method for sorting
                next_beams[batch_idx] = candidates[:beam_size]
            
            # Update beams
            beams = next_beams
            
            # Early stopping if all beams are finished
            if early_stopping and all(
                all(beam.sequence[0, -1].item() == self.eos_token_id for beam in batch_beams)
                for batch_beams in beams
            ):
                break
        
        # Select output sequences
        output_sequences = []
        for batch_beams in beams:
            # Sort beams by score and logical consistency if enabled
            if use_logical:
                sorted_beams = sorted(
                    batch_beams,
                    key=lambda x: (x.score, x.consistency_score, -x.logical_violations),
                    reverse=True
                )
            else:
                sorted_beams = sorted(batch_beams, key=lambda x: x.score, reverse=True)
            
            # Take the requested number of sequences
            for i in range(min(num_return_sequences, len(sorted_beams))):
                output_sequences.append(sorted_beams[i].sequence)
        
        # Pad all sequences to the same length
        max_gen_length = max(seq.size(1) for seq in output_sequences)
        padded_sequences = []
        
        for seq in output_sequences:
            # Pad if needed
            if seq.size(1) < max_gen_length:
                padding = torch.full(
                    (1, max_gen_length - seq.size(1)),
                    self.eos_token_id,
                    dtype=seq.dtype,
                    device=seq.device
                )
                seq = torch.cat([seq, padding], dim=1)
            padded_sequences.append(seq)
        
        # Concatenate all sequences
        result = torch.cat(padded_sequences, dim=0)
        
        if self.verbose:
            logger.info(f"Generated {len(output_sequences)} sequences")
            if use_logical:
                logger.info(f"Avoided {generation_stats['logical_violations_avoided']} logical violations")
        
        return result 