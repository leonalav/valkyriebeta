import torch
import torch.nn.functional as F
import math
import logging
from typing import List, Tuple, Dict, Optional, Union, Any, Callable, Set
from dataclasses import dataclass, field
from contextlib import nullcontext
from enum import Enum

class RuleType(Enum):
    CONTRADICTION = "contradiction"
    SYLLOGISM = "syllogism"
    EQUIVALENCE = "equivalence"
    IMPLICATION = "implication"
    MUTUAL_EXCLUSION = "mutual_exclusion"
    SUBSET_RELATION = "subset_relation"

@dataclass
class BeamHypothesis:
    """Class to hold beam search hypothesis information"""
    sequence: torch.Tensor
    score: float
    reasoning_trace: List[float] = field(default_factory=list)
    uncertainty: float = 0.0
    logical_violations: int = 0
    premise_satisfaction: float = 1.0
    reasoning_states: Optional[Any] = None

@dataclass
class LogicalRule:
    """Class to define a logical rule for consistency checking"""
    rule_type: RuleType
    name: str
    description: str
    condition_fn: Callable
    penalty_weight: float = 1.0
    activation_threshold: float = 0.5
    dependent_rules: Set[str] = field(default_factory=set)
    
    def check_violation(self, context, generated_text, states=None) -> Tuple[bool, float]:
        """Check if this rule is violated and return violation severity"""
        return self.condition_fn(context, generated_text, states)

@dataclass
class Premise:
    """Class to represent a logical premise for consistency checking"""
    id: str
    text: str
    keywords: List[str]
    importance: float = 1.0
    negation_terms: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    satisfaction_score: float = 0.0
    validated: bool = False
    
    def check_satisfaction(self, generated_text: str) -> float:
        """
        Check if the premise is satisfied in the generated text
        
        Returns:
            Float between 0.0 and 1.0 indicating satisfaction level
        """
        text_lower = generated_text.lower()
        
        # Basic keyword presence check
        keyword_matches = sum(keyword.lower() in text_lower for keyword in self.keywords)
        keyword_score = keyword_matches / max(1, len(self.keywords))
        
        # Check for negations that would invalidate the premise
        negation_present = any(neg.lower() in text_lower for neg in self.negation_terms)
        negation_factor = 0.2 if negation_present else 1.0
        
        # Check for implications being honored
        implication_score = 0.0
        if self.implications:
            impl_matches = sum(impl.lower() in text_lower for impl in self.implications)
            implication_score = impl_matches / len(self.implications)
        
        # Check for contradictions being avoided
        contradiction_score = 1.0
        if self.contradictions:
            contr_matches = sum(contr.lower() in text_lower for contr in self.contradictions)
            contradiction_score = 1.0 - (contr_matches / len(self.contradictions))
        
        # Combined satisfaction score
        satisfaction = keyword_score * negation_factor * 0.4 + implication_score * 0.3 + contradiction_score * 0.3
        self.satisfaction_score = satisfaction
        
        # Mark as validated if satisfaction is high enough
        self.validated = satisfaction > 0.7
        
        return satisfaction
        
    def __hash__(self):
        return hash(self.id)

class LogicalBeamSearch:
    """
    Enhanced beam search with logical consistency checking for RWKV and transformer models.
    
    This implementation incorporates:
    1. Logical consistency evaluation using state tracking
    2. Support for RWKV recurrent state propagation
    3. Premise-based reasoning validation
    4. Uncertainty estimation and penalization
    5. Mixed precision support for faster generation
    6. Efficient batch processing
    """
    def __init__(self, model, config):
        self.model = model
        self.beam_size = getattr(config, 'beam_size', 4)
        self.max_length = getattr(config, 'max_seq_length', 2048)
        self.length_penalty = getattr(config, 'length_penalty', 1.0)
        self.consistency_threshold = getattr(config, 'consistency_threshold', 0.7)
        self.logical_reward_weight = getattr(config, 'logical_reward_weight', 0.5)
        self.logical_penalty_weight = getattr(config, 'logical_penalty_weight', 0.8)
        self.uncertainty_penalty = getattr(config, 'uncertainty_penalty', 0.3)
        self.early_stopping = getattr(config, 'early_stopping', True)
        self.do_sample = getattr(config, 'do_sample', False)
        self.top_k = getattr(config, 'top_k', 50)
        self.top_p = getattr(config, 'top_p', 0.9)
        
        # Check if model is RWKV type
        self.is_rwkv_model = hasattr(self.model, 'states') or hasattr(self.model, 'rwkv_states')
        
        # Special tokens
        self.eos_token_id = getattr(config, 'eos_token_id', None)
        if self.eos_token_id is None:
            # Try to get from model config
            if hasattr(self.model, 'config'):
                self.eos_token_id = getattr(self.model.config, 'eos_token_id', 2)  # 2 is common for EOS
            else:
                self.eos_token_id = 2  # Default
        
        # Mixed precision
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', False)
        self.mixed_precision_dtype = getattr(config, 'mixed_precision_dtype', torch.float16)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Logical consistency rules
        self.logical_rules = self._initialize_logical_rules()
        
        # Track metadata
        self.generation_metadata = {
            "total_generations": 0,
            "logical_violations_avoided": 0,
            "high_uncertainty_tokens_avoided": 0
        }
    
    def _initialize_logical_rules(self):
        """
        Initialize a sophisticated set of logical rules for consistency checking.
        """
        rules = {}
        
        # Contradiction detection rule
        def contradiction_condition(context, generated_text, states=None):
            # Example implementation checking for direct contradictions
            # In a real implementation, this would use NLI or semantic analysis
            if states is not None and isinstance(states, dict) and 'contradiction_score' in states:
                return states['contradiction_score'] > 0.7, states['contradiction_score']
            
            # Simplified heuristic based on negation words
            negation_terms = ["not", "never", "no", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't"]
            affirmation_count = 0
            negation_count = 0
            
            tokens = generated_text.split()
            for i, token in enumerate(tokens):
                if token.lower() in negation_terms:
                    negation_count += 1
                    # Check if a previous affirmation is being negated
                    if i > 0 and tokens[i-1].lower() not in negation_terms:
                        return True, 0.8
            
            return False, 0.0
        
        rules["contradiction"] = LogicalRule(
            rule_type=RuleType.CONTRADICTION,
            name="contradiction_detection",
            description="Detects direct contradictions in generated text",
            condition_fn=contradiction_condition,
            penalty_weight=2.0,
            activation_threshold=0.6
        )
        
        # Syllogism validation rule
        def syllogism_condition(context, generated_text, states=None):
            if states is not None and isinstance(states, dict) and 'syllogism_validity' in states:
                return not states['syllogism_validity'], 1.0 - states['syllogism_validity']
            
            # Basic pattern detection for syllogistic reasoning errors
            # This is a placeholder for more sophisticated logic
            syllogism_patterns = [
                (["all", "are"], ["none", "are"]),
                (["if", "then"], ["if", "not"]),
                (["must be"], ["cannot be"])
            ]
            
            text_lower = generated_text.lower()
            
            # Check for pattern conflicts
            for premise, conclusion in syllogism_patterns:
                if all(p in text_lower for p in premise) and all(c in text_lower for c in conclusion):
                    # Simple check for logical form issues
                    premise_idx = max(text_lower.find(p) for p in premise)
                    conclusion_idx = min(text_lower.find(c) for c in conclusion)
                    
                    if premise_idx < conclusion_idx:
                        return True, 0.7
            
            return False, 0.0
        
        rules["syllogism"] = LogicalRule(
            rule_type=RuleType.SYLLOGISM,
            name="syllogism_validation",
            description="Validates syllogistic reasoning patterns",
            condition_fn=syllogism_condition,
            penalty_weight=1.5,
            activation_threshold=0.5
        )
        
        # Mutual exclusion rule
        def mutual_exclusion_condition(context, generated_text, states=None):
            if states is not None and isinstance(states, dict) and 'exclusion_violation' in states:
                return states['exclusion_violation'] > 0.6, states['exclusion_violation']
            
            # Check for mutually exclusive concepts appearing as compatible
            mutually_exclusive_pairs = [
                (["true", "false"], 0.9),
                (["alive", "dead"], 0.9),
                (["always", "never"], 0.8),
                (["all", "none"], 0.8),
                (["increase", "decrease"], 0.6)
            ]
            
            text_lower = generated_text.lower()
            
            for pair, severity in mutually_exclusive_pairs:
                # Check if both terms appear in a "both X and Y" construction
                if all(term in text_lower for term in pair):
                    if "both " in text_lower or " and " in text_lower:
                        # Simple window check to see if they're being presented as compatible
                        term_positions = [text_lower.find(term) for term in pair]
                        if abs(term_positions[0] - term_positions[1]) < 20:  # Within a small window
                            return True, severity
            
            return False, 0.0
        
        rules["mutual_exclusion"] = LogicalRule(
            rule_type=RuleType.MUTUAL_EXCLUSION, 
            name="mutual_exclusion_check",
            description="Checks that mutually exclusive concepts aren't presented as compatible",
            condition_fn=mutual_exclusion_condition,
            penalty_weight=1.7,
            activation_threshold=0.5
        )
        
        # Add more rules as needed...
        
        return rules
    
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
    
    def _extract_reasoning_info(self, model_outputs: Dict[str, Any], sequence: torch.Tensor = None) -> Dict[str, float]:
        """
        Extract comprehensive reasoning information and metrics from model outputs
        
        Args:
            model_outputs: Outputs from the model forward pass
            sequence: The generated sequence so far
            
        Returns:
            Dictionary of reasoning metrics including:
            - logical_consistency: How logically consistent the generation is
            - uncertainty: Model's uncertainty in generation
            - violations: Count of logical violations
            - coherence: Text coherence metric
            - specificity: How specific the reasoning is
            - relevance: Relevance to the original query/context
            - depth: Reasoning depth metric
        """
        # Initialize metrics with default values
        metrics = {
            "logical_consistency": 0.5,  # Neutral consistency
            "uncertainty": 0.1,  # Low uncertainty
            "logical_violations": 0,
            "coherence": 0.5,  # Neutral coherence
            "specificity": 0.5,  # Neutral specificity 
            "relevance": 0.5,  # Neutral relevance
            "depth": 0.3,  # Low-medium depth
            "temporal_consistency": 0.5,  # Neutral temporal consistency
            "factuality": 0.5,  # Neutral factuality score
        }
        
        # Check if reasoning-related outputs exist
        if 'reasoning_trace' in model_outputs and model_outputs['reasoning_trace']:
            # Extract from reasoning trace if available
            trace = model_outputs['reasoning_trace']
            if isinstance(trace, list) and trace:
                metrics["logical_consistency"] = trace[-1].get('confidence', 0.5) if isinstance(trace[-1], dict) else 0.5
                metrics["uncertainty"] = trace[-1].get('uncertainty', 0.1) if isinstance(trace[-1], dict) else 0.1
                metrics["logical_violations"] = trace[-1].get('violations', 0) if isinstance(trace[-1], dict) else 0
                
                # Extract additional metrics if available
                if isinstance(trace[-1], dict):
                    for key in ["coherence", "specificity", "relevance", "depth", "temporal_consistency", "factuality"]:
                        if key in trace[-1]:
                            metrics[key] = trace[-1][key]
        
        # Extract from reasoning states if available
        elif 'reasoning_states' in model_outputs and model_outputs['reasoning_states'] is not None:
            # Try to extract confidence from reasoning states
            states = model_outputs['reasoning_states']
            if isinstance(states, torch.Tensor):
                # Assuming dimensions contain different metrics
                if states.dim() > 1 and states.size(-1) >= 4:
                    # Use different dimensions for different metrics
                    metrics["logical_consistency"] = torch.sigmoid(states[..., 0]).item()
                    metrics["uncertainty"] = torch.sigmoid(states[..., 1]).item() if states.size(-1) > 1 else 0.1
                    metrics["coherence"] = torch.sigmoid(states[..., 2]).item() if states.size(-1) > 2 else 0.5
                    metrics["specificity"] = torch.sigmoid(states[..., 3]).item() if states.size(-1) > 3 else 0.5
                    
                    if states.size(-1) > 4:
                        metrics["relevance"] = torch.sigmoid(states[..., 4]).item()
                    if states.size(-1) > 5:
                        metrics["depth"] = torch.sigmoid(states[..., 5]).item()
                    if states.size(-1) > 6:
                        metrics["temporal_consistency"] = torch.sigmoid(states[..., 6]).item()
                    if states.size(-1) > 7:
                        metrics["factuality"] = torch.sigmoid(states[..., 7]).item()
            
            elif isinstance(states, dict):
                # If states are provided as a dictionary, extract metrics directly
                for key in metrics:
                    if key in states:
                        metrics[key] = states[key]
        
        # Extract from standard model hidden states with enhanced analysis
        elif 'hidden_states' in model_outputs and model_outputs['hidden_states'] is not None:
            hidden = model_outputs['hidden_states']
            if isinstance(hidden, torch.Tensor):
                # Use norm as a proxy for confidence
                state_norm = torch.norm(hidden[:, -1, :], dim=-1)
                metrics["logical_consistency"] = min(1.0, state_norm.item() / math.sqrt(hidden.size(-1)))
                
                # Estimate uncertainty from variance
                if hidden.size(1) > 1:  # At least 2 tokens
                    variance = torch.var(hidden[:, -2:, :], dim=1).mean().item()
                    metrics["uncertainty"] = min(1.0, variance)
                
                # Estimate coherence from hidden state consistency
                if hidden.size(1) > 3:  # At least 4 tokens for meaningful analysis
                    # Compute cosine similarity between adjacent hidden states
                    cos_sims = []
                    for i in range(1, min(hidden.size(1), 10)):  # Look at up to 10 positions
                        cos_sim = F.cosine_similarity(
                            hidden[:, -i, :], 
                            hidden[:, -(i+1), :], 
                            dim=1
                        ).mean().item()
                        cos_sims.append(cos_sim)
                    
                    # Average cosine similarity as coherence metric
                    if cos_sims:
                        metrics["coherence"] = (sum(cos_sims) / len(cos_sims) + 1) / 2  # Scale to [0,1]
                    
                    # Specificity estimated from hidden state entropy
                    logits = model_outputs.get('logits', None)
                    if logits is not None and isinstance(logits, torch.Tensor):
                        probs = F.softmax(logits[:, -1, :], dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                        max_entropy = math.log(probs.size(-1))
                        # Lower entropy means higher specificity
                        metrics["specificity"] = 1.0 - min(1.0, entropy.item() / max_entropy)
        
        # Extract from RWKV states with enhanced metrics
        elif self.is_rwkv_model and 'states' in model_outputs and model_outputs['states']:
            states = model_outputs['states']
            # Try to estimate from state values
            if isinstance(states, list) and states:
                # Use the norm of the last layer's state as a heuristic
                last_state = states[-1]
                if isinstance(last_state, tuple) and len(last_state) >= 2:
                    k_state, v_state = last_state[:2]
                    if isinstance(k_state, torch.Tensor) and isinstance(v_state, torch.Tensor):
                        k_norm = torch.norm(k_state).item()
                        v_norm = torch.norm(v_state).item()
                        metrics["logical_consistency"] = min(1.0, (k_norm + v_norm) / (2 * math.sqrt(k_state.size(-1))))
                        metrics["uncertainty"] = abs(k_norm - v_norm) / max(k_norm, v_norm, 1e-5)
                        
                        # Estimate coherence from state evolution
                        if len(states) > 1 and isinstance(states[-2], tuple) and len(states[-2]) >= 2:
                            prev_k, prev_v = states[-2][:2]
                            if isinstance(prev_k, torch.Tensor) and prev_k.shape == k_state.shape:
                                k_change = torch.norm(k_state - prev_k) / torch.norm(prev_k).clamp(min=1e-8)
                                v_change = torch.norm(v_state - prev_v) / torch.norm(prev_v).clamp(min=1e-8)
                                
                                # Smooth change indicates coherence
                                avg_change = (k_change + v_change) / 2
                                metrics["coherence"] = 1.0 - min(1.0, avg_change)
                                
                                # Depth estimated from recurrent state evolution magnitude
                                if len(states) > 2:
                                    metrics["depth"] = min(1.0, avg_change * 3)  # Scale for reasonable values
        
        # Analyze sequence for additional metrics if available
        if sequence is not None:
            # Sequence-based metrics could be implemented here
            # This would likely require tokenized text or embedding analysis
            pass
        
        return metrics
    
    def _evaluate_logical_consistency(
        self,
        model_outputs: Dict[str, Any],
        prev_trace: List[float],
        prev_hypo: Optional[BeamHypothesis] = None
    ) -> Tuple[float, float, int, float]:
        """
        Evaluate logical consistency of the reasoning
        
        Args:
            model_outputs: Model outputs containing reasoning information
            prev_trace: Previous reasoning trace scores
            prev_hypo: Previous hypothesis information
            
        Returns:
            Tuple of (consistency_score, uncertainty, logical_violations, premise_satisfaction)
        """
        # Extract basic reasoning info
        logical_consistency, uncertainty, logical_violations = self._extract_reasoning_info(model_outputs)
        
        # Default premise satisfaction
        premise_satisfaction = 1.0
        
        # Enhanced logical analysis based on previous trace
        if prev_trace and len(prev_trace) >= 2:
            # Detect oscillations or inconsistencies in reasoning trace
            recent_trace = prev_trace[-min(5, len(prev_trace)):]
            trace_variance = torch.tensor(recent_trace).var().item()
            
            # Penalize high variance in reasoning confidence (consistency should be stable)
            if trace_variance > 0.1:
                logical_consistency *= (1.0 - min(0.5, trace_variance))
                logical_violations += 1
            
            # Check for declining confidence trend
            if len(recent_trace) >= 3 and sorted(recent_trace, reverse=True) == recent_trace:
                # Confidence is strictly decreasing
                logical_consistency *= 0.8
            
            # If previous hypothesis available, leverage its information
            if prev_hypo is not None:
                # Check for increasing logical violations
                if logical_violations > prev_hypo.logical_violations:
                    # Growing violations indicates logical deterioration
                    logical_consistency *= 0.7
                    
                # Factor in premise satisfaction from previous state
                premise_satisfaction = prev_hypo.premise_satisfaction
                
                # For RWKV models, use state evolution to assess consistency
                if self.is_rwkv_model and prev_hypo.reasoning_states is not None:
                    if 'states' in model_outputs and model_outputs['states']:
                        current_states = model_outputs['states']
                        prev_states = prev_hypo.reasoning_states
                        
                        # Compare state evolution if types match
                        if (isinstance(current_states, list) and isinstance(prev_states, list) and 
                            len(current_states) == len(prev_states)):
                            
                            # Measure state change magnitude
                            state_changes = []
                            for prev_s, curr_s in zip(prev_states, current_states):
                                if isinstance(prev_s, tuple) and isinstance(curr_s, tuple) and len(prev_s) == len(curr_s):
                                    for p, c in zip(prev_s, curr_s):
                                        if isinstance(p, torch.Tensor) and isinstance(c, torch.Tensor) and p.shape == c.shape:
                                            # Calculate normalized state change
                                            change = torch.norm(c - p) / torch.norm(p).clamp(min=1e-8)
                                            state_changes.append(change.item())
                            
                            # Excessive or insufficient state change can indicate issues
                            if state_changes:
                                avg_change = sum(state_changes) / len(state_changes)
                                if avg_change > 0.5:  # Dramatic state change
                                    logical_consistency *= 0.8
                                    uncertainty += 0.1
                                elif avg_change < 0.01:  # Almost no change
                                    premise_satisfaction *= 0.9  # Model might not be using the context
        
        # Apply consistency threshold
        final_consistency = logical_consistency if logical_consistency > self.consistency_threshold else 0.0
        
        return final_consistency, uncertainty, logical_violations, premise_satisfaction
        
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
        use_rwkv_state: bool = True,
        premise_texts: Optional[List[str]] = None,
        per_example_params: Optional[List[Dict[str, Any]]] = None,
        parallel_beam_search: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using logical beam search with enhanced batch processing
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            max_length: Maximum generation length (overrides self.max_length)
            temperature: Sampling temperature
            top_k: Keep only top k tokens with highest probability
            top_p: Keep the top tokens with cumulative probability >= top_p
            num_return_sequences: Number of sequences to return
            use_rwkv_state: Whether to use RWKV recurrent state
            premise_texts: Optional list of explicit premise texts to consider
            per_example_params: Optional list of parameter dictionaries for each example
            parallel_beam_search: Whether to process beams in parallel when possible
            **kwargs: Additional arguments to pass to model
            
        Returns:
            Generated token IDs [batch_size * num_return_sequences, max_length]
        """
        # Store original temperature
        self.temperature = temperature
        
        # Update generation tracking
        self.generation_metadata["total_generations"] += 1
        
        # Initialize generation parameters
        max_length = max_length or self.max_length
        effective_max_length = min(max_length, self.max_length)
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Set sampling parameters
        do_sample = self.do_sample
        cur_top_k = top_k if top_k is not None else self.top_k
        cur_top_p = top_p if top_p is not None else self.top_p
        
        # Initialize premises if provided
        batch_premises = []
        if premise_texts is not None:
            for ptext in premise_texts:
                batch_premises.append(self._initialize_premises_from_context(ptext))
        else:
            # Try to extract premises from input_ids (would need tokenizer and detokenization)
            batch_premises = [[] for _ in range(batch_size)]
        
        # Check for per-example parameters
        if per_example_params is None:
            per_example_params = [{}] * batch_size
        elif len(per_example_params) != batch_size:
            raise ValueError(f"Expected {batch_size} parameter dictionaries, got {len(per_example_params)}")
        
        # Mixed precision context
        if self.use_mixed_precision and torch.cuda.is_available():
            mp_ctx = torch.cuda.amp.autocast(dtype=self.mixed_precision_dtype)
        else:
            mp_ctx = nullcontext()
        
        # Initialize beams for each batch item
        beams = [[] for _ in range(batch_size)]
        
        # Track beam statistics for dynamic adjustment
        beam_stats = {
            "logical_consistency": [],
            "uncertainty": [],
            "coherence": [],
            "premise_satisfaction": []
        }
        
        # Initialize beams
        for batch_idx in range(batch_size):
            # Apply per-example parameters
            example_params = per_example_params[batch_idx]
            example_temperature = example_params.get('temperature', temperature)
            
            # Create initial hypothesis for each batch item
            batch_input = input_ids[batch_idx:batch_idx+1]
            
            # If using RWKV, initialize states
            reasoning_states = None
            if self.is_rwkv_model and use_rwkv_state:
                # Run model on prefix to get initial state
                with mp_ctx:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=batch_input,
                            attention_mask=attention_mask[batch_idx:batch_idx+1] if attention_mask is not None else None,
                            position_ids=position_ids[batch_idx:batch_idx+1] if position_ids is not None else None,
                            use_states=True,  # Enable state tracking
                            return_dict=True,
                            **kwargs
                        )
                # Extract states for RWKV models
                if 'states' in outputs:
                    reasoning_states = outputs['states']
                elif hasattr(self.model, 'states'):
                    reasoning_states = self.model.states
            
            # Create initial beam hypothesis
            beams[batch_idx].append(
                BeamHypothesis(
                    sequence=batch_input,
                    score=0.0,
                    reasoning_trace=[],
                    uncertainty=0.0,
                    logical_violations=0,
                    premise_satisfaction=1.0,
                    reasoning_states=reasoning_states
                )
            )
        
        # Generate tokens
        for step in range(effective_max_length - input_ids.size(1)):
            next_beams = [[] for _ in range(batch_size)]
            step_metrics = {key: [] for key in beam_stats.keys()}
            
            # Prepare parallel beam processing if enabled
            if parallel_beam_search and torch.cuda.is_available():
                # Group beams across batches for parallel processing
                all_beam_sequences = []
                all_beam_indices = []  # Track (batch_idx, beam_idx)
                all_beam_states = []   # Track reasoning states for each beam
                
                for batch_idx, batch_beams in enumerate(beams):
                    for beam_idx, beam in enumerate(batch_beams):
                        if beam.sequence[0, -1].item() != self.eos_token_id:
                            all_beam_sequences.append(beam.sequence)
                            all_beam_indices.append((batch_idx, beam_idx))
                            all_beam_states.append(beam.reasoning_states)
                
                if all_beam_sequences:
                    # Process batched beams in parallel
                    batched_sequences = torch.cat(all_beam_sequences, dim=0)
                    
                    # Create attention mask and position IDs for the batch if needed
                    batched_attention_mask = None
                    if attention_mask is not None:
                        batched_attention_mask = torch.cat([
                            attention_mask[batch_idx:batch_idx+1, :beam.sequence.size(1)]
                            for batch_idx, beam_idx in all_beam_indices
                            for beam in [beams[batch_idx][beam_idx]]
                        ], dim=0)
                    
                    batched_position_ids = None
                    if position_ids is not None:
                        batched_position_ids = torch.cat([
                            position_ids[batch_idx:batch_idx+1, :beam.sequence.size(1)]
                            for batch_idx, beam_idx in all_beam_indices
                            for beam in [beams[batch_idx][beam_idx]]
                        ], dim=0)
                    
                    # Prepare RWKV states if needed
                    if self.is_rwkv_model and use_rwkv_state:
                        # Save original model states
                        original_states = None
                        if hasattr(self.model, 'states'):
                            original_states = self.model.states
                        elif hasattr(self.model, 'rwkv_states'):
                            original_states = self.model.rwkv_states
                    
                    # Forward pass
                    with mp_ctx:
                        with torch.no_grad():
                            # Prepare forward pass inputs
                            forward_kwargs = {
                                'input_ids': batched_sequences,
                                'return_dict': True
                            }
                            
                            if batched_attention_mask is not None:
                                forward_kwargs['attention_mask'] = batched_attention_mask
                            
                            if batched_position_ids is not None:
                                forward_kwargs['position_ids'] = batched_position_ids
                            
                            # Add RWKV specific handling for batched processing
                            if self.is_rwkv_model and use_rwkv_state:
                                forward_kwargs['use_states'] = False  # Don't use global states for parallel processing
                                forward_kwargs['return_states'] = True  # Return states explicitly
                            
                            # Add additional kwargs
                            forward_kwargs.update(kwargs)
                            
                            # Perform forward pass
                            outputs = self.model(**forward_kwargs)
                    
                    # Process each beam result
                    logits_batch = outputs.get('logits', None)
                    states_batch = outputs.get('states', None)
                    
                    # Process each beam individually
                    for beam_idx, (batch_idx, original_beam_idx) in enumerate(all_beam_indices):
                        beam = beams[batch_idx][original_beam_idx]
                        
                        # Extract this beam's logits
                        next_token_logits = logits_batch[beam_idx, -1, :] / temperature
                        
                        # Extract this beam's states if available
                        beam_outputs = {'logits': logits_batch[beam_idx:beam_idx+1]}
                        if states_batch is not None:
                            # Extract appropriate states for this beam
                            if isinstance(states_batch, list):
                                beam_outputs['states'] = [state[beam_idx:beam_idx+1] for state in states_batch]
                            else:
                                beam_outputs['states'] = states_batch[beam_idx:beam_idx+1]
                        
                        # Get beam's premises
                        beam_premises = batch_premises[batch_idx] if batch_idx < len(batch_premises) else []
                        
                        # Extract reasoning metrics for this beam
                        reasoning_metrics = self._extract_reasoning_info(beam_outputs, beam.sequence)
                        
                        # Check premise satisfaction if available
                        premise_satisfaction = beam.premise_satisfaction
                        if beam_premises:
                            # We'd need tokenized text here, approximating with empty string
                            generated_text = ""  # Placeholder - in real implementation, would decode tokens
                            premise_satisfaction = self._evaluate_premise_satisfaction(beam_premises, generated_text)
                        
                        # Apply dynamic parameter adjustment
                        adjusted_params = self._adjust_parameters_dynamically(
                            reasoning_metrics,
                            step,
                            effective_max_length - input_ids.size(1),
                            beam_stats
                        )
                        
                        # Apply logical reasoning evaluation with premise satisfaction
                        violations = reasoning_metrics.get("logical_violations", 0)
                        logical_consistency = reasoning_metrics.get("logical_consistency", 0.5)
                        uncertainty = reasoning_metrics.get("uncertainty", 0.1)
                        
                        # Update beam statistics
                        for key in step_metrics:
                            if key in reasoning_metrics:
                                step_metrics[key].append(reasoning_metrics[key])
                        step_metrics["premise_satisfaction"].append(premise_satisfaction)
                        
                        # Apply logical reward/penalty with adjusted weights
                        logical_reward_weight = adjusted_params["logical_reward_weight"]
                        logical_penalty_weight = adjusted_params["logical_penalty_weight"]
                        uncertainty_penalty = adjusted_params["uncertainty_penalty"]
                        
                        if logical_consistency > self.consistency_threshold:
                            # Boost logits for consistent generations
                            next_token_logits = next_token_logits + logical_reward_weight
                        elif violations > beam.logical_violations:
                            # Penalize inconsistent generations
                            next_token_logits = next_token_logits - logical_penalty_weight
                            self.generation_metadata["logical_violations_avoided"] += 1
                        
                        # Apply uncertainty penalty
                        if uncertainty > 0.3:  # High uncertainty threshold
                            next_token_logits = next_token_logits - (uncertainty * uncertainty_penalty)
                            self.generation_metadata["high_uncertainty_tokens_avoided"] += 1
                        
                        # Apply premise satisfaction scaling
                        next_token_logits = next_token_logits * premise_satisfaction
                        
                        # Apply sampling with adjusted parameters
                        adjusted_top_k = adjusted_params["top_k"]
                        adjusted_top_p = adjusted_params["top_p"]
                        
                        if do_sample:
                            # Apply top-k and top-p filtering
                            filtered_logits = self._top_k_top_p_filtering(
                                next_token_logits, top_k=adjusted_top_k, top_p=adjusted_top_p
                            )
                            
                            # Sample from the filtered distribution
                            probs = F.softmax(filtered_logits, dim=-1)
                            next_tokens = torch.multinomial(probs, num_samples=self.beam_size)
                            token_scores = torch.gather(probs, -1, next_tokens)
                        else:
                            # Get top-k candidates deterministically
                            token_scores, next_tokens = next_token_logits.topk(
                                min(self.beam_size, next_token_logits.size(-1)), dim=-1
                            )
                            token_scores = F.softmax(token_scores, dim=-1)
                        
                        # Get reasoning states for next step
                        next_reasoning_states = None
                        if states_batch is not None:
                            if isinstance(states_batch, list):
                                next_reasoning_states = [state[beam_idx:beam_idx+1] for state in states_batch]
                            else:
                                next_reasoning_states = states_batch[beam_idx:beam_idx+1]
                        
                        # Create candidates for each possible next token
                        for token_idx, (score, token) in enumerate(zip(token_scores[0], next_tokens[0])):
                            # Create new sequence with the next token
                            new_sequence = torch.cat([
                                beam.sequence, token.unsqueeze(0).unsqueeze(0)
                            ], dim=1)
                            
                            # Calculate sequence score with length penalty
                            sequence_score = beam.score + math.log(score.item() + 1e-8) 
                            normalized_score = sequence_score / ((5 + step + 1) / 6) ** self.length_penalty
                            
                            # Create new candidate
                            next_beams[batch_idx].append(
                                BeamHypothesis(
                                    sequence=new_sequence,
                                    score=normalized_score,
                                    reasoning_trace=beam.reasoning_trace + [logical_consistency],
                                    uncertainty=uncertainty,
                                    logical_violations=violations,
                                    premise_satisfaction=premise_satisfaction,
                                    reasoning_states=next_reasoning_states
                                )
                            )
                        
                        # Restore original RWKV states if needed
                        if self.is_rwkv_model and use_rwkv_state and original_states is not None:
                            if hasattr(self.model, 'states'):
                                self.model.states = original_states
                            elif hasattr(self.model, 'rwkv_states'):
                                self.model.rwkv_states = original_states
            
            else:
                # Standard sequential processing
                for batch_idx in range(batch_size):
                    current_beams = beams[batch_idx]
                    candidates = []
                    
                    # Get batch-specific premises
                    batch_premises_list = batch_premises[batch_idx] if batch_idx < len(batch_premises) else []
                    
                    # Apply per-example parameters
                    example_params = per_example_params[batch_idx]
                    example_temperature = example_params.get('temperature', temperature)
                    example_top_k = example_params.get('top_k', cur_top_k)
                    example_top_p = example_params.get('top_p', cur_top_p)
                    
                    # Process each beam in the current batch
                    for beam in current_beams:
                        # Skip completed sequences
                        if beam.sequence[0, -1].item() == self.eos_token_id:
                            candidates.append(beam)
                            continue
                        
                        # Forward pass with proper state handling
                        with mp_ctx:
                            with torch.no_grad():
                                # Prepare forward pass inputs
                                forward_kwargs = {
                                    'input_ids': beam.sequence,
                                    'return_dict': True
                                }
                                
                                # Add attention mask if provided
                                if attention_mask is not None:
                                    forward_kwargs['attention_mask'] = attention_mask[batch_idx:batch_idx+1, :beam.sequence.size(1)]
                                
                                # Add position IDs if provided
                                if position_ids is not None:
                                    forward_kwargs['position_ids'] = position_ids[batch_idx:batch_idx+1, :beam.sequence.size(1)]
                                
                                # Add RWKV specific handling
                                if self.is_rwkv_model and use_rwkv_state:
                                    # Set use_states flag
                                    forward_kwargs['use_states'] = True
                                    
                                    # For hybrid models, add rwkv_states flag
                                    if hasattr(self.model, 'hybrid_model') and self.model.hybrid_model:
                                        forward_kwargs['use_rwkv_states'] = True
                                    
                                    # Restore states if needed
                                    if beam.reasoning_states is not None:
                                        # Save original model states
                                        original_states = None
                                        if hasattr(self.model, 'states'):
                                            original_states = self.model.states
                                            self.model.states = beam.reasoning_states
                                        elif hasattr(self.model, 'rwkv_states'):
                                            original_states = self.model.rwkv_states
                                            self.model.rwkv_states = beam.reasoning_states
                                
                                # Add additional kwargs
                                forward_kwargs.update(kwargs)
                                
                                # Perform forward pass
                                outputs = self.model(**forward_kwargs)
                        
                        # Get next token logits
                        next_token_logits = outputs['logits'][:, -1, :] / example_temperature
                        
                        # Extract reasoning metrics for this beam
                        reasoning_metrics = self._extract_reasoning_info(outputs, beam.sequence)
                        
                        # Check premise satisfaction if available
                        premise_satisfaction = beam.premise_satisfaction
                        if batch_premises_list:
                            # We'd need tokenized text here, approximating with empty string
                            generated_text = ""  # Placeholder - in real implementation, would decode tokens
                            premise_satisfaction = self._evaluate_premise_satisfaction(batch_premises_list, generated_text)
                        
                        # Apply dynamic parameter adjustment
                        adjusted_params = self._adjust_parameters_dynamically(
                            reasoning_metrics,
                            step,
                            effective_max_length - input_ids.size(1),
                            beam_stats
                        )
                        
                        # Apply logical reasoning evaluation with premise satisfaction
                        violations = reasoning_metrics.get("logical_violations", 0)
                        logical_consistency = reasoning_metrics.get("logical_consistency", 0.5)
                        uncertainty = reasoning_metrics.get("uncertainty", 0.1)
                        
                        # Update beam statistics
                        for key in step_metrics:
                            if key in reasoning_metrics:
                                step_metrics[key].append(reasoning_metrics[key])
                        step_metrics["premise_satisfaction"].append(premise_satisfaction)
                        
                        # Apply logical reward/penalty with adjusted weights
                        logical_reward_weight = adjusted_params["logical_reward_weight"]
                        logical_penalty_weight = adjusted_params["logical_penalty_weight"]
                        uncertainty_penalty = adjusted_params["uncertainty_penalty"]
                        
                        if logical_consistency > self.consistency_threshold:
                            # Boost logits for consistent generations
                            next_token_logits = next_token_logits + logical_reward_weight
                        elif violations > beam.logical_violations:
                            # Penalize inconsistent generations
                            next_token_logits = next_token_logits - logical_penalty_weight
                            self.generation_metadata["logical_violations_avoided"] += 1
                        
                        # Apply uncertainty penalty
                        if uncertainty > 0.3:  # High uncertainty threshold
                            next_token_logits = next_token_logits - (uncertainty * uncertainty_penalty)
                            self.generation_metadata["high_uncertainty_tokens_avoided"] += 1
                        
                        # Apply premise satisfaction scaling
                        next_token_logits = next_token_logits * premise_satisfaction
                        
                        # Apply sampling with adjusted parameters
                        adjusted_top_k = adjusted_params["top_k"]
                        adjusted_top_p = adjusted_params["top_p"]
                        
                        if do_sample:
                            # Apply top-k and top-p filtering
                            filtered_logits = self._top_k_top_p_filtering(
                                next_token_logits, top_k=adjusted_top_k, top_p=adjusted_top_p
                            )
                            
                            # Sample from the filtered distribution
                            probs = F.softmax(filtered_logits, dim=-1)
                            next_tokens = torch.multinomial(probs, num_samples=self.beam_size)
                            token_scores = torch.gather(probs, -1, next_tokens)
                        else:
                            # Get top-k candidates deterministically
                            token_scores, next_tokens = next_token_logits.topk(
                                min(self.beam_size, next_token_logits.size(-1)), dim=-1
                            )
                            token_scores = F.softmax(token_scores, dim=-1)
                        
                        # Get reasoning states for next step
                        next_reasoning_states = None
                        if self.is_rwkv_model and use_rwkv_state:
                            if 'states' in outputs:
                                next_reasoning_states = outputs['states']
                            elif hasattr(self.model, 'states'):
                                next_reasoning_states = self.model.states
                            elif hasattr(self.model, 'rwkv_states'):
                                next_reasoning_states = self.model.rwkv_states
                        
                        # Create candidates for each possible next token
                        for token_idx, (score, token) in enumerate(zip(token_scores[0], next_tokens[0])):
                            # Create new sequence with the next token
                            new_sequence = torch.cat([
                                beam.sequence, token.unsqueeze(0).unsqueeze(0)
                            ], dim=1)
                            
                            # Calculate sequence score with length penalty
                            sequence_score = beam.score + math.log(score.item() + 1e-8) 
                            normalized_score = sequence_score / ((5 + step + 1) / 6) ** self.length_penalty
                            
                            # Create new candidate
                            candidates.append(
                                BeamHypothesis(
                                    sequence=new_sequence,
                                    score=normalized_score,
                                    reasoning_trace=beam.reasoning_trace + [logical_consistency],
                                    uncertainty=uncertainty,
                                    logical_violations=violations,
                                    premise_satisfaction=premise_satisfaction,
                                    reasoning_states=next_reasoning_states
                                )
                            )
                        
                        # Select top beams according to score
                        candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
                        next_beams[batch_idx] = candidates[:self.beam_size]
                
            # Update overall beam statistics
            for key in beam_stats:
                if step_metrics[key]:
                    beam_stats[key].append(sum(step_metrics[key]) / len(step_metrics[key]))
            
            # Update beams
            beams = next_beams
            
            # Early stopping if all beams are finished
            if self.early_stopping and all(
                all(beam.sequence[0, -1].item() == self.eos_token_id for beam in batch_beams)
                for batch_beams in beams
            ):
                break
        
        # Select output sequences
        output_sequences = []
        for batch_beams in beams:
            # Sort beams by score and logical consistency
            sorted_beams = sorted(
                batch_beams, 
                key=lambda x: (x.score, sum(x.reasoning_trace)/max(1, len(x.reasoning_trace)), x.premise_satisfaction), 
                reverse=True
            )
            
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
        
        return result

    def _adjust_parameters_dynamically(
        self, 
        metrics: Dict[str, float], 
        step: int, 
        max_steps: int,
        beam_stats: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Dynamically adjust beam search parameters based on generation metrics
        
        Args:
            metrics: Dictionary of current reasoning metrics
            step: Current generation step
            max_steps: Maximum number of generation steps
            beam_stats: Historical statistics about beam search performance
            
        Returns:
            Dictionary of adjusted parameters
        """
        adjusted_params = {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "logical_reward_weight": self.logical_reward_weight,
            "logical_penalty_weight": self.logical_penalty_weight,
            "uncertainty_penalty": self.uncertainty_penalty
        }
        
        # 1. Temperature adjustment based on uncertainty and generation progress
        progress = step / max_steps
        uncertainty = metrics.get("uncertainty", 0.1)
        logical_consistency = metrics.get("logical_consistency", 0.5)
        coherence = metrics.get("coherence", 0.5)
        
        # Lower temperature as we progress to improve coherence
        progression_temp_factor = 1.0 - (progress * 0.3)  # Reduce up to 30% by the end
        
        # Increase temperature when uncertain to explore more, decrease when certain
        uncertainty_temp_factor = 1.0 + ((uncertainty - 0.5) * 0.4)  # ±20% based on uncertainty
        
        # Decrease temperature when logical consistency is high to exploit good paths
        consistency_temp_factor = 1.0 - ((logical_consistency - 0.5) * 0.4)  # Reduce up to 20% for high consistency
        
        # Combine factors with reasonable bounds
        adjusted_params["temperature"] = adjusted_params["temperature"] * progression_temp_factor * uncertainty_temp_factor * consistency_temp_factor
        adjusted_params["temperature"] = max(0.5, min(1.5, adjusted_params["temperature"]))  # Keep in reasonable range
        
        # 2. Top-k and top-p adjustments
        # Tighten sampling as coherence increases
        if coherence > 0.7:
            # More focused sampling for coherent generations
            adjusted_params["top_k"] = max(10, int(adjusted_params["top_k"] * 0.8))
            adjusted_params["top_p"] = max(0.5, adjusted_params["top_p"] * 0.9)
        elif coherence < 0.3:
            # Broader sampling for less coherent generations
            adjusted_params["top_k"] = min(100, int(adjusted_params["top_k"] * 1.2))
            adjusted_params["top_p"] = min(0.95, adjusted_params["top_p"] * 1.1)
        
        # 3. Logical weight adjustments
        # Increase logical rewards for consistent generations, increase penalties for inconsistent ones
        if logical_consistency > 0.7:
            adjusted_params["logical_reward_weight"] = min(2.0, adjusted_params["logical_reward_weight"] * 1.1)
        elif logical_consistency < 0.3:
            adjusted_params["logical_penalty_weight"] = min(2.0, adjusted_params["logical_penalty_weight"] * 1.2)
        
        # 4. Uncertainty penalty adjustment
        # Make uncertainty more impactful later in generation
        adjusted_params["uncertainty_penalty"] = adjusted_params["uncertainty_penalty"] * (1.0 + progress * 0.5)
        
        # 5. Analyze beam statistics for trends
        if beam_stats and all(len(stats) > 3 for stats in beam_stats.values()):
            # Check consistency trend (last 3 steps)
            consistency_trend = beam_stats["logical_consistency"][-3:]
            if all(a < b for a, b in zip(consistency_trend, consistency_trend[1:])):
                # Consistency is improving, reduce exploration
                adjusted_params["temperature"] = max(0.5, adjusted_params["temperature"] * 0.9)
            elif all(a > b for a, b in zip(consistency_trend, consistency_trend[1:])):
                # Consistency is declining, increase exploration
                adjusted_params["temperature"] = min(1.5, adjusted_params["temperature"] * 1.1)
                adjusted_params["logical_penalty_weight"] = min(2.0, adjusted_params["logical_penalty_weight"] * 1.2)
        
        return adjusted_params 

    def _initialize_premises_from_context(self, context: str) -> List[Premise]:
        """
        Extract logical premises from the generation context
        
        Args:
            context: The input context/prompt
            
        Returns:
            List of Premise objects
        """
        premises = []
        
        # In a real implementation, this would use NLP to extract premises
        # This is a simple placeholder implementation based on keywords and patterns
        
        # Look for explicit premises
        premise_indicators = [
            "given that", "assume that", "assuming", "premise:", "premises:",
            "we know that", "it is known that", "suppose that", "let's assume"
        ]
        
        text_lower = context.lower()
        
        # Extract sentences that might contain premises
        sentences = context.split('.')
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains premise indicators
            is_potential_premise = any(indicator in sentence.lower() for indicator in premise_indicators)
            
            if is_potential_premise:
                # Extract keywords (simplified approach)
                words = sentence.split()
                keywords = [word for word in words if len(word) > 3 and word.lower() not in ["that", "than", "this", "with", "from", "have", "what"]]
                
                # Create premise with basic properties
                premise = Premise(
                    id=f"premise_{i}",
                    text=sentence,
                    keywords=keywords[:5],  # Use top 5 keywords
                    importance=1.0,
                    negation_terms=["not", "never", "no", "isn't", "aren't", "wasn't", "weren't"],
                    implications=[],
                    contradictions=[]
                )
                
                premises.append(premise)
        
        # If no explicit premises were found, try to infer from the context
        if not premises and len(sentences) > 0:
            # Use the first sentence as a general premise
            first_sentence = sentences[0].strip()
            if first_sentence:
                words = first_sentence.split()
                keywords = [word for word in words if len(word) > 3 and word.lower() not in ["that", "than", "this", "with", "from", "have", "what"]]
                
                premise = Premise(
                    id="premise_general",
                    text=first_sentence,
                    keywords=keywords[:5],  # Use top 5 keywords
                    importance=0.8,
                    negation_terms=["not", "never", "no"],
                    implications=[],
                    contradictions=[]
                )
                
                premises.append(premise)
        
        return premises

    def _evaluate_premise_satisfaction(self, premises: List[Premise], generated_text: str) -> float:
        """
        Evaluate how well the generated text satisfies all premises
        
        Args:
            premises: List of premises to check
            generated_text: Text generated so far
            
        Returns:
            Overall premise satisfaction score between 0.0 and 1.0
        """
        if not premises:
            return 1.0  # No premises to satisfy
        
        total_importance = sum(premise.importance for premise in premises)
        
        if total_importance == 0:
            return 1.0  # No important premises
        
        weighted_satisfaction = 0.0
        
        for premise in premises:
            satisfaction = premise.check_satisfaction(generated_text)
            weighted_satisfaction += satisfaction * premise.importance
        
        return weighted_satisfaction / total_importance 