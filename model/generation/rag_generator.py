import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Tuple
from collections import OrderedDict
from ..rag import EnhancedRAG, EnhancedRAGConfig
from ..neural_symbolic import NeuralSymbolicConfig
from .speculative import SpeculativeGenerator, SpeculativeConfig

class EnhancedRAGGenerator(nn.Module):
    """Enhanced knowledge-aware generation with neural-symbolic RAG integration"""
    
    def __init__(
        self,
        base_model: nn.Module,
        rag_config: EnhancedRAGConfig,
        knowledge_bank: Optional[torch.Tensor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_speculative: bool = True,
        sparse_training: bool = True
    ):
        super().__init__()
        self.base_model = base_model
        self.rag = EnhancedRAG(rag_config)
        self.device = device
        self.sparse_training = sparse_training
        self.sparsity_threshold = 0.01
        
        # Initialize speculative generator if enabled
        self.speculative_generator = None
        if use_speculative:
            spec_config = SpeculativeConfig(
                num_draft_models=3,
                acceptance_threshold=0.7,
                temperature=0.7
            )
            self.speculative_generator = SpeculativeGenerator(
                target_model=self,
                num_draft_models=spec_config.num_draft_models,
                acceptance_threshold=spec_config.acceptance_threshold,
                temperature=spec_config.temperature
            )
        
        # Register knowledge bank buffer
        if knowledge_bank is not None:
            self.register_buffer('knowledge_bank', knowledge_bank.to(device))
            if self.sparse_training:
                self._init_sparse_mask(knowledge_bank.shape)
            
        # Track reasoned knowledge cache with LRU eviction
        self.knowledge_cache = OrderedDict()
        self.max_cache_size = 1000
        
    def update_knowledge_bank(self, knowledge_bank: torch.Tensor):
        """Update the knowledge bank"""
        self.register_buffer('knowledge_bank', knowledge_bank.to(self.device))
        # Clear cache when knowledge bank changes
        self.knowledge_cache.clear()
        
    def _init_sparse_mask(self, shape: Tuple[int, ...]):
        """Initialize sparse mask for dynamic sparse training"""
        self.register_buffer('sparse_mask', torch.ones(shape, device=self.device))
        self.sparse_update_freq = 100
        self.sparse_update_counter = 0
        
    def _update_sparse_mask(self):
        """Update sparse mask based on knowledge bank usage"""
        if not hasattr(self, 'sparse_mask'):
            return
            
        # Calculate usage statistics
        usage = torch.mean(torch.abs(self.knowledge_bank), dim=1)
        threshold = torch.quantile(usage, self.sparsity_threshold)
        
        # Update mask
        self.sparse_mask = (usage > threshold).float()
        self.sparse_update_counter = 0
        
    def _maintain_cache(self):
        """Maintain reasoned knowledge cache size with LRU eviction"""
        while len(self.knowledge_cache) > self.max_cache_size:
            self.knowledge_cache.popitem(last=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        return_dict: bool = True,
        use_checkpointing: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced RAG integration"""
        # Get base model outputs with optional gradient checkpointing
        def _base_forward(*args, **kwargs):
            return self.base_model(*args, **kwargs)
            
        if use_checkpointing:
            base_outputs = torch.utils.checkpoint.checkpoint(
                _base_forward,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                return_dict=True,
                **kwargs
            )
        else:
            base_outputs = _base_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                return_dict=True,
                **kwargs
            )
        
        outputs = {}
        
        # Apply RAG if knowledge bank exists
        if hasattr(self, 'knowledge_bank') and self.knowledge_bank is not None:
            # Apply sparse mask if enabled
            if self.sparse_training:
                if not hasattr(self, 'sparse_mask'):
                    self._init_sparse_mask(self.knowledge_bank.shape)
                knowledge_bank = self.knowledge_bank * self.sparse_mask
                
                # Update sparse mask periodically
                self.sparse_update_counter += 1
                if self.sparse_update_counter >= self.sparse_update_freq:
                    self._update_sparse_mask()
            else:
                knowledge_bank = self.knowledge_bank
                
            # Get verified and uncertainty-aware knowledge
            rag_outputs = self.rag(
                hidden_states=base_outputs.hidden_states,
                knowledge_bank=knowledge_bank,
                attention_mask=attention_mask
            )
            
            # Get enhanced hidden states with knowledge integration
            enhanced_hidden_states = rag_outputs["hidden_states"]
            
            # Cache verified knowledge if cache is enabled
            if use_cache and "verification_scores" in rag_outputs:
                batch_size = input_ids.size(0)
                for b in range(batch_size):
                    key = input_ids[b].cpu().tolist()
                    if rag_outputs["verification_scores"][b].mean() > 0.8:  # High confidence threshold
                        self.knowledge_cache[str(key)] = {
                            "hidden_states": enhanced_hidden_states[b].detach(),
                            "verification": rag_outputs["verification_scores"][b].detach()
                        }
                self._maintain_cache()
            
            # Project to vocabulary space
            if hasattr(self.base_model, 'lm_head'):
                logits = self.base_model.lm_head(enhanced_hidden_states)
            else:
                logits = base_outputs.logits
            
            # Compute loss if labels provided
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
            
            # Include RAG-specific outputs
            outputs.update({
                "loss": loss,
                "logits": logits,
                "hidden_states": enhanced_hidden_states,
                "retrieval_scores": rag_outputs.get("retrieval_scores"),
                "knowledge_weights": rag_outputs.get("knowledge_weights"),
                "uncertainties": rag_outputs.get("uncertainties"),
                "verification_scores": rag_outputs.get("verification_scores")
            })
        else:
            outputs = {
                "loss": base_outputs.loss,
                "logits": base_outputs.logits,
                "hidden_states": base_outputs.hidden_states
            }
        
        if not return_dict:
            return tuple(outputs.values())
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        min_length: int = 0,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        use_cache: bool = True,
        use_speculative: Optional[bool] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate text with enhanced RAG integration"""
        # Determine if using speculative decoding
        use_speculative = use_speculative if use_speculative is not None else (
            self.speculative_generator is not None and num_beams == 1
        )
        
        if use_speculative:
            return self.speculative_generator.generate(
                input_ids=input_ids,
                max_length=max_length,
                attention_mask=attention_mask,
                temperature=temperature,
                **kwargs
            )
            
        # Set up generation config
        gen_kwargs = {
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "use_cache": use_cache,
            **kwargs
        }
        
        # Track knowledge utilization metrics
        knowledge_scores = []
        verification_scores = []
        
        def _knowledge_scoring_fn(batch_idx: int, beam_idx: int, token_ids: torch.Tensor) -> float:
            """Score function for knowledge-aware beam search"""
            if not hasattr(self, 'knowledge_bank') or self.knowledge_bank is None:
                return 0.0
            
            # Check knowledge cache first
            cache_key = str(token_ids.cpu().tolist())
            if cache_key in self.knowledge_cache:
                cached = self.knowledge_cache[cache_key]
                knowledge_scores.append(1.0)  # Known verified knowledge
                verification_scores.append(cached["verification"].mean().item())
                return cached["verification"].mean().item()
            
            # Get hidden states for current sequence
            with torch.no_grad():
                outputs = self.base_model(token_ids.unsqueeze(0), return_dict=True)
                hidden_states = outputs.hidden_states
                
                # Get knowledge relevance and verification scores
                rag_outputs = self.rag(
                    hidden_states=hidden_states,
                    knowledge_bank=self.knowledge_bank
                )
                
                # Use verification score if available, otherwise use retrieval score
                if "verification_scores" in rag_outputs:
                    score = rag_outputs["verification_scores"].mean().item()
                    verification_scores.append(score)
                else:
                    score = rag_outputs["retrieval_scores"].max().item()
                
                knowledge_scores.append(score)
                return score
        
        # Add knowledge scoring to beam search
        if num_beams > 1:
            gen_kwargs["scoring_fn"] = _knowledge_scoring_fn
        
        # Generate with knowledge integration
        outputs = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )
        
        # Log knowledge utilization metrics
        if knowledge_scores:
            mean_knowledge_score = sum(knowledge_scores) / len(knowledge_scores)
            mean_verification_score = sum(verification_scores) / len(verification_scores) if verification_scores else 0
            
            # Could log these metrics for monitoring
            if hasattr(self, 'training_stats'):
                self.training_stats.update({
                    'mean_knowledge_score': mean_knowledge_score,
                    'mean_verification_score': mean_verification_score
                })
        
        return outputs
