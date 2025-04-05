import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, List, Dict, Any, Tuple
from .nanogpt import Block, GPTConfig, LayerNorm
from .reasoning import ChainOfThoughtReasoner
from .math_reasoning import (
    SymbolicMathTransformer,
    FormalVerificationModule,
    HierarchicalSparseGatedMoE,
    DynamicReasoningRouter,
    MathReasoningConfig
)
from .advanced_reasoning import (
    EnhancedReasoningConfig,
    AdvancedReasoningModule,
    NeuralSymbolicIntegration,
    RecursiveReasoningTransformer,
    KnowledgeAugmentedReasoning,
    TreeOfThoughtReasoner,
    VerifiableComputationLayer
)
from utils.optimization import ModelOptimizer, OptimizationConfig

class TreeLSTM(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Input transformations
        self.ioux = nn.Linear(hidden_size, 3 * hidden_size)
        self.iouh = nn.Linear(hidden_size, 3 * hidden_size)
        
        # Forget gate transformations
        self.fx = nn.Linear(hidden_size, hidden_size)
        self.fh = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, tree_structure: Optional[Dict] = None) -> torch.Tensor:
        if tree_structure is None:
            return x
            
        batch_size, seq_length, _ = x.size()
        h = torch.zeros_like(x)
        c = torch.zeros_like(x)
        
        def process_node(node_id: int, node: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
            if isinstance(node, str):  # Leaf node
                return h[:, node_id], c[:, node_id]
                
            # Process children first
            child_h = []
            child_c = []
            for child in node['children']:
                ch, cc = process_node(child['id'], child)
                child_h.append(ch)
                child_c.append(cc)
                
            # Combine children information
            iou = self.ioux(x[:, node_id])
            i, o, u = torch.split(iou, self.hidden_size, dim=-1)
            
            # Calculate forget gates for each child
            f_sum = torch.zeros_like(c[:, node_id])
            for ch, cc in zip(child_h, child_c):
                f = torch.sigmoid(self.fx(x[:, node_id]) + self.fh(ch))
                f_sum = f_sum + f * cc
                
            # Update cell state and hidden state
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            u = torch.tanh(u)
            c_new = i * u + f_sum
            h_new = o * torch.tanh(c_new)
            
            h[:, node_id] = h_new
            c[:, node_id] = c_new
            return h_new, c_new
            
        # Process the entire tree
        for root in tree_structure['roots']:
            process_node(root['id'], root)
            
        return h

class MemoryAugmentedNetwork(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.memory_size = config.n_embd
        self.num_memory_slots = 32
        
        # Memory components
        self.memory = nn.Parameter(torch.randn(1, self.num_memory_slots, self.memory_size))
        self.query_proj = nn.Linear(config.n_embd, self.memory_size)
        self.memory_proj = nn.Linear(self.memory_size, config.n_embd)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Expand memory for batch size
        memory = self.memory.expand(batch_size, -1, -1)
        
        # Create query from input
        query = self.query_proj(x)
        
        # Calculate attention weights
        attention = torch.matmul(query, memory.transpose(-2, -1)) / math.sqrt(self.memory_size)
        attention_weights = F.softmax(attention, dim=-1)
        
        # Read from memory
        memory_output = torch.matmul(attention_weights, memory)
        memory_output = self.memory_proj(memory_output)
        
        # Combine with input through residual connection
        return x + memory_output

class EnhancedMathReasoner(nn.Module):
    """Enhanced mathematical reasoning component for LogicalGPT"""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_size = config.n_embd
        
        # Create math reasoning config
        math_config = MathReasoningConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=config.n_head
        )
        
        # Create enhanced reasoning config (with advanced capabilities)
        self.enhanced_config = EnhancedReasoningConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=config.n_head,
            # Additional advanced parameters
            use_neural_symbolic=True,
            use_knowledge_augmentation=True,
            use_tree_of_thought=True
        )
        
        # Symbolic math transformer
        self.symbolic_transformer = SymbolicMathTransformer(
            hidden_size=hidden_size,
            num_heads=config.n_head,
            intermediate_size=hidden_size * 4
        )
        
        # Formal verification module
        self.verification = FormalVerificationModule(math_config)
        
        # Math domain experts
        self.math_experts = HierarchicalSparseGatedMoE(
            num_experts=math_config.num_math_experts,
            expert_capacity=hidden_size,
            num_expert_layers=math_config.num_expert_layers,
            router_jitter=math_config.router_jitter
        )
        
        # Dynamic reasoning router
        self.reasoning_router = DynamicReasoningRouter(math_config)
        
        # Advanced reasoning components
        self.advanced_reasoning = AdvancedReasoningModule(self.enhanced_config)
        
        # Integration layer
        self.integration_layer = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, reasoning_steps: Optional[List[torch.Tensor]] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        # Apply symbolic transformer
        symbolic_output = self.symbolic_transformer(x)
        
        # Apply math experts
        expert_output = self.math_experts(x)
        
        # Route through specialized reasoning paths
        routed_output, routing_info = self.reasoning_router(x)
        
        # Apply advanced reasoning components
        advanced_output = self.advanced_reasoning(x, attention_mask)
        
        # Combine outputs with weighted integration
        # Give more weight to advanced reasoning components
        combined = symbolic_output * 0.2 + expert_output * 0.2 + routed_output * 0.2 + advanced_output * 0.4
        final_output = self.integration_layer(combined)
        
        # Verify reasoning if steps are provided
        verification_info = {}
        if reasoning_steps is not None:
            is_valid, verification_score, verification_details = self.verification(reasoning_steps)
            verification_info = {
                "is_valid": is_valid,
                "verification_score": verification_score,
                "verification_details": verification_details
            }
            
        return final_output, {
            "routing_info": routing_info,
            "verification_info": verification_info,
            "advanced_reasoning_applied": True
        }

class LogicalGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Add logical reasoning specific components
        self.tree_lstm = TreeLSTM(config.n_embd)
        self.memory_network = MemoryAugmentedNetwork(config)
        
        # Add reasoning components
        self.reasoner = ChainOfThoughtReasoner(config)
        
        # Add enhanced mathematical reasoning with advanced capabilities
        self.math_reasoner = EnhancedMathReasoner(config)
        
        # Add dedicated advanced reasoning capabilities - separate module for even more reasoning power
        self.enhanced_config = EnhancedReasoningConfig(
            hidden_size=config.n_embd,
            intermediate_size=config.n_embd * 4,
            num_attention_heads=config.n_head
        )
        self.advanced_reasoner = AdvancedReasoningModule(self.enhanced_config)
        
        # Output projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Reasoning quality estimation
        self.reasoning_quality_estimator = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Apply optimizations
        opt_config = OptimizationConfig(
            quantization_bits=8,
            use_gradient_checkpointing=True,
            use_parameter_sharing=True,
            use_lora=True,
            use_dynamic_memory=True,
            use_sparse_attention=True,
            pruning_threshold=0.1,
            activation_sparsity=True
        )
        
        self.model = ModelOptimizer(self, opt_config).optimize()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
                input_ids: torch.Tensor, 
                targets: Optional[torch.Tensor] = None,
                logical_tree: Optional[Dict] = None,
                attention_mask: Optional[torch.Tensor] = None,
                use_math_reasoning: bool = True,
                use_advanced_reasoning: bool = True,
                problem_type: Optional[str] = None) -> Dict[str, Any]:
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Apply transformer blocks with residual connections
        block_outputs = []
        for block in self.transformer.h:
            x = block(x)
            block_outputs.append(x)
        x = self.transformer.ln_f(x)

        # Apply logical reasoning components
        if logical_tree is not None:
            x = self.tree_lstm(x, logical_tree)
        x = self.memory_network(x)

        # Track reasoning steps for verification
        all_reasoning_steps = []

        # Apply chain-of-thought reasoning
        reasoned_output, reasoning_steps = self.reasoner(x, attention_mask)
        all_reasoning_steps.extend(reasoning_steps)
        
        # Calculate reasoning quality for later adaptive reasoning
        reasoning_quality = self.reasoning_quality_estimator(reasoned_output).mean()
        
        # Initialize reasoning outputs and info dictionaries
        math_output = torch.zeros_like(reasoned_output)
        advanced_output = torch.zeros_like(reasoned_output)
        math_reasoning_info = {}
        advanced_reasoning_info = {}
        
        # Apply mathematical reasoning if enabled
        if use_math_reasoning:
            math_output, math_reasoning_info = self.math_reasoner(
                reasoned_output, 
                reasoning_steps=all_reasoning_steps,
                attention_mask=attention_mask
            )
        
        # Apply advanced reasoning if enabled and needed based on reasoning quality
        # Only apply advanced reasoning if reasoning quality is below threshold
        if use_advanced_reasoning and reasoning_quality < 0.8:
            advanced_output = self.advanced_reasoner(reasoned_output, attention_mask)
            advanced_reasoning_info = {"applied": True, "reasoning_quality": reasoning_quality.item()}
        
        # Dynamic weighted combination based on reasoning quality and problem type
        # Give more weight to advanced reasoning for complex problems
        if problem_type in ["algebra", "calculus", "geometry"]:
            # Math-heavy problems get more weight on math reasoning
            math_weight = 0.5
            advanced_weight = 0.3
            base_weight = 0.2
        else:
            # General problems balance the weights
            math_weight = 0.3
            advanced_weight = 0.3
            base_weight = 0.4
            
        # Dynamic adjustment based on reasoning quality
        if reasoning_quality < 0.5:
            # Poor reasoning quality, lean more on advanced reasoning
            advanced_weight += 0.2
            base_weight -= 0.1
            math_weight -= 0.1
            
        # Combine all reasoning outputs with weights
        if use_math_reasoning and use_advanced_reasoning:
            reasoned_output = (
                reasoned_output * base_weight + 
                math_output * math_weight +
                advanced_output * advanced_weight
            )
        elif use_math_reasoning:
            reasoned_output = reasoned_output * 0.3 + math_output * 0.7
        elif use_advanced_reasoning:
            reasoned_output = reasoned_output * 0.3 + advanced_output * 0.7
        
        # Final output with uncertainty estimation
        if targets is not None:
            # Project back to vocabulary for training
            logits = self.lm_head(reasoned_output)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
        else:
            # For inference, only compute predictions for the last token
            logits = self.lm_head(reasoned_output[:, [-1], :])
            loss = None
            
        # Determine if the output is mathematically valid using verification
        is_mathematically_valid = math_reasoning_info.get("verification_info", {}).get("is_valid", None)
            
        return {
            'logits': logits,
            'loss': loss,
            'reasoning_steps': reasoning_steps,
            'uncertainty': reasoning_steps[-1].uncertainty if reasoning_steps else None,
            'math_reasoning_info': math_reasoning_info,
            'advanced_reasoning_info': advanced_reasoning_info,
            'is_mathematically_valid': is_mathematically_valid,
            'reasoning_quality': reasoning_quality.item()
        }

    def generate(self, 
                input_ids: torch.Tensor,
                max_new_tokens: int = 100,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                use_math_reasoning: bool = True,
                use_advanced_reasoning: bool = True,
                problem_type: Optional[str] = None) -> torch.Tensor:
        """
        Generate text using the model
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            use_math_reasoning: Whether to use math reasoning components
            use_advanced_reasoning: Whether to use advanced reasoning components
            problem_type: Type of math problem for specialized handling
            
        Returns:
            Generated token IDs
        """
        for _ in range(max_new_tokens):
            # Crop input_ids to block_size if needed
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(
                    idx_cond, 
                    use_math_reasoning=use_math_reasoning,
                    use_advanced_reasoning=use_advanced_reasoning,
                    problem_type=problem_type
                )
                logits = outputs['logits']
                
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
            
        return input_ids 