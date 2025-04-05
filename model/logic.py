import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
import torch.nn.functional as F

@dataclass
class LogicOperator:
    name: str
    priority: int
    arity: int

class NeuralLogicMachine(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        
        # Neural predicate embeddings
        self.predicate_embeddings = nn.Parameter(
            torch.randn(config.num_predicates, hidden_size)
        )
        
        # Core logical operators
        self.operators = {
            'AND': LogicOperator('AND', 2, 2),
            'OR': LogicOperator('OR', 2, 2),
            'NOT': LogicOperator('NOT', 3, 1),
            'IMPLIES': LogicOperator('IMPLIES', 1, 2)
        }
        
        # Neural implementations of logical operators
        self.logic_modules = nn.ModuleDict({
            'AND': nn.Bilinear(hidden_size, hidden_size, hidden_size),
            'OR': nn.Bilinear(hidden_size, hidden_size, hidden_size),
            'NOT': nn.Linear(hidden_size, hidden_size),
            'IMPLIES': nn.Bilinear(hidden_size, hidden_size, hidden_size)
        })
        
        # Predicate composition
        self.compose_predicates = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x: torch.Tensor, operator: str, 
                y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if operator not in self.operators:
            raise ValueError(f"Unknown operator: {operator}")
            
        if self.operators[operator].arity == 2 and y is None:
            raise ValueError(f"Operator {operator} requires two operands")
            
        if operator == 'NOT':
            return torch.sigmoid(self.logic_modules[operator](x))
            
        return torch.sigmoid(self.logic_modules[operator](x, y))
        
    def evaluate_expression(self, expression: List[Dict]) -> torch.Tensor:
        """Evaluates a logical expression in postfix notation"""
        stack = []
        
        for token in expression:
            if token['type'] == 'PREDICATE':
                predicate_idx = token['value']
                stack.append(self.predicate_embeddings[predicate_idx])
            else:
                operator = token['value']
                if self.operators[operator].arity == 1:
                    x = stack.pop()
                    result = self(x, operator)
                else:
                    y = stack.pop()
                    x = stack.pop()
                    result = self(x, operator, y)
                stack.append(result)
                
        return stack[0]


class LogicModule(nn.Module):
    """
    Neural module for logical reasoning and inference.
    Implements differentiable logical operations and inference mechanisms.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        
        # Neural logic machine for core logical operations
        self.neural_logic_machine = NeuralLogicMachine(config)
        
        # Logical reasoning layers
        self.reasoning_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.num_attention_heads if hasattr(config, 'num_attention_heads') else 8,
                dim_feedforward=hidden_size * 4,
                dropout=config.dropout if hasattr(config, 'dropout') else 0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(config.num_logic_layers if hasattr(config, 'num_logic_layers') else 3)
        ])
        
        # Logical connective embeddings
        self.connective_embeddings = nn.Embedding(
            num_embeddings=5,  # AND, OR, NOT, IMPLIES, EQUIVALENT
            embedding_dim=hidden_size
        )
        
        # Logical quantifier embeddings (∀, ∃)
        self.quantifier_embeddings = nn.Embedding(
            num_embeddings=2,  # Universal, Existential
            embedding_dim=hidden_size
        )
        
        # Logical variable binding
        self.variable_binding = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Logical inference head
        self.inference_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Logical consistency checker
        self.consistency_checker = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Logical rule application
        self.rule_application = nn.Bilinear(hidden_size, hidden_size, hidden_size)
        
        # Logical memory for tracking context
        self.logical_memory = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.1,
            batch_first=True
        )
        
        # Initialize logical memory state
        self.memory_state = None
        
    def forward(self, hidden_states, attention_mask=None, logical_context=None):
        """
        Apply logical reasoning to input hidden states
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            logical_context: Optional logical context from previous reasoning
            
        Returns:
            processed_states: Processed hidden states with logical reasoning applied
            logical_outputs: Dictionary containing logical reasoning outputs
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Process through reasoning layers
        logical_states = hidden_states
        layer_outputs = []
        
        for layer in self.reasoning_layers:
            logical_states = layer(logical_states, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
            layer_outputs.append(logical_states)
        
        # Update logical memory
        if logical_context is not None:
            # Use provided logical context
            memory_input = torch.cat([hidden_states, logical_context], dim=1)
        else:
            memory_input = logical_states
            
        if self.memory_state is None:
            # Initialize memory state if not present
            h0 = torch.zeros(2, batch_size, hidden_size, device=device)
            c0 = torch.zeros(2, batch_size, hidden_size, device=device)
            self.memory_state = (h0, c0)
            
        # Update memory with current logical states
        memory_output, self.memory_state = self.logical_memory(memory_input, self.memory_state)
        
        # Apply logical inference
        inference_scores = self.inference_head(logical_states)
        
        # Check logical consistency
        if logical_context is not None:
            # Compare current logical states with context
            consistency_input = torch.cat([logical_states, logical_context], dim=-1)
            consistency_scores = self.consistency_checker(consistency_input)
        else:
            # Check internal consistency
            avg_state = logical_states.mean(dim=1, keepdim=True).expand_as(logical_states)
            consistency_input = torch.cat([logical_states, avg_state], dim=-1)
            consistency_scores = self.consistency_checker(consistency_input)
        
        # Prepare output
        logical_outputs = {
            'inference_scores': inference_scores,
            'consistency_scores': consistency_scores,
            'memory_output': memory_output,
            'layer_outputs': layer_outputs
        }
        
        return logical_states, logical_outputs
    
    def apply_rule(self, premise, rule):
        """Apply a logical rule to a premise to derive a conclusion"""
        return self.rule_application(premise, rule)
    
    def check_entailment(self, premise, conclusion):
        """Check if premise entails conclusion"""
        # Create IMPLIES relationship
        implies_idx = 3  # Index for IMPLIES in connective_embeddings
        implies_embedding = self.connective_embeddings(torch.tensor([implies_idx], device=premise.device))
        
        # Combine premise and conclusion with IMPLIES
        entailment = self.neural_logic_machine(
            premise, 
            'IMPLIES', 
            conclusion
        )
        
        return entailment
    
    def resolve_contradiction(self, statements):
        """Identify and resolve contradictions in a set of statements"""
        batch_size, num_statements, hidden_size = statements.shape
        
        # Check pairwise consistency
        consistency_matrix = torch.zeros(batch_size, num_statements, num_statements, device=statements.device)
        
        for i in range(num_statements):
            for j in range(i+1, num_statements):
                stmt_i = statements[:, i]
                stmt_j = statements[:, j]
                
                # Check consistency between statements i and j
                consistency_input = torch.cat([stmt_i, stmt_j], dim=-1)
                consistency_score = self.consistency_checker(consistency_input)
                
                consistency_matrix[:, i, j] = consistency_score.squeeze(-1)
                consistency_matrix[:, j, i] = consistency_score.squeeze(-1)
        
        # Identify most consistent subset of statements
        statement_scores = consistency_matrix.sum(dim=-1)  # Sum consistency scores for each statement
        
        # Sort statements by consistency score
        _, sorted_indices = torch.sort(statement_scores, dim=-1, descending=True)
        
        # Return most consistent statements (top half)
        top_k = max(1, num_statements // 2)
        consistent_indices = sorted_indices[:, :top_k]
        
        # Gather consistent statements
        batch_indices = torch.arange(batch_size).unsqueeze(-1).expand(-1, top_k)
        consistent_statements = statements[batch_indices, consistent_indices]
        
        return consistent_statements, consistency_matrix


class FOLProcessor(nn.Module):
    """
    First-Order Logic Processor for handling complex logical expressions
    with quantifiers, variables, and predicates.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        
        # Core logic module
        self.logic_module = LogicModule(config)
        
        # Variable binding mechanism
        self.variable_binder = nn.Bilinear(hidden_size, hidden_size, hidden_size)
        
        # Quantifier handling
        self.universal_quantifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.existential_quantifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Predicate application
        self.predicate_applier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Unification mechanism for variable matching
        self.unification = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Substitution mechanism
        self.substitution = nn.Bilinear(hidden_size, hidden_size, hidden_size)
        
        # FOL expression encoder
        self.expression_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.num_attention_heads if hasattr(config, 'num_attention_heads') else 8,
                dim_feedforward=hidden_size * 4,
                dropout=config.dropout if hasattr(config, 'dropout') else 0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=config.num_fol_layers if hasattr(config, 'num_fol_layers') else 2
        )
        
        # FOL expression decoder
        self.expression_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=config.num_attention_heads if hasattr(config, 'num_attention_heads') else 8,
                dim_feedforward=hidden_size * 4,
                dropout=config.dropout if hasattr(config, 'dropout') else 0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=config.num_fol_layers if hasattr(config, 'num_fol_layers') else 2
        )
        
        # Logical form classifier
        self.form_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.num_logical_forms if hasattr(config, 'num_logical_forms') else 8)
        )
        
    def forward(self, hidden_states, attention_mask=None, variables=None, predicates=None):
        """
        Process first-order logic expressions
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            variables: Optional tensor of variable representations
            predicates: Optional tensor of predicate representations
            
        Returns:
            processed_states: Processed hidden states with FOL reasoning applied
            fol_outputs: Dictionary containing FOL processing outputs
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Encode FOL expression
        encoded_expr = self.expression_encoder(
            hidden_states, 
            src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Process with core logic module
        logical_states, logical_outputs = self.logic_module(encoded_expr, attention_mask)
        
        # Handle variables if provided
        if variables is not None:
            # Bind variables to their values
            var_bindings = []
            for var in variables:
                # Compute binding scores between variable and all tokens
                binding_scores = torch.matmul(var.unsqueeze(1), logical_states.transpose(-2, -1))
                binding_scores = binding_scores / math.sqrt(hidden_size)
                binding_scores = F.softmax(binding_scores, dim=-1)
                
                # Bind variable
                bound_var = torch.matmul(binding_scores, logical_states)
                var_bindings.append(bound_var.squeeze(1))
                
            # Stack variable bindings
            var_bindings = torch.stack(var_bindings, dim=1)
        else:
            # Create dummy variable bindings
            var_bindings = torch.zeros(batch_size, 1, hidden_size, device=device)
        
        # Handle predicates if provided
        if predicates is not None:
            # Apply predicates to variables
            pred_applications = []
            for pred in predicates:
                for var_binding in var_bindings:
                    # Apply predicate to variable
                    pred_app = self.predicate_applier(
                        torch.cat([pred.unsqueeze(1).expand(-1, var_bindings.size(1), -1), 
                                  var_bindings], dim=-1)
                    )
                    pred_applications.append(pred_app)
                    
            # Stack predicate applications
            if pred_applications:
                pred_applications = torch.cat(pred_applications, dim=1)
            else:
                pred_applications = torch.zeros(batch_size, 1, hidden_size, device=device)
        else:
            # Create dummy predicate applications
            pred_applications = torch.zeros(batch_size, 1, hidden_size, device=device)
        
        # Apply universal quantifier (∀)
        universal_expr = self.universal_quantifier(logical_states.mean(dim=1, keepdim=True))
        
        # Apply existential quantifier (∃)
        existential_expr = self.existential_quantifier(logical_states.max(dim=1, keepdim=True)[0])
        
        # Decode FOL expression
        memory = torch.cat([
            logical_states,
            var_bindings.expand(-1, seq_len, -1) if var_bindings.size(1) == 1 else var_bindings,
            pred_applications.expand(-1, seq_len, -1) if pred_applications.size(1) == 1 else pred_applications,
            universal_expr.expand(-1, seq_len, -1),
            existential_expr.expand(-1, seq_len, -1)
        ], dim=1)
        
        decoded_expr = self.expression_decoder(
            logical_states,
            memory,
            tgt_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Classify logical form
        logical_form_logits = self.form_classifier(decoded_expr.mean(dim=1))
        logical_form_probs = F.softmax(logical_form_logits, dim=-1)
        
        # Prepare output
        fol_outputs = {
            'logical_form_probs': logical_form_probs,
            'universal_expr': universal_expr,
            'existential_expr': existential_expr,
            'var_bindings': var_bindings,
            'pred_applications': pred_applications,
            'logical_outputs': logical_outputs
        }
        
        return decoded_expr, fol_outputs
    
    def unify_terms(self, term1, term2):
        """Unify two logical terms"""
        # Compute unification score
        unification_input = torch.cat([term1, term2], dim=-1)
        unification_score = self.unification(unification_input)
        
        # Compute unified term
        if unification_score.mean() > 0.5:
            unified_term = self.substitution(term1, term2)
            return unified_term, unification_score
        else:
            return None, unification_score
    
    def apply_resolution(self, clause1, clause2):
        """Apply resolution rule to two clauses"""
        batch_size, clause1_len, hidden_size = clause1.shape
        _, clause2_len, _ = clause2.shape
        
        # Compute pairwise unification scores
        unification_scores = torch.zeros(batch_size, clause1_len, clause2_len, device=clause1.device)
        
        for i in range(clause1_len):
            for j in range(clause2_len):
                term1 = clause1[:, i]
                term2 = clause2[:, j]
                
                # Check if terms can be unified (one is negation of other)
                _, unification_score = self.unify_terms(term1, term2)
                unification_scores[:, i, j] = unification_score.squeeze(-1)
        
        # Find best unification
        best_i, best_j = torch.where(unification_scores == unification_scores.max())
        
        if len(best_i) > 0:
            # Create resolvent by removing unified terms and combining clauses
            resolvent_terms = []
            
            for i in range(clause1_len):
                if i != best_i[0]:
                    resolvent_terms.append(clause1[:, i])
                    
            for j in range(clause2_len):
                if j != best_j[0]:
                    resolvent_terms.append(clause2[:, j])
            
            if resolvent_terms:
                resolvent = torch.stack(resolvent_terms, dim=1)
            else:
                # Empty resolvent (contradiction found)
                resolvent = torch.zeros(batch_size, 1, hidden_size, device=clause1.device)
                
            return resolvent, True
        else:
            # No resolution possible
            return None, False
