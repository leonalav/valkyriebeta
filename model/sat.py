import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import math

class NeuralSATSolver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        
        # Variable and clause embeddings
        self.var_embedding = nn.Embedding(config.num_predicates, hidden_size)
        self.clause_embedding = nn.Linear(hidden_size, hidden_size)
        
        # Message passing networks
        self.var_to_clause = nn.Linear(hidden_size, hidden_size)
        self.clause_to_var = nn.Linear(hidden_size, hidden_size)
        
        # Solution predictor
        self.solution_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, clause_indices: torch.Tensor, 
                variable_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_clauses, clause_size = clause_indices.shape
        
        # Initialize variable states if not provided
        if variable_states is None:
            variable_states = self.var_embedding.weight.unsqueeze(0).expand(
                batch_size, -1, -1)
        
        # Message passing iterations
        for _ in range(self.config.sat_iterations):
            # Variables to clauses
            clause_states = []
            for i in range(num_clauses):
                clause_vars = clause_indices[:, i]
                clause_var_states = self.var_to_clause(
                    variable_states[torch.arange(batch_size).unsqueeze(1), clause_vars]
                )
                clause_states.append(self.clause_embedding(clause_var_states.mean(1)))
            clause_states = torch.stack(clause_states, dim=1)
            
            # Clauses to variables
            for var_idx in range(self.config.num_predicates):
                relevant_clauses = (clause_indices == var_idx).any(-1)
                if relevant_clauses.any():
                    messages = self.clause_to_var(
                        clause_states[relevant_clauses]
                    ).mean(0)
                    variable_states[:, var_idx] += messages
        
        # Predict satisfiability
        solution_probs = self.solution_head(variable_states)
        return solution_probs.squeeze(-1)


class SATSolver(nn.Module):
    """
    Neural SAT solver that learns to solve boolean satisfiability problems.
    Implements a differentiable version of DPLL and message passing for SAT solving.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        
        # Core neural SAT solver
        self.neural_sat_solver = NeuralSATSolver(config)
        
        # Variable assignment embeddings
        self.assignment_embeddings = nn.Embedding(3, hidden_size)  # True, False, Unassigned
        
        # Literal embeddings (variables and their negations)
        max_literals = getattr(config, 'max_literals', 1024)
        self.literal_embeddings = nn.Embedding(max_literals * 2, hidden_size)  # x and ¬x
        
        # Clause state encoder
        self.clause_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Variable selection network
        self.variable_selector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Assignment predictor
        self.assignment_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),  # True or False
            nn.Softmax(dim=-1)
        )
        
        # Conflict analyzer
        self.conflict_analyzer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Learned clause generator
        self.clause_generator = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=config.num_attention_heads if hasattr(config, 'num_attention_heads') else 8,
            dim_feedforward=hidden_size * 4,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.1,
            activation='gelu',
            batch_first=True
        )
        
        # Satisfiability predictor
        self.sat_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Attention for clause-variable interactions
        self.clause_var_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=config.num_attention_heads if hasattr(config, 'num_attention_heads') else 8,
            dropout=config.dropout if hasattr(config, 'dropout') else 0.1,
            batch_first=True
        )
        
        # Initialize parameters
        self.max_iterations = getattr(config, 'max_sat_iterations', 50)
        self.restart_probability = getattr(config, 'restart_probability', 0.1)
        
    def forward(self, formula, literal_indices=None, initial_assignments=None):
        """
        Solve a boolean satisfiability problem
        
        Args:
            formula: Tensor representation of the formula in CNF
                    [batch_size, num_clauses, max_literals_per_clause]
            literal_indices: Indices of literals in the formula
            initial_assignments: Optional initial variable assignments
            
        Returns:
            is_sat: Probability that the formula is satisfiable
            assignments: Variable assignments that satisfy the formula (if satisfiable)
            trace: Dictionary containing the solving trace
        """
        batch_size = formula.size(0)
        device = formula.device
        
        # Get formula dimensions
        if literal_indices is not None:
            num_clauses, max_literals_per_clause = literal_indices.shape[1:3]
            num_variables = literal_indices.max().item() // 2 + 1
        else:
            num_clauses, max_literals_per_clause = formula.shape[1:3]
            num_variables = formula.max().item() // 2 + 1
        
        # Initialize variable assignments
        if initial_assignments is None:
            # Start with all variables unassigned (2)
            assignments = torch.full((batch_size, num_variables), 2, device=device)
        else:
            assignments = initial_assignments
            
        # Initialize trace
        trace = {
            'variable_selections': [],
            'assignment_predictions': [],
            'conflicts': [],
            'learned_clauses': [],
            'satisfiability_scores': []
        }
        
        # Initialize state
        var_states = torch.zeros(batch_size, num_variables, self.config.hidden_size, device=device)
        clause_states = torch.zeros(batch_size, num_clauses, self.config.hidden_size, device=device)
        
        # Initialize literal embeddings
        if literal_indices is not None:
            # Use provided literal indices
            literal_embeds = self.literal_embeddings(literal_indices)
        else:
            # Create literal embeddings from formula
            literal_embeds = self.literal_embeddings(formula)
        
        # Main solving loop
        for iteration in range(self.max_iterations):
            # Update clause states based on current assignments
            for c in range(num_clauses):
                # Get literals in this clause
                if literal_indices is not None:
                    clause_literals = literal_indices[:, c]
                else:
                    clause_literals = formula[:, c]
                
                # Get corresponding variable indices (literal // 2)
                var_indices = clause_literals // 2
                
                # Get literal polarities (literal % 2 == 0 for positive, 1 for negative)
                polarities = clause_literals % 2
                
                # Get current assignments for variables in this clause
                var_assignments = assignments.gather(1, var_indices)
                
                # Compute clause satisfaction status
                # A clause is satisfied if any literal matches its polarity:
                # - If polarity is 0 (positive) and assignment is 0 (True) -> satisfied
                # - If polarity is 1 (negative) and assignment is 1 (False) -> satisfied
                clause_satisfied = (var_assignments == polarities) & (var_assignments != 2)
                clause_satisfied = clause_satisfied.any(dim=1, keepdim=True)
                
                # Update clause state
                clause_literals_embed = literal_embeds[:, c].mean(dim=1)
                clause_state = self.clause_encoder(clause_literals_embed)
                
                # Incorporate satisfaction status
                satisfaction_factor = clause_satisfied.float().unsqueeze(-1)
                clause_states[:, c] = clause_state * (1 - satisfaction_factor) + clause_state * 0.1 * satisfaction_factor
            
            # Update variable states using attention
            var_states_flat = var_states.reshape(batch_size, -1, self.config.hidden_size)
            clause_states_flat = clause_states.reshape(batch_size, -1, self.config.hidden_size)
            
            var_states_updated, _ = self.clause_var_attention(
                var_states_flat, clause_states_flat, clause_states_flat
            )
            var_states = var_states_updated.reshape(batch_size, num_variables, self.config.hidden_size)
            
            # Select variable to assign
            var_scores = self.variable_selector(var_states).squeeze(-1)
            
            # Mask already assigned variables
            var_mask = (assignments != 2).float() * -1e9
            var_scores = var_scores + var_mask
            
            # Select variable with highest score
            selected_var_idx = var_scores.argmax(dim=1)
            trace['variable_selections'].append(selected_var_idx)
            
            # Predict assignment for selected variable
            selected_var_states = var_states[torch.arange(batch_size), selected_var_idx]
            assignment_probs = self.assignment_predictor(selected_var_states)
            trace['assignment_predictions'].append(assignment_probs)
            
            # Make assignment
            new_assignment = assignment_probs.argmax(dim=1)
            assignments[torch.arange(batch_size), selected_var_idx] = new_assignment
            
            # Check for conflicts
            all_clauses_satisfied = True
            conflicts = []
            
            for c in range(num_clauses):
                # Get literals in this clause
                if literal_indices is not None:
                    clause_literals = literal_indices[:, c]
                else:
                    clause_literals = formula[:, c]
                
                # Get corresponding variable indices
                var_indices = clause_literals // 2
                
                # Get literal polarities
                polarities = clause_literals % 2
                
                # Get current assignments for variables in this clause
                var_assignments = assignments.gather(1, var_indices)
                
                # A clause is satisfied if any literal matches its polarity
                clause_satisfied = (var_assignments == polarities) & (var_assignments != 2)
                clause_satisfied = clause_satisfied.any(dim=1)
                
                # A clause is conflicting if all literals are assigned and none satisfy the clause
                all_assigned = (var_assignments != 2).all(dim=1)
                clause_conflicting = all_assigned & ~clause_satisfied
                
                if clause_conflicting.any():
                    # Record conflict
                    conflict_indices = torch.where(clause_conflicting)[0]
                    conflicts.extend([(idx.item(), c) for idx in conflict_indices])
                    all_clauses_satisfied = False
            
            trace['conflicts'].append(conflicts)
            
            # Handle conflicts
            if conflicts:
                # Analyze conflict
                conflict_states = []
                for batch_idx, clause_idx in conflicts:
                    conflict_state = clause_states[batch_idx, clause_idx]
                    conflict_states.append(conflict_state)
                
                if conflict_states:
                    conflict_state = torch.stack(conflict_states, dim=0)
                    
                    # Generate learned clause
                    learned_clause_state = self.conflict_analyzer(
                        torch.cat([conflict_state, var_states.mean(dim=1)], dim=-1)
                    )
                    trace['learned_clauses'].append(learned_clause_state)
                    
                    # Backtrack (unassign the last variable)
                    assignments[torch.arange(batch_size), selected_var_idx] = 2
                    
                    # Random restart with some probability
                    if torch.rand(1).item() < self.restart_probability:
                        assignments = torch.full((batch_size, num_variables), 2, device=device)
            
            # Check if all variables are assigned
            all_assigned = (assignments != 2).all(dim=1)
            
            # Predict satisfiability
            sat_score = self.sat_predictor(var_states.mean(dim=1))
            trace['satisfiability_scores'].append(sat_score)
            
            # Early stopping if all formulas are solved
            if all_assigned.all() and all_clauses_satisfied:
                break
        
        # Final satisfiability prediction
        is_sat = self.sat_predictor(var_states.mean(dim=1))
        
        return is_sat, assignments, trace
    
    def check_solution(self, formula, assignments, literal_indices=None):
        """
        Check if the given assignments satisfy the formula
        
        Args:
            formula: Formula in CNF
            assignments: Variable assignments
            literal_indices: Optional literal indices
            
        Returns:
            is_satisfied: Boolean tensor indicating if each formula is satisfied
        """
        batch_size = formula.size(0)
        num_clauses = formula.size(1)
        
        # Initialize clause satisfaction
        clause_satisfied = torch.zeros(batch_size, num_clauses, dtype=torch.bool, device=formula.device)
        
        for c in range(num_clauses):
            # Get literals in this clause
            if literal_indices is not None:
                clause_literals = literal_indices[:, c]
            else:
                clause_literals = formula[:, c]
            
            # Get corresponding variable indices
            var_indices = clause_literals // 2
            
            # Get literal polarities
            polarities = clause_literals % 2
            
            # Get assignments for variables in this clause
            var_assignments = assignments.gather(1, var_indices)
            
            # A clause is satisfied if any literal matches its polarity
            literal_satisfied = (var_assignments == polarities) & (var_assignments != 2)
            clause_satisfied[:, c] = literal_satisfied.any(dim=1)
        
        # Formula is satisfied if all clauses are satisfied
        is_satisfied = clause_satisfied.all(dim=1)
        
        return is_satisfied
    
    def generate_assignments(self, formula, literal_indices=None):
        """
        Generate variable assignments for a formula without running the full solver
        
        Args:
            formula: Formula in CNF
            literal_indices: Optional literal indices
            
        Returns:
            assignments: Generated variable assignments
        """
        batch_size = formula.size(0)
        device = formula.device
        
        # Get number of variables
        if literal_indices is not None:
            num_variables = literal_indices.max().item() // 2 + 1
        else:
            num_variables = formula.max().item() // 2 + 1
        
        # Initialize variable states
        var_states = torch.zeros(batch_size, num_variables, self.config.hidden_size, device=device)
        
        # Initialize literal embeddings
        if literal_indices is not None:
            # Use provided literal indices
            literal_embeds = self.literal_embeddings(literal_indices)
        else:
            # Create literal embeddings from formula
            literal_embeds = self.literal_embeddings(formula)
        
        # Update variable states based on formula structure
        for c in range(formula.size(1)):
            # Get literals in this clause
            if literal_indices is not None:
                clause_literals = literal_indices[:, c]
            else:
                clause_literals = formula[:, c]
            
            # Get corresponding variable indices
            var_indices = clause_literals // 2
            
            # Update variable states
            clause_embed = literal_embeds[:, c].mean(dim=1, keepdim=True)
            for i in range(var_indices.size(1)):
                var_idx = var_indices[:, i]
                var_states[torch.arange(batch_size), var_idx] += clause_embed.squeeze(1) / formula.size(1)
        
        # Predict assignments for all variables
        assignments = []
        for v in range(num_variables):
            assignment_probs = self.assignment_predictor(var_states[:, v])
            assignment = assignment_probs.argmax(dim=1)
            assignments.append(assignment)
        
        # Stack assignments
        assignments = torch.stack(assignments, dim=1)
        
        return assignments 