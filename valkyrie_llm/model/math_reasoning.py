import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MathProblemConfig:
    """Configuration for math problem generation and curriculum."""
    difficulty_levels: int = 5
    operations: List[str] = field(default_factory=lambda: ["+", "-", "*", "/"])
    max_operands: int = 5
    min_operand_value: float = 0.0
    max_operand_value: float = 1000.0
    include_decimals: bool = True
    include_fractions: bool = True
    include_word_problems: bool = True
    include_algebraic: bool = True
    include_multi_step: bool = True


class MathReasoner(nn.Module):
    """
    Math reasoning module that enhances model capabilities for mathematical tasks.
    
    This module uses a combination of symbolic reasoning and neural networks
    to improve mathematical reasoning capabilities.
    """
    
    def __init__(self, config):
        """
        Initialize math reasoner.
        
        Args:
            config: Configuration for the math reasoner
        """
        super().__init__()
        self.config = config
        
        # Embedding dimension
        self.hidden_size = getattr(config, 'hidden_size', 768)
        
        # Math-specific projection
        self.math_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Number representation layers
        self.number_embedder = nn.Sequential(
            nn.Linear(1, self.hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.hidden_size // 4, self.hidden_size)
        )
        
        # Operation embeddings
        self.operation_embeddings = nn.Embedding(5, self.hidden_size)  # +, -, *, /, =
        
        # Math reasoning networks
        self.math_reasoning_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Step tracking for multi-step problems
        self.max_steps = getattr(config, 'math_max_steps', 10)
        
        # Initialize symbolic calculator
        self._init_symbolic_calculator()
    
    def _init_symbolic_calculator(self):
        """Initialize symbolic calculation components"""
        # Simple symbolic calculator for verification
        self.operations = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y if y != 0 else float('nan'),
            '^': lambda x, y: x ** y,
        }
    
    def embed_math_expression(self, expression_tokens, token_embeddings):
        """
        Embed a mathematical expression for reasoning.
        
        Args:
            expression_tokens: Tokenized mathematical expression
            token_embeddings: Base token embeddings
            
        Returns:
            Enhanced math expression embeddings
        """
        # Extract numbers and operations from tokens
        numbers = []
        operations = []
        
        for token in expression_tokens:
            # Check if token is a number
            if token.replace('.', '', 1).isdigit() or (token[0] == '-' and token[1:].replace('.', '', 1).isdigit()):
                numbers.append(float(token))
            # Check if token is an operation
            elif token in self.operations:
                operations.append(token)
        
        # Embed numbers
        number_tensors = torch.tensor(numbers, dtype=torch.float32).unsqueeze(1)
        number_embeddings = self.number_embedder(number_tensors)
        
        # Embed operations
        operation_indices = [list(self.operations.keys()).index(op) for op in operations]
        operation_tensors = torch.tensor(operation_indices, dtype=torch.long)
        operation_embeddings = self.operation_embeddings(operation_tensors)
        
        # Combine enhanced embeddings with base token embeddings
        enhanced_embeddings = token_embeddings.clone()
        
        # Replace embeddings for numbers and operations
        num_idx = 0
        op_idx = 0
        
        for i, token in enumerate(expression_tokens):
            if token.replace('.', '', 1).isdigit() or (token[0] == '-' and token[1:].replace('.', '', 1).isdigit()):
                enhanced_embeddings[i] = enhanced_embeddings[i] + number_embeddings[num_idx]
                num_idx += 1
            elif token in self.operations:
                enhanced_embeddings[i] = enhanced_embeddings[i] + operation_embeddings[op_idx]
                op_idx += 1
        
        return enhanced_embeddings
    
    def forward(self, hidden_states, attention_mask=None, math_tokens=None):
        """
        Forward pass through the math reasoning module.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            math_tokens: Optional tokenized math expressions
            
        Returns:
            Enhanced hidden states with mathematical reasoning
        """
        # Apply math-specific projection
        math_hidden = self.math_proj(hidden_states)
        
        # Apply mathematical reasoning network
        reasoned_hidden = self.math_reasoning_ffn(math_hidden)
        
        # Skip connection
        enhanced_hidden = hidden_states + self.output_proj(reasoned_hidden)
        
        return enhanced_hidden
    
    def verify_calculation(self, expression: str) -> Tuple[float, bool]:
        """
        Verify a mathematical calculation using symbolic computation.
        
        Args:
            expression: Mathematical expression as a string
            
        Returns:
            result: Calculated result
            is_correct: Whether the calculation is correct
        """
        try:
            # Parse the expression
            parts = expression.replace(' ', '').split('=')
            
            if len(parts) != 2:
                return float('nan'), False
            
            left_side, right_side = parts
            
            # Evaluate left side
            left_result = self._evaluate_expression(left_side)
            
            # Evaluate right side
            if right_side.replace('.', '', 1).isdigit() or (right_side[0] == '-' and right_side[1:].replace('.', '', 1).isdigit()):
                right_result = float(right_side)
            else:
                right_result = self._evaluate_expression(right_side)
            
            # Check if results match
            is_correct = math.isclose(left_result, right_result, rel_tol=1e-9, abs_tol=1e-9)
            
            return left_result, is_correct
        
        except Exception as e:
            logger.warning(f"Error verifying calculation: {e}")
            return float('nan'), False
    
    def _evaluate_expression(self, expr: str) -> float:
        """
        Evaluate a mathematical expression.
        
        Args:
            expr: Expression to evaluate
            
        Returns:
            Calculated result
        """
        # Simple parser for basic expressions
        # This is a simplified implementation - a real one would use a proper parser
        from ast import literal_eval
        
        # Replace operation symbols with Python equivalents
        expr = expr.replace('^', '**')
        
        try:
            # Use Python's ast.literal_eval for safe evaluation
            result = literal_eval(expr)
            return float(result)
        except:
            # Fallback to eval (not safe for production)
            result = eval(expr)
            return float(result)


class MathCurriculum:
    """
    Math curriculum for progressively increasing difficulty in math problems.
    
    This class helps generate mathematical problems with increasing complexity
    for curriculum learning.
    """
    
    def __init__(self, config: MathProblemConfig = None):
        """
        Initialize math curriculum.
        
        Args:
            config: Configuration for math problem generation
        """
        self.config = config or MathProblemConfig()
        self.difficulty_level = 0
        self.max_difficulty = self.config.difficulty_levels - 1
    
    def generate_problem(self, difficulty: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a math problem with specified difficulty.
        
        Args:
            difficulty: Difficulty level (0-4)
            
        Returns:
            Dictionary with problem text and solution
        """
        if difficulty is None:
            difficulty = self.difficulty_level
        
        # Clamp difficulty to valid range
        difficulty = max(0, min(difficulty, self.max_difficulty))
        
        # Scale parameters based on difficulty
        num_operands = min(2 + difficulty, self.config.max_operands)
        
        # Increase numeric range with difficulty
        value_scale = 10 ** difficulty
        min_value = self.config.min_operand_value * (difficulty + 1)
        max_value = min(self.config.max_operand_value, value_scale * 100)
        
        # Determine whether to include more complex features based on difficulty
        use_decimals = self.config.include_decimals and difficulty >= 1
        use_fractions = self.config.include_fractions and difficulty >= 2
        use_word_problem = self.config.include_word_problems and difficulty >= 1
        use_algebraic = self.config.include_algebraic and difficulty >= 3
        use_multi_step = self.config.include_multi_step and difficulty >= 2
        
        # Generate problem based on type
        if use_algebraic:
            return self._generate_algebraic_problem(difficulty)
        elif use_word_problem:
            return self._generate_word_problem(difficulty, num_operands, min_value, max_value, use_decimals)
        else:
            return self._generate_arithmetic_problem(num_operands, min_value, max_value, use_decimals, use_fractions)
    
    def _generate_arithmetic_problem(self, num_operands, min_value, max_value, use_decimals, use_fractions):
        """Generate a basic arithmetic problem"""
        import random
        
        # Generate operands
        operands = []
        for _ in range(num_operands):
            if use_decimals and random.random() < 0.3:
                # Generate decimal number with appropriate precision
                value = random.uniform(min_value, max_value)
                decimals = random.randint(1, 2)
                value = round(value, decimals)
            elif use_fractions and random.random() < 0.3:
                # Generate fraction as a decimal
                numerator = random.randint(1, 20)
                denominator = random.randint(1, 20)
                value = numerator / denominator
            else:
                # Generate integer
                value = random.randint(int(min_value), int(max_value))
            
            operands.append(value)
        
        # Generate operations
        operations = []
        for _ in range(num_operands - 1):
            operations.append(random.choice(self.config.operations))
        
        # Build expression
        expression = ""
        for i in range(num_operands):
            expression += str(operands[i])
            if i < num_operands - 1:
                expression += f" {operations[i]} "
        
        # Calculate solution
        solution = operands[0]
        for i in range(num_operands - 1):
            if operations[i] == "+":
                solution += operands[i + 1]
            elif operations[i] == "-":
                solution -= operands[i + 1]
            elif operations[i] == "*":
                solution *= operands[i + 1]
            elif operations[i] == "/":
                if operands[i + 1] == 0:
                    return self._generate_arithmetic_problem(num_operands, min_value, max_value, use_decimals, use_fractions)
                solution /= operands[i + 1]
        
        # Round the solution to avoid floating point issues
        solution = round(solution, 6)
        
        return {
            "problem": f"Calculate: {expression}",
            "solution": str(solution),
            "expression": expression,
            "type": "arithmetic",
            "difficulty": min(max(round(num_operands / 2), 1), self.max_difficulty)
        }
    
    def _generate_word_problem(self, difficulty, num_operands, min_value, max_value, use_decimals):
        """Generate a word problem"""
        import random
        
        # Templates for different operations
        templates = {
            "+": [
                "If X has {a} apples and receives {b} more, how many apples does X have in total?",
                "A store sold {a} items in the morning and {b} items in the afternoon. How many items were sold in total?",
                "If a train travels {a} kilometers on Monday and {b} kilometers on Tuesday, what is the total distance traveled?"
            ],
            "-": [
                "If X has {a} dollars and spends {b} dollars, how much money does X have left?",
                "A tank contains {a} liters of water. If {b} liters are used, how much water remains in the tank?",
                "If a store has {a} items and sells {b} of them, how many items are left?"
            ],
            "*": [
                "If each box contains {a} items and there are {b} boxes, how many items are there in total?",
                "If a car travels at {a} kilometers per hour for {b} hours, how far does it travel?",
                "If each ticket costs ${a} and {b} tickets are sold, what is the total revenue?"
            ],
            "/": [
                "If {a} items are shared equally among {b} people, how many items does each person get?",
                "If a car travels {a} kilometers in {b} hours, what is its average speed in kilometers per hour?",
                "If a project takes {a} hours of work and {b} workers share the work equally, how many hours does each worker need to work?"
            ]
        }
        
        # Choose operation
        operation = random.choice(self.config.operations)
        
        # Make sure we don't divide by zero
        if operation == "/":
            a = random.uniform(min_value, max_value) if use_decimals else random.randint(int(min_value), int(max_value))
            b = random.uniform(1, max_value / 10) if use_decimals else random.randint(1, int(max_value / 10))
        else:
            a = random.uniform(min_value, max_value) if use_decimals else random.randint(int(min_value), int(max_value))
            b = random.uniform(min_value, max_value) if use_decimals else random.randint(int(min_value), int(max_value))
        
        # Round numbers to make them more readable
        if use_decimals:
            decimals = random.randint(1, 2)
            a = round(a, decimals)
            b = round(b, decimals)
        else:
            a = int(a)
            b = int(b)
        
        # Choose template for the operation
        template = random.choice(templates[operation])
        
        # Calculate answer
        if operation == "+":
            answer = a + b
        elif operation == "-":
            answer = a - b
        elif operation == "*":
            answer = a * b
        elif operation == "/":
            answer = a / b
        
        # Round the answer to avoid floating point issues
        answer = round(answer, 6)
        
        # Generate problem text
        problem_text = template.replace("{a}", str(a)).replace("{b}", str(b))
        
        return {
            "problem": problem_text,
            "solution": str(answer),
            "expression": f"{a} {operation} {b}",
            "type": "word_problem",
            "difficulty": difficulty
        }
    
    def _generate_algebraic_problem(self, difficulty):
        """Generate an algebraic problem"""
        import random
        import string
        
        # Choose variable name
        variable = random.choice(string.ascii_lowercase)
        
        # Generate coefficients based on difficulty
        a = random.randint(1, 10)
        b = random.randint(-20, 20)
        c = random.randint(-50, 50)
        
        # Choose problem type based on difficulty
        if difficulty <= 3:
            # Linear equation: ax + b = c
            solution = (c - b) / a
            equation = f"{a}{variable} + {b} = {c}" if b >= 0 else f"{a}{variable} - {abs(b)} = {c}"
            problem_type = "linear_equation"
        else:
            # Quadratic equation: ax^2 + bx + c = 0
            # Ensure it has real solutions
            while b**2 - 4*a*c < 0:
                b = random.randint(-20, 20)
                c = random.randint(-50, 50)
            
            disc = b**2 - 4*a*c
            solution1 = (-b + math.sqrt(disc)) / (2*a)
            solution2 = (-b - math.sqrt(disc)) / (2*a)
            
            # Format equation
            equation = f"{a}{variable}² "
            if b > 0:
                equation += f"+ {b}{variable} "
            elif b < 0:
                equation += f"- {abs(b)}{variable} "
            
            if c > 0:
                equation += f"+ {c} = 0"
            elif c < 0:
                equation += f"- {abs(c)} = 0"
            else:
                equation += "= 0"
            
            # For quadratics, return both solutions
            solution = [round(solution1, 4), round(solution2, 4)]
            solution.sort()  # Convention: smallest solution first
            solution = [str(s) for s in solution]
            problem_type = "quadratic_equation"
        
        return {
            "problem": f"Solve for {variable}: {equation}",
            "solution": str(solution) if not isinstance(solution, list) else solution,
            "expression": equation,
            "type": problem_type,
            "difficulty": difficulty
        }
    
    def step(self):
        """Increase difficulty level"""
        if self.difficulty_level < self.max_difficulty:
            self.difficulty_level += 1
    
    def generate_dataset(self, size: int) -> List[Dict[str, Any]]:
        """
        Generate a dataset of math problems.
        
        Args:
            size: Number of problems to generate
            
        Returns:
            List of math problems
        """
        problems = []
        for _ in range(size):
            # Generate problems with varying difficulty
            difficulty = random.randint(0, self.difficulty_level)
            problems.append(self.generate_problem(difficulty))
        
        return problems


class CurriculumScheduler:
    """
    Scheduler for curriculum learning that manages stages and progression.
    
    This scheduler manages transitions between different curriculum stages
    based on epochs or validation metrics.
    """
    
    def __init__(self, stages, data_loader=None):
        """
        Initialize curriculum scheduler.
        
        Args:
            stages: List of curriculum stages
            data_loader: Initial data loader
        """
        self.stages = stages
        self.current_stage = 0
        self.data_loader = data_loader
        self.epoch_counter = 0
    
    def step(self, metrics=None):
        """
        Advance to the next curriculum stage if needed.
        
        Args:
            metrics: Optional dictionary of metrics for metric-based advancement
            
        Returns:
            Boolean indicating whether the stage was advanced
        """
        # Get current stage configuration
        if self.current_stage >= len(self.stages):
            return False
        
        stage = self.stages[self.current_stage]
        
        # Check if we should advance based on epochs
        self.epoch_counter += 1
        if self.epoch_counter >= stage.get('epochs', 1) and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.epoch_counter = 0
            return True
        
        # Could also implement metric-based advancement here
        
        return False
    
    def get_data_loader(self):
        """Get data loader for current stage"""
        return self.data_loader
    
    def get_current_stage(self):
        """Get current stage information"""
        if self.current_stage < len(self.stages):
            return self.stages[self.current_stage]
        return None


def build_curriculum(data_loader, difficulty_fn, num_epochs):
    """
    Build a curriculum learning scheduler.
    
    This function creates a curriculum with progressively increasing difficulty
    levels distributed across the total training epochs.
    
    Args:
        data_loader: Base data loader
        difficulty_fn: Function to compute example difficulty
        num_epochs: Total number of training epochs
        
    Returns:
        Configured curriculum scheduler
    """
    # Create difficulty-based stages
    stages = []
    
    # Calculate epochs per stage, ensuring at least 1 epoch per stage
    num_stages = 5  # Default to 5 difficulty stages
    epochs_per_stage = max(1, num_epochs // num_stages)
    
    # Create progressive stages with increasing difficulty
    for i in range(num_stages):
        min_difficulty = i / num_stages
        max_difficulty = (i + 1) / num_stages
        
        # Final stage includes all difficulties
        if i == num_stages - 1:
            max_difficulty = 1.0
        
        # Add stage
        stages.append({
            'difficulty_range': (min_difficulty, max_difficulty),
            'epochs': epochs_per_stage,
            'sample_ratio': min(1.0, 0.6 + 0.1 * i)  # Start with 60% of data, increase gradually
        })
    
    # Ensure total epochs match the specified number
    remaining_epochs = num_epochs - (epochs_per_stage * num_stages)
    if remaining_epochs > 0:
        stages[-1]['epochs'] += remaining_epochs
    
    return CurriculumScheduler(stages, data_loader) 