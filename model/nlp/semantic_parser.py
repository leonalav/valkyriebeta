import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class SemanticParserConfig:
    """Configuration for Semantic Parser module."""
    # Architecture parameters
    hidden_size: int = 768
    intermediate_size: int = 1024
    num_layers: int = 2
    num_attention_heads: int = 8
    dropout: float = 0.1
    
    # Grammar and structure parameters
    max_parse_depth: int = 16
    num_semantic_labels: int = 128
    use_typed_dependencies: bool = True
    use_constituency_parsing: bool = True
    use_semantic_roles: bool = True
    
    # Learning parameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    
    # Training parameters
    batch_size: int = 16
    max_seq_length: int = 512


class SemanticAttention(nn.Module):
    """Multi-head attention mechanism for semantic parsing."""
    
    def __init__(self, config: SemanticParserConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape for multi-head attention."""
        batch_size, seq_length, hidden_size = x.size()
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute multi-head attention.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask tensor
            
        Returns:
            Contextualized representation
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Compute Q, K, V
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Broadcast to right shape
            attention_scores = attention_scores + attention_mask
        
        # Apply softmax to get attention weights
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_length, num_heads, head_size)
        context = context.view(batch_size, seq_length, self.all_head_size)
        
        # Apply output projection
        output = self.out(context)
        
        return output


class RecursiveParsingLayer(nn.Module):
    """Layer that enables recursive parsing of hierarchical structures."""
    
    def __init__(self, config: SemanticParserConfig):
        super().__init__()
        self.max_depth = config.max_parse_depth
        
        # Attention for composition
        self.attention = SemanticAttention(config)
        
        # Feed-forward for transformation
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Parameters for recursive application
        self.recursion_gate = nn.Linear(config.hidden_size, 1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        parent_states: Optional[torch.Tensor] = None,
        depth: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply recursive parsing layer.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask tensor
            parent_states: Optional parent states for recursive parsing
            depth: Current recursion depth
            
        Returns:
            Tuple of (hidden_states, recursion_weights)
        """
        # Apply attention
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm(hidden_states + attention_output)
        
        # Apply feed-forward transformation
        intermediate_output = F.gelu(self.intermediate(hidden_states))
        hidden_states = self.layer_norm(hidden_states + self.dropout(self.output(intermediate_output)))
        
        # Compute recursion gate values
        recursion_weights = torch.sigmoid(self.recursion_gate(hidden_states))
        
        # If at max depth or parent_states is None, don't recurse further
        if depth >= self.max_depth or parent_states is None:
            return hidden_states, recursion_weights
        
        # Recursive application on suitable nodes
        # This is a simplified version - a real implementation would be more complex
        recursion_mask = (recursion_weights > 0.5).float()
        
        # Apply parent states where recursion is indicated
        hidden_states = hidden_states * (1 - recursion_mask) + parent_states * recursion_mask
        
        return hidden_states, recursion_weights


class SemanticParser(nn.Module):
    """
    Semantic Parser for structured language understanding.
    
    This module performs semantic parsing to extract structured meaning from text:
    - Syntactic parsing (constituency and dependency)
    - Semantic role labeling
    - Formal meaning representations
    """
    
    def __init__(
        self,
        config: SemanticParserConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.config = config
        self.device = device
        
        # Define semantic parsing components
        self.recursive_layers = nn.ModuleList([
            RecursiveParsingLayer(config) for _ in range(config.num_layers)
        ])
        
        # Output heads for different parsing tasks
        if config.use_typed_dependencies:
            self.dependency_head = self._create_dependency_head(config)
        
        if config.use_constituency_parsing:
            self.constituency_head = self._create_constituency_head(config)
        
        if config.use_semantic_roles:
            self.semantic_role_head = self._create_semantic_role_head(config)
        
        # Final semantic representation layer
        self.semantic_output = nn.Linear(config.hidden_size, config.num_semantic_labels)
        
        # Move to device
        self.to(device)
    
    def _create_dependency_head(self, config: SemanticParserConfig) -> nn.Module:
        """Create the dependency parsing head."""
        return nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    def _create_constituency_head(self, config: SemanticParserConfig) -> nn.Module:
        """Create the constituency parsing head."""
        return nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    def _create_semantic_role_head(self, config: SemanticParserConfig) -> nn.Module:
        """Create the semantic role labeling head."""
        return nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process hidden states with semantic parsing.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask tensor
            
        Returns:
            Dictionary of semantic parsing outputs
        """
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Apply recursive parsing layers
        intermediate_states = []
        recursion_weights_list = []
        
        # Process with recursive layers
        current_states = hidden_states
        for layer in self.recursive_layers:
            current_states, recursion_weights = layer(
                current_states, 
                attention_mask=attention_mask,
                parent_states=hidden_states,  # Use original states as parent context
                depth=0  # Start at depth 0
            )
            intermediate_states.append(current_states)
            recursion_weights_list.append(recursion_weights)
        
        # Final hidden states
        final_states = current_states
        
        # Generate parsing outputs
        outputs = {
            "final_hidden_states": final_states,
            "recursion_weights": recursion_weights_list[-1]
        }
        
        # Apply dependency parsing if enabled
        if hasattr(self, "dependency_head"):
            # Compute dependency scores between tokens
            dependency_features = self.dependency_head(final_states)
            # Simple pairwise scoring for dependencies (simplified)
            dependency_scores = torch.bmm(
                dependency_features,  # [batch, seq, hidden]
                dependency_features.transpose(1, 2)  # [batch, hidden, seq]
            )  # [batch, seq, seq]
            outputs["dependency_scores"] = dependency_scores
        
        # Apply constituency parsing if enabled
        if hasattr(self, "constituency_head"):
            constituency_features = self.constituency_head(final_states)
            # For constituency parsing, we'd need a more complex mechanism
            # This is a simplified placeholder
            outputs["constituency_features"] = constituency_features
        
        # Apply semantic role labeling if enabled
        if hasattr(self, "semantic_role_head"):
            semantic_role_features = self.semantic_role_head(final_states)
            outputs["semantic_role_features"] = semantic_role_features
        
        # Generate final semantic representations
        semantic_labels = self.semantic_output(final_states)
        outputs["semantic_labels"] = semantic_labels
        
        return outputs
    
    def parse_text(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Parse input text using the language model and semantic parser.
        
        Args:
            model: The language model to integrate with
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary containing parsed structures
        """
        # Get model device
        device = next(model.parameters()).device
        self.to(device)
        
        # Get hidden states from language model
        with torch.no_grad():
            model_outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                output_hidden_states=True
            )
            hidden_states = model_outputs.last_hidden_state
        
        # Apply semantic parsing
        parsing_outputs = self.forward(hidden_states, attention_mask)
        
        # Convert to structured output
        structured_output = self._convert_to_structured_output(
            parsing_outputs, input_ids, model.tokenizer
        )
        
        return structured_output
    
    def _convert_to_structured_output(
        self,
        parsing_outputs: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        tokenizer
    ) -> Dict[str, Any]:
        """
        Convert raw parsing outputs to structured representations.
        
        Args:
            parsing_outputs: Outputs from the forward pass
            input_ids: Token IDs
            tokenizer: Tokenizer for decoding
            
        Returns:
            Dictionary with structured parsing results
        """
        # This would be a complex conversion from tensor outputs to structured formats
        # For now, return a simplified placeholder
        
        structured_output = {
            "dependencies": [],
            "constituency_tree": {},
            "semantic_representation": {}
        }
        
        return structured_output 