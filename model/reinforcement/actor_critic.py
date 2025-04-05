import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class ActorCritic:
    """
    Actor-Critic implementation for reinforcement learning with language models.
    """
    def __init__(self, model=None, tokenizer=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.actor_lr = kwargs.get('actor_lr', 1e-5)
        self.critic_lr = kwargs.get('critic_lr', 1e-5)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        
        # Create actor and critic heads if model is provided
        if model is not None:
            hidden_size = model.config.hidden_size if hasattr(model, 'config') else 768
            self.actor_head = nn.Linear(hidden_size, 1)
            self.critic_head = nn.Linear(hidden_size, 1)
            logger.info(f"Initialized ActorCritic with model (hidden size: {hidden_size})")
        else:
            logger.warning("Initialized ActorCritic without model (mock mode)")
            
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        Forward pass through the actor-critic network.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for input
            
        Returns:
            Dictionary with actor and critic outputs
        """
        if self.model is None:
            # Mock implementation
            batch_size = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
            seq_len = input_ids.shape[1] if hasattr(input_ids, 'shape') else 10
            
            if isinstance(input_ids, torch.Tensor):
                device = input_ids.device
                # Create mock outputs
                actor_logits = torch.randn(batch_size, seq_len, 1, device=device)
                critic_values = torch.randn(batch_size, seq_len, 1, device=device)
            else:
                # Handle non-tensor inputs
                actor_logits = torch.randn(1, 10, 1)
                critic_values = torch.randn(1, 10, 1)
                
            return {
                "actor_logits": actor_logits,
                "critic_values": critic_values
            }
        else:
            # Use the actual model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )
            
            # Get the last hidden state
            hidden_states = outputs.hidden_states[-1]
            
            # Apply actor and critic heads
            actor_logits = self.actor_head(hidden_states)
            critic_values = self.critic_head(hidden_states)
            
            return {
                "actor_logits": actor_logits,
                "critic_values": critic_values,
                "hidden_states": hidden_states
            } 