"""
Test script for optimized MCTS implementation.
This script runs a simple test to verify the optimized MCTS reasoner.
"""

import torch
import time
import argparse
from mcts_reasoner import MCTSReasoner, MCTSConfig

def test_memory_efficiency():
    """Test memory efficiency of optimized MCTS"""
    print("Testing memory efficiency...")
    
    # Create states and actions for testing
    batch_size = 1
    seq_len = 64
    hidden_size = 768
    num_actions = 100
    
    # Create random states and actions
    states = torch.randn(batch_size, seq_len, hidden_size)
    action_embeddings = torch.randn(num_actions, hidden_size)
    available_actions = list(range(num_actions))
    
    # Define simple transition and reward functions
    def transition_fn(state, action):
        # Simple transition: add action embedding to state
        action_emb = action_embeddings[action].unsqueeze(0).unsqueeze(0)
        
        # Debug dimensions
        if isinstance(state, torch.Tensor):
            state_dims = state.shape
            if len(state_dims) == 2:  # If state is [batch_size, hidden_size]
                return torch.cat([state, action_emb.squeeze(0)], dim=0).unsqueeze(0)
            elif len(state_dims) == 3:  # If state is [batch_size, seq_len, hidden_size]
                return torch.cat([state[:, 1:], action_emb], dim=1)
        return state  # Fallback
        
    def reward_fn(state, action):
        # Simple reward: dot product with action embedding
        return torch.matmul(state[:, -1], action_embeddings[action]).item() * 0.01
    
    # Test with and without optimizations
    configs = [
        ("Baseline", MCTSConfig(
            use_state_compression=False,
            use_value_cache=False,
            use_hybrid_search=False,
            use_action_space_reduction=False,
            use_dynamic_confidence=False,
            max_simulations=50
        )),
        ("Optimized", MCTSConfig(
            use_state_compression=True,
            use_value_cache=True,
            use_hybrid_search=True,
            use_action_space_reduction=True,
            use_dynamic_confidence=True,
            state_compression_ratio=0.25,
            value_cache_capacity=1000,
            beam_size=4,
            max_action_space_size=50,
            max_simulations=50
        ))
    ]
    
    results = {}
    
    for name, config in configs:
        # Create MCTS reasoner
        mcts = MCTSReasoner(config)
        
        # Run search
        start_time = time.time()
        mcts(states[0], available_actions, action_embeddings, transition_fn, reward_fn)
        end_time = time.time()
        
        # Get performance report
        perf_report = mcts.get_performance_report()
        
        # Store results
        results[name] = {
            "time": end_time - start_time,
            "report": perf_report
        }
        
        print(f"\n{name} MCTS:")
        print(f"Time: {results[name]['time']:.4f} seconds")
        print(perf_report)
        
    # Compare results
    speedup = results["Baseline"]["time"] / results["Optimized"]["time"]
    print(f"\nSpeedup: {speedup:.2f}x")
    
def test_hybrid_search():
    """Test hybrid MCTS-Beam search implementation"""
    print("\nTesting hybrid search...")
    
    # Create states and actions for testing
    batch_size = 1
    seq_len = 32
    hidden_size = 768
    num_actions = 1000  # Very large action space
    
    # Create random states and actions
    states = torch.randn(batch_size, seq_len, hidden_size)
    action_embeddings = torch.randn(num_actions, hidden_size)
    available_actions = list(range(num_actions))
    
    # Create logits with a few clearly better actions
    logits = torch.randn(num_actions)
    # Make a few actions much better
    best_actions = [10, 25, 42, 77]
    for action in best_actions:
        logits[action] = 10.0
    
    # Create optimized MCTS
    config = MCTSConfig(
        use_state_compression=True,
        use_value_cache=True,
        use_hybrid_search=True,
        use_action_space_reduction=True,
        beam_size=4,
        max_simulations=100
    )
    
    mcts = MCTSReasoner(config)
    
    # Override policy network to return our logits
    def mock_policy(state):
        return logits.clone()
    
    mcts.policy_network = mock_policy
    
    # Define simple transition and reward functions
    def transition_fn(state, action):
        # Simple transition: add action embedding to state
        action_emb = action_embeddings[action].unsqueeze(0).unsqueeze(0)
        
        # Debug dimensions
        if isinstance(state, torch.Tensor):
            state_dims = state.shape
            if len(state_dims) == 2:  # If state is [batch_size, hidden_size]
                return torch.cat([state, action_emb.squeeze(0)], dim=0).unsqueeze(0)
            elif len(state_dims) == 3:  # If state is [batch_size, seq_len, hidden_size]
                return torch.cat([state[:, 1:], action_emb], dim=1)
        return state  # Fallback
        
    def reward_fn(state, action):
        # Simple reward: higher for best actions
        if action in best_actions:
            return 1.0
        return 0.0
    
    # Run search
    start_time = time.time()
    action, value = mcts(states[0], available_actions, action_embeddings, transition_fn, reward_fn)
    end_time = time.time()
    
    print(f"Selected action: {action}")
    print(f"Action value: {value:.4f}")
    print(f"Time: {end_time - start_time:.4f} seconds")
    print(f"Best action found: {action in best_actions}")
    print(mcts.get_performance_report())

def main():
    parser = argparse.ArgumentParser(description="Test MCTS optimizations")
    parser.add_argument("--test", choices=["memory", "hybrid", "all"], default="all",
                       help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "memory" or args.test == "all":
        test_memory_efficiency()
        
    if args.test == "hybrid" or args.test == "all":
        test_hybrid_search()

if __name__ == "__main__":
    main() 