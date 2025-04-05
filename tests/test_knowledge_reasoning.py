import unittest
import torch
import os
from model.knowledge_reasoning import (
    KnowledgeReasoningConfig,
    KnowledgeRetriever,
    KnowledgeFusion,
    MultiHopAttention,
    KnowledgeVerifier,
    KnowledgeRouter,
    KnowledgeComposer,
    MultiHopKnowledgeReasoner,
    KnowledgeReasoningModule
)

class TestKnowledgeReasoning(unittest.TestCase):
    def setUp(self):
        # Ensure knowledge graph directory exists
        os.makedirs('data/knowledge_graphs', exist_ok=True)

        self.config = KnowledgeReasoningConfig(
            hidden_size=64,
            num_heads=4,
            dropout=0.1,
            max_hops=2,
            use_adaptive_hops=True,
            early_stopping_threshold=0.9,
            knowledge_source="conceptnet",
            knowledge_embedding_dim=64,
            knowledge_graph_path="data/knowledge_graphs/",
            max_knowledge_items=10,
            use_knowledge_fusion=True,
            fusion_layers=1,
            use_multi_hop_attention=True,
            use_knowledge_routing=True,
            num_knowledge_experts=2,
            use_knowledge_composition=True,
            composition_depth=1
        )
        self.batch_size = 2
        self.seq_len = 8
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)

    def test_knowledge_retriever(self):
        """Test the knowledge retriever component"""
        retriever = KnowledgeRetriever(self.config)

        knowledge_embeddings, knowledge_mask, knowledge_scores = retriever(self.hidden_states)

        # Check output shapes
        self.assertEqual(knowledge_embeddings.shape, (self.batch_size, self.seq_len, self.config.max_knowledge_items, self.config.knowledge_embedding_dim))
        self.assertEqual(knowledge_mask.shape, (self.batch_size, self.seq_len, self.config.max_knowledge_items))
        self.assertEqual(knowledge_scores.shape, (self.batch_size, self.seq_len, self.config.max_knowledge_items))

    def test_knowledge_fusion(self):
        """Test the knowledge fusion component"""
        fusion = KnowledgeFusion(self.config)

        # Create mock knowledge data
        knowledge_embeddings = torch.randn(self.batch_size, self.seq_len, self.config.max_knowledge_items, self.config.knowledge_embedding_dim)
        knowledge_mask = torch.ones(self.batch_size, self.seq_len, self.config.max_knowledge_items)
        knowledge_scores = torch.ones(self.batch_size, self.seq_len, self.config.max_knowledge_items)

        fused_states = fusion(self.hidden_states, knowledge_embeddings, knowledge_mask, knowledge_scores)

        # Check output shape
        self.assertEqual(fused_states.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

    def test_multi_hop_attention(self):
        """Test the multi-hop attention component"""
        attention = MultiHopAttention(self.config)

        # Create mock knowledge states
        knowledge_states = torch.randn(self.batch_size, self.seq_len * 5, self.config.hidden_size)

        # Test without hop level
        output, attn_weights = attention(self.hidden_states, knowledge_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

        # Test with hop level
        output, attn_weights = attention(self.hidden_states, knowledge_states, hop_level=1)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

    def test_knowledge_verifier(self):
        """Test the knowledge verifier component"""
        verifier = KnowledgeVerifier(self.config)

        output, verification_scores = verifier(self.hidden_states, self.hidden_states)

        # Check output shapes
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))
        self.assertEqual(verification_scores.shape, (self.batch_size, self.seq_len, 1))

    def test_knowledge_router(self):
        """Test the knowledge router component"""
        router = KnowledgeRouter(self.config)

        output, routing_weights = router(self.hidden_states)

        # Check output shapes
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))
        self.assertEqual(routing_weights.shape, (self.batch_size, self.seq_len, self.config.num_knowledge_experts))

    def test_knowledge_composer(self):
        """Test the knowledge composer component"""
        composer = KnowledgeComposer(self.config)

        # Test without past states
        output = composer(self.hidden_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

        # Test with past states
        past_states = [torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)]
        output = composer(self.hidden_states, past_states)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

    def test_multi_hop_knowledge_reasoner(self):
        """Test the multi-hop knowledge reasoner"""
        reasoner = MultiHopKnowledgeReasoner(self.config)

        output, reasoning_info = reasoner(self.hidden_states)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

        # Check reasoning info
        self.assertIn('knowledge_scores', reasoning_info)
        if self.config.use_knowledge_routing:
            self.assertIn('routing_weights', reasoning_info)
        if self.config.use_knowledge_verification:
            self.assertIn('verification_scores', reasoning_info)

    def test_knowledge_reasoning_module(self):
        """Test the complete knowledge reasoning module"""
        module = KnowledgeReasoningModule(self.config)

        output, reasoning_info = module(self.hidden_states)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.config.hidden_size))

        # Check reasoning info
        self.assertIn('knowledge_scores', reasoning_info)
        if self.config.use_knowledge_routing:
            self.assertIn('routing_weights', reasoning_info)
        if self.config.use_knowledge_verification:
            self.assertIn('verification_scores', reasoning_info)

if __name__ == '__main__':
    unittest.main()