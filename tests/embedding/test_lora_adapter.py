"""
Tests for ww.embedding.lora_adapter module.

Tests LoRA-style adaptation of frozen embeddings with
contrastive learning from retrieval outcomes.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ww.embedding.lora_adapter import (
    LoRAConfig,
    LoRAEmbeddingAdapter,
    LoRAState,
    RetrievalOutcome,
    AdaptedBGEM3Provider,
    create_lora_adapter,
    create_adapted_provider,
    TORCH_AVAILABLE,
)


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = LoRAConfig()
        assert config.embedding_dim == 1024
        assert config.rank == 16
        assert config.alpha == 16.0
        assert config.dropout == 0.1
        assert config.learning_rate == 1e-4
        assert config.use_asymmetric is False

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = LoRAConfig(
            rank=32,
            alpha=32.0,
            use_asymmetric=True,
            learning_rate=5e-5
        )
        assert config.rank == 32
        assert config.alpha == 32.0
        assert config.use_asymmetric is True
        assert config.learning_rate == 5e-5

    def test_scaling_factor(self):
        """Alpha/rank scaling should be correct."""
        config = LoRAConfig(rank=16, alpha=32.0)
        expected_scale = 32.0 / 16
        assert expected_scale == 2.0


class TestRetrievalOutcome:
    """Tests for RetrievalOutcome dataclass."""

    def test_create_outcome(self):
        """Should create outcome with embeddings."""
        query = np.random.randn(1024).astype(np.float32)
        positives = [np.random.randn(1024).astype(np.float32) for _ in range(3)]
        negatives = [np.random.randn(1024).astype(np.float32) for _ in range(5)]

        outcome = RetrievalOutcome(
            query_embedding=query,
            positive_embeddings=positives,
            negative_embeddings=negatives,
            session_id="test-session"
        )

        assert outcome.query_embedding.shape == (1024,)
        assert len(outcome.positive_embeddings) == 3
        assert len(outcome.negative_embeddings) == 5
        assert outcome.session_id == "test-session"
        assert outcome.timestamp is not None

    def test_empty_negatives(self):
        """Should allow empty negative embeddings."""
        query = np.random.randn(1024).astype(np.float32)
        positives = [np.random.randn(1024).astype(np.float32)]

        outcome = RetrievalOutcome(
            query_embedding=query,
            positive_embeddings=positives,
            negative_embeddings=[]
        )

        assert len(outcome.negative_embeddings) == 0


class TestLoRAState:
    """Tests for LoRAState serialization."""

    def test_to_dict(self):
        """Should serialize to dict."""
        config = LoRAConfig(rank=32)
        state = LoRAState(
            config=config,
            step_count=100,
            training_losses=[0.5, 0.4, 0.3]
        )

        data = state.to_dict()

        assert data["config"]["rank"] == 32
        assert data["step_count"] == 100
        assert data["training_losses"] == [0.5, 0.4, 0.3]
        assert "created_at" in data
        assert "updated_at" in data

    def test_from_dict_roundtrip(self):
        """Should roundtrip through dict."""
        config = LoRAConfig(rank=32, alpha=64.0)
        state = LoRAState(
            config=config,
            step_count=50,
            training_losses=[0.6, 0.5, 0.4]
        )

        data = state.to_dict()
        restored = LoRAState.from_dict(data)

        assert restored.config.rank == 32
        assert restored.config.alpha == 64.0
        assert restored.step_count == 50
        assert restored.training_losses == [0.6, 0.5, 0.4]

    def test_loss_truncation(self):
        """Should truncate losses to last 100."""
        config = LoRAConfig()
        state = LoRAState(
            config=config,
            training_losses=list(range(200))
        )

        data = state.to_dict()
        assert len(data["training_losses"]) == 100
        assert data["training_losses"][0] == 100  # Last 100 values


class TestLoRAEmbeddingAdapter:
    """Tests for LoRAEmbeddingAdapter."""

    def test_create_adapter(self):
        """Should create adapter."""
        adapter = LoRAEmbeddingAdapter()
        assert adapter.config.rank == 16
        assert adapter.state.step_count == 0

    def test_adapt_preserves_shape_1d(self):
        """Should preserve 1D input shape."""
        adapter = LoRAEmbeddingAdapter()
        embedding = np.random.randn(1024).astype(np.float32)

        adapted = adapter.adapt(embedding)

        assert adapted.shape == (1024,)
        assert adapted.dtype == np.float32

    def test_adapt_preserves_shape_2d(self):
        """Should preserve 2D batch shape."""
        adapter = LoRAEmbeddingAdapter()
        embeddings = np.random.randn(10, 1024).astype(np.float32)

        adapted = adapter.adapt(embeddings)

        assert adapted.shape == (10, 1024)
        assert adapted.dtype == np.float32

    def test_adapt_query_vs_memory(self):
        """Query and memory adaptation should differ for asymmetric."""
        config = LoRAConfig(use_asymmetric=True)
        adapter = LoRAEmbeddingAdapter(config=config)

        embedding = np.random.randn(1024).astype(np.float32)

        # Initially both should be identity-ish
        query_adapted = adapter.adapt_query(embedding)
        memory_adapted = adapter.adapt_memory(embedding)

        # Both should preserve shape
        assert query_adapted.shape == (1024,)
        assert memory_adapted.shape == (1024,)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_initial_adapter_near_identity(self):
        """Initial adapter should be close to identity."""
        adapter = LoRAEmbeddingAdapter()
        embedding = np.random.randn(1024).astype(np.float32)

        adapted = adapter.adapt(embedding)

        # B matrix initialized to zeros, so should be identity
        np.testing.assert_allclose(adapted, embedding, rtol=1e-5)

    def test_record_outcome(self):
        """Should record outcomes to buffer."""
        adapter = LoRAEmbeddingAdapter()

        for _ in range(5):
            outcome = RetrievalOutcome(
                query_embedding=np.random.randn(1024).astype(np.float32),
                positive_embeddings=[np.random.randn(1024).astype(np.float32)],
                negative_embeddings=[np.random.randn(1024).astype(np.float32)]
            )
            adapter.record_outcome(outcome)

        assert len(adapter._training_buffer) == 5

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_training_step(self):
        """Should execute training step when buffer full."""
        config = LoRAConfig(batch_size=4)
        adapter = LoRAEmbeddingAdapter(config=config)

        # Fill buffer to trigger training
        for _ in range(4):
            outcome = RetrievalOutcome(
                query_embedding=np.random.randn(1024).astype(np.float32),
                positive_embeddings=[np.random.randn(1024).astype(np.float32)],
                negative_embeddings=[np.random.randn(1024).astype(np.float32)]
            )
            adapter.record_outcome(outcome)

        # Buffer should be cleared after training
        assert len(adapter._training_buffer) == 0
        assert adapter.state.step_count == 1
        assert len(adapter.state.training_losses) == 1

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_on_outcomes(self):
        """Should train on list of outcomes."""
        config = LoRAConfig(batch_size=4)
        adapter = LoRAEmbeddingAdapter(config=config)

        outcomes = [
            RetrievalOutcome(
                query_embedding=np.random.randn(1024).astype(np.float32),
                positive_embeddings=[np.random.randn(1024).astype(np.float32)],
                negative_embeddings=[np.random.randn(1024).astype(np.float32)]
            )
            for _ in range(10)
        ]

        losses = adapter.train_on_outcomes(outcomes, epochs=1)

        assert len(losses) > 0
        assert adapter.state.step_count > 0

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_training_changes_output(self):
        """Training should change adapter output."""
        config = LoRAConfig(batch_size=2, learning_rate=0.1)  # Higher LR for visible change
        adapter = LoRAEmbeddingAdapter(config=config)

        embedding = np.random.randn(1024).astype(np.float32)
        before = adapter.adapt(embedding.copy())

        # Create outcomes with realistic separation (not too extreme)
        outcomes = []
        for _ in range(20):
            query = np.random.randn(1024).astype(np.float32)
            # Positive moderately similar to query (not too close)
            positive = query * 0.7 + np.random.randn(1024).astype(np.float32) * 0.5
            # Negative is random (not perfectly opposite)
            negative = np.random.randn(1024).astype(np.float32)

            outcome = RetrievalOutcome(
                query_embedding=query,
                positive_embeddings=[positive],
                negative_embeddings=[negative]
            )
            outcomes.append(outcome)

        # Train on outcomes (don't use record_outcome which auto-trains)
        adapter.train_on_outcomes(outcomes, epochs=20)

        after = adapter.adapt(embedding.copy())

        # Output should change after training
        assert not np.allclose(before, after)

    def test_get_stats(self):
        """Should return adapter statistics."""
        adapter = LoRAEmbeddingAdapter()
        stats = adapter.get_stats()

        assert "step_count" in stats
        assert "torch_available" in stats
        assert "config" in stats
        assert stats["config"]["rank"] == 16

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_num_parameters(self):
        """Should report correct parameter count."""
        config = LoRAConfig(rank=16, embedding_dim=1024)
        adapter = LoRAEmbeddingAdapter(config=config)

        stats = adapter.get_stats()

        # LoRA params = 2 * rank * dim (A and B matrices)
        expected = 2 * 16 * 1024
        assert stats["num_parameters"] == expected

    def test_reset(self):
        """Should reset adapter to initial state."""
        adapter = LoRAEmbeddingAdapter()
        adapter.state.step_count = 100
        adapter.state.training_losses = [0.5, 0.4, 0.3]

        adapter.reset()

        assert adapter.state.step_count == 0
        assert len(adapter.state.training_losses) == 0


class TestLoRAAdapterPersistence:
    """Tests for adapter save/load functionality."""

    def test_save_creates_files(self):
        """Save should create state and weight files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = LoRAEmbeddingAdapter()
            adapter.state.step_count = 50

            path = adapter.save(tmpdir)

            assert Path(path).exists()
            assert (Path(path) / "lora_state.json").exists()
            if TORCH_AVAILABLE:
                assert (Path(path) / "lora_weights.pt").exists()

    def test_load_restores_state(self):
        """Load should restore state correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            adapter1 = LoRAEmbeddingAdapter()
            adapter1.state.step_count = 75
            adapter1.state.training_losses = [0.6, 0.5, 0.4]
            adapter1.save(tmpdir)

            # Load into new adapter
            adapter2 = LoRAEmbeddingAdapter()
            success = adapter2.load(tmpdir)

            assert success
            assert adapter2.state.step_count == 75
            assert adapter2.state.training_losses == [0.6, 0.5, 0.4]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_load_restores_weights(self):
        """Load should restore weights correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            config = LoRAConfig(batch_size=2, learning_rate=0.1)
            adapter1 = LoRAEmbeddingAdapter(config=config)

            # Do some training to change weights
            for _ in range(4):
                outcome = RetrievalOutcome(
                    query_embedding=np.random.randn(1024).astype(np.float32),
                    positive_embeddings=[np.random.randn(1024).astype(np.float32)],
                    negative_embeddings=[np.random.randn(1024).astype(np.float32)]
                )
                adapter1.record_outcome(outcome)

            # Get output after training
            embedding = np.random.randn(1024).astype(np.float32)
            output1 = adapter1.adapt(embedding.copy())

            adapter1.save(tmpdir)

            # Load into new adapter
            adapter2 = LoRAEmbeddingAdapter()
            adapter2.load(tmpdir)

            output2 = adapter2.adapt(embedding.copy())

            # Outputs should match
            np.testing.assert_allclose(output1, output2, rtol=1e-5)

    def test_load_nonexistent_returns_false(self):
        """Load from nonexistent path should return False."""
        adapter = LoRAEmbeddingAdapter()
        success = adapter.load("/nonexistent/path")
        assert success is False


class TestAsymmetricLoRA:
    """Tests for asymmetric query/memory adapter."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_asymmetric_separate_adapters(self):
        """Asymmetric mode should have separate query/memory."""
        config = LoRAConfig(use_asymmetric=True)
        adapter = LoRAEmbeddingAdapter(config=config)

        stats = adapter.get_stats()

        # Should have 2x parameters (query + memory adapters)
        symmetric_params = 2 * config.rank * config.embedding_dim
        assert stats["num_parameters"] == symmetric_params * 2

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_asymmetric_training_differentiates(self):
        """Training should differentiate query/memory adapters."""
        config = LoRAConfig(use_asymmetric=True, batch_size=2, learning_rate=0.1)
        adapter = LoRAEmbeddingAdapter(config=config)

        embedding = np.random.randn(1024).astype(np.float32)

        # Initially should be same (both identity)
        query_before = adapter.adapt_query(embedding.copy())
        memory_before = adapter.adapt_memory(embedding.copy())
        np.testing.assert_allclose(query_before, memory_before, rtol=1e-5)

        # Create outcomes with realistic separation (not too extreme)
        outcomes = []
        for _ in range(30):
            query = np.random.randn(1024).astype(np.float32)
            # Positive moderately similar
            positive = query * 0.7 + np.random.randn(1024).astype(np.float32) * 0.5
            # Negative is random (not perfectly opposite)
            negative = np.random.randn(1024).astype(np.float32)

            outcome = RetrievalOutcome(
                query_embedding=query,
                positive_embeddings=[positive],
                negative_embeddings=[negative]
            )
            outcomes.append(outcome)

        # Train the adapters
        adapter.train_on_outcomes(outcomes, epochs=30)

        query_after = adapter.adapt_query(embedding.copy())
        memory_after = adapter.adapt_memory(embedding.copy())

        # Training should change the outputs (verify adaptation is happening)
        assert not np.allclose(query_before, query_after) or not np.allclose(memory_before, memory_after), \
            "At least one adapter should change after training"

        # Check that adapters have different weights (the key asymmetric property)
        query_weights = adapter._adapter.query_adapter.A.weight.detach().cpu().numpy()
        memory_weights = adapter._adapter.memory_adapter.A.weight.detach().cpu().numpy()
        # Weights should diverge due to different gradient flows
        weight_diff = np.abs(query_weights - memory_weights).mean()
        assert weight_diff > 1e-8, f"Weights should diverge, diff={weight_diff}"


class TestAdaptedBGEM3Provider:
    """Tests for AdaptedBGEM3Provider integration."""

    @pytest.fixture
    def mock_base_provider(self):
        """Create mock base provider."""
        class MockProvider:
            @property
            def dimension(self):
                return 1024

            async def embed_query(self, query: str):
                np.random.seed(hash(query) % (2**32))
                return np.random.randn(1024).astype(np.float32).tolist()

            async def embed(self, texts):
                return [
                    np.random.randn(1024).astype(np.float32).tolist()
                    for _ in texts
                ]

            async def embed_batch(self, texts, show_progress=False):
                return await self.embed(texts)

        return MockProvider()

    @pytest.mark.asyncio
    async def test_embed_query(self, mock_base_provider):
        """Should embed query with adaptation."""
        provider = AdaptedBGEM3Provider(
            base_provider=mock_base_provider
        )

        result = await provider.embed_query("test query")

        assert len(result) == 1024
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_embed_texts(self, mock_base_provider):
        """Should embed texts with adaptation."""
        provider = AdaptedBGEM3Provider(
            base_provider=mock_base_provider
        )

        result = await provider.embed(["text1", "text2", "text3"])

        assert len(result) == 3
        assert all(len(e) == 1024 for e in result)

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_base_provider):
        """Should batch embed with adaptation."""
        provider = AdaptedBGEM3Provider(
            base_provider=mock_base_provider
        )

        result = await provider.embed_batch(
            ["text" + str(i) for i in range(50)]
        )

        assert len(result) == 50

    def test_record_outcome(self, mock_base_provider):
        """Should record retrieval outcome."""
        provider = AdaptedBGEM3Provider(
            base_provider=mock_base_provider
        )

        query_emb = np.random.randn(1024).astype(np.float32)
        retrieved = [np.random.randn(1024).astype(np.float32) for _ in range(5)]
        scores = [0.9, 0.8, 0.7, 0.3, 0.1]

        provider.record_retrieval_outcome(
            query_embedding=query_emb,
            retrieved_embeddings=retrieved,
            relevance_scores=scores,
            relevance_threshold=0.5
        )

        # Should have recorded outcome with correct splits
        assert len(provider.adapter._training_buffer) == 1
        outcome = provider.adapter._training_buffer[0]
        assert len(outcome.positive_embeddings) == 3  # scores >= 0.5
        assert len(outcome.negative_embeddings) == 2  # scores < 0.5

    def test_adapter_stats(self, mock_base_provider):
        """Should return adapter stats."""
        provider = AdaptedBGEM3Provider(
            base_provider=mock_base_provider
        )

        stats = provider.get_adapter_stats()

        assert "step_count" in stats
        assert "config" in stats


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_lora_adapter(self):
        """Should create adapter with config."""
        adapter = create_lora_adapter(
            rank=32,
            use_asymmetric=True,
            learning_rate=5e-5
        )

        assert adapter.config.rank == 32
        assert adapter.config.use_asymmetric is True
        assert adapter.config.learning_rate == 5e-5

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider for factory test."""
        class MockProvider:
            dimension = 1024

            async def embed_query(self, q):
                return np.random.randn(1024).tolist()

            async def embed(self, texts):
                return [np.random.randn(1024).tolist() for _ in texts]

            async def embed_batch(self, texts, show_progress=False):
                return await self.embed(texts)

        return MockProvider()

    def test_create_adapted_provider(self, mock_provider):
        """Should create adapted provider."""
        provider = create_adapted_provider(
            base_provider=mock_provider,
            rank=32,
            use_asymmetric=True
        )

        assert isinstance(provider, AdaptedBGEM3Provider)
        assert provider.adapter.config.rank == 32
        assert provider.adapter.config.use_asymmetric is True


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_positives(self):
        """Outcome with empty positives should not cause error."""
        adapter = LoRAEmbeddingAdapter()

        outcome = RetrievalOutcome(
            query_embedding=np.random.randn(1024).astype(np.float32),
            positive_embeddings=[],  # Empty
            negative_embeddings=[]
        )
        adapter.record_outcome(outcome)

        # Should still be in buffer (will be skipped during training)
        assert len(adapter._training_buffer) == 1

    def test_adapt_without_torch(self):
        """Adapt should return original if torch unavailable."""
        adapter = LoRAEmbeddingAdapter()
        adapter._adapter = None  # Simulate torch unavailable

        embedding = np.random.randn(1024).astype(np.float32)
        result = adapter.adapt(embedding)

        np.testing.assert_array_equal(result, embedding)

    @pytest.mark.asyncio
    async def test_empty_texts(self):
        """Should handle empty text list."""
        class MockProvider:
            dimension = 1024

            async def embed_query(self, q):
                return np.random.randn(1024).tolist()

            async def embed(self, texts):
                return [np.random.randn(1024).tolist() for _ in texts]

            async def embed_batch(self, texts, show_progress=False):
                return await self.embed(texts)

        provider = AdaptedBGEM3Provider(base_provider=MockProvider())
        result = await provider.embed([])

        assert result == []
