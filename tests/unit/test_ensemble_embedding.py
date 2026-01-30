"""
Unit tests for ensemble embedding adapter.

Tests EnsembleEmbeddingAdapter with multiple backends and strategies.
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock

from ww.embedding.adapter import MockEmbeddingAdapter, EmbeddingAdapter
from ww.embedding.ensemble import (
    EnsembleStrategy,
    AdapterWeight,
    EnsembleEmbeddingAdapter,
    create_ensemble_adapter,
)


class TestEnsembleStrategy:
    """Tests for EnsembleStrategy enum."""

    def test_strategy_values(self):
        assert EnsembleStrategy.MEAN.value == "mean"
        assert EnsembleStrategy.WEIGHTED_MEAN.value == "weighted_mean"
        assert EnsembleStrategy.CONCAT.value == "concat"
        assert EnsembleStrategy.VOTING.value == "voting"
        assert EnsembleStrategy.BEST.value == "best"


class TestAdapterWeight:
    """Tests for AdapterWeight dataclass."""

    def test_default_values(self):
        weight = AdapterWeight()
        assert weight.base_weight == 1.0
        assert weight.health_weight == 1.0

    def test_effective_weight(self):
        weight = AdapterWeight(base_weight=0.8, health_weight=0.5)
        assert weight.effective_weight == 0.4


class TestEnsembleEmbeddingAdapter:
    """Tests for EnsembleEmbeddingAdapter."""

    @pytest.fixture
    def mock_adapters(self):
        """Create list of mock adapters."""
        return [
            MockEmbeddingAdapter(dimension=128, seed=42),
            MockEmbeddingAdapter(dimension=128, seed=43),
            MockEmbeddingAdapter(dimension=128, seed=44),
        ]

    @pytest.fixture
    def ensemble(self, mock_adapters):
        """Create ensemble with default settings."""
        return EnsembleEmbeddingAdapter(
            adapters=mock_adapters,
            strategy=EnsembleStrategy.WEIGHTED_MEAN,
        )

    def test_creation(self, ensemble):
        assert ensemble.dimension == 128
        assert len(ensemble.adapters) == 3
        assert ensemble.strategy == EnsembleStrategy.WEIGHTED_MEAN

    def test_creation_with_weights(self, mock_adapters):
        ensemble = EnsembleEmbeddingAdapter(
            adapters=mock_adapters,
            weights=[0.5, 0.3, 0.2],
        )
        assert ensemble._weights[0].base_weight == 0.5
        assert ensemble._weights[2].base_weight == 0.2

    def test_creation_requires_adapters(self):
        with pytest.raises(ValueError, match="At least one adapter"):
            EnsembleEmbeddingAdapter(adapters=[])

    def test_creation_validates_weights_count(self, mock_adapters):
        with pytest.raises(ValueError, match="Weights must match"):
            EnsembleEmbeddingAdapter(
                adapters=mock_adapters,
                weights=[0.5, 0.5],  # Only 2 weights for 3 adapters
            )

    def test_dimension_mismatch_error(self):
        adapters = [
            MockEmbeddingAdapter(dimension=128),
            MockEmbeddingAdapter(dimension=256),
        ]
        with pytest.raises(ValueError, match="Dimension mismatch"):
            EnsembleEmbeddingAdapter(adapters=adapters)

    def test_concat_allows_different_dimensions(self):
        adapters = [
            MockEmbeddingAdapter(dimension=128),
            MockEmbeddingAdapter(dimension=256),
        ]
        ensemble = EnsembleEmbeddingAdapter(
            adapters=adapters,
            strategy=EnsembleStrategy.CONCAT,
        )
        assert ensemble.dimension == 384  # 128 + 256

    @pytest.mark.asyncio
    async def test_embed_query_returns_correct_dimension(self, ensemble):
        result = await ensemble.embed_query("test query")
        assert len(result) == 128

    @pytest.mark.asyncio
    async def test_embed_query_normalized(self, ensemble):
        result = await ensemble.embed_query("test query")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, ensemble):
        result = await ensemble.embed([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, ensemble):
        texts = ["text one", "text two", "text three"]
        results = await ensemble.embed(texts)
        assert len(results) == 3
        for emb in results:
            assert len(emb) == 128

    @pytest.mark.asyncio
    async def test_mean_strategy(self, mock_adapters):
        ensemble = EnsembleEmbeddingAdapter(
            adapters=mock_adapters,
            strategy=EnsembleStrategy.MEAN,
        )
        result = await ensemble.embed_query("test")
        assert len(result) == 128
        assert abs(np.linalg.norm(result) - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_best_strategy(self, mock_adapters):
        ensemble = EnsembleEmbeddingAdapter(
            adapters=mock_adapters,
            strategy=EnsembleStrategy.BEST,
            weights=[1.0, 0.5, 0.2],
        )
        result = await ensemble.embed_query("test")
        assert len(result) == 128

    @pytest.mark.asyncio
    async def test_concat_strategy(self):
        adapters = [
            MockEmbeddingAdapter(dimension=64),
            MockEmbeddingAdapter(dimension=64),
        ]
        ensemble = EnsembleEmbeddingAdapter(
            adapters=adapters,
            strategy=EnsembleStrategy.CONCAT,
        )
        result = await ensemble.embed_query("test")
        assert len(result) == 128  # 64 + 64


class TestEnsembleFaultTolerance:
    """Tests for ensemble fault tolerance."""

    @pytest.fixture
    def failing_adapter(self):
        """Create an adapter that fails."""
        adapter = AsyncMock(spec=EmbeddingAdapter)
        adapter.dimension = 128
        adapter.is_healthy.return_value = False
        adapter.embed_query = AsyncMock(side_effect=Exception("Connection failed"))
        adapter.embed = AsyncMock(side_effect=Exception("Connection failed"))
        return adapter

    @pytest.fixture
    def healthy_adapter(self):
        return MockEmbeddingAdapter(dimension=128)

    @pytest.fixture
    def ensemble(self):
        """Create ensemble for fault tolerance tests."""
        adapters = [
            MockEmbeddingAdapter(dimension=128, seed=42),
            MockEmbeddingAdapter(dimension=128, seed=43),
            MockEmbeddingAdapter(dimension=128, seed=44),
        ]
        return EnsembleEmbeddingAdapter(adapters=adapters)

    @pytest.mark.asyncio
    async def test_continues_with_healthy_adapters(self, failing_adapter, healthy_adapter):
        """Ensemble should continue when some adapters fail."""
        ensemble = EnsembleEmbeddingAdapter(
            adapters=[failing_adapter, healthy_adapter],
            fallback_on_failure=True,
        )

        result = await ensemble.embed_query("test")
        assert len(result) == 128

    @pytest.mark.asyncio
    async def test_fails_when_all_adapters_fail(self, failing_adapter):
        """Ensemble should fail when all adapters fail."""
        ensemble = EnsembleEmbeddingAdapter(
            adapters=[failing_adapter],
        )

        with pytest.raises(RuntimeError, match="All adapters failed"):
            await ensemble.embed_query("test")

    @pytest.mark.asyncio
    async def test_health_weight_updates(self, healthy_adapter):
        """Health weights should update based on performance."""
        adapters = [
            MockEmbeddingAdapter(dimension=128),
            MockEmbeddingAdapter(dimension=128),
        ]
        ensemble = EnsembleEmbeddingAdapter(adapters=adapters)

        # Run some queries
        for _ in range(5):
            await ensemble.embed_query("test")

        stats = ensemble.get_ensemble_stats()
        assert stats["adapters"][0]["successes"] == 5
        assert stats["adapters"][1]["successes"] == 5

    def test_reset_health_tracking(self, ensemble):
        """Reset should clear health tracking."""
        ensemble._failure_counts = [5, 3, 1]
        ensemble._success_counts = [10, 10, 10]

        ensemble.reset_health_tracking()

        assert ensemble._failure_counts == [0, 0, 0]
        assert ensemble._success_counts == [0, 0, 0]


class TestGetEnsembleStats:
    """Tests for ensemble statistics."""

    @pytest.fixture
    def ensemble(self):
        adapters = [
            MockEmbeddingAdapter(dimension=128),
            MockEmbeddingAdapter(dimension=128),
        ]
        return EnsembleEmbeddingAdapter(adapters=adapters)

    def test_stats_structure(self, ensemble):
        stats = ensemble.get_ensemble_stats()

        assert "strategy" in stats
        assert "output_dimension" in stats
        assert "num_adapters" in stats
        assert "num_healthy" in stats
        assert "adapters" in stats
        assert len(stats["adapters"]) == 2

    def test_adapter_stats_structure(self, ensemble):
        stats = ensemble.get_ensemble_stats()

        adapter_stat = stats["adapters"][0]
        assert "index" in adapter_stat
        assert "backend" in adapter_stat
        assert "dimension" in adapter_stat
        assert "healthy" in adapter_stat
        assert "effective_weight" in adapter_stat
        assert "successes" in adapter_stat
        assert "failures" in adapter_stat


class TestCreateEnsembleAdapter:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        adapters = [
            MockEmbeddingAdapter(dimension=64),
            MockEmbeddingAdapter(dimension=64),
        ]
        ensemble = create_ensemble_adapter(adapters)

        assert ensemble.strategy == EnsembleStrategy.WEIGHTED_MEAN
        assert len(ensemble.adapters) == 2

    def test_create_with_strategy(self):
        adapters = [MockEmbeddingAdapter(dimension=64)]
        ensemble = create_ensemble_adapter(
            adapters,
            strategy=EnsembleStrategy.BEST,
        )

        assert ensemble.strategy == EnsembleStrategy.BEST

    def test_create_with_weights(self):
        adapters = [
            MockEmbeddingAdapter(dimension=64),
            MockEmbeddingAdapter(dimension=64),
        ]
        ensemble = create_ensemble_adapter(
            adapters,
            weights=[0.7, 0.3],
        )

        assert ensemble._weights[0].base_weight == 0.7
        assert ensemble._weights[1].base_weight == 0.3


class TestEnsembleDeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.asyncio
    async def test_same_input_same_output(self):
        """Same query should produce consistent results."""
        adapters = [
            MockEmbeddingAdapter(dimension=128, seed=42),
            MockEmbeddingAdapter(dimension=128, seed=43),
        ]
        ensemble = EnsembleEmbeddingAdapter(
            adapters=adapters,
            strategy=EnsembleStrategy.MEAN,
        )

        result1 = await ensemble.embed_query("test query")
        result2 = await ensemble.embed_query("test query")

        np.testing.assert_array_almost_equal(result1, result2)

    @pytest.mark.asyncio
    async def test_different_queries_different_results(self):
        """Different queries should produce different results."""
        adapters = [
            MockEmbeddingAdapter(dimension=128, seed=42),
            MockEmbeddingAdapter(dimension=128, seed=43),
        ]
        ensemble = EnsembleEmbeddingAdapter(adapters=adapters)

        result1 = await ensemble.embed_query("query one")
        result2 = await ensemble.embed_query("query two")

        assert not np.allclose(result1, result2)
