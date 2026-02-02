"""
Ensemble Embedding Adapter for T4DM.

Implements multi-scale redundancy through ensemble voting across multiple
embedding providers. This mirrors biological neural redundancy where
multiple pathways encode the same information for fault tolerance.

CompBio-inspired: Real neural systems use redundant pathways and population
coding for robust information representation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from t4dm.embedding.adapter import EmbeddingAdapter, EmbeddingBackend

logger = logging.getLogger(__name__)


class EnsembleStrategy(Enum):
    """Strategy for combining embeddings from multiple adapters."""
    MEAN = "mean"              # Simple average
    WEIGHTED_MEAN = "weighted_mean"  # Health-weighted average
    CONCAT = "concat"          # Concatenate (increases dimension)
    VOTING = "voting"          # Weighted voting on similarity
    BEST = "best"              # Use best healthy adapter only


@dataclass
class AdapterWeight:
    """Weight configuration for an adapter in the ensemble."""
    base_weight: float = 1.0        # Base contribution weight
    health_weight: float = 1.0      # Dynamic weight from health status
    latency_penalty: float = 0.1    # Penalty per 100ms latency

    @property
    def effective_weight(self) -> float:
        """Calculate effective weight including health."""
        return self.base_weight * self.health_weight


class EnsembleEmbeddingAdapter(EmbeddingAdapter):
    """
    Ensemble of multiple embedding adapters with health-aware voting.

    Provides fault tolerance through redundant encoding paths.
    If one adapter fails or becomes unhealthy, others compensate.

    Example:
        # Create ensemble from multiple backends
        bge_adapter = BGEM3Adapter()
        mock_adapter = MockEmbeddingAdapter(dimension=1024)

        ensemble = EnsembleEmbeddingAdapter(
            adapters=[bge_adapter, mock_adapter],
            strategy=EnsembleStrategy.WEIGHTED_MEAN,
        )

        # Embeddings combine both, weighted by health
        result = await ensemble.embed_query("test")
    """

    def __init__(
        self,
        adapters: list[EmbeddingAdapter],
        strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_MEAN,
        weights: list[float] | None = None,
        require_dimension_match: bool = True,
        fallback_on_failure: bool = True,
    ):
        """
        Initialize ensemble adapter.

        Args:
            adapters: List of embedding adapters
            strategy: How to combine embeddings
            weights: Optional per-adapter weights (default: equal)
            require_dimension_match: Require same dimension (except CONCAT)
            fallback_on_failure: Continue with healthy adapters on failure
        """
        if not adapters:
            raise ValueError("At least one adapter required")

        # Validate dimensions
        dimensions = [a.dimension for a in adapters]
        if require_dimension_match and strategy != EnsembleStrategy.CONCAT:
            if len(set(dimensions)) > 1:
                raise ValueError(
                    f"Dimension mismatch: {dimensions}. "
                    "Set require_dimension_match=False or use CONCAT strategy."
                )

        # Calculate output dimension
        if strategy == EnsembleStrategy.CONCAT:
            output_dim = sum(dimensions)
        else:
            output_dim = dimensions[0]

        super().__init__(dimension=output_dim)
        self._backend = EmbeddingBackend.MOCK  # Ensemble is a meta-backend

        self._adapters = adapters
        self._strategy = strategy
        self._fallback_on_failure = fallback_on_failure

        # Initialize weights
        if weights:
            if len(weights) != len(adapters):
                raise ValueError("Weights must match number of adapters")
            self._weights = [
                AdapterWeight(base_weight=w) for w in weights
            ]
        else:
            self._weights = [
                AdapterWeight(base_weight=1.0) for _ in adapters
            ]

        # Track adapter health
        self._failure_counts = [0] * len(adapters)
        self._success_counts = [0] * len(adapters)

    @property
    def adapters(self) -> list[EmbeddingAdapter]:
        """Get list of adapters in ensemble."""
        return self._adapters

    @property
    def strategy(self) -> EnsembleStrategy:
        """Get ensemble strategy."""
        return self._strategy

    def _update_health_weights(self) -> None:
        """Update health weights based on recent performance."""
        for i, adapter in enumerate(self._adapters):
            # Base health from adapter's own assessment
            if adapter.is_healthy():
                base_health = 1.0
            else:
                base_health = 0.5

            # Adjust based on recent success/failure ratio
            total = self._success_counts[i] + self._failure_counts[i]
            if total > 0:
                success_rate = self._success_counts[i] / total
                self._weights[i].health_weight = base_health * success_rate
            else:
                self._weights[i].health_weight = base_health

    def _get_healthy_adapters(self) -> list[tuple[int, EmbeddingAdapter, float]]:
        """Get list of healthy adapters with their weights."""
        self._update_health_weights()

        healthy = []
        for i, (adapter, weight) in enumerate(zip(self._adapters, self._weights)):
            if weight.effective_weight > 0.1:  # Minimum threshold
                healthy.append((i, adapter, weight.effective_weight))

        return healthy

    async def _embed_with_adapter(
        self,
        adapter_idx: int,
        adapter: EmbeddingAdapter,
        query: str,
    ) -> np.ndarray | None:
        """
        Embed with single adapter, handling failures.

        Args:
            adapter_idx: Index in adapter list
            adapter: The adapter to use
            query: Query to embed

        Returns:
            Embedding array or None on failure
        """
        try:
            result = await adapter.embed_query(query)
            self._success_counts[adapter_idx] += 1
            return np.array(result, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Adapter {adapter_idx} failed: {e}")
            self._failure_counts[adapter_idx] += 1
            return None

    async def _embed_batch_with_adapter(
        self,
        adapter_idx: int,
        adapter: EmbeddingAdapter,
        texts: list[str],
    ) -> np.ndarray | None:
        """
        Batch embed with single adapter, handling failures.

        Args:
            adapter_idx: Index in adapter list
            adapter: The adapter to use
            texts: Texts to embed

        Returns:
            Embedding array (N x dim) or None on failure
        """
        try:
            result = await adapter.embed(texts)
            self._success_counts[adapter_idx] += 1
            return np.array(result, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Adapter {adapter_idx} batch failed: {e}")
            self._failure_counts[adapter_idx] += 1
            return None

    def _combine_embeddings(
        self,
        embeddings: list[tuple[int, np.ndarray]],
    ) -> np.ndarray:
        """
        Combine embeddings according to strategy.

        Args:
            embeddings: List of (adapter_idx, embedding) tuples

        Returns:
            Combined embedding
        """
        if not embeddings:
            raise ValueError("No embeddings to combine")

        if self._strategy == EnsembleStrategy.BEST:
            # Use embedding from highest-weighted adapter
            best_idx = max(embeddings, key=lambda x: self._weights[x[0]].effective_weight)
            return best_idx[1]

        if self._strategy == EnsembleStrategy.CONCAT:
            # Concatenate all embeddings
            return np.concatenate([e[1] for e in embeddings])

        if self._strategy == EnsembleStrategy.MEAN:
            # Simple average
            all_embs = np.stack([e[1] for e in embeddings])
            return np.mean(all_embs, axis=0)

        if self._strategy == EnsembleStrategy.WEIGHTED_MEAN:
            # Weighted average by health
            all_embs = []
            all_weights = []
            for idx, emb in embeddings:
                all_embs.append(emb)
                all_weights.append(self._weights[idx].effective_weight)

            all_embs = np.stack(all_embs)
            all_weights = np.array(all_weights, dtype=np.float32)
            all_weights /= all_weights.sum()  # Normalize

            weighted = np.average(all_embs, axis=0, weights=all_weights)
            return weighted

        if self._strategy == EnsembleStrategy.VOTING:
            # For voting, we use weighted mean (voting applies to similarity)
            return self._combine_embeddings(
                [(idx, emb) for idx, emb in embeddings]
            )

        raise ValueError(f"Unknown strategy: {self._strategy}")

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed query using ensemble of adapters.

        Args:
            query: Query text to embed

        Returns:
            Combined embedding vector
        """
        import time

        start = time.perf_counter()

        # Get healthy adapters
        healthy = self._get_healthy_adapters()
        if not healthy:
            raise RuntimeError("No healthy adapters available")

        # Run all healthy adapters in parallel
        tasks = [
            self._embed_with_adapter(idx, adapter, query)
            for idx, adapter, _ in healthy
        ]
        results = await asyncio.gather(*tasks)

        # Collect successful results
        embeddings = [
            (healthy[i][0], result)
            for i, result in enumerate(results)
            if result is not None
        ]

        if not embeddings:
            raise RuntimeError("All adapters failed")

        # Combine embeddings
        combined = self._combine_embeddings(embeddings)

        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 1e-8:
            combined /= norm

        latency = (time.perf_counter() - start) * 1000
        self._record_query(latency, cache_hit=False)

        return combined.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts using ensemble.

        Args:
            texts: List of texts to embed

        Returns:
            List of combined embedding vectors
        """
        import time

        if not texts:
            return []

        start = time.perf_counter()

        # Get healthy adapters
        healthy = self._get_healthy_adapters()
        if not healthy:
            raise RuntimeError("No healthy adapters available")

        # Run all healthy adapters in parallel
        tasks = [
            self._embed_batch_with_adapter(idx, adapter, texts)
            for idx, adapter, _ in healthy
        ]
        results = await asyncio.gather(*tasks)

        # Collect successful results
        valid_results = [
            (healthy[i][0], result)
            for i, result in enumerate(results)
            if result is not None
        ]

        if not valid_results:
            raise RuntimeError("All adapters failed for batch")

        # Combine per-text embeddings
        n_texts = len(texts)
        combined_embeddings = []

        for text_idx in range(n_texts):
            text_embeddings = [
                (adapter_idx, result[text_idx])
                for adapter_idx, result in valid_results
            ]
            combined = self._combine_embeddings(text_embeddings)

            # Normalize
            norm = np.linalg.norm(combined)
            if norm > 1e-8:
                combined /= norm

            combined_embeddings.append(combined.tolist())

        latency = (time.perf_counter() - start) * 1000
        self._record_documents(len(texts), latency)

        return combined_embeddings

    def get_ensemble_stats(self) -> dict:
        """Get detailed ensemble statistics."""
        adapter_stats = []
        for i, adapter in enumerate(self._adapters):
            adapter_stats.append({
                "index": i,
                "backend": adapter.backend.value,
                "dimension": adapter.dimension,
                "healthy": adapter.is_healthy(),
                "base_weight": self._weights[i].base_weight,
                "health_weight": self._weights[i].health_weight,
                "effective_weight": self._weights[i].effective_weight,
                "successes": self._success_counts[i],
                "failures": self._failure_counts[i],
            })

        return {
            "strategy": self._strategy.value,
            "output_dimension": self._dimension,
            "num_adapters": len(self._adapters),
            "num_healthy": sum(1 for a in self._adapters if a.is_healthy()),
            "adapters": adapter_stats,
            "stats": self._stats.to_dict(),
        }

    def reset_health_tracking(self) -> None:
        """Reset health tracking counters."""
        self._failure_counts = [0] * len(self._adapters)
        self._success_counts = [0] * len(self._adapters)
        for weight in self._weights:
            weight.health_weight = 1.0


def create_ensemble_adapter(
    adapters: list[EmbeddingAdapter],
    strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_MEAN,
    weights: list[float] | None = None,
) -> EnsembleEmbeddingAdapter:
    """
    Factory function to create ensemble adapter.

    Args:
        adapters: List of adapters to combine
        strategy: Combination strategy
        weights: Optional per-adapter weights

    Returns:
        Configured ensemble adapter
    """
    return EnsembleEmbeddingAdapter(
        adapters=adapters,
        strategy=strategy,
        weights=weights,
    )


__all__ = [
    "AdapterWeight",
    "EnsembleEmbeddingAdapter",
    "EnsembleStrategy",
    "create_ensemble_adapter",
]
