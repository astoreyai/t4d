"""
Uncertainty-Aware Memory Storage (W2-01).

MC Dropout-based uncertainty estimation for memory embeddings
following Friston's Free Energy Principle.

Evidence Base: Friston (2010) "The free-energy principle: a unified brain theory?"

Key Insight:
    Uncertainty quantification enables:
    1. Confidence-weighted retrieval
    2. Selective consolidation (high uncertainty → needs more examples)
    3. Active learning (query for items with high uncertainty)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty estimation.

    Attributes:
        n_samples: Number of MC dropout samples (default 10).
        use_diagonal_covariance: Use diagonal covariance for efficiency (default True).
        method: Uncertainty estimation method ("mc_dropout" or "ensemble").
    """

    n_samples: int = 10
    use_diagonal_covariance: bool = True
    method: str = "mc_dropout"


@dataclass
class UncertaintyAwareItem:
    """Memory item with uncertainty quantification.

    Stores both the embedding mean and covariance (diagonal for efficiency).

    Attributes:
        id: Unique identifier.
        vector_mean: Embedding mean (1024-dim).
        vector_cov: Covariance (1024-dim diagonal or 1024x1024 full).
        content: Text content.
        kappa: Consolidation level [0, 1].
        importance: Importance weight.
    """

    id: UUID
    vector_mean: np.ndarray
    vector_cov: np.ndarray
    content: str
    kappa: float
    importance: float

    @property
    def uncertainty(self) -> float:
        """Scalar uncertainty measure.

        For diagonal covariance: sum of variances.
        For full covariance: trace of covariance matrix.
        """
        if self.vector_cov.ndim == 1:
            return float(np.sum(self.vector_cov))
        return float(np.trace(self.vector_cov))

    @property
    def confidence(self) -> float:
        """Confidence score = 1 / (1 + uncertainty).

        Returns value in (0, 1] where 1 = maximum confidence.
        """
        return 1.0 / (1.0 + self.uncertainty)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "vector_mean": self.vector_mean.tolist(),
            "vector_cov": self.vector_cov.tolist(),
            "content": self.content,
            "kappa": self.kappa,
            "importance": self.importance,
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
        }


class UncertaintyEstimator:
    """Estimate embedding uncertainty via MC Dropout.

    Uses Monte Carlo Dropout: sample embeddings with dropout enabled,
    compute mean and variance across samples.

    Example:
        >>> estimator = UncertaintyEstimator(embedding_model)
        >>> mean, var = estimator.embed_with_uncertainty("Hello world")
        >>> print(f"Uncertainty: {np.sum(var):.4f}")
    """

    def __init__(
        self,
        embedding_model: Any,
        method: str = "mc_dropout",
        n_samples: int = 10,
    ):
        """Initialize uncertainty estimator.

        Args:
            embedding_model: Model with encode(), train(), eval() methods.
            method: "mc_dropout" or "ensemble".
            n_samples: Number of samples for MC Dropout.
        """
        self.model = embedding_model
        self.method = method
        self.n_samples = n_samples

    def embed_with_uncertainty(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute embedding mean and variance.

        Uses MC Dropout: sample embeddings with dropout enabled,
        compute mean and variance across samples.

        Args:
            text: Text to embed.

        Returns:
            Tuple of (mean, variance) where both are 1D arrays.

        Raises:
            ValueError: If method is unknown.
        """
        if self.method == "mc_dropout":
            return self._mc_dropout_estimate(text)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _mc_dropout_estimate(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Estimate uncertainty via MC Dropout."""
        # Enable dropout
        self.model.train()

        samples = []
        for _ in range(self.n_samples):
            emb = self.model.encode(text)
            samples.append(emb)

        # Disable dropout
        self.model.eval()

        samples = np.stack(samples)

        # Compute mean and variance
        mean = np.mean(samples, axis=0)
        var = np.var(samples, axis=0)

        return mean, var


class UncertaintyAwareSearch:
    """Search with uncertainty consideration.

    Combines similarity score with confidence for ranking.
    """

    def __init__(self, engine: Any):
        """Initialize search.

        Args:
            engine: Storage engine with search() method returning UncertaintyAwareItem.
        """
        self.engine = engine

    def search_with_confidence(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        confidence_weight: float = 0.0,
    ) -> list[UncertaintyAwareItem]:
        """Search with confidence-weighted ranking.

        Args:
            query_vector: Query embedding.
            k: Number of results.
            confidence_weight: Weight for confidence in ranking (0-1).
                0 = pure similarity, 1 = only confidence.

        Returns:
            List of items with confidence scores.
        """
        # Get base results
        results = self.engine.search(query_vector, k)

        if confidence_weight > 0:
            # Re-rank by combined score
            # Assume engine returns items with similarity in some property
            # For now, just sort by confidence
            results = sorted(results, key=lambda x: x.confidence, reverse=True)

        return results
