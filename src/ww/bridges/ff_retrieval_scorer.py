"""
P6.4: Forward-Forward Retrieval Scoring Bridge.

Uses Hinton's Forward-Forward algorithm to score retrieval candidates based on
pattern match confidence. High goodness indicates confident pattern match.

Biological Basis:
- FF goodness maps to hippocampal CA3 pattern completion confidence
- High goodness = strong pattern match (familiar/expected)
- Goodness reflects "resonance" between query and memory
- Integrates with neuromodulator state for context-dependent scoring

Integration Flow:
    Query Embedding + Candidate Embeddings
         |
    [ForwardForwardLayer] → per-candidate goodness
         |
    [FFRetrievalScorer]
         |
    ├── Confidence Scores (goodness-based ranking boost)
    ├── Batch Scoring (efficient multi-candidate)
    └── Outcome Learning (train from retrieval success/failure)
         |
    [Memory Recall Pipeline] → enhanced ranking

References:
- Hinton, G. (2022): The Forward-Forward Algorithm
- Norman & O'Reilly (2003): Hippocampal pattern completion
- McClelland et al. (1995): Why there are complementary learning systems
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class ForwardForwardLayerProtocol(Protocol):
    """Protocol for ForwardForward layer interface."""

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass computing activations."""
        ...

    def compute_goodness(self, h: np.ndarray) -> float:
        """Compute goodness (sum of squared activations)."""
        ...

    def learn_positive(self, x: np.ndarray, h: np.ndarray) -> dict[str, Any]:
        """Learn from positive sample (increase goodness)."""
        ...

    def learn_negative(self, x: np.ndarray, h: np.ndarray) -> dict[str, Any]:
        """Learn from negative sample (decrease goodness)."""
        ...

    @property
    def config(self) -> Any:
        """Access layer config."""
        ...

    @property
    def state(self) -> Any:
        """Access layer state."""
        ...


@dataclass
class FFRetrievalConfig:
    """Configuration for FF-Retrieval scoring.

    Attributes:
        normalize_goodness: Normalize by layer size for comparability
        confidence_scale: Scale factor for confidence scores (0-1 range)
        min_confidence: Floor for confidence scores
        max_boost: Maximum retrieval score boost from FF
        learn_from_outcomes: Enable learning from retrieval outcomes
        positive_outcome_threshold: Outcome score above which to treat as positive
        learning_rate_multiplier: Multiplier for outcome-based learning
        cache_size: Maximum cached goodness values
    """
    normalize_goodness: bool = True
    confidence_scale: float = 0.5  # Sigmoid scaling
    min_confidence: float = 0.0
    max_boost: float = 0.3  # Cap on retrieval boost
    learn_from_outcomes: bool = True
    positive_outcome_threshold: float = 0.6
    learning_rate_multiplier: float = 1.0
    cache_size: int = 1000


@dataclass
class FFRetrievalState:
    """Current state of the FF-Retrieval scorer.

    Attributes:
        n_candidates_scored: Total candidates scored
        n_positive_outcomes: Positive outcome learning events
        n_negative_outcomes: Negative outcome learning events
        mean_goodness: Running mean of goodness values
        mean_confidence: Running mean of confidence scores
        last_batch_size: Size of last batch scored
        timestamp: Last update time
    """
    n_candidates_scored: int = 0
    n_positive_outcomes: int = 0
    n_negative_outcomes: int = 0
    mean_goodness: float = 0.0
    mean_confidence: float = 0.0
    last_batch_size: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievalScore:
    """FF-based retrieval score for a candidate.

    Attributes:
        memory_id: ID of the scored memory
        goodness: Raw FF goodness value
        normalized_goodness: Goodness / layer_size (if enabled)
        confidence: Sigmoid-transformed confidence [0, 1]
        boost: Actual boost to apply to retrieval score
        is_confident: Whether this is a confident match
    """
    memory_id: str
    goodness: float
    normalized_goodness: float
    confidence: float
    boost: float
    is_confident: bool


class FFRetrievalScorer:
    """
    Bridge connecting ForwardForward to retrieval ranking.

    Implements P6.4: FF-based confidence scoring for retrieval.
    Uses FF goodness to boost candidates that strongly match learned patterns.

    Confident matches (high goodness):
    - Strong pattern completion in FF layer
    - Higher retrieval score boost
    - Indicates familiar/reliable memory

    Uncertain matches (low goodness):
    - Weak pattern completion
    - Minimal or no boost
    - Candidate may be novel or mismatched

    Example:
        ```python
        from ww.nca.forward_forward import ForwardForwardLayer, ForwardForwardConfig
        from ww.bridges import FFRetrievalScorer

        # Create FF layer
        ff_config = ForwardForwardConfig(input_dim=1024, hidden_dim=512)
        ff_layer = ForwardForwardLayer(ff_config)

        # Create scorer
        scorer = FFRetrievalScorer(ff_layer=ff_layer)

        # Score candidates
        scores = scorer.score_candidates(
            query_embedding=query_emb,
            candidate_embeddings=candidates,
            candidate_ids=["mem1", "mem2", "mem3"]
        )

        # Apply boosts to base retrieval scores
        for score in scores:
            base_scores[score.memory_id] += score.boost
        ```
    """

    def __init__(
        self,
        ff_layer: ForwardForwardLayerProtocol | None = None,
        config: FFRetrievalConfig | None = None
    ):
        """
        Initialize FF-Retrieval scorer.

        Args:
            ff_layer: ForwardForwardLayer instance
            config: Scorer configuration
        """
        self.ff_layer = ff_layer
        self.config = config or FFRetrievalConfig()
        self.state = FFRetrievalState()

        # Simple LRU cache for goodness values
        self._goodness_cache: dict[str, tuple[float, datetime]] = {}

        # Running statistics for adaptive thresholding
        self._goodness_history: list[float] = []
        self._max_history = 1000

        logger.info(
            f"P6.4: FFRetrievalScorer initialized "
            f"(max_boost={self.config.max_boost}, "
            f"learn_from_outcomes={self.config.learn_from_outcomes})"
        )

    def set_ff_layer(self, ff_layer: ForwardForwardLayerProtocol) -> None:
        """Set or update the FF layer.

        Args:
            ff_layer: ForwardForwardLayer instance
        """
        self.ff_layer = ff_layer
        logger.info("P6.4: FF layer updated")

    def _compute_goodness(self, embedding: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Compute goodness for an embedding.

        Args:
            embedding: Input embedding

        Returns:
            Tuple of (goodness, activations)
        """
        if self.ff_layer is not None:
            activations = self.ff_layer.forward(embedding, training=False)
            goodness = self.ff_layer.compute_goodness(activations)
        else:
            # Fallback: use embedding norm as proxy
            activations = embedding
            goodness = float(np.sum(embedding ** 2))

        return goodness, activations

    def _normalize_goodness(self, goodness: float) -> float:
        """Normalize goodness by layer size if configured."""
        if not self.config.normalize_goodness or self.ff_layer is None:
            return goodness

        try:
            layer_size = self.ff_layer.config.hidden_dim
            return goodness / layer_size
        except AttributeError:
            return goodness

    def _goodness_to_confidence(self, goodness: float) -> float:
        """
        Convert goodness to confidence score [0, 1].

        Uses sigmoid transformation centered at the mean goodness.

        Args:
            goodness: Normalized goodness value

        Returns:
            Confidence score in [0, 1]
        """
        # Use running mean as center for sigmoid
        if self._goodness_history:
            center = np.mean(self._goodness_history)
        else:
            center = 1.0  # Default center

        # Sigmoid transformation
        x = (goodness - center) * self.config.confidence_scale
        confidence = 1.0 / (1.0 + np.exp(-x))

        return float(np.clip(confidence, self.config.min_confidence, 1.0))

    def _confidence_to_boost(self, confidence: float) -> float:
        """
        Convert confidence to retrieval score boost.

        Only high-confidence matches get substantial boost.

        Args:
            confidence: Confidence score [0, 1]

        Returns:
            Boost value [0, max_boost]
        """
        # Threshold at 0.5 - below that, no boost
        if confidence < 0.5:
            return 0.0

        # Scale [0.5, 1.0] to [0, max_boost]
        scaled = (confidence - 0.5) * 2.0
        return float(scaled * self.config.max_boost)

    def score_candidate(
        self,
        embedding: np.ndarray,
        memory_id: str,
        use_cache: bool = True
    ) -> RetrievalScore:
        """
        Score a single retrieval candidate.

        Args:
            embedding: Candidate memory embedding
            memory_id: Memory identifier
            use_cache: Whether to use cached goodness

        Returns:
            RetrievalScore with goodness, confidence, and boost
        """
        # Check cache
        if use_cache and memory_id in self._goodness_cache:
            cached_goodness, cache_time = self._goodness_cache[memory_id]
            # Cache valid for 1 hour
            if (datetime.now() - cache_time).total_seconds() < 3600:
                goodness = cached_goodness
                activations = None
            else:
                goodness, activations = self._compute_goodness(embedding)
        else:
            goodness, activations = self._compute_goodness(embedding)

        # Update cache
        if use_cache:
            self._goodness_cache[memory_id] = (goodness, datetime.now())
            self._cleanup_cache()

        # Normalize
        normalized = self._normalize_goodness(goodness)

        # Track history for adaptive thresholding
        self._goodness_history.append(normalized)
        if len(self._goodness_history) > self._max_history:
            self._goodness_history = self._goodness_history[-self._max_history:]

        # Convert to confidence and boost
        confidence = self._goodness_to_confidence(normalized)
        boost = self._confidence_to_boost(confidence)

        # Update state
        self.state.n_candidates_scored += 1
        alpha = 0.01  # Running average decay
        self.state.mean_goodness = (
            (1 - alpha) * self.state.mean_goodness + alpha * normalized
        )
        self.state.mean_confidence = (
            (1 - alpha) * self.state.mean_confidence + alpha * confidence
        )
        self.state.timestamp = datetime.now()

        return RetrievalScore(
            memory_id=memory_id,
            goodness=goodness,
            normalized_goodness=normalized,
            confidence=confidence,
            boost=boost,
            is_confident=confidence > 0.6
        )

    def score_candidates(
        self,
        candidate_embeddings: list[np.ndarray],
        candidate_ids: list[str],
        query_embedding: np.ndarray | None = None,
        use_cache: bool = True
    ) -> list[RetrievalScore]:
        """
        Score multiple retrieval candidates.

        Args:
            candidate_embeddings: List of candidate memory embeddings
            candidate_ids: List of memory identifiers
            query_embedding: Optional query (for future query-aware scoring)
            use_cache: Whether to use cached goodness

        Returns:
            List of RetrievalScore objects, ordered by input order
        """
        if len(candidate_embeddings) != len(candidate_ids):
            raise ValueError("Embeddings and IDs must have same length")

        scores = []
        for embedding, memory_id in zip(candidate_embeddings, candidate_ids):
            score = self.score_candidate(
                embedding=embedding,
                memory_id=memory_id,
                use_cache=use_cache
            )
            scores.append(score)

        self.state.last_batch_size = len(scores)

        logger.debug(
            f"P6.4: Scored {len(scores)} candidates, "
            f"mean_confidence={np.mean([s.confidence for s in scores]):.3f}"
        )

        return scores

    def learn_from_outcome(
        self,
        embeddings: list[np.ndarray],
        memory_ids: list[str],
        outcome_score: float,
        learning_rate: float | None = None
    ) -> dict[str, Any]:
        """
        Train FF layer from retrieval outcome.

        Positive outcomes (success) → treat as positive samples (increase goodness)
        Negative outcomes (failure) → treat as negative samples (decrease goodness)

        Args:
            embeddings: Memory embeddings that were retrieved
            memory_ids: Memory identifiers
            outcome_score: Task outcome [0, 1] (1 = success)
            learning_rate: Optional learning rate override

        Returns:
            Dictionary with learning statistics
        """
        if not self.config.learn_from_outcomes or self.ff_layer is None:
            return {"skipped": True, "reason": "learning disabled or no FF layer"}

        stats = {
            "n_samples": len(embeddings),
            "outcome_score": outcome_score,
            "positive_learning": 0,
            "negative_learning": 0,
            "mean_gradient_norm": 0.0
        }

        try:
            is_positive = outcome_score >= self.config.positive_outcome_threshold
            gradient_norms = []

            for embedding in embeddings:
                # Forward pass
                activations = self.ff_layer.forward(embedding, training=True)

                if is_positive:
                    # Positive outcome → increase goodness for these patterns
                    result = self.ff_layer.learn_positive(embedding, activations)
                    stats["positive_learning"] += 1
                    self.state.n_positive_outcomes += 1
                else:
                    # Negative outcome → decrease goodness (these patterns led to failure)
                    result = self.ff_layer.learn_negative(embedding, activations)
                    stats["negative_learning"] += 1
                    self.state.n_negative_outcomes += 1

                if "gradient_norm" in result:
                    gradient_norms.append(result["gradient_norm"])

            if gradient_norms:
                stats["mean_gradient_norm"] = float(np.mean(gradient_norms))

            # Invalidate cache for updated memories
            for memory_id in memory_ids:
                self._goodness_cache.pop(memory_id, None)

            logger.info(
                f"P6.4: FF learning from {'positive' if is_positive else 'negative'} "
                f"outcome ({len(embeddings)} samples, score={outcome_score:.2f})"
            )

        except Exception as e:
            logger.warning(f"P6.4: FF outcome learning failed: {e}")
            stats["error"] = str(e)

        return stats

    def _cleanup_cache(self) -> None:
        """Remove old cache entries if over limit."""
        if len(self._goodness_cache) <= self.config.cache_size:
            return

        # Remove oldest entries
        sorted_entries = sorted(
            self._goodness_cache.items(),
            key=lambda x: x[1][1],  # Sort by timestamp
            reverse=True
        )
        self._goodness_cache = dict(sorted_entries[:self.config.cache_size])

    def get_stats(self) -> dict[str, Any]:
        """Get current scorer statistics.

        Returns:
            Dictionary with state and performance metrics
        """
        return {
            "n_candidates_scored": self.state.n_candidates_scored,
            "n_positive_outcomes": self.state.n_positive_outcomes,
            "n_negative_outcomes": self.state.n_negative_outcomes,
            "mean_goodness": self.state.mean_goodness,
            "mean_confidence": self.state.mean_confidence,
            "cache_size": len(self._goodness_cache),
            "history_size": len(self._goodness_history),
            "last_batch_size": self.state.last_batch_size,
        }

    def clear_cache(self) -> None:
        """Clear the goodness cache."""
        self._goodness_cache.clear()
        logger.info("P6.4: Goodness cache cleared")

    def get_adaptive_threshold(self) -> float:
        """
        Get adaptive goodness threshold based on recent history.

        Uses mean + 1 std as threshold for "high confidence".

        Returns:
            Adaptive threshold value
        """
        if len(self._goodness_history) < 10:
            return 1.0  # Default

        mean = np.mean(self._goodness_history)
        std = np.std(self._goodness_history)

        return float(mean + std)
