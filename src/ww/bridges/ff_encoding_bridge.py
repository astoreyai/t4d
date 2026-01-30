"""
P6.3: Bridge between ForwardForward and Encoding Pipeline.

Integrates Hinton's Forward-Forward algorithm with memory encoding to provide
local learning signals and novelty detection without backpropagation.

Biological Basis:
- FF goodness maps to local metabolic activity / neural coherence
- High goodness = familiar pattern (expected)
- Low goodness = novel pattern (unexpected) → triggers enhanced encoding
- Maps to hippocampal novelty detection and encoding prioritization

Integration Flow:
    Embedding (from BGE-M3)
         |
    [ForwardForwardLayer] → goodness, is_positive, confidence
         |
    [FFEncodingBridge]
         |
    ├── Novelty Detection (low goodness = novel)
    ├── Encoding Priority (novel → stronger encoding)
    └── Local Learning (update FF weights)
         |
    [SparseEncoder / Memory System]

References:
- Hinton, G. (2022): The Forward-Forward Algorithm
- Lisman & Grace (2005): Hippocampal-VTA loop and novelty
- Kumaran & Maguire (2007): Hippocampal novelty detection
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

    def positive_phase(
        self,
        x: np.ndarray,
        learning_rate: float | None = None
    ) -> float:
        """Positive phase: increase goodness for real data."""
        ...

    def negative_phase(
        self,
        x: np.ndarray,
        learning_rate: float | None = None
    ) -> float:
        """Negative phase: decrease goodness for fake data."""
        ...

    @property
    def state(self) -> Any:
        """Access layer state."""
        ...


@dataclass
class FFEncodingConfig:
    """Configuration for FF-Encoding bridge.

    Attributes:
        goodness_threshold: Threshold for novel vs familiar
        novelty_boost: How much to boost encoding for novel patterns
        familiar_dampening: How much to dampen encoding for familiar patterns
        online_learning: Enable FF layer learning during encoding
        generate_negatives: Generate negative samples for learning
        negative_noise_scale: Noise scale for negative generation
        max_history: Maximum samples in history
    """
    goodness_threshold: float = 2.0
    novelty_boost: float = 0.5  # Boost for novel patterns
    familiar_dampening: float = 0.2  # Dampening for very familiar
    online_learning: bool = True
    generate_negatives: bool = True
    negative_noise_scale: float = 0.3
    max_history: int = 1000


@dataclass
class FFEncodingState:
    """Current state of the FF-Encoding bridge.

    Attributes:
        n_embeddings_processed: Total embeddings processed
        n_novel_detected: Number of novel patterns detected
        n_familiar_detected: Number of familiar patterns detected
        mean_goodness: Running mean of goodness values
        mean_novelty_boost: Running mean of novelty boosts applied
        novelty_ratio: Ratio of novel to total patterns
        last_goodness: Most recent goodness value
        timestamp: Last update time
    """
    n_embeddings_processed: int = 0
    n_novel_detected: int = 0
    n_familiar_detected: int = 0
    mean_goodness: float = 0.0
    mean_novelty_boost: float = 0.0
    novelty_ratio: float = 0.5
    last_goodness: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EncodingGuidance:
    """Guidance from FF layer for encoding pipeline.

    Attributes:
        is_novel: Whether this pattern is novel
        goodness: Raw goodness value
        encoding_multiplier: Multiplier for encoding strength
        priority: Encoding priority (higher = more important)
        confidence: FF layer confidence in classification
        enhanced_embedding: Optional FF-enhanced embedding
    """
    is_novel: bool
    goodness: float
    encoding_multiplier: float
    priority: float
    confidence: float
    enhanced_embedding: np.ndarray | None = None


class FFEncodingBridge:
    """
    Bridge connecting ForwardForward to encoding pipeline.

    Implements P6.3: FF-based novelty detection for encoding prioritization.
    Uses FF goodness to detect novel patterns and modulate encoding strength.

    Novel patterns (low goodness):
    - Haven't been seen before
    - Trigger enhanced encoding (stronger memory traces)
    - Prioritized for consolidation

    Familiar patterns (high goodness):
    - Match existing representations
    - Standard encoding strength
    - Lower consolidation priority

    Example:
        ```python
        from ww.nca.forward_forward import ForwardForwardLayer, ForwardForwardConfig
        from ww.bridges import FFEncodingBridge

        # Create FF layer
        ff_config = ForwardForwardConfig(input_dim=1024, hidden_dim=512)
        ff_layer = ForwardForwardLayer(ff_config)

        # Create bridge
        bridge = FFEncodingBridge(ff_layer=ff_layer)

        # Process embedding
        guidance = bridge.process(embedding)
        if guidance.is_novel:
            logger.info(f"Novel pattern! Priority: {guidance.priority}")
            # Enhanced encoding
            encoded = encoder(embedding * guidance.encoding_multiplier)
        ```
    """

    def __init__(
        self,
        ff_layer: ForwardForwardLayerProtocol | None = None,
        config: FFEncodingConfig | None = None
    ):
        """
        Initialize FF-Encoding bridge.

        Args:
            ff_layer: ForwardForwardLayer instance
            config: Bridge configuration
        """
        self.ff_layer = ff_layer
        self.config = config or FFEncodingConfig()
        self.state = FFEncodingState()

        # History for adaptive thresholding
        self._goodness_history: list[float] = []

        logger.info(
            f"P6.3: FFEncodingBridge initialized "
            f"(threshold={self.config.goodness_threshold}, "
            f"novelty_boost={self.config.novelty_boost})"
        )

    def set_ff_layer(self, ff_layer: ForwardForwardLayerProtocol) -> None:
        """Set or update the FF layer.

        Args:
            ff_layer: ForwardForwardLayer instance
        """
        self.ff_layer = ff_layer
        logger.info("P6.3: FF layer updated")

    def _compute_goodness(self, embedding: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Compute goodness for an embedding.

        Args:
            embedding: Input embedding

        Returns:
            Tuple of (goodness, activations)
        """
        if self.ff_layer is not None:
            # Use FF layer
            activations = self.ff_layer.forward(embedding, training=False)
            goodness = self.ff_layer.compute_goodness(activations)
        else:
            # Fallback: use embedding norm as proxy
            activations = embedding
            goodness = float(np.sum(embedding ** 2) / len(embedding))

        return goodness, activations

    def _generate_negative(self, embedding: np.ndarray) -> np.ndarray:
        """
        Generate negative sample for FF learning.

        Args:
            embedding: Original embedding

        Returns:
            Corrupted/negative embedding
        """
        noise = np.random.randn(*embedding.shape) * self.config.negative_noise_scale
        negative = embedding + noise
        # Normalize to maintain embedding properties
        negative = negative / (np.linalg.norm(negative) + 1e-8)
        return negative

    def _online_learn(self, embedding: np.ndarray) -> None:
        """
        Perform online FF learning.

        Args:
            embedding: Input embedding (treated as positive)
        """
        if self.ff_layer is None or not self.config.online_learning:
            return

        try:
            # Positive phase: this is real data
            self.ff_layer.positive_phase(embedding)

            # Negative phase: generate and learn from corrupted data
            if self.config.generate_negatives:
                negative = self._generate_negative(embedding)
                self.ff_layer.negative_phase(negative)
        except AttributeError:
            # Layer might not have these methods
            pass

    def _compute_encoding_multiplier(self, goodness: float) -> float:
        """
        Compute encoding strength multiplier based on goodness.

        Low goodness (novel) → higher multiplier
        High goodness (familiar) → lower multiplier

        Args:
            goodness: FF goodness value

        Returns:
            Encoding multiplier (typically 0.8 - 1.5)
        """
        # Distance from threshold (negative = novel, positive = familiar)
        distance = goodness - self.config.goodness_threshold

        if distance < 0:
            # Novel: boost encoding
            # More novel (more negative) = higher boost
            boost = 1.0 + self.config.novelty_boost * min(-distance / 2.0, 1.0)
        else:
            # Familiar: slight dampening
            dampening = 1.0 - self.config.familiar_dampening * min(distance / 2.0, 0.5)
            boost = max(dampening, 0.8)  # Don't dampen too much

        return float(np.clip(boost, 0.8, 1.0 + self.config.novelty_boost))

    def _compute_priority(self, goodness: float, confidence: float) -> float:
        """
        Compute encoding priority.

        Novel patterns with high confidence get highest priority.

        Args:
            goodness: FF goodness value
            confidence: Classification confidence

        Returns:
            Priority score (0-1)
        """
        # Distance from threshold
        distance = abs(goodness - self.config.goodness_threshold)

        # Novelty contribution (novel = high priority)
        novelty_score = 1.0 if goodness < self.config.goodness_threshold else 0.5

        # Combine with confidence
        priority = novelty_score * (0.5 + 0.5 * min(confidence, 1.0))

        # Distance contribution (clear classifications get higher priority)
        priority *= (0.5 + 0.5 * min(distance / 2.0, 1.0))

        return float(np.clip(priority, 0.0, 1.0))

    def process(self, embedding: np.ndarray) -> EncodingGuidance:
        """
        Process embedding through FF layer for encoding guidance.

        Args:
            embedding: Input embedding vector

        Returns:
            EncodingGuidance with novelty detection and encoding modulation
        """
        # Compute goodness
        goodness, activations = self._compute_goodness(embedding)

        # Determine if novel
        is_novel = goodness < self.config.goodness_threshold

        # Compute confidence (distance from threshold)
        confidence = abs(goodness - self.config.goodness_threshold)

        # Compute encoding multiplier and priority
        encoding_multiplier = self._compute_encoding_multiplier(goodness)
        priority = self._compute_priority(goodness, confidence)

        # Online learning
        if self.config.online_learning:
            self._online_learn(embedding)

        # Update state
        self.state.n_embeddings_processed += 1
        if is_novel:
            self.state.n_novel_detected += 1
        else:
            self.state.n_familiar_detected += 1

        self.state.last_goodness = goodness

        # Update running means
        alpha = 0.01
        self.state.mean_goodness = alpha * goodness + (1 - alpha) * self.state.mean_goodness
        self.state.mean_novelty_boost = (
            alpha * (encoding_multiplier - 1.0) +
            (1 - alpha) * self.state.mean_novelty_boost
        )

        # Update novelty ratio
        total = self.state.n_novel_detected + self.state.n_familiar_detected
        if total > 0:
            self.state.novelty_ratio = self.state.n_novel_detected / total

        self.state.timestamp = datetime.now()

        # Track goodness history (for adaptive thresholding)
        self._goodness_history.append(goodness)
        if len(self._goodness_history) > self.config.max_history:
            self._goodness_history.pop(0)

        logger.debug(
            f"P6.3: FF encoding - goodness={goodness:.3f}, "
            f"novel={is_novel}, priority={priority:.3f}"
        )

        return EncodingGuidance(
            is_novel=is_novel,
            goodness=goodness,
            encoding_multiplier=encoding_multiplier,
            priority=priority,
            confidence=confidence,
            enhanced_embedding=activations if activations is not embedding else None
        )

    def process_batch(
        self,
        embeddings: list[np.ndarray]
    ) -> list[EncodingGuidance]:
        """
        Process batch of embeddings.

        Args:
            embeddings: List of embedding vectors

        Returns:
            List of EncodingGuidance for each embedding
        """
        return [self.process(emb) for emb in embeddings]

    def get_adaptive_threshold(self) -> float:
        """
        Get adaptive threshold based on recent goodness distribution.

        Adjusts threshold to maintain desired novelty ratio.

        Returns:
            Suggested threshold value
        """
        if len(self._goodness_history) < 100:
            return self.config.goodness_threshold

        # Use median as adaptive threshold
        sorted_history = sorted(self._goodness_history)
        target_percentile = 0.3  # 30% novel
        idx = int(len(sorted_history) * (1 - target_percentile))

        return sorted_history[idx]

    def update_threshold(self, new_threshold: float) -> None:
        """
        Update the goodness threshold.

        Args:
            new_threshold: New threshold value
        """
        old_threshold = self.config.goodness_threshold
        self.config.goodness_threshold = new_threshold
        logger.info(
            f"P6.3: Updated threshold {old_threshold:.3f} → {new_threshold:.3f}"
        )

    def adapt_threshold(self, target_novelty_ratio: float = 0.3) -> None:
        """
        Automatically adapt threshold to achieve target novelty ratio.

        Args:
            target_novelty_ratio: Desired ratio of novel detections
        """
        adaptive_threshold = self.get_adaptive_threshold()
        self.update_threshold(adaptive_threshold)

    def get_statistics(self) -> dict[str, Any]:
        """Get bridge statistics."""
        return {
            "n_embeddings_processed": self.state.n_embeddings_processed,
            "n_novel_detected": self.state.n_novel_detected,
            "n_familiar_detected": self.state.n_familiar_detected,
            "novelty_ratio": self.state.novelty_ratio,
            "mean_goodness": self.state.mean_goodness,
            "mean_novelty_boost": self.state.mean_novelty_boost,
            "current_threshold": self.config.goodness_threshold,
            "adaptive_threshold": self.get_adaptive_threshold(),
            "config": {
                "novelty_boost": self.config.novelty_boost,
                "familiar_dampening": self.config.familiar_dampening,
                "online_learning": self.config.online_learning,
            }
        }

    def reset_statistics(self) -> None:
        """Reset bridge statistics."""
        self.state = FFEncodingState()
        self._goodness_history.clear()
        logger.info("P6.3: FFEncodingBridge statistics reset")


def create_ff_encoding_bridge(
    ff_layer: ForwardForwardLayerProtocol | None = None,
    goodness_threshold: float = 2.0,
    novelty_boost: float = 0.5,
    online_learning: bool = True
) -> FFEncodingBridge:
    """
    Factory function for FF-Encoding bridge.

    Args:
        ff_layer: Optional ForwardForwardLayer instance
        goodness_threshold: Threshold for novelty detection
        novelty_boost: Boost factor for novel patterns
        online_learning: Enable online FF learning

    Returns:
        Configured FFEncodingBridge
    """
    config = FFEncodingConfig(
        goodness_threshold=goodness_threshold,
        novelty_boost=novelty_boost,
        online_learning=online_learning
    )
    return FFEncodingBridge(
        ff_layer=ff_layer,
        config=config
    )
