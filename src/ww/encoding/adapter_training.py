"""
Adapter Training System for World Weaver.

Phase 2C: Training pipeline for OnlineEmbeddingAdapter.

This module connects the feedback collection system with the adapter training,
enabling continuous learning from retrieval outcomes:

1. AdapterTrainer: Orchestrates training using feedback and embeddings
2. Training batches constructed from RetrievalFeedbackCollector
3. Supports async epoch training with configurable scheduling

Biological Mapping:
- Feedback collector = hippocampal replay buffer
- Adapter trainer = sleep consolidation learning system
- Training epoch = offline replay/consolidation phase
- Continuous training = online plasticity during waking

Integration:
    ```python
    from ww.encoding.adapter_training import AdapterTrainer, TrainingConfig
    from ww.encoding.online_adapter import OnlineEmbeddingAdapter
    from ww.learning.retrieval_feedback import RetrievalFeedbackCollector
    from ww.embedding.adapter import EmbeddingAdapter

    # Setup components
    adapter = OnlineEmbeddingAdapter()
    feedback = RetrievalFeedbackCollector(db_path="feedback.db")
    embedder = get_embedder()

    # Create trainer
    trainer = AdapterTrainer(adapter, feedback, embedder)

    # Train epoch (async)
    stats = await trainer.train_epoch(batch_size=32)
    ```
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from ww.encoding.online_adapter import OnlineEmbeddingAdapter

if TYPE_CHECKING:
    from ww.embedding.adapter import EmbeddingAdapter
    from ww.learning.retrieval_feedback import RetrievalFeedbackCollector, RetrievalFeedback

logger = logging.getLogger(__name__)

# Limits
MAX_EPOCH_BATCHES = 100
MAX_EMBEDDING_CACHE = 10000


@dataclass
class TrainingConfig:
    """
    Configuration for adapter training.

    Attributes:
        batch_size: Number of feedback samples per training batch
        min_confidence: Minimum confidence threshold for feedback inclusion
        relevance_threshold: Threshold for positive/negative classification
        max_batches_per_epoch: Maximum batches per epoch
        feedback_lookback_hours: How far back to look for feedback
        embedding_cache_size: Size of embedding cache
        warmup_batches: Number of batches before counting loss
        early_stop_patience: Batches without improvement before stopping
        early_stop_min_delta: Minimum loss improvement to reset patience
    """
    batch_size: int = 32
    min_confidence: float = 0.5
    relevance_threshold: float = 0.5
    max_batches_per_epoch: int = MAX_EPOCH_BATCHES
    feedback_lookback_hours: int = 24
    embedding_cache_size: int = MAX_EMBEDDING_CACHE
    warmup_batches: int = 5
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.001


@dataclass
class TrainingStats:
    """Statistics from a training run."""
    epoch: int = 0
    batches_trained: int = 0
    total_samples: int = 0
    mean_loss: float = 0.0
    final_loss: float = 0.0
    best_loss: float = float("inf")
    positive_samples: int = 0
    negative_samples: int = 0
    training_time_seconds: float = 0.0
    stopped_early: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "epoch": self.epoch,
            "batches_trained": self.batches_trained,
            "total_samples": self.total_samples,
            "mean_loss": self.mean_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "positive_samples": self.positive_samples,
            "negative_samples": self.negative_samples,
            "training_time_seconds": self.training_time_seconds,
            "stopped_early": self.stopped_early,
            "timestamp": self.timestamp.isoformat(),
        }


class EmbeddingCache:
    """
    LRU cache for embeddings to avoid redundant computation.

    Stores result_id -> embedding mappings with bounded size.
    """

    def __init__(self, max_size: int = MAX_EMBEDDING_CACHE):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.max_size = max_size
        self._cache: dict[str, tuple[np.ndarray, datetime]] = {}

    def get(self, result_id: str) -> np.ndarray | None:
        """Get cached embedding if available."""
        if result_id in self._cache:
            emb, _ = self._cache[result_id]
            # Update access time
            self._cache[result_id] = (emb, datetime.now())
            return emb
        return None

    def put(self, result_id: str, embedding: np.ndarray) -> None:
        """Cache an embedding."""
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        self._cache[result_id] = (embedding.copy(), datetime.now())

    def _evict_oldest(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        oldest = min(self._cache.keys(), key=lambda k: self._cache[k][1])
        del self._cache[oldest]

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)


class AdapterTrainer:
    """
    Train adapter using feedback collector.

    Orchestrates the training pipeline:
    1. Fetch feedback batches from collector
    2. Retrieve embeddings for query/result IDs
    3. Construct contrastive training batches
    4. Update adapter weights
    5. Track training statistics
    """

    def __init__(
        self,
        adapter: OnlineEmbeddingAdapter,
        feedback: RetrievalFeedbackCollector,
        embedder: EmbeddingAdapter,
        config: TrainingConfig | None = None,
    ):
        """
        Initialize adapter trainer.

        Args:
            adapter: The OnlineEmbeddingAdapter to train
            feedback: RetrievalFeedbackCollector with training data
            embedder: EmbeddingAdapter for computing embeddings
            config: Training configuration
        """
        self.adapter = adapter
        self.feedback = feedback
        self.embedder = embedder
        self.config = config or TrainingConfig()

        # Embedding cache
        self._cache = EmbeddingCache(self.config.embedding_cache_size)

        # Training history
        self._epoch_count = 0
        self._total_batches = 0
        self._training_history: list[TrainingStats] = []

        logger.info(
            f"AdapterTrainer initialized: "
            f"batch_size={self.config.batch_size}, "
            f"relevance_threshold={self.config.relevance_threshold}"
        )

    async def train_epoch(
        self,
        batch_size: int | None = None,
        max_batches: int | None = None,
    ) -> TrainingStats:
        """
        Train one epoch from collected feedback.

        Fetches feedback batches and trains the adapter on contrastive pairs
        constructed from retrieval outcomes.

        Args:
            batch_size: Override config batch size
            max_batches: Override max batches per epoch

        Returns:
            Training statistics for this epoch
        """
        batch_size = batch_size or self.config.batch_size
        max_batches = max_batches or self.config.max_batches_per_epoch

        start_time = datetime.now()
        self._epoch_count += 1

        stats = TrainingStats(epoch=self._epoch_count)
        losses: list[float] = []
        patience_counter = 0
        best_loss = float("inf")

        # Calculate feedback lookback
        since = datetime.now() - timedelta(hours=self.config.feedback_lookback_hours)

        batch_num = 0
        while batch_num < max_batches:
            # Fetch feedback batch
            feedbacks = self.feedback.get_training_batch(
                batch_size=batch_size,
                min_confidence=self.config.min_confidence,
                since=since,
            )

            if not feedbacks:
                logger.debug("No more feedback available for training")
                break

            # Group feedback by query
            query_feedbacks = self._group_by_query(feedbacks)

            # Train on each query's feedback
            for query_id, query_fb_list in query_feedbacks.items():
                # Get query embedding
                query_emb = await self._get_query_embedding(query_id, query_fb_list)
                if query_emb is None:
                    continue

                # Split into positive/negative based on relevance
                positives, negatives = self._split_by_relevance(query_fb_list)
                stats.positive_samples += len(positives)
                stats.negative_samples += len(negatives)

                # Get embeddings for results
                pos_embs = await self._get_result_embeddings(positives)
                neg_embs = await self._get_result_embeddings(negatives)

                if not pos_embs:
                    continue

                # Train step
                loss = self.adapter.train_step(
                    query_emb=query_emb,
                    positive_embs=pos_embs,
                    negative_embs=neg_embs,
                )

                losses.append(loss)
                stats.total_samples += len(pos_embs) + len(neg_embs)

            batch_num += 1
            self._total_batches += 1

            # Early stopping check (after warmup)
            if batch_num >= self.config.warmup_batches and losses:
                recent_loss = np.mean(losses[-5:]) if len(losses) >= 5 else losses[-1]

                if recent_loss < best_loss - self.config.early_stop_min_delta:
                    best_loss = recent_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stop_patience:
                    logger.info(f"Early stopping at batch {batch_num}")
                    stats.stopped_early = True
                    break

        # Compute final stats
        stats.batches_trained = batch_num
        stats.training_time_seconds = (datetime.now() - start_time).total_seconds()

        if losses:
            stats.mean_loss = float(np.mean(losses))
            stats.final_loss = losses[-1]
            stats.best_loss = min(losses)

        self._training_history.append(stats)

        logger.info(
            f"Epoch {stats.epoch} complete: "
            f"batches={stats.batches_trained}, "
            f"samples={stats.total_samples}, "
            f"mean_loss={stats.mean_loss:.4f}, "
            f"time={stats.training_time_seconds:.1f}s"
        )

        return stats

    def train_on_feedback(
        self,
        feedbacks: list[RetrievalFeedback],
        query_embeddings: dict[str, np.ndarray],
        result_embeddings: dict[str, np.ndarray],
    ) -> float:
        """
        Synchronous training on pre-prepared feedback batch.

        Used when embeddings are already available (e.g., from a retrieval operation).

        Args:
            feedbacks: List of feedback events
            query_embeddings: Mapping of query_id to embedding
            result_embeddings: Mapping of result_id to embedding

        Returns:
            Training loss
        """
        if not feedbacks:
            return 0.0

        # Group by query
        query_feedbacks = self._group_by_query(feedbacks)

        total_loss = 0.0
        batch_count = 0

        for query_id, fb_list in query_feedbacks.items():
            if query_id not in query_embeddings:
                continue

            query_emb = query_embeddings[query_id]

            # Split by relevance
            positives, negatives = self._split_by_relevance(fb_list)

            # Collect embeddings
            pos_embs = [
                result_embeddings[fb.result_id]
                for fb in positives
                if fb.result_id in result_embeddings
            ]
            neg_embs = [
                result_embeddings[fb.result_id]
                for fb in negatives
                if fb.result_id in result_embeddings
            ]

            if not pos_embs:
                continue

            loss = self.adapter.train_step(
                query_emb=query_emb,
                positive_embs=pos_embs,
                negative_embs=neg_embs,
            )

            total_loss += loss
            batch_count += 1
            self._total_batches += 1

        return total_loss / batch_count if batch_count > 0 else 0.0

    def _group_by_query(
        self,
        feedbacks: list[RetrievalFeedback],
    ) -> dict[str, list[RetrievalFeedback]]:
        """Group feedback events by query ID."""
        groups: dict[str, list[RetrievalFeedback]] = {}
        for fb in feedbacks:
            if fb.query_id not in groups:
                groups[fb.query_id] = []
            groups[fb.query_id].append(fb)
        return groups

    def _split_by_relevance(
        self,
        feedbacks: list[RetrievalFeedback],
    ) -> tuple[list[RetrievalFeedback], list[RetrievalFeedback]]:
        """Split feedback into positive and negative based on relevance score."""
        positives = []
        negatives = []

        for fb in feedbacks:
            if fb.relevance >= self.config.relevance_threshold:
                positives.append(fb)
            else:
                negatives.append(fb)

        return positives, negatives

    async def _get_query_embedding(
        self,
        query_id: str,
        feedbacks: list[RetrievalFeedback],
    ) -> np.ndarray | None:
        """
        Get embedding for a query.

        First checks cache, then tries to reconstruct from feedback data.
        """
        # Check cache
        cached = self._cache.get(f"query:{query_id}")
        if cached is not None:
            return cached

        # For now, we don't have the original query text stored in feedback
        # In production, this would require storing query text or using
        # pre-computed query embeddings from the retrieval outcome
        # We'll generate a representative embedding from result embeddings

        # Get embeddings for results that were clicked (proxy for query intent)
        clicked_embs = []
        for fb in feedbacks:
            if fb.clicked:
                emb = self._cache.get(f"result:{fb.result_id}")
                if emb is not None:
                    clicked_embs.append(emb)

        if clicked_embs:
            # Average of clicked results as proxy for query
            query_emb = np.mean(clicked_embs, axis=0).astype(np.float32)
            self._cache.put(f"query:{query_id}", query_emb)
            return query_emb

        return None

    async def _get_result_embeddings(
        self,
        feedbacks: list[RetrievalFeedback],
    ) -> list[np.ndarray]:
        """Get embeddings for result IDs."""
        embeddings = []

        for fb in feedbacks:
            result_id = fb.result_id

            # Check cache
            cached = self._cache.get(f"result:{result_id}")
            if cached is not None:
                embeddings.append(cached)
                continue

            # In production, would retrieve from memory store
            # For now, skip if not cached
            pass

        return embeddings

    def cache_embeddings(
        self,
        query_id: str,
        query_embedding: np.ndarray,
        result_embeddings: dict[str, np.ndarray],
    ) -> None:
        """
        Pre-cache embeddings for upcoming training.

        Call this after retrieval to enable training without re-computing embeddings.

        Args:
            query_id: ID of the query
            query_embedding: Query embedding vector
            result_embeddings: Mapping of result_id to embedding
        """
        self._cache.put(f"query:{query_id}", query_embedding)
        for result_id, emb in result_embeddings.items():
            self._cache.put(f"result:{result_id}", emb)

    def get_training_history(self) -> list[dict]:
        """Get history of training runs."""
        return [stats.to_dict() for stats in self._training_history]

    def get_stats(self) -> dict:
        """Get trainer statistics."""
        return {
            "epoch_count": self._epoch_count,
            "total_batches": self._total_batches,
            "cache_size": len(self._cache),
            "adapter_stats": self.adapter.get_stats(),
            "config": {
                "batch_size": self.config.batch_size,
                "min_confidence": self.config.min_confidence,
                "relevance_threshold": self.config.relevance_threshold,
            },
        }

    def reset_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()


class ContinuousTrainer:
    """
    Background trainer for continuous adapter updates.

    Runs training in background at configurable intervals,
    enabling online learning during system operation.
    """

    def __init__(
        self,
        trainer: AdapterTrainer,
        interval_seconds: float = 300.0,
        min_feedback_count: int = 10,
    ):
        """
        Initialize continuous trainer.

        Args:
            trainer: AdapterTrainer instance
            interval_seconds: Time between training runs
            min_feedback_count: Minimum feedback before training
        """
        self.trainer = trainer
        self.interval_seconds = interval_seconds
        self.min_feedback_count = min_feedback_count

        self._running = False
        self._task: asyncio.Task | None = None
        self._last_train_time: datetime | None = None

    async def start(self) -> None:
        """Start continuous training loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._training_loop())
        logger.info(f"ContinuousTrainer started (interval={self.interval_seconds}s)")

    async def stop(self) -> None:
        """Stop continuous training loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("ContinuousTrainer stopped")

    async def _training_loop(self) -> None:
        """Background training loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_seconds)

                # Check if enough feedback
                stats = self.trainer.feedback.get_statistics()
                if stats.get("persisted_feedbacks", 0) < self.min_feedback_count:
                    continue

                # Run training
                training_stats = await self.trainer.train_epoch()
                self._last_train_time = datetime.now()

                logger.debug(
                    f"Continuous training: "
                    f"loss={training_stats.mean_loss:.4f}, "
                    f"samples={training_stats.total_samples}"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Continuous training error: {e}")

    @property
    def is_running(self) -> bool:
        """Check if trainer is running."""
        return self._running

    def get_stats(self) -> dict:
        """Get continuous trainer statistics."""
        return {
            "running": self._running,
            "interval_seconds": self.interval_seconds,
            "min_feedback_count": self.min_feedback_count,
            "last_train_time": (
                self._last_train_time.isoformat()
                if self._last_train_time else None
            ),
            "trainer_stats": self.trainer.get_stats(),
        }


# Factory functions

def create_adapter_trainer(
    adapter: OnlineEmbeddingAdapter,
    feedback: RetrievalFeedbackCollector,
    embedder: EmbeddingAdapter,
    batch_size: int = 32,
    relevance_threshold: float = 0.5,
    **kwargs,
) -> AdapterTrainer:
    """
    Factory function for AdapterTrainer.

    Args:
        adapter: The adapter to train
        feedback: Feedback collector
        embedder: Embedding provider
        batch_size: Training batch size
        relevance_threshold: Positive/negative threshold
        **kwargs: Additional TrainingConfig options

    Returns:
        Configured AdapterTrainer
    """
    config = TrainingConfig(
        batch_size=batch_size,
        relevance_threshold=relevance_threshold,
        **kwargs,
    )
    return AdapterTrainer(adapter, feedback, embedder, config)


def create_continuous_trainer(
    trainer: AdapterTrainer,
    interval_minutes: float = 5.0,
    min_feedback: int = 10,
) -> ContinuousTrainer:
    """
    Factory function for ContinuousTrainer.

    Args:
        trainer: AdapterTrainer to wrap
        interval_minutes: Minutes between training runs
        min_feedback: Minimum feedback before training

    Returns:
        Configured ContinuousTrainer
    """
    return ContinuousTrainer(
        trainer=trainer,
        interval_seconds=interval_minutes * 60,
        min_feedback_count=min_feedback,
    )


__all__ = [
    "TrainingConfig",
    "TrainingStats",
    "EmbeddingCache",
    "AdapterTrainer",
    "ContinuousTrainer",
    "create_adapter_trainer",
    "create_continuous_trainer",
]
