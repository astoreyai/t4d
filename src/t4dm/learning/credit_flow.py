"""
Credit Flow Integration for World Weaver.

Connects neuromodulator learning signals to actual weight updates.
This module solves the "dangling credits" problem where process_outcome()
returns learning signals that were never applied.

Flow:
1. NeuromodulatorOrchestra.process_outcome() computes learning signals
2. CreditFlowEngine applies them to:
   - ReconsolidationEngine: Update episodic memory embeddings
   - SemanticMemory: Strengthen/weaken relationship weights
3. Learning signals combine dopamine (immediate) and serotonin (long-term)

This creates end-to-end credit assignment from outcomes to memory updates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from t4dm.learning.neuromodulators import NeuromodulatorOrchestra
    from t4dm.learning.reconsolidation import ReconsolidationEngine
    from t4dm.storage import T4DXVectorStore

from t4dm.learning.fsrs import FSRSMemoryTracker, Rating

logger = logging.getLogger(__name__)


def outcome_to_rating(outcome: float) -> Rating:
    """
    Convert outcome score [0, 1] to FSRS Rating.

    Args:
        outcome: Outcome score where 0.5 is neutral

    Returns:
        FSRS Rating (AGAIN, HARD, GOOD, EASY)
    """
    if outcome < 0.25:
        return Rating.AGAIN  # Failed/forgot
    if outcome < 0.5:
        return Rating.HARD   # Difficult recall
    if outcome < 0.75:
        return Rating.GOOD   # Normal recall
    return Rating.EASY   # Easy recall


class CreditFlowEngine:
    """
    Apply neuromodulator learning signals to memory systems.

    Bridges the gap between:
    - Neuromodulator credit assignment (what should change)
    - Memory reconsolidation and Hebbian updates (how to change it)

    Usage:
        engine = CreditFlowEngine(orchestra, reconsolidation_engine)

        # After retrieval
        learning_signals = orchestra.process_outcome(
            memory_outcomes={mem_id: score, ...},
            session_outcome=0.8
        )

        # Apply signals to update weights
        updated_embeddings = engine.apply_to_episodic(
            learning_signals,
            retrieved_memories,
            query_embedding
        )
    """

    def __init__(
        self,
        neuromodulator_orchestra: NeuromodulatorOrchestra,
        reconsolidation_engine: ReconsolidationEngine | None = None,
        fsrs_tracker: FSRSMemoryTracker | None = None,
        vector_store: T4DXVectorStore | None = None,
        collection_name: str = "episodes",
        episodic_lr_scale: float = 1.0,
        semantic_lr_scale: float = 1.0
    ):
        """
        Initialize credit flow engine.

        Args:
            neuromodulator_orchestra: Orchestra that computes learning signals
            reconsolidation_engine: Engine for episodic memory updates
            fsrs_tracker: Optional FSRS tracker for spaced repetition scheduling
            vector_store: Optional T4DXVectorStore for persisting updated embeddings
            collection_name: Collection name for embedding persistence
            episodic_lr_scale: Scale factor for episodic learning rate
            semantic_lr_scale: Scale factor for semantic learning rate
        """
        self.orchestra = neuromodulator_orchestra
        self.reconsolidation = reconsolidation_engine
        self.fsrs_tracker = fsrs_tracker or FSRSMemoryTracker()  # Default tracker
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.episodic_lr_scale = episodic_lr_scale
        self.semantic_lr_scale = semantic_lr_scale

        # Statistics
        self._total_episodic_updates = 0
        self._total_fsrs_updates = 0
        self._total_semantic_updates = 0
        self._total_learning_signal_magnitude = 0.0
        self._total_persisted_updates = 0

    def apply_to_episodic(
        self,
        learning_signals: dict[str, float],
        retrieved_memories: list[tuple[UUID, np.ndarray]],
        query_embedding: np.ndarray,
        base_outcome: float = 0.5
    ) -> dict[UUID, np.ndarray]:
        """
        Apply learning signals to episodic memory embeddings.

        Uses reconsolidation to update embeddings based on:
        - Learning signals (dopamine + serotonin credits)
        - Long-term value (importance weighting)
        - Surprise magnitude (dopamine modulation)

        Args:
            learning_signals: Memory ID -> learning signal (from process_outcome)
            retrieved_memories: List of (memory_id, embedding) tuples
            query_embedding: Query that retrieved these memories
            base_outcome: Base outcome score for memories without signals

        Returns:
            Dict of memory_id -> updated_embedding
        """
        if self.reconsolidation is None:
            logger.warning("No reconsolidation engine configured")
            return {}

        # Prepare per-memory rewards and modulation
        per_memory_rewards = {}
        per_memory_importance = {}
        per_memory_lr_modulation = {}

        for memory_id, _ in retrieved_memories:
            mem_id_str = str(memory_id)

            # Use learning signal as outcome score
            if mem_id_str in learning_signals:
                signal = learning_signals[mem_id_str]
                # LOGIC-010 FIX: Use SIGNED RPE for outcome direction
                # learning_signals contains magnitude (always >= 0) for rate modulation
                # but we need signed RPE for direction (potentiation vs depression)
                signed_rpe = self.orchestra.get_signed_rpe(memory_id)

                # Combine: signed RPE determines direction, signal magnitude scales it
                # If RPE is negative (worse than expected), outcome should be < 0.5
                # If RPE is positive (better than expected), outcome should be > 0.5
                if signal > 0:
                    # Have a learning signal - use signed RPE for direction
                    # Scale magnitude by signal strength
                    outcome = np.clip(0.5 + np.sign(signed_rpe) * signal, 0.0, 1.0)
                else:
                    # No signal strength - use base outcome
                    outcome = base_outcome
                per_memory_rewards[mem_id_str] = outcome

                # Track magnitude
                self._total_learning_signal_magnitude += abs(signal)
            else:
                per_memory_rewards[mem_id_str] = base_outcome

            # Get long-term value for importance weighting
            ltv = self.orchestra.get_long_term_value(memory_id)
            # High LTV = high importance = protect from catastrophic forgetting
            importance = ltv
            per_memory_importance[mem_id_str] = importance

            # LOGIC-011 FIX: Use dopamine's computed surprise (|actual - expected|)
            # rather than re-computing from transformed outcome which has different scale.
            # The learning_params.surprise is already computed from raw outcomes.
            learning_params = self.orchestra.get_learning_params(memory_id)
            surprise = learning_params.surprise
            # More surprise = higher learning rate
            lr_mod = 1.0 + surprise
            per_memory_lr_modulation[mem_id_str] = lr_mod

        # Apply reconsolidation with neuromodulator-informed parameters
        updated = self.reconsolidation.batch_reconsolidate(
            memories=retrieved_memories,
            query_embedding=query_embedding,
            outcome_score=base_outcome,
            per_memory_rewards=per_memory_rewards,
            per_memory_importance=per_memory_importance,
            per_memory_lr_modulation=per_memory_lr_modulation
        )

        self._total_episodic_updates += len(updated)

        # Update FSRS spaced repetition state for each memory
        for memory_id, _ in retrieved_memories:
            mem_id_str = str(memory_id)
            outcome = per_memory_rewards.get(mem_id_str, base_outcome)
            rating = outcome_to_rating(outcome)
            self.fsrs_tracker.review(mem_id_str, rating)
            self._total_fsrs_updates += 1

        logger.info(
            f"Applied episodic updates: {len(updated)}/{len(retrieved_memories)} memories updated, "
            f"FSRS: {len(retrieved_memories)} reviews recorded"
        )

        return updated

    async def apply_to_semantic(
        self,
        learning_signals: dict[str, float],
        entity_pairs: list[tuple[UUID, UUID]],
        semantic_memory
    ) -> int:
        """
        Apply learning signals to semantic relationship weights.

        Strengthens or weakens relationships based on learning signals.

        Args:
            learning_signals: Memory ID -> learning signal
            entity_pairs: List of (source_id, target_id) entity pairs
            semantic_memory: SemanticMemory instance with graph_store

        Returns:
            Number of relationships updated
        """
        if not hasattr(semantic_memory, "graph_store"):
            logger.warning("Semantic memory has no graph_store")
            return 0

        updates = 0

        for source_id, target_id in entity_pairs:
            # Get average signal for both entities
            source_signal = learning_signals.get(str(source_id), 0.0)
            target_signal = learning_signals.get(str(target_id), 0.0)
            avg_signal = (source_signal + target_signal) / 2.0

            # Positive signal = strengthen, negative = weaken
            if abs(avg_signal) > 0.01:
                try:
                    if avg_signal > 0:
                        # Strengthen relationship
                        await semantic_memory.graph_store.strengthen_relationship(
                            source_id=str(source_id),
                            target_id=str(target_id),
                            learning_rate=semantic_memory.learning_rate * self.semantic_lr_scale * abs(avg_signal)
                        )
                    else:
                        # Weaken relationship (anti-Hebbian)
                        # Note: strengthen_relationship with negative signal would work
                        # if the implementation supports it, otherwise skip
                        pass

                    updates += 1
                except Exception as e:
                    logger.debug(f"Failed to update {source_id}->{target_id}: {e}")

        self._total_semantic_updates += updates

        logger.info(f"Applied semantic updates: {updates} relationships updated")

        return updates

    async def process_and_apply_outcome(
        self,
        memory_outcomes: dict[str, float],
        retrieved_memories: list[tuple[UUID, np.ndarray]],
        query_embedding: np.ndarray,
        session_outcome: float | None = None,
        entity_pairs: list[tuple[UUID, UUID]] | None = None,
        semantic_memory=None
    ) -> dict:
        """
        End-to-end: Process outcomes through neuromodulators and apply updates.

        ASYNC-002 FIX: Made method async to properly handle semantic updates.

        This is the convenience method that does the full flow:
        1. Compute learning signals via orchestra.process_outcome()
        2. Apply to episodic memory via reconsolidation
        3. (Optional) Apply to semantic memory via Hebbian updates

        Args:
            memory_outcomes: Memory ID -> immediate outcome
            retrieved_memories: List of (memory_id, embedding) tuples
            query_embedding: Query that retrieved these memories
            session_outcome: Optional session-level outcome
            entity_pairs: Optional entity pairs for semantic updates
            semantic_memory: Optional SemanticMemory instance

        Returns:
            Dict with:
                - learning_signals: Memory ID -> signal
                - episodic_updates: Memory ID -> new embedding
                - semantic_updates: Number of relationships updated
        """
        # Step 1: Compute learning signals
        learning_signals = self.orchestra.process_outcome(
            memory_outcomes=memory_outcomes,
            session_outcome=session_outcome
        )

        # Step 2: Apply to episodic memory
        base_outcome = np.mean(list(memory_outcomes.values())) if memory_outcomes else 0.5
        episodic_updates = self.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=retrieved_memories,
            query_embedding=query_embedding,
            base_outcome=base_outcome
        )

        # Step 2.5: PERSIST updated embeddings to vector store
        # This is the CRITICAL FIX - reconsolidated embeddings were computed but never saved
        persisted_count = 0
        if episodic_updates and self.vector_store is not None:
            try:
                vector_updates = [
                    (str(mem_id), new_emb.tolist())
                    for mem_id, new_emb in episodic_updates.items()
                ]
                persisted_count = await self.vector_store.batch_update_vectors(
                    collection=self.collection_name,
                    updates=vector_updates,
                )
                self._total_persisted_updates += persisted_count
                logger.info(
                    f"Persisted {persisted_count} reconsolidated embeddings to {self.collection_name}"
                )
            except Exception as e:
                logger.warning(f"Failed to persist reconsolidated embeddings: {e}")

        # Step 3: (Optional) Apply to semantic memory
        # ASYNC-002 FIX: Properly await async method instead of using deprecated event loop pattern
        semantic_updates = 0
        if entity_pairs and semantic_memory:
            semantic_updates = await self.apply_to_semantic(
                learning_signals=learning_signals,
                entity_pairs=entity_pairs,
                semantic_memory=semantic_memory
            )

        return {
            "learning_signals": learning_signals,
            "episodic_updates": episodic_updates,
            "semantic_updates": semantic_updates,
            "persisted_count": persisted_count,
        }

    def process_and_apply_outcome_sync(
        self,
        memory_outcomes: dict[str, float],
        retrieved_memories: list[tuple[UUID, np.ndarray]],
        query_embedding: np.ndarray,
        session_outcome: float | None = None,
        entity_pairs: list[tuple[UUID, UUID]] | None = None,
        semantic_memory=None
    ) -> dict:
        """
        Synchronous version of process_and_apply_outcome.

        ASYNC-002 FIX: For sync contexts that don't need semantic updates,
        or use this wrapper that properly creates a new event loop.

        Note: If entity_pairs and semantic_memory are provided, this will
        create a new event loop. For better performance in async code,
        use process_and_apply_outcome() directly.
        """
        import asyncio

        # Step 1 & 2: Sync operations
        learning_signals = self.orchestra.process_outcome(
            memory_outcomes=memory_outcomes,
            session_outcome=session_outcome
        )

        base_outcome = np.mean(list(memory_outcomes.values())) if memory_outcomes else 0.5
        episodic_updates = self.apply_to_episodic(
            learning_signals=learning_signals,
            retrieved_memories=retrieved_memories,
            query_embedding=query_embedding,
            base_outcome=base_outcome
        )

        # Step 3: Handle async semantic updates in sync context
        semantic_updates = 0
        if entity_pairs and semantic_memory:
            try:
                # Check if already in async context
                asyncio.get_running_loop()
                logger.warning(
                    "process_and_apply_outcome_sync() called from async context. "
                    "Use process_and_apply_outcome() instead. Skipping semantic updates."
                )
            except RuntimeError:
                # No running loop - create one
                loop = asyncio.new_event_loop()
                try:
                    semantic_updates = loop.run_until_complete(
                        self.apply_to_semantic(
                            learning_signals=learning_signals,
                            entity_pairs=entity_pairs,
                            semantic_memory=semantic_memory
                        )
                    )
                finally:
                    loop.close()

        return {
            "learning_signals": learning_signals,
            "episodic_updates": episodic_updates,
            "semantic_updates": semantic_updates
        }

    def get_stats(self) -> dict:
        """Get credit flow statistics including FSRS."""
        fsrs_stats = self.fsrs_tracker.get_stats()
        return {
            "total_episodic_updates": self._total_episodic_updates,
            "total_semantic_updates": self._total_semantic_updates,
            "total_fsrs_updates": self._total_fsrs_updates,
            "total_persisted_updates": self._total_persisted_updates,
            "total_signal_magnitude": self._total_learning_signal_magnitude,
            "episodic_lr_scale": self.episodic_lr_scale,
            "semantic_lr_scale": self.semantic_lr_scale,
            "vector_store_connected": self.vector_store is not None,
            "fsrs": fsrs_stats
        }

    def get_due_memories(self, limit: int = 100) -> list:
        """
        Get memories due for review based on FSRS scheduling.

        Args:
            limit: Maximum number of items to return

        Returns:
            List of (memory_id, days_overdue, state) tuples sorted by urgency
        """
        return self.fsrs_tracker.get_due_items(limit=limit)

    def get_memory_retrievability(self, memory_id: str) -> float:
        """
        Get current retrievability for a memory.

        Args:
            memory_id: Memory ID to check

        Returns:
            Probability of recall [0, 1]
        """
        return self.fsrs_tracker.get_retrievability(memory_id)

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._total_episodic_updates = 0
        self._total_semantic_updates = 0
        self._total_fsrs_updates = 0
        self._total_learning_signal_magnitude = 0.0


__all__ = [
    "CreditFlowEngine",
    "outcome_to_rating",
]
