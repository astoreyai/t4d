"""
BufferManager - CA1-like temporary storage for uncertain memories.

This module implements a biological analog to hippocampal CA1 temporary storage:
items with BUFFER decisions are held here while evidence accumulates through
retrieval probing and contextual signals before promotion or discard.

Key insight: BUFFER != "delayed STORE". It means "candidate under observation."
Evidence emerges from:
- Retrieval matches (implicit utility signal)
- Contextual co-occurrence with stored items
- Neuromodulator signals (DA, 5HT)
- Time decay (prevents indefinite buffering)

Per Hinton: The buffer participates in retrieval, gathering implicit evidence
of utility rather than waiting for explicit feedback.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np

if TYPE_CHECKING:
    from t4dm.core.learned_gate import LearnedMemoryGate
    from t4dm.learning.neuromodulators import NeuromodulatorState

logger = logging.getLogger(__name__)


class PromotionAction(Enum):
    """Result of buffer item evaluation."""
    PROMOTE = "promote"  # Sufficient evidence - store to long-term
    DISCARD = "discard"  # Insufficient evidence - drop
    WAIT = "wait"        # Continue accumulating evidence


@dataclass
class BufferedItem:
    """A memory candidate awaiting evidence accumulation."""

    id: UUID
    content: str
    embedding: np.ndarray
    features: np.ndarray  # Gate features for training upon promotion
    context: dict  # Original context dict
    created_at: datetime = field(default_factory=datetime.now)
    evidence_score: float = 0.5  # Starts neutral
    evidence_count: int = 0
    retrieval_hits: int = 0  # Times matched in recall queries
    co_occurrence_boost: float = 0.0  # Proximity to stored items
    last_evidence_at: datetime | None = None
    outcome: str = "neutral"
    valence: float = 0.5


@dataclass
class PromotionDecision:
    """Result of evaluating a buffered item for promotion."""

    item_id: UUID
    action: PromotionAction
    final_evidence: float
    reason: str


class BufferManager:
    """
    CA1-like temporary storage for BUFFER decisions.

    Implements evidence accumulation before promotion to long-term storage.
    The buffer actively participates in retrieval (via probe()) and
    accumulates evidence from matches, neuromodulator signals, and context.

    Lifecycle:
    1. Item added with BUFFER decision
    2. During recall(), buffer is probed for matches
    3. Matches accumulate evidence
    4. tick() evaluates items for promotion/discard
    5. Promoted items go to long-term storage with gate training
    6. Discarded items provide soft negative signal to gate

    Per Hinton: This creates "observation before commitment" rather than
    immediate storage or discard.
    """

    # Default thresholds
    DEFAULT_PROMOTION_THRESHOLD = 0.65  # Higher than gate's 0.6 (need more evidence)
    DEFAULT_DISCARD_THRESHOLD = 0.25    # Below gate's 0.3 buffer threshold
    DEFAULT_MAX_RESIDENCE_SECONDS = 300  # 5 minutes
    DEFAULT_MAX_BUFFER_SIZE = 50

    # Evidence signal magnitudes
    RETRIEVAL_HIT_SIGNAL = 0.25      # Direct match during recall
    CO_RETRIEVAL_SIGNAL = 0.15       # Matched alongside stored item
    OUTCOME_SIGNAL_SCALE = 0.2       # Scaled by (utility - 0.5)
    NEUROMOD_SIGNAL_SCALE = 0.1      # Scaled by combined DA+5HT
    CONTEXT_MATCH_SIGNAL = 0.05      # Same project still active
    TIME_DECAY_PER_SECOND = 0.0003   # ~0.02 per minute

    # Promotion utility mapping
    PROMOTED_UTILITY_BASE = 0.5
    PROMOTED_UTILITY_SCALE = 0.5     # Maps evidence 0-1 to utility 0.5-1.0
    # MEMORY-HIGH-002 FIX: Discards now also scale by evidence score
    # Previously used constant 0.3, missing [0.3, 0.5] range in training
    DISCARDED_UTILITY_BASE = 0.1     # Base utility for discards
    DISCARDED_UTILITY_SCALE = 0.35   # Maps evidence 0-0.8 to utility 0.1-0.45

    def __init__(
        self,
        promotion_threshold: float = DEFAULT_PROMOTION_THRESHOLD,
        discard_threshold: float = DEFAULT_DISCARD_THRESHOLD,
        max_residence_time: timedelta = timedelta(seconds=DEFAULT_MAX_RESIDENCE_SECONDS),
        max_buffer_size: int = DEFAULT_MAX_BUFFER_SIZE,
        learned_gate: LearnedMemoryGate | None = None,
        stagger_limit: int = 5,  # Max promotions per tick
    ):
        """
        Initialize buffer manager.

        Args:
            promotion_threshold: Evidence threshold for promotion (0-1)
            discard_threshold: Evidence threshold for discard (0-1)
            max_residence_time: Max time before forced decision
            max_buffer_size: Max buffered items (prevents OOM)
            learned_gate: Gate to train on promotion/discard
            stagger_limit: Max items to promote in single tick (prevents catastrophic forgetting)
        """
        self.promotion_threshold = promotion_threshold
        self.discard_threshold = discard_threshold
        self.max_residence_time = max_residence_time
        self.max_buffer_size = max_buffer_size
        self.learned_gate = learned_gate
        self.stagger_limit = stagger_limit

        # MEMORY-CRITICAL-001 FIX: Thread-safe buffer access
        self._lock = threading.RLock()  # RLock allows nested locking

        # Buffer storage
        self._buffer: dict[UUID, BufferedItem] = {}

        # Recently promoted cooldown (prevents feedback amplification)
        self._recently_promoted: dict[UUID, datetime] = {}
        self._promotion_cooldown = timedelta(seconds=60)
        # MEMORY-HIGH-001 FIX: Limit cooldown dict size
        self._max_cooldown_size = 100

        # Statistics
        self.stats = {
            "items_added": 0,
            "items_promoted": 0,
            "items_discarded": 0,
            "items_timed_out": 0,
            "retrieval_probes": 0,
            "retrieval_hits": 0,
        }

        logger.info(
            f"BufferManager initialized: promote_thresh={promotion_threshold}, "
            f"discard_thresh={discard_threshold}, max_size={max_buffer_size}"
        )

    @property
    def size(self) -> int:
        """Current buffer size."""
        return len(self._buffer)

    @property
    def pressure(self) -> float:
        """Buffer pressure (0-1), used for threshold adjustment."""
        return len(self._buffer) / self.max_buffer_size

    def add(
        self,
        content: str,
        embedding: np.ndarray,
        features: np.ndarray,
        context: dict,
        outcome: str = "neutral",
        valence: float = 0.5,
    ) -> UUID:
        """
        Add item to buffer for evidence accumulation.

        MEMORY-CRITICAL-001 FIX: Thread-safe with RLock.

        Args:
            content: Memory content
            embedding: Content embedding vector
            features: Gate features (for training on promotion/discard)
            context: Original context dict
            outcome: Outcome string
            valence: Importance valence

        Returns:
            UUID of buffered item
        """
        with self._lock:
            # Handle buffer overflow
            if len(self._buffer) >= self.max_buffer_size:
                self._evict_lowest_evidence()

            item = BufferedItem(
                id=uuid4(),
                content=content,
                embedding=np.asarray(embedding),
                features=np.asarray(features),
                context=context,
                outcome=outcome,
                valence=valence,
            )

            self._buffer[item.id] = item
            self.stats["items_added"] += 1

            logger.debug(f"Added to buffer: {item.id} (size={len(self._buffer)})")
            return item.id

    def probe(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.6,
        limit: int = 5,
    ) -> list[BufferedItem]:
        """
        Search buffer for relevant items during recall.

        MEMORY-CRITICAL-001 FIX: Thread-safe with RLock.

        This is the key mechanism for implicit evidence accumulation:
        items that match recall queries are proving their utility.

        Args:
            query_embedding: Query vector
            threshold: Similarity threshold for matching
            limit: Max matches to return

        Returns:
            List of matching buffered items (sorted by similarity)
        """
        with self._lock:
            if not self._buffer:
                return []

            self.stats["retrieval_probes"] += 1
            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return []

            matches: list[tuple[float, BufferedItem]] = []

            # RACE-BUFFER-002 FIX: Use snapshot to prevent RuntimeError during concurrent tick()
            for item in list(self._buffer.values()):
                # Compute cosine similarity
                item_norm = np.linalg.norm(item.embedding)
                if item_norm == 0:
                    continue

                similarity = float(
                    np.dot(query_embedding, item.embedding) / (query_norm * item_norm)
                )

                if similarity >= threshold:
                    matches.append((similarity, item))

                    # Accumulate evidence from retrieval hit (lock already held via RLock)
                    self._accumulate_evidence_unlocked(
                        item.id,
                        signal=self.RETRIEVAL_HIT_SIGNAL,
                        reason="retrieval_hit"
                    )
                    self.stats["retrieval_hits"] += 1

            # Sort by similarity descending
            matches.sort(key=lambda x: x[0], reverse=True)

            return [item for _, item in matches[:limit]]

    def _accumulate_evidence_unlocked(
        self,
        item_id: UUID,
        signal: float,
        reason: str = "unknown"
    ) -> None:
        """
        Internal: Accumulate evidence without acquiring lock.

        Used when caller already holds the lock (e.g., from probe()).
        """
        if item_id not in self._buffer:
            return

        item = self._buffer[item_id]
        old_score = item.evidence_score

        # Apply signal with bounds checking
        item.evidence_score = float(np.clip(item.evidence_score + signal, 0.0, 1.0))
        item.evidence_count += 1
        item.last_evidence_at = datetime.now()

        if reason == "retrieval_hit":
            item.retrieval_hits += 1

        logger.debug(
            f"Evidence for {item_id}: {old_score:.3f} -> {item.evidence_score:.3f} "
            f"(signal={signal:+.3f}, reason={reason})"
        )

    def accumulate_evidence(
        self,
        item_id: UUID,
        signal: float,
        reason: str = "unknown"
    ) -> None:
        """
        Accumulate evidence for a buffered item.

        MEMORY-CRITICAL-001 FIX: Thread-safe with RLock.

        Args:
            item_id: ID of buffered item
            signal: Evidence delta (positive or negative)
            reason: Source of evidence (for logging)
        """
        with self._lock:
            self._accumulate_evidence_unlocked(item_id, signal, reason)

    def accumulate_from_outcome(
        self,
        query_embedding: np.ndarray,
        combined_signal: float,
        similarity_threshold: float = 0.5
    ) -> None:
        """
        Propagate outcome signals to related buffer items.

        MEMORY-CRITICAL-001 FIX: Thread-safe with RLock.

        Called from learn_from_outcome() to share neuromodulator
        signals with semantically related buffered items.

        Args:
            query_embedding: Context embedding from outcome
            combined_signal: DA+5HT combined signal
            similarity_threshold: Min similarity to propagate
        """
        with self._lock:
            if not self._buffer:
                return

            query_norm = np.linalg.norm(query_embedding)
            if query_norm == 0:
                return

            # RACE-BUFFER-002 FIX: Use snapshot to prevent RuntimeError during concurrent tick()
            for item in list(self._buffer.values()):
                item_norm = np.linalg.norm(item.embedding)
                if item_norm == 0:
                    continue

                similarity = float(
                    np.dot(query_embedding, item.embedding) / (query_norm * item_norm)
                )

                if similarity > similarity_threshold:
                    # Scale signal by neuromod and similarity
                    scaled_signal = self.NEUROMOD_SIGNAL_SCALE * combined_signal * similarity
                    self._accumulate_evidence_unlocked(
                        item.id,
                        signal=scaled_signal,
                        reason="outcome_propagation"
                    )

    def tick(
        self,
        neuromod_state: NeuromodulatorState | None = None
    ) -> list[PromotionDecision]:
        """
        Evaluate all buffer items for promotion/discard.

        MEMORY-CRITICAL-001 FIX: Thread-safe with RLock.

        Should be called periodically (e.g., after each interaction or
        every 30 seconds).

        Args:
            neuromod_state: Current neuromodulator state for threshold adjustment

        Returns:
            List of decisions made this tick
        """
        with self._lock:
            if not self._buffer:
                return []

            now = datetime.now()
            decisions: list[PromotionDecision] = []
            to_promote: list[BufferedItem] = []
            to_discard: list[BufferedItem] = []

            # Clean up old promotion cooldowns
            # MEMORY-HIGH-001 FIX: Limit cooldown dict size
            expired_cooldowns = [
                uid for uid, ts in self._recently_promoted.items()
                if now - ts > self._promotion_cooldown
            ]
            for uid in expired_cooldowns:
                del self._recently_promoted[uid]

            # Additional limit check for cooldown dict
            if len(self._recently_promoted) > self._max_cooldown_size:
                # Remove oldest entries
                sorted_items = sorted(self._recently_promoted.items(), key=lambda x: x[1])
                for uid, _ in sorted_items[:len(self._recently_promoted) - self._max_cooldown_size]:
                    del self._recently_promoted[uid]

            # Get adjusted thresholds
            promote_thresh = self._get_adjusted_threshold(
                self.promotion_threshold,
                neuromod_state,
                is_promote=True
            )
            discard_thresh = self._get_adjusted_threshold(
                self.discard_threshold,
                neuromod_state,
                is_promote=False
            )

            # Evaluate each item
            for item in list(self._buffer.values()):
                decision = self._evaluate_item(item, promote_thresh, discard_thresh, now)

                if decision.action == PromotionAction.PROMOTE:
                    to_promote.append(item)
                    decisions.append(decision)
                elif decision.action == PromotionAction.DISCARD:
                    to_discard.append(item)
                    decisions.append(decision)
                # WAIT items stay in buffer

            # Sort promotions by evidence (highest first)
            to_promote.sort(key=lambda x: x.evidence_score, reverse=True)

            # Stagger promotions to prevent catastrophic forgetting
            if len(to_promote) > self.stagger_limit:
                # Defer lower-evidence items to next tick
                deferred = to_promote[self.stagger_limit:]
                to_promote = to_promote[:self.stagger_limit]

                # Update decisions to reflect deferral
                deferred_ids = {item.id for item in deferred}
                decisions = [
                    d if d.item_id not in deferred_ids else
                    PromotionDecision(
                        item_id=d.item_id,
                        action=PromotionAction.WAIT,
                        final_evidence=d.final_evidence,
                        reason="staggered - queued for next tick"
                    )
                    for d in decisions
                ]

            # Execute promotions
            for item in to_promote:
                self._execute_promotion(item)

            # Execute discards
            for item in to_discard:
                self._execute_discard(item)

            return decisions

    def _evaluate_item(
        self,
        item: BufferedItem,
        promote_thresh: float,
        discard_thresh: float,
        now: datetime
    ) -> PromotionDecision:
        """Evaluate a single item for promotion/discard."""

        # Calculate effective evidence with retrieval boost
        evidence = item.evidence_score

        # Nonlinear retrieval boost (multiple hits are strong signal)
        if item.retrieval_hits > 0:
            retrieval_boost = min(0.3, 0.12 * np.sqrt(item.retrieval_hits))
            evidence += retrieval_boost

        # Apply time decay
        age_seconds = (now - item.created_at).total_seconds()
        time_decay = age_seconds * self.TIME_DECAY_PER_SECOND
        evidence -= time_decay
        evidence = max(0.0, evidence)

        # Check urgency (force decision after max residence)
        max_seconds = self.max_residence_time.total_seconds()
        urgency = age_seconds / max_seconds

        # Evaluate against thresholds
        if evidence >= promote_thresh:
            return PromotionDecision(
                item_id=item.id,
                action=PromotionAction.PROMOTE,
                final_evidence=evidence,
                reason=f"evidence {evidence:.3f} >= threshold {promote_thresh:.3f}"
            )
        if evidence <= discard_thresh:
            return PromotionDecision(
                item_id=item.id,
                action=PromotionAction.DISCARD,
                final_evidence=evidence,
                reason=f"evidence {evidence:.3f} <= threshold {discard_thresh:.3f}"
            )
        if urgency >= 1.0:
            # Force decision based on evidence vs midpoint
            midpoint = (promote_thresh + discard_thresh) / 2
            if evidence >= midpoint:
                return PromotionDecision(
                    item_id=item.id,
                    action=PromotionAction.PROMOTE,
                    final_evidence=evidence,
                    reason=f"timeout forced promotion (evidence {evidence:.3f} >= midpoint)"
                )
            self.stats["items_timed_out"] += 1
            return PromotionDecision(
                item_id=item.id,
                action=PromotionAction.DISCARD,
                final_evidence=evidence,
                reason=f"timeout forced discard (evidence {evidence:.3f} < midpoint)"
            )
        return PromotionDecision(
            item_id=item.id,
            action=PromotionAction.WAIT,
            final_evidence=evidence,
            reason=f"accumulating (evidence={evidence:.3f}, urgency={urgency:.2f})"
        )

    def _get_adjusted_threshold(
        self,
        base: float,
        neuromod_state: NeuromodulatorState | None,
        is_promote: bool
    ) -> float:
        """
        Get neuromodulator-adjusted threshold.

        Per Hinton: Neuromodulators should modulate storage decisions.
        High arousal = be more inclusive; encoding mode = store now.
        """
        if neuromod_state is None:
            return base

        adjustment = 0.0

        # High NE (arousal) -> lower promotion threshold
        if neuromod_state.norepinephrine_gain > 1.3:
            adjustment -= 0.1 if is_promote else -0.05

        # Encoding ACh mode -> lower promotion threshold
        if neuromod_state.acetylcholine_mode == "encoding":
            adjustment -= 0.05 if is_promote else 0.0
        elif neuromod_state.acetylcholine_mode == "retrieval":
            adjustment += 0.05 if is_promote else 0.0  # Focus on recall, not storage

        # High DA surprise -> lower promotion threshold
        if abs(neuromod_state.dopamine_rpe) > 0.3:
            adjustment -= 0.08 if is_promote else 0.0

        # 5-HT mood affects patience
        if neuromod_state.serotonin_mood > 0.6:
            adjustment += 0.05 if is_promote else 0.0  # High mood = patience
        elif neuromod_state.serotonin_mood < 0.4:
            adjustment -= 0.05 if is_promote else 0.0  # Low mood = urgency

        # Buffer pressure adjustment
        pressure = self.pressure
        if pressure > 0.7:
            adjustment -= 0.1 * pressure if is_promote else 0.05 * pressure

        return float(np.clip(base + adjustment, 0.1, 0.95))

    def _execute_promotion(self, item: BufferedItem) -> None:
        """Execute promotion of buffered item to long-term storage."""

        # Remove from buffer
        del self._buffer[item.id]

        # Add to cooldown to prevent feedback amplification
        self._recently_promoted[item.id] = datetime.now()

        # Train gate with positive signal
        if self.learned_gate is not None:
            # Map evidence to utility (0.5-1.0 range)
            utility = self.PROMOTED_UTILITY_BASE + self.PROMOTED_UTILITY_SCALE * item.evidence_score
            utility = float(np.clip(utility, 0.5, 1.0))

            try:
                # P0a: Register with raw embedding for content projection learning
                self.learned_gate.register_pending(
                    item.id,
                    item.features,
                    raw_content_embedding=item.embedding
                )
                # Then update with utility
                self.learned_gate.update(item.id, utility)
            except Exception as e:
                logger.warning(f"Failed to train gate on promotion: {e}")

        self.stats["items_promoted"] += 1
        logger.info(
            f"Promoted {item.id}: evidence={item.evidence_score:.3f}, "
            f"retrieval_hits={item.retrieval_hits}"
        )

    def _execute_discard(self, item: BufferedItem) -> None:
        """Execute discard of buffered item."""

        # Remove from buffer
        del self._buffer[item.id]

        # Train gate with scaled negative based on evidence
        # MEMORY-HIGH-002 FIX: Discards also scale by evidence to cover [0.1, 0.45] range
        if self.learned_gate is not None:
            try:
                # Map evidence to utility (0.1-0.45 range for discards)
                utility = self.DISCARDED_UTILITY_BASE + self.DISCARDED_UTILITY_SCALE * item.evidence_score
                utility = float(np.clip(utility, 0.1, 0.45))

                # P0a: Register with raw embedding for content projection learning
                self.learned_gate.register_pending(
                    item.id,
                    item.features,
                    raw_content_embedding=item.embedding
                )
                self.learned_gate.update(item.id, utility)
            except Exception as e:
                logger.warning(f"Failed to train gate on discard: {e}")

        self.stats["items_discarded"] += 1
        logger.debug(
            f"Discarded {item.id}: evidence={item.evidence_score:.3f}, "
            f"retrieval_hits={item.retrieval_hits}"
        )

    def _evict_lowest_evidence(self) -> None:
        """Evict lowest evidence item when buffer is full."""
        if not self._buffer:
            return

        # RACE-BUFFER-002 FIX: Use snapshot to prevent RuntimeError during concurrent access
        # Find item with lowest evidence score
        buffer_snapshot = list(self._buffer.values())
        if not buffer_snapshot:
            return
        min_item = min(buffer_snapshot, key=lambda x: x.evidence_score)

        logger.warning(
            f"Buffer overflow eviction: {min_item.id} "
            f"(evidence={min_item.evidence_score:.3f})"
        )

        # Treat as discard
        self._execute_discard(min_item)

    def get_item(self, item_id: UUID) -> BufferedItem | None:
        """Get buffered item by ID."""
        return self._buffer.get(item_id)

    def get_all_items(self) -> list[BufferedItem]:
        """Get all buffered items."""
        return list(self._buffer.values())

    def get_item_ids(self) -> list[UUID]:
        """Get all buffered item IDs."""
        return list(self._buffer.keys())

    def is_recently_promoted(self, item_id: UUID) -> bool:
        """Check if item was recently promoted (within cooldown)."""
        return item_id in self._recently_promoted

    def clear(self) -> None:
        """Clear all buffered items (for testing/shutdown)."""
        # Discard all remaining items
        for item in list(self._buffer.values()):
            self._execute_discard(item)

        self._buffer.clear()
        self._recently_promoted.clear()

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            **self.stats,
            "current_size": len(self._buffer),
            "pressure": self.pressure,
            "promotion_rate": (
                self.stats["items_promoted"] / self.stats["items_added"]
                if self.stats["items_added"] > 0 else 0.0
            ),
            "discard_rate": (
                self.stats["items_discarded"] / self.stats["items_added"]
                if self.stats["items_added"] > 0 else 0.0
            ),
            "timeout_rate": (
                self.stats["items_timed_out"] / self.stats["items_discarded"]
                if self.stats["items_discarded"] > 0 else 0.0
            ),
            "probe_hit_rate": (
                self.stats["retrieval_hits"] / self.stats["retrieval_probes"]
                if self.stats["retrieval_probes"] > 0 else 0.0
            ),
        }


__all__ = [
    "BufferManager",
    "BufferedItem",
    "PromotionAction",
    "PromotionDecision",
]
