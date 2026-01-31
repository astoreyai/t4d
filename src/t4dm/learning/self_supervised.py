"""
Self-Supervised Credit Estimation for World Weaver.

P1-2: Addresses gap where credit assignment requires explicit outcomes.
Enables learning from implicit signals when no outcome is provided.

Biological Basis:
- The brain assigns value even without explicit feedback
- Co-retrieval implies relatedness (Hebbian: "fire together, wire together")
- Retrieval frequency reflects implicit importance
- Temporal contiguity suggests causal relationships

Implementation:
1. Contrastive Credit: Retrieved memories get positive credit, non-retrieved negative
2. Co-activation Credit: Memories retrieved together boost each other
3. Temporal Chain Credit: Memory chains that lead to outcomes propagate credit
4. Frequency-Based Credit: Often-retrieved memories accumulate implicit value

References:
- Chen et al. (2020): SimCLR contrastive learning
- Grill et al. (2020): BYOL self-supervised learning
- Nair & Silver (2010): Curiosity-driven exploration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ImplicitCredit:
    """
    Credit signal derived without explicit outcomes.

    Represents implicit value from retrieval patterns.
    """
    memory_id: UUID
    credit_type: str  # "contrastive", "coactivation", "temporal", "frequency"
    credit_value: float  # [-1, 1]
    confidence: float  # [0, 1] - how confident in this credit
    timestamp: datetime = field(default_factory=datetime.now)
    source_ids: list[UUID] = field(default_factory=list)  # Related memories


@dataclass
class RetrievalEvent:
    """Record of a memory retrieval for self-supervised learning."""
    query_embedding: np.ndarray
    retrieved_ids: list[UUID]
    retrieved_embeddings: list[np.ndarray]
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = "default"


class SelfSupervisedCredit:
    """
    Self-supervised credit estimation without explicit outcomes.

    P1-2: Provides credit signals when no outcome is available,
    enabling continuous learning from retrieval patterns alone.

    Methods:
    1. Contrastive Credit: Query-memory similarity as implicit success
    2. Co-activation Credit: Co-retrieved memories boost each other
    3. Temporal Chain Credit: Propagate credit along retrieval sequences
    4. Frequency Credit: Retrieval frequency as implicit importance
    """

    def __init__(
        self,
        # Contrastive parameters
        contrastive_temperature: float = 0.1,
        positive_weight: float = 1.0,
        negative_weight: float = 0.1,
        # Co-activation parameters
        coactivation_boost: float = 0.1,
        coactivation_decay: float = 0.95,
        # Temporal chain parameters
        chain_discount: float = 0.9,
        chain_max_depth: int = 5,
        # Frequency parameters
        frequency_baseline: float = 1.0,
        frequency_scaling: float = 0.1,
        # General
        credit_decay_rate: float = 0.99,
        max_history: int = 10000,
    ):
        """
        Initialize self-supervised credit estimator.

        Args:
            contrastive_temperature: Temperature for contrastive scoring
            positive_weight: Weight for positive (retrieved) examples
            negative_weight: Weight for negative (not retrieved) examples
            coactivation_boost: Credit boost for co-retrieved memories
            coactivation_decay: Decay rate for coactivation over time
            chain_discount: Discount factor for temporal credit propagation
            chain_max_depth: Maximum depth for temporal chain credit
            frequency_baseline: Baseline frequency for new memories
            frequency_scaling: How much frequency affects credit
            credit_decay_rate: Daily decay rate for accumulated credit
            max_history: Maximum retrieval events to track
        """
        self.contrastive_temperature = contrastive_temperature
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.coactivation_boost = coactivation_boost
        self.coactivation_decay = coactivation_decay
        self.chain_discount = chain_discount
        self.chain_max_depth = chain_max_depth
        self.frequency_baseline = frequency_baseline
        self.frequency_scaling = frequency_scaling
        self.credit_decay_rate = credit_decay_rate
        self.max_history = max_history

        # State
        self._retrieval_history: list[RetrievalEvent] = []
        self._accumulated_credit: dict[str, float] = {}  # memory_id -> credit
        self._retrieval_counts: dict[str, int] = {}  # memory_id -> count
        self._coactivation_matrix: dict[tuple[str, str], float] = {}
        self._last_decay: datetime = datetime.now()

    def record_retrieval(
        self,
        query_embedding: np.ndarray,
        retrieved_memories: list[tuple[UUID, np.ndarray]],
        session_id: str = "default",
    ) -> list[ImplicitCredit]:
        """
        Record a retrieval event and compute implicit credits.

        Args:
            query_embedding: Query vector used for retrieval
            retrieved_memories: List of (id, embedding) for retrieved memories
            session_id: Session identifier

        Returns:
            List of implicit credits computed from this retrieval
        """
        if not retrieved_memories:
            return []

        retrieved_ids = [m[0] for m in retrieved_memories]
        retrieved_embeddings = [m[1] for m in retrieved_memories]

        event = RetrievalEvent(
            query_embedding=query_embedding,
            retrieved_ids=retrieved_ids,
            retrieved_embeddings=retrieved_embeddings,
            session_id=session_id,
        )
        self._retrieval_history.append(event)

        # Enforce history limit
        if len(self._retrieval_history) > self.max_history:
            self._retrieval_history = self._retrieval_history[-self.max_history:]

        credits = []

        # 1. Contrastive credit based on query-memory similarity
        contrastive_credits = self._compute_contrastive_credit(event)
        credits.extend(contrastive_credits)

        # 2. Co-activation credit for co-retrieved memories
        coactivation_credits = self._compute_coactivation_credit(event)
        credits.extend(coactivation_credits)

        # 3. Update frequency counts
        frequency_credits = self._compute_frequency_credit(event)
        credits.extend(frequency_credits)

        # Apply periodic decay
        self._apply_decay()

        return credits

    def _compute_contrastive_credit(
        self,
        event: RetrievalEvent,
    ) -> list[ImplicitCredit]:
        """
        Compute contrastive credit from query-memory similarity.

        High similarity between query and retrieved memory = positive credit.
        Being retrieved when others weren't = relative positive signal.
        """
        credits = []
        query = event.query_embedding

        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        for i, (mem_id, embedding) in enumerate(zip(event.retrieved_ids, event.retrieved_embeddings)):
            # Compute similarity
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            similarity = float(np.dot(query_norm, emb_norm))

            # Temperature scaling
            score = similarity / self.contrastive_temperature

            # Softmax-like probability (higher = better match to query)
            # Clamp to prevent overflow
            score = min(score, 10.0)
            prob = np.exp(score)

            # Credit is scaled probability, weighted by position (earlier = better)
            position_weight = 1.0 / (i + 1)
            credit_value = float(self.positive_weight * prob * position_weight / (1 + prob))

            # Clamp to [-1, 1]
            credit_value = np.clip(credit_value, -1.0, 1.0)

            credit = ImplicitCredit(
                memory_id=mem_id,
                credit_type="contrastive",
                credit_value=credit_value,
                confidence=min(1.0, similarity + 0.5),  # Higher similarity = more confident
            )
            credits.append(credit)

            # Accumulate
            mem_key = str(mem_id)
            self._accumulated_credit[mem_key] = self._accumulated_credit.get(mem_key, 0.0) + credit_value

        return credits

    def _compute_coactivation_credit(
        self,
        event: RetrievalEvent,
    ) -> list[ImplicitCredit]:
        """
        Compute co-activation credit for memories retrieved together.

        Hebbian principle: Memories that fire together wire together.
        Co-retrieved memories get mutual credit boost.
        """
        credits = []
        ids = event.retrieved_ids

        if len(ids) < 2:
            return credits

        # Update coactivation matrix for all pairs
        for i, id1 in enumerate(ids):
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue

                key = (str(id1), str(id2)) if str(id1) < str(id2) else (str(id2), str(id1))
                current = self._coactivation_matrix.get(key, 0.0)
                self._coactivation_matrix[key] = current + self.coactivation_boost

                # Credit for this co-activation
                credit_value = self.coactivation_boost

                credit1 = ImplicitCredit(
                    memory_id=id1,
                    credit_type="coactivation",
                    credit_value=credit_value,
                    confidence=0.8,  # Medium-high confidence
                    source_ids=[id2],
                )
                credit2 = ImplicitCredit(
                    memory_id=id2,
                    credit_type="coactivation",
                    credit_value=credit_value,
                    confidence=0.8,
                    source_ids=[id1],
                )
                credits.extend([credit1, credit2])

                # Accumulate
                self._accumulated_credit[str(id1)] = self._accumulated_credit.get(str(id1), 0.0) + credit_value
                self._accumulated_credit[str(id2)] = self._accumulated_credit.get(str(id2), 0.0) + credit_value

        return credits

    def _compute_frequency_credit(
        self,
        event: RetrievalEvent,
    ) -> list[ImplicitCredit]:
        """
        Compute frequency-based credit.

        Often-retrieved memories are implicitly important.
        """
        credits = []

        for mem_id in event.retrieved_ids:
            mem_key = str(mem_id)

            # Update count
            count = self._retrieval_counts.get(mem_key, 0) + 1
            self._retrieval_counts[mem_key] = count

            # Credit based on log frequency (diminishing returns)
            log_freq = np.log1p(count)
            credit_value = float(self.frequency_scaling * log_freq / (log_freq + self.frequency_baseline))

            credit = ImplicitCredit(
                memory_id=mem_id,
                credit_type="frequency",
                credit_value=credit_value,
                confidence=min(1.0, count / 10.0),  # More retrievals = more confident
            )
            credits.append(credit)

            # Accumulate
            self._accumulated_credit[mem_key] = self._accumulated_credit.get(mem_key, 0.0) + credit_value

        return credits

    def propagate_outcome_credit(
        self,
        outcome_memory_id: UUID,
        outcome_value: float,
        lookback_window: timedelta = timedelta(hours=1),
    ) -> list[ImplicitCredit]:
        """
        Propagate explicit outcome credit to recently co-retrieved memories.

        When an outcome arrives for one memory, credit flows back along
        the temporal chain of retrievals.

        Args:
            outcome_memory_id: Memory that received explicit outcome
            outcome_value: The outcome value [0, 1]
            lookback_window: How far back to propagate credit

        Returns:
            List of propagated credits
        """
        credits = []
        now = datetime.now()
        cutoff = now - lookback_window

        # Find relevant retrieval events
        relevant_events = [
            e for e in self._retrieval_history
            if e.timestamp > cutoff and outcome_memory_id in e.retrieved_ids
        ]

        if not relevant_events:
            return credits

        # For each event where outcome memory appeared, credit co-retrieved memories
        for depth, event in enumerate(reversed(relevant_events)):
            if depth >= self.chain_max_depth:
                break

            discount = self.chain_discount ** depth

            for mem_id in event.retrieved_ids:
                if mem_id == outcome_memory_id:
                    continue

                credit_value = float(outcome_value * discount * 0.5)  # Half credit to associates

                credit = ImplicitCredit(
                    memory_id=mem_id,
                    credit_type="temporal",
                    credit_value=credit_value,
                    confidence=discount,  # Deeper in chain = less confident
                    source_ids=[outcome_memory_id],
                )
                credits.append(credit)

                # Accumulate
                mem_key = str(mem_id)
                self._accumulated_credit[mem_key] = self._accumulated_credit.get(mem_key, 0.0) + credit_value

        return credits

    def get_implicit_credit(self, memory_id: UUID) -> float:
        """
        Get accumulated implicit credit for a memory.

        Args:
            memory_id: Memory to query

        Returns:
            Accumulated credit value
        """
        return self._accumulated_credit.get(str(memory_id), 0.0)

    def get_retrieval_count(self, memory_id: UUID) -> int:
        """
        Get retrieval count for a memory.

        Args:
            memory_id: Memory to query

        Returns:
            Number of times retrieved
        """
        return self._retrieval_counts.get(str(memory_id), 0)

    def get_coactivation_strength(self, id1: UUID, id2: UUID) -> float:
        """
        Get coactivation strength between two memories.

        Args:
            id1: First memory
            id2: Second memory

        Returns:
            Coactivation strength
        """
        key = (str(id1), str(id2)) if str(id1) < str(id2) else (str(id2), str(id1))
        return self._coactivation_matrix.get(key, 0.0)

    def _apply_decay(self) -> None:
        """Apply daily decay to accumulated credits and coactivation."""
        now = datetime.now()
        days_elapsed = (now - self._last_decay).total_seconds() / 86400

        if days_elapsed < 0.1:  # Only decay every ~2.4 hours
            return

        decay_factor = self.credit_decay_rate ** days_elapsed

        # Decay accumulated credits
        self._accumulated_credit = {
            k: v * decay_factor
            for k, v in self._accumulated_credit.items()
            if abs(v * decay_factor) > 0.001  # Remove negligible credits
        }

        # Decay coactivation matrix
        self._coactivation_matrix = {
            k: v * self.coactivation_decay ** days_elapsed
            for k, v in self._coactivation_matrix.items()
            if v * self.coactivation_decay ** days_elapsed > 0.001
        }

        self._last_decay = now

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about self-supervised credit state.

        Returns:
            Dictionary of statistics
        """
        credits = list(self._accumulated_credit.values())
        return {
            "num_tracked_memories": len(self._accumulated_credit),
            "num_coactivation_pairs": len(self._coactivation_matrix),
            "total_retrievals": sum(self._retrieval_counts.values()),
            "retrieval_history_size": len(self._retrieval_history),
            "mean_credit": float(np.mean(credits)) if credits else 0.0,
            "max_credit": float(np.max(credits)) if credits else 0.0,
            "min_credit": float(np.min(credits)) if credits else 0.0,
        }

    def save_state(self) -> dict:
        """Save state for persistence."""
        return {
            "accumulated_credit": self._accumulated_credit.copy(),
            "retrieval_counts": self._retrieval_counts.copy(),
            "coactivation_matrix": {str(k): v for k, v in self._coactivation_matrix.items()},
            "last_decay": self._last_decay.isoformat(),
        }

    def load_state(self, state: dict) -> None:
        """Load state from persistence."""
        self._accumulated_credit = state.get("accumulated_credit", {})
        self._retrieval_counts = state.get("retrieval_counts", {})

        # Reconstruct coactivation matrix with tuple keys
        coact = state.get("coactivation_matrix", {})
        self._coactivation_matrix = {}
        for k, v in coact.items():
            # Parse string key back to tuple
            if isinstance(k, str) and k.startswith("("):
                parts = k[1:-1].replace("'", "").split(", ")
                if len(parts) == 2:
                    self._coactivation_matrix[(parts[0], parts[1])] = v
            else:
                self._coactivation_matrix[k] = v

        last_decay = state.get("last_decay")
        if last_decay:
            self._last_decay = datetime.fromisoformat(last_decay)
