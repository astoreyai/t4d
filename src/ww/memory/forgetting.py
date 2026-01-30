"""
PO-2: Active Forgetting System.

Implements biologically-inspired forgetting mechanisms:
- Interference-based forgetting (similar memories compete)
- Value-based forgetting (low importance memories pruned first)
- Decay-based forgetting (recency + access frequency)
- Configurable retention policies

References:
- Anderson & Neely (1996) - Interference and inhibition in memory retrieval
- Hardt et al. (2013) - Decay happens: The role of active forgetting
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class ForgettingStrategy(Enum):
    """Forgetting strategy types."""

    DECAY = "decay"  # Time-based decay
    INTERFERENCE = "interference"  # Similar memory competition
    VALUE = "value"  # Importance-based
    HYBRID = "hybrid"  # Combination of all


@dataclass
class RetentionPolicy:
    """Configuration for memory retention."""

    # Maximum memory count (hard limit)
    max_episodes: int = 100_000
    max_semantic_entities: int = 50_000
    max_procedures: int = 10_000

    # Soft limits (trigger forgetting when exceeded)
    soft_limit_ratio: float = 0.8  # Trigger at 80% of max

    # Age-based retention
    max_age_days: int = 365  # Oldest memories to keep
    archive_age_days: int = 90  # Age to archive rather than delete

    # Access-based retention
    min_access_count: int = 1  # Minimum accesses to retain
    access_decay_days: float = 30.0  # Half-life for access count

    # Importance thresholds
    min_importance: float = 0.1  # Below this = candidate for forgetting
    critical_importance: float = 0.9  # Never forget above this

    # Interference settings
    interference_threshold: float = 0.85  # Similarity to consider interference
    max_similar_memories: int = 5  # Keep only top N similar memories


@dataclass
class ForgettingCandidate:
    """A memory identified as a forgetting candidate."""

    memory_id: str
    memory_type: str  # "episode", "entity", "procedure"
    forgetting_score: float  # Higher = more likely to forget
    reasons: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    last_accessed: datetime | None = None
    access_count: int = 0
    importance: float = 0.5


@dataclass
class ForgettingResult:
    """Result of a forgetting cycle."""

    candidates_evaluated: int
    memories_forgotten: int
    memories_archived: int
    memories_retained: int
    bytes_freed: int = 0
    duration_seconds: float = 0.0
    strategy_used: ForgettingStrategy = ForgettingStrategy.HYBRID


class MemoryStore(Protocol):
    """Protocol for memory stores that support forgetting."""

    async def get_memory_count(self) -> int:
        """Get total memory count."""
        ...

    async def get_oldest_memories(
        self, limit: int, memory_type: str | None = None
    ) -> list[dict]:
        """Get oldest memories."""
        ...

    async def get_least_accessed(
        self, limit: int, memory_type: str | None = None
    ) -> list[dict]:
        """Get least accessed memories."""
        ...

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        ...

    async def archive_memory(self, memory_id: str, archive_path: str) -> bool:
        """Archive a memory to cold storage."""
        ...


class ActiveForgettingSystem:
    """
    Active forgetting system for bounded memory growth.

    Implements biologically-inspired forgetting to maintain
    a healthy memory system with bounded resource usage.
    """

    def __init__(
        self,
        policy: RetentionPolicy | None = None,
        strategy: ForgettingStrategy = ForgettingStrategy.HYBRID,
    ):
        self.policy = policy or RetentionPolicy()
        self.strategy = strategy

        # Statistics
        self._total_forgotten = 0
        self._total_archived = 0
        self._cycles_run = 0

        logger.info(
            f"ActiveForgettingSystem initialized: "
            f"strategy={strategy.value}, max_episodes={self.policy.max_episodes}"
        )

    def compute_decay_score(
        self,
        created_at: datetime,
        last_accessed: datetime | None,
        access_count: int,
    ) -> float:
        """
        Compute decay-based forgetting score.

        Higher score = more likely to forget.

        Args:
            created_at: When the memory was created
            last_accessed: Last access time (or None)
            access_count: Total access count

        Returns:
            Decay score (0-1)
        """
        now = datetime.now()

        # Age component (older = higher score)
        age_days = (now - created_at).total_seconds() / 86400
        age_score = min(1.0, age_days / self.policy.max_age_days)

        # Recency component (longer since access = higher score)
        if last_accessed:
            recency_days = (now - last_accessed).total_seconds() / 86400
        else:
            recency_days = age_days  # Never accessed = age since creation

        recency_score = min(1.0, recency_days / self.policy.access_decay_days)

        # Access frequency component (fewer accesses = higher score)
        # Use log scale for access count
        access_score = 1.0 / (1.0 + np.log1p(access_count))

        # Combine components (weighted average)
        decay_score = 0.3 * age_score + 0.4 * recency_score + 0.3 * access_score

        return float(decay_score)

    def compute_interference_score(
        self,
        memory_embedding: np.ndarray,
        similar_embeddings: list[np.ndarray],
        similar_importances: list[float],
    ) -> float:
        """
        Compute interference-based forgetting score.

        Memories that are highly similar to more important memories
        are candidates for forgetting (interference).

        Args:
            memory_embedding: Embedding of target memory
            similar_embeddings: Embeddings of similar memories
            similar_importances: Importance scores of similar memories

        Returns:
            Interference score (0-1)
        """
        if not similar_embeddings:
            return 0.0

        # Compute similarities
        memory_norm = memory_embedding / np.linalg.norm(memory_embedding)
        similarities = []

        for emb in similar_embeddings:
            emb_norm = emb / np.linalg.norm(emb)
            sim = float(np.dot(memory_norm, emb_norm))
            similarities.append(sim)

        # Count how many similar memories are more important
        more_important_count = 0
        max_importance_diff = 0.0

        for sim, imp in zip(similarities, similar_importances):
            if sim >= self.policy.interference_threshold:
                # This memory interferes
                importance_diff = imp - 0.5  # Assume target has 0.5 importance
                if importance_diff > 0:
                    more_important_count += 1
                    max_importance_diff = max(max_importance_diff, importance_diff)

        if more_important_count == 0:
            return 0.0

        # Score based on count and importance difference
        count_factor = min(1.0, more_important_count / self.policy.max_similar_memories)
        interference_score = 0.5 * count_factor + 0.5 * max_importance_diff

        return float(interference_score)

    def compute_value_score(self, importance: float) -> float:
        """
        Compute value-based forgetting score.

        Low importance = high forgetting score.

        Args:
            importance: Memory importance (0-1)

        Returns:
            Value score (0-1, higher = more likely to forget)
        """
        if importance >= self.policy.critical_importance:
            return 0.0  # Never forget critical memories

        # Invert importance: low importance = high score
        value_score = 1.0 - importance

        # Scale by threshold
        if importance < self.policy.min_importance:
            value_score = 1.0  # Definitely forget

        return float(value_score)

    def compute_forgetting_score(
        self,
        created_at: datetime,
        last_accessed: datetime | None,
        access_count: int,
        importance: float,
        memory_embedding: np.ndarray | None = None,
        similar_embeddings: list[np.ndarray] | None = None,
        similar_importances: list[float] | None = None,
    ) -> tuple[float, list[str]]:
        """
        Compute overall forgetting score using configured strategy.

        Args:
            created_at: Memory creation time
            last_accessed: Last access time
            access_count: Total accesses
            importance: Memory importance
            memory_embedding: Optional embedding for interference
            similar_embeddings: Optional similar memory embeddings
            similar_importances: Optional similar memory importances

        Returns:
            Tuple of (score, reasons)
        """
        reasons = []

        if self.strategy == ForgettingStrategy.DECAY:
            score = self.compute_decay_score(created_at, last_accessed, access_count)
            if score > 0.5:
                reasons.append(f"decay_score={score:.2f}")
            return score, reasons

        elif self.strategy == ForgettingStrategy.VALUE:
            score = self.compute_value_score(importance)
            if score > 0.5:
                reasons.append(f"low_importance={importance:.2f}")
            return score, reasons

        elif self.strategy == ForgettingStrategy.INTERFERENCE:
            if memory_embedding is None or not similar_embeddings:
                return 0.0, []
            score = self.compute_interference_score(
                memory_embedding,
                similar_embeddings,
                similar_importances or [],
            )
            if score > 0.5:
                reasons.append(f"interference_score={score:.2f}")
            return score, reasons

        else:  # HYBRID
            # Compute all components
            decay = self.compute_decay_score(created_at, last_accessed, access_count)
            value = self.compute_value_score(importance)

            interference = 0.0
            if memory_embedding is not None and similar_embeddings:
                interference = self.compute_interference_score(
                    memory_embedding,
                    similar_embeddings,
                    similar_importances or [],
                )

            # Weighted combination
            score = 0.4 * decay + 0.4 * value + 0.2 * interference

            if decay > 0.5:
                reasons.append(f"decay={decay:.2f}")
            if value > 0.5:
                reasons.append(f"low_value={value:.2f}")
            if interference > 0.3:
                reasons.append(f"interference={interference:.2f}")

            return float(score), reasons

    def identify_candidates(
        self,
        memories: list[dict],
        embeddings: dict[str, np.ndarray] | None = None,
        threshold: float = 0.6,
    ) -> list[ForgettingCandidate]:
        """
        Identify forgetting candidates from a list of memories.

        Args:
            memories: List of memory dicts with metadata
            embeddings: Optional dict of memory_id -> embedding
            threshold: Minimum score to be a candidate

        Returns:
            List of forgetting candidates, sorted by score (highest first)
        """
        candidates = []

        for mem in memories:
            memory_id = str(mem.get("id", mem.get("memory_id", "")))
            memory_type = mem.get("type", "episode")

            created_at = mem.get("created_at") or mem.get("timestamp")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            elif created_at is None:
                created_at = datetime.now() - timedelta(days=30)  # Default

            last_accessed = mem.get("last_accessed")
            if isinstance(last_accessed, str):
                last_accessed = datetime.fromisoformat(
                    last_accessed.replace("Z", "+00:00")
                )

            access_count = mem.get("access_count", 0)
            importance = mem.get("importance", 0.5)

            # Get embedding if available
            memory_embedding = None
            similar_embeddings = None
            similar_importances = None

            if embeddings and memory_id in embeddings:
                memory_embedding = embeddings[memory_id]
                # Find similar memories (simplified: would use vector index in practice)
                similar_embeddings = []
                similar_importances = []

            score, reasons = self.compute_forgetting_score(
                created_at=created_at,
                last_accessed=last_accessed,
                access_count=access_count,
                importance=importance,
                memory_embedding=memory_embedding,
                similar_embeddings=similar_embeddings,
                similar_importances=similar_importances,
            )

            if score >= threshold:
                candidates.append(
                    ForgettingCandidate(
                        memory_id=memory_id,
                        memory_type=memory_type,
                        forgetting_score=score,
                        reasons=reasons,
                        created_at=created_at,
                        last_accessed=last_accessed,
                        access_count=access_count,
                        importance=importance,
                    )
                )

        # Sort by score (highest first = forget first)
        candidates.sort(key=lambda c: c.forgetting_score, reverse=True)

        return candidates

    async def run_forgetting_cycle(
        self,
        store: MemoryStore,
        archive_path: str | None = None,
        max_to_forget: int = 100,
        dry_run: bool = False,
    ) -> ForgettingResult:
        """
        Run a forgetting cycle.

        Args:
            store: Memory store to operate on
            archive_path: Optional path for archiving (None = delete)
            max_to_forget: Maximum memories to forget this cycle
            dry_run: If True, don't actually delete/archive

        Returns:
            ForgettingResult with statistics
        """
        import time

        start_time = time.time()

        # Get current memory count
        current_count = await store.get_memory_count()
        soft_limit = int(self.policy.max_episodes * self.policy.soft_limit_ratio)

        if current_count < soft_limit:
            logger.debug(
                f"Memory count {current_count} below soft limit {soft_limit}, "
                f"skipping forgetting cycle"
            )
            return ForgettingResult(
                candidates_evaluated=0,
                memories_forgotten=0,
                memories_archived=0,
                memories_retained=current_count,
                duration_seconds=time.time() - start_time,
                strategy_used=self.strategy,
            )

        # Get candidate memories
        oldest = await store.get_oldest_memories(limit=max_to_forget * 2)
        least_accessed = await store.get_least_accessed(limit=max_to_forget * 2)

        # Combine and deduplicate
        all_candidates = {m.get("id"): m for m in oldest}
        for m in least_accessed:
            all_candidates[m.get("id")] = m

        candidates = self.identify_candidates(list(all_candidates.values()))

        forgotten = 0
        archived = 0

        for candidate in candidates[:max_to_forget]:
            if dry_run:
                logger.debug(
                    f"[DRY RUN] Would forget {candidate.memory_id}: "
                    f"score={candidate.forgetting_score:.2f}, "
                    f"reasons={candidate.reasons}"
                )
                forgotten += 1
                continue

            # Decide: archive or delete
            should_archive = (
                archive_path is not None
                and candidate.created_at is not None
                and (datetime.now() - candidate.created_at).days
                < self.policy.archive_age_days
            )

            if should_archive:
                success = await store.archive_memory(candidate.memory_id, archive_path)
                if success:
                    archived += 1
                    self._total_archived += 1
            else:
                success = await store.delete_memory(candidate.memory_id)
                if success:
                    forgotten += 1
                    self._total_forgotten += 1

        self._cycles_run += 1

        result = ForgettingResult(
            candidates_evaluated=len(candidates),
            memories_forgotten=forgotten,
            memories_archived=archived,
            memories_retained=current_count - forgotten - archived,
            duration_seconds=time.time() - start_time,
            strategy_used=self.strategy,
        )

        logger.info(
            f"Forgetting cycle complete: "
            f"evaluated={result.candidates_evaluated}, "
            f"forgotten={result.memories_forgotten}, "
            f"archived={result.memories_archived}"
        )

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get forgetting system statistics."""
        return {
            "strategy": self.strategy.value,
            "total_forgotten": self._total_forgotten,
            "total_archived": self._total_archived,
            "cycles_run": self._cycles_run,
            "policy": {
                "max_episodes": self.policy.max_episodes,
                "max_age_days": self.policy.max_age_days,
                "min_importance": self.policy.min_importance,
            },
        }


# Singleton instance
_forgetting_system: ActiveForgettingSystem | None = None


def get_forgetting_system(
    policy: RetentionPolicy | None = None,
    strategy: ForgettingStrategy = ForgettingStrategy.HYBRID,
) -> ActiveForgettingSystem:
    """Get or create the forgetting system singleton."""
    global _forgetting_system
    if _forgetting_system is None:
        _forgetting_system = ActiveForgettingSystem(policy, strategy)
    return _forgetting_system


def reset_forgetting_system() -> None:
    """Reset the forgetting system singleton."""
    global _forgetting_system
    _forgetting_system = None
