"""
Fast Episodic Store for rapid memory encoding.

Biological inspiration: Hippocampal CA3 rapid encoding.

The Fast Episodic Store (FES) provides:
- 10K episode capacity
- One-shot learning (100x faster than semantic store)
- Salience-based eviction (DA/NE/ACh weighted)
- Consolidation flagging for high-replay episodes
"""

import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from t4dm.core.types import Episode

logger = logging.getLogger(__name__)


# Security limits
MAX_CAPACITY = 100000  # Maximum capacity to prevent memory exhaustion
MAX_TOP_K = 1000  # Maximum top_k for retrieval
MAX_CONSOLIDATION_CANDIDATES = 1000  # Maximum candidates per cycle


@dataclass
class FastEpisodicConfig:
    """Configuration for Fast Episodic Store."""
    capacity: int = 10000
    learning_rate: float = 0.1  # 100x faster than semantic (0.001)
    eviction_strategy: str = "lru_salience"
    consolidation_threshold: float = 0.7
    salience_weights: dict[str, float] = field(default_factory=lambda: {
        "da": 0.4,  # Dopamine - reward/novelty
        "ne": 0.3,  # Norepinephrine - arousal/attention
        "ach": 0.3  # Acetylcholine - learning signal
    })
    embedding_dim: int = 1024


@dataclass
class FESEntry:
    """Entry in Fast Episodic Store."""
    episode: Episode
    encoding: torch.Tensor
    salience: float
    timestamp: float
    access_count: int = 0
    consolidated: bool = False


class FastEpisodicStore:
    """
    Fast episodic memory with rapid learning and consolidation.

    Biological inspiration: Hippocampal CA3 rapid encoding.

    Features:
    - 10K episode capacity (configurable)
    - 100x faster learning than semantic store
    - Salience-based eviction (DA/NE/ACh weighted)
    - Consolidation flagging for high-replay episodes

    Integration with bioinspired encoding pipeline:
    - Uses SparseEncoder for pattern separation
    - Uses AttractorNetwork for pattern completion
    """

    def __init__(
        self,
        capacity: int = 10000,
        learning_rate: float = 0.1,
        eviction_strategy: str = "lru_salience",
        consolidation_threshold: float = 0.7,
        salience_weights: dict[str, float] | None = None,
        device: str | None = None
    ):
        # Security validation
        if capacity > MAX_CAPACITY:
            raise ValueError(f"capacity ({capacity}) exceeds MAX_CAPACITY ({MAX_CAPACITY})")
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if not 0 < learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if not 0 < consolidation_threshold <= 1:
            raise ValueError(f"consolidation_threshold must be in (0, 1], got {consolidation_threshold}")

        self.capacity = capacity
        self.learning_rate = learning_rate
        self.eviction_strategy = eviction_strategy
        self.consolidation_threshold = consolidation_threshold
        self.salience_weights = salience_weights or {"da": 0.4, "ne": 0.3, "ach": 0.3}

        # Device handling
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Storage
        self.entries: dict[str, FESEntry] = {}
        self.access_counts: dict[str, int] = defaultdict(int)

        # RACE-008 FIX: Thread lock for capacity management
        self._lock = threading.Lock()

        # Statistics
        self._total_writes = 0
        self._total_reads = 0
        self._eviction_count = 0

    def write(
        self,
        episode: Episode,
        encoding: torch.Tensor | None = None,
        neuromod_state: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """
        One-shot storage of episode.

        This is 100x faster than semantic store writes due to:
        - No embedding computation (done upstream if needed)
        - Simple dict storage vs vector DB
        - Lightweight salience computation

        RACE-008 FIX: Uses thread lock to prevent concurrent writes from
        exceeding capacity.

        Args:
            episode: Episode to store
            encoding: Pre-computed encoding (1024-dim BGE-M3 or sparse code)
            neuromod_state: Neuromodulator levels for salience

        Returns:
            Dict with episode_id, salience, storage status
        """
        episode_id = str(episode.id) if episode.id else str(uuid.uuid4())

        # Get encoding from episode if not provided
        if encoding is None:
            if episode.embedding is not None:
                encoding = torch.tensor(episode.embedding, device=self.device)
            else:
                # Create dummy encoding if none available
                encoding = torch.randn(1024, device=self.device)
                logger.warning(f"No encoding for episode {episode_id}, using random")

        encoding = encoding.to(self.device)

        # Compute salience
        salience = self._compute_salience(episode, neuromod_state)

        # RACE-008 FIX: Lock during capacity check and modification
        with self._lock:
            # CAP-001 FIX: Enforce capacity strictly
            # Check capacity and evict if needed
            while len(self.entries) >= self.capacity:
                evicted = self._evict()
                if evicted is None:
                    # All entries consolidated - force eviction by removing oldest
                    if self.entries:
                        oldest_id = min(
                            self.entries.keys(),
                            key=lambda k: self.entries[k].timestamp
                        )
                        logger.warning(
                            f"Forcing eviction of consolidated episode {oldest_id} "
                            f"to maintain capacity"
                        )
                        del self.entries[oldest_id]
                        if oldest_id in self.access_counts:
                            del self.access_counts[oldest_id]
                        self._eviction_count += 1
                    else:
                        # This shouldn't happen but be safe
                        break

            # Create entry
            entry = FESEntry(
                episode=episode,
                encoding=encoding,
                salience=salience,
                timestamp=time.time(),
                access_count=0,
                consolidated=False
            )

            self.entries[episode_id] = entry
            self._total_writes += 1

            return {
                "stored": True,
                "episode_id": episode_id,
                "salience": salience,
                "capacity_usage": len(self.entries) / self.capacity
            }

    def read(
        self,
        cue: str | torch.Tensor,
        top_k: int = 5
    ) -> list[tuple[Episode, float]]:
        """
        Retrieve episodes matching cue.

        Args:
            cue: Query string or encoding tensor
            top_k: Number of results to return

        Returns:
            List of (episode, similarity_score) tuples
        """
        # Security validation
        top_k = min(top_k, MAX_TOP_K, len(self.entries))

        if len(self.entries) == 0:
            return []

        # Handle string cue (would need embedding in real use)
        if isinstance(cue, str):
            logger.warning("String cue provided without embedding - using random vector")
            cue_encoding = torch.randn(1024, device=self.device)
        else:
            cue_encoding = cue.to(self.device)

        # Normalize cue
        cue_encoding = cue_encoding / (cue_encoding.norm() + 1e-8)

        # Compute similarities
        similarities = []
        for eid, entry in self.entries.items():
            entry_encoding = entry.encoding / (entry.encoding.norm() + 1e-8)
            sim = F.cosine_similarity(
                cue_encoding.unsqueeze(0),
                entry_encoding.unsqueeze(0)
            ).item()
            similarities.append((eid, sim))

        # Sort by similarity
        top_episodes = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

        # Update access counts
        for eid, _ in top_episodes:
            self.access_counts[eid] += 1
            self.entries[eid].access_count += 1

        self._total_reads += 1

        return [(self.entries[eid].episode, score) for eid, score in top_episodes]

    def get_consolidation_candidates(
        self,
        max_candidates: int = 100
    ) -> list[tuple[str, Episode, float]]:
        """
        Select episodes for consolidation to semantic store.

        Criteria:
        - High access count (replay frequency)
        - High salience
        - Age > threshold

        Args:
            max_candidates: Maximum number of candidates

        Returns:
            List of (episode_id, episode, consolidation_score) tuples
        """
        max_candidates = min(max_candidates, MAX_CONSOLIDATION_CANDIDATES)

        candidates = []
        current_time = time.time()

        for eid, entry in self.entries.items():
            if entry.consolidated:
                continue

            # Compute composite score
            replay_score = self.access_counts[eid]
            salience = entry.salience
            age_days = (current_time - entry.timestamp) / 86400

            # Consolidation score formula
            score = (
                0.5 * min(replay_score / 10, 1.0) +  # Normalize replay (cap at 10)
                0.3 * salience +
                0.2 * min(age_days, 7) / 7  # Normalize age (cap at 7 days)
            )

            if score > self.consolidation_threshold:
                candidates.append((eid, entry.episode, score))

        # Sort by score
        return sorted(candidates, key=lambda x: x[2], reverse=True)[:max_candidates]

    def mark_consolidated(self, episode_id: str) -> bool:
        """
        Mark episode as consolidated.

        Args:
            episode_id: Episode to mark

        Returns:
            True if marked, False if not found
        """
        if episode_id not in self.entries:
            return False

        self.entries[episode_id].consolidated = True
        return True

    def remove(self, episode_id: str) -> bool:
        """
        Remove episode from store.

        Args:
            episode_id: Episode to remove

        Returns:
            True if removed, False if not found
        """
        if episode_id not in self.entries:
            return False

        del self.entries[episode_id]
        if episode_id in self.access_counts:
            del self.access_counts[episode_id]
        return True

    def _compute_salience(
        self,
        episode: Episode,
        neuromod_state: dict[str, float] | None
    ) -> float:
        """
        Compute salience as weighted combination of neuromodulator levels.

        High salience = more important, less likely to be evicted.

        Args:
            episode: Episode being stored
            neuromod_state: Dict with 'dopamine', 'norepinephrine', 'acetylcholine'

        Returns:
            Salience score in [0, 1]
        """
        # Base salience from episode emotional valence
        base_salience = episode.emotional_valence if episode.emotional_valence else 0.5

        if not neuromod_state:
            return base_salience

        # Weighted neuromodulator combination
        neuro_salience = (
            self.salience_weights["da"] * neuromod_state.get("dopamine", 0) +
            self.salience_weights["ne"] * neuromod_state.get("norepinephrine", 0) +
            self.salience_weights["ach"] * neuromod_state.get("acetylcholine", 0)
        )

        # Combine base and neuro salience
        return 0.3 * base_salience + 0.7 * neuro_salience

    def _evict(self) -> str | None:
        """
        Evict lowest-value episode based on strategy.

        Strategies:
        - 'lru_salience': Lowest (salience × recency)
        - 'lru': Least recently used
        - 'salience': Lowest salience

        Returns:
            ID of evicted episode, or None
        """
        if not self.entries:
            return None

        current_time = time.time()
        scores = {}

        for eid, entry in self.entries.items():
            if entry.consolidated:
                # Don't evict consolidated episodes
                scores[eid] = float("inf")
                continue

            if self.eviction_strategy == "lru_salience":
                # Lower score = more likely to evict
                recency_hours = (current_time - entry.timestamp) / 3600
                scores[eid] = entry.salience / (1 + recency_hours)
            elif self.eviction_strategy == "lru":
                scores[eid] = -entry.timestamp  # Negative so older = lower
            elif self.eviction_strategy == "salience":
                scores[eid] = entry.salience
            else:
                scores[eid] = entry.salience / (1 + (current_time - entry.timestamp) / 3600)

        # Find victim (lowest score)
        victim_id = min(scores, key=scores.get)

        # Don't evict if victim has infinite score (consolidated)
        if scores[victim_id] == float("inf"):
            logger.warning("Cannot evict - all episodes are consolidated")
            return None

        del self.entries[victim_id]
        if victim_id in self.access_counts:
            del self.access_counts[victim_id]
        self._eviction_count += 1

        return victim_id

    def clear(self):
        """Clear all entries from store."""
        self.entries.clear()
        self.access_counts.clear()

    @property
    def count(self) -> int:
        """Number of stored episodes."""
        return len(self.entries)

    @property
    def capacity_usage(self) -> float:
        """Capacity usage ratio."""
        return len(self.entries) / self.capacity

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        consolidated = sum(1 for e in self.entries.values() if e.consolidated)
        avg_salience = (
            sum(e.salience for e in self.entries.values()) / len(self.entries)
            if self.entries else 0
        )
        avg_access = (
            sum(self.access_counts.values()) / len(self.access_counts)
            if self.access_counts else 0
        )

        return {
            "count": len(self.entries),
            "capacity": self.capacity,
            "capacity_usage": self.capacity_usage,
            "consolidated_count": consolidated,
            "total_writes": self._total_writes,
            "total_reads": self._total_reads,
            "eviction_count": self._eviction_count,
            "average_salience": avg_salience,
            "average_access_count": avg_access
        }
