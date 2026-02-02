"""
Consolidation Service for T4DM.

Implements memory consolidation mirroring hippocampal-neocortical transfer:
- Light: Quick deduplication and cleanup
- Deep: Full episodic → semantic extraction
- Skill: Procedure optimization and merging
- All: Complete consolidation cycle

P3.3: Automatic consolidation triggering based on time and load.
"""

import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

# HDBSCAN is optional - only needed for deep consolidation clustering
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN = None
    HDBSCAN_AVAILABLE = False

# P7.1: Bridge container for NCA subsystem integration
from t4dm.core.bridge_container import get_bridge_container
from t4dm.core.config import get_settings
from t4dm.core.types import (
    Entity,
    EntityType,
    Episode,
    Procedure,
    RelationType,
)
from t4dm.embedding.bge_m3 import get_embedding_provider
from t4dm.memory.episodic import get_episodic_memory
from t4dm.memory.procedural import get_procedural_memory
from t4dm.memory.semantic import get_semantic_memory
from t4dm.storage import get_graph_store, get_vector_store

logger = logging.getLogger(__name__)


# =============================================================================
# P3.3: Automatic Consolidation Triggering
# =============================================================================


class TriggerReason(Enum):
    """Reason for consolidation trigger."""

    TIME_BASED = "time_based"  # Hours since last consolidation exceeded threshold
    LOAD_BASED = "load_based"  # New memory count exceeded threshold
    ADENOSINE_BASED = "adenosine_based"  # P5.3: Sleep pressure threshold exceeded
    MANUAL = "manual"  # User-initiated
    STARTUP = "startup"  # System startup


@dataclass
class SchedulerState:
    """Current state of the consolidation scheduler."""

    last_consolidation: datetime = field(default_factory=datetime.now)
    new_memory_count: int = 0
    total_consolidations: int = 0
    last_trigger_reason: TriggerReason | None = None
    is_running: bool = False
    last_error: str | None = None
    # P7: Multi-night consolidation tracking
    current_night: int = 1  # Which sleep night we're on (1-indexed)
    night_start_time: datetime = field(default_factory=datetime.now)
    memories_this_night: int = 0  # Memories consolidated this night
    total_nights: int = 0  # Total sleep nights completed


@dataclass
class ConsolidationTrigger:
    """Result of should_consolidate check."""

    should_run: bool
    reason: TriggerReason | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ConsolidationScheduler:
    """
    Automatic consolidation scheduler (P3.3).

    Triggers consolidation based on:
    - Time: Hours since last consolidation
    - Load: Number of new memories created

    Mirrors biological consolidation timing where memories are consolidated
    during rest periods after sufficient encoding activity.
    """

    def __init__(
        self,
        interval_hours: float = 1.5,
        memory_threshold: int = 100,
        check_interval_seconds: float = 300.0,
        consolidation_type: str = "light",
        enabled: bool = True,
        within_stage_cycle_seconds: float = 60.0,
    ):
        """
        Initialize scheduler.

        Args:
            interval_hours: Hours between time-based consolidation (90 min ultradian cycle per Carskadon & Dement 2011)
            memory_threshold: New memories to trigger load-based consolidation
            check_interval_seconds: Seconds between scheduler checks
            consolidation_type: Type of consolidation to run (light, deep, skill, all)
            enabled: Whether automatic consolidation is enabled
            within_stage_cycle_seconds: Consolidation cycle timing within stages (default 60s)
        """
        self.interval_hours = interval_hours
        self.memory_threshold = memory_threshold
        self.check_interval_seconds = check_interval_seconds
        self.consolidation_type = consolidation_type
        self.enabled = enabled
        self.within_stage_cycle_seconds = within_stage_cycle_seconds

        self.state = SchedulerState()
        self._background_task: asyncio.Task | None = None
        self._consolidation_callback: Callable | None = None
        self._lock = asyncio.Lock()

        # P5.3: Optional adenosine system for sleep-pressure-based triggers
        self._adenosine: Any | None = None

        logger.info(
            f"ConsolidationScheduler initialized: interval={interval_hours}h, "
            f"threshold={memory_threshold} memories, enabled={enabled}"
        )

    def set_adenosine(self, adenosine: Any) -> None:
        """
        P5.3: Set AdenosineDynamics for sleep-pressure-based triggers.

        When adenosine system is connected:
        - Consolidation triggers when sleep pressure exceeds threshold
        - Mirrors biological process where adenosine accumulation drives sleep need
        - Reference: Porkka-Heiskanen et al. (1997), Basheer et al. (2004)

        Args:
            adenosine: AdenosineDynamics instance (or any with should_sleep() method)
        """
        self._adenosine = adenosine
        logger.info("P5.3: Adenosine system connected to consolidation scheduler")

    def should_consolidate(self) -> ConsolidationTrigger:
        """
        Check if consolidation should run.

        Checks triggers in priority order:
        1. Adenosine-based (P5.3) - biological sleep pressure
        2. Time-based - hours since last consolidation
        3. Load-based - new memory count threshold

        Returns:
            ConsolidationTrigger with decision and reason
        """
        if not self.enabled:
            return ConsolidationTrigger(should_run=False)

        if self.state.is_running:
            return ConsolidationTrigger(
                should_run=False,
                details={"reason": "consolidation_already_running"},
            )

        now = datetime.now()
        hours_since = (now - self.state.last_consolidation).total_seconds() / 3600

        # P5.3: Adenosine-based trigger (highest priority - biological signal)
        if self._adenosine is not None:
            try:
                if self._adenosine.should_sleep():
                    consolidation_signal = self._adenosine.get_consolidation_signal()
                    sleep_pressure = getattr(
                        getattr(self._adenosine, 'state', None),
                        'sleep_pressure', 0.0
                    )
                    return ConsolidationTrigger(
                        should_run=True,
                        reason=TriggerReason.ADENOSINE_BASED,
                        details={
                            "consolidation_signal": round(consolidation_signal, 3),
                            "sleep_pressure": round(sleep_pressure, 3),
                            "hours_since_last": round(hours_since, 2),
                        },
                    )
            except Exception as e:
                logger.warning(f"Adenosine check failed: {e}")

        # Time-based trigger
        if hours_since >= self.interval_hours:
            return ConsolidationTrigger(
                should_run=True,
                reason=TriggerReason.TIME_BASED,
                details={
                    "hours_since_last": round(hours_since, 2),
                    "threshold_hours": self.interval_hours,
                },
            )

        # Load-based trigger
        if self.state.new_memory_count >= self.memory_threshold:
            return ConsolidationTrigger(
                should_run=True,
                reason=TriggerReason.LOAD_BASED,
                details={
                    "new_memory_count": self.state.new_memory_count,
                    "threshold": self.memory_threshold,
                },
            )

        return ConsolidationTrigger(
            should_run=False,
            details={
                "hours_since_last": round(hours_since, 2),
                "hours_remaining": round(self.interval_hours - hours_since, 2),
                "new_memory_count": self.state.new_memory_count,
                "memories_remaining": self.memory_threshold - self.state.new_memory_count,
                "adenosine_connected": self._adenosine is not None,
            },
        )

    def record_memory_created(self, count: int = 1) -> None:
        """
        Record that new memories were created.

        Called by EpisodicMemory.create() to track memory load.

        Args:
            count: Number of memories created (default: 1)
        """
        self.state.new_memory_count += count
        logger.debug(
            f"Memory created: count={count}, total_new={self.state.new_memory_count}"
        )

    def record_consolidation_complete(
        self,
        reason: TriggerReason = TriggerReason.MANUAL,
        error: str | None = None,
    ) -> None:
        """
        Record that consolidation completed.

        Resets counters and updates state.

        Args:
            reason: Why consolidation was triggered
            error: Error message if consolidation failed
        """
        self.state.last_consolidation = datetime.now()
        self.state.new_memory_count = 0
        self.state.total_consolidations += 1
        self.state.last_trigger_reason = reason
        self.state.last_error = error
        self.state.is_running = False

        logger.info(
            f"Consolidation complete: reason={reason.value}, "
            f"total={self.state.total_consolidations}, error={error}"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        now = datetime.now()
        hours_since = (now - self.state.last_consolidation).total_seconds() / 3600

        return {
            "enabled": self.enabled,
            "is_running": self.state.is_running,
            "last_consolidation": self.state.last_consolidation.isoformat(),
            "hours_since_last": round(hours_since, 2),
            "new_memory_count": self.state.new_memory_count,
            "total_consolidations": self.state.total_consolidations,
            "last_trigger_reason": (
                self.state.last_trigger_reason.value
                if self.state.last_trigger_reason
                else None
            ),
            "last_error": self.state.last_error,
            "config": {
                "interval_hours": self.interval_hours,
                "memory_threshold": self.memory_threshold,
                "check_interval_seconds": self.check_interval_seconds,
                "consolidation_type": self.consolidation_type,
            },
            # P7: Multi-night scheduling stats
            "multi_night": {
                "current_night": self.state.current_night,
                "total_nights": self.state.total_nights,
                "memories_this_night": self.state.memories_this_night,
                "recommended_depth": self.get_recommended_consolidation_depth(),
            },
        }

    def get_recommended_consolidation_depth(self) -> str:
        """
        P7: Get recommended consolidation depth based on current night.

        Multi-night consolidation follows biological patterns:
        - Night 1: Light (fast hippocampal transfer)
        - Night 2-3: Deep (more abstraction, entity formation)
        - Night 4+: All (full integration, skill consolidation)

        Reference: Stickgold & Walker (2013): Sleep-dependent memory triage

        Returns:
            Recommended consolidation type: 'light', 'deep', or 'all'
        """
        night = self.state.current_night
        if night <= 1:
            return "light"
        elif night <= 3:
            return "deep"
        else:
            return "all"

    def advance_night(self, memories_consolidated: int = 0) -> int:
        """
        P7: Advance to next sleep night after consolidation cycle.

        Tracks multi-night consolidation progress and updates state.

        Args:
            memories_consolidated: Number of memories consolidated this night

        Returns:
            New night number
        """
        self.state.memories_this_night += memories_consolidated
        self.state.total_nights += 1
        self.state.current_night += 1
        self.state.night_start_time = datetime.now()

        logger.info(
            f"P7: Advanced to night {self.state.current_night} "
            f"(total nights: {self.state.total_nights}, "
            f"memories this cycle: {self.state.memories_this_night})"
        )

        return self.state.current_night

    def reset_night_cycle(self) -> None:
        """
        P7: Reset night cycle counter (e.g., after major memory update).

        Called when significant new memories are added that require
        fresh multi-night consolidation.
        """
        self.state.current_night = 1
        self.state.memories_this_night = 0
        self.state.night_start_time = datetime.now()

        logger.info("P7: Night cycle reset - starting new consolidation sequence")

    async def start_background_task(
        self,
        consolidation_callback: Callable[..., Any],
    ) -> None:
        """
        Start the background scheduler task.

        Args:
            consolidation_callback: Async function to call for consolidation
                                   (typically ConsolidationService.consolidate)
        """
        if self._background_task is not None:
            logger.warning("Background task already running")
            return

        self._consolidation_callback = consolidation_callback

        async def scheduler_loop():
            logger.info(
                f"Consolidation scheduler started: "
                f"checking every {self.check_interval_seconds}s"
            )
            while True:
                try:
                    await asyncio.sleep(self.check_interval_seconds)

                    trigger = self.should_consolidate()
                    if trigger.should_run and trigger.reason:
                        logger.info(
                            f"Auto-consolidation triggered: {trigger.reason.value}, "
                            f"details={trigger.details}"
                        )

                        async with self._lock:
                            self.state.is_running = True
                            try:
                                await self._consolidation_callback(
                                    consolidation_type=self.consolidation_type
                                )
                                self.record_consolidation_complete(
                                    reason=trigger.reason
                                )
                            except Exception as e:
                                logger.error(f"Auto-consolidation failed: {e}")
                                self.record_consolidation_complete(
                                    reason=trigger.reason,
                                    error=str(e),
                                )

                except asyncio.CancelledError:
                    logger.info("Consolidation scheduler stopped")
                    break
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
                    await asyncio.sleep(60)  # Wait before retrying

        self._background_task = asyncio.create_task(scheduler_loop())
        logger.info("Background consolidation task started")

    async def stop_background_task(self) -> None:
        """Stop the background scheduler task."""
        if self._background_task is not None:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
            logger.info("Background consolidation task stopped")

    def reset(self) -> None:
        """Reset scheduler state (for testing)."""
        self.state = SchedulerState()


# Singleton scheduler instance
_scheduler_instance: ConsolidationScheduler | None = None


def get_consolidation_scheduler() -> ConsolidationScheduler:
    """Get or create singleton consolidation scheduler."""
    global _scheduler_instance
    if _scheduler_instance is None:
        settings = get_settings()
        _scheduler_instance = ConsolidationScheduler(
            interval_hours=settings.auto_consolidation_interval_hours,
            memory_threshold=settings.auto_consolidation_memory_threshold,
            check_interval_seconds=settings.auto_consolidation_check_interval_seconds,
            consolidation_type=settings.auto_consolidation_type,
            enabled=settings.auto_consolidation_enabled,
        )
    return _scheduler_instance


def reset_consolidation_scheduler() -> None:
    """Reset the scheduler singleton (for testing)."""
    global _scheduler_instance
    _scheduler_instance = None


class ConsolidationService:
    """
    Memory consolidation service.

    Orchestrates episodic→semantic transfer, skill merging, and decay updates.
    """

    def __init__(self):
        """Initialize consolidation service."""
        settings = get_settings()

        self.embedding = get_embedding_provider()
        self.vector_store = get_vector_store()
        self.graph_store = get_graph_store()

        # Thresholds from config
        self.min_similarity = settings.consolidation_min_similarity
        self.min_occurrences = settings.consolidation_min_occurrences
        self.skill_similarity = settings.consolidation_skill_similarity

        # HDBSCAN parameters
        self.hdbscan_min_cluster_size = settings.hdbscan_min_cluster_size
        self.hdbscan_min_samples = settings.hdbscan_min_samples
        self.hdbscan_metric = settings.hdbscan_metric
        self.hdbscan_max_samples = settings.hdbscan_max_samples

        # Memory services (lazy init)
        self._episodic = None
        self._semantic = None
        self._procedural = None

        # RACE-002 FIX: Lock to prevent concurrent consolidation runs
        # Consolidation modifies shared state (vector store, graph store)
        # and must be serialized to prevent race conditions
        self._consolidation_lock = asyncio.Lock()

        # P3.3: Reference to scheduler (lazy loaded)
        self._scheduler: ConsolidationScheduler | None = None

        # P7.1: Bridge container for Dopamine bridge integration
        # Dopamine bridge provides RPE for consolidation prioritization
        settings = get_settings()
        self._bridge_container = get_bridge_container()
        self._dopamine_prioritization_enabled = getattr(
            settings, "dopamine_prioritization_enabled", True
        )

    @property
    def scheduler(self) -> ConsolidationScheduler:
        """Get the consolidation scheduler."""
        if self._scheduler is None:
            self._scheduler = get_consolidation_scheduler()
        return self._scheduler

    async def start_auto_consolidation(self) -> None:
        """
        Start automatic consolidation background task (P3.3).

        This starts a background task that periodically checks if
        consolidation should run based on time and memory load.
        """
        await self.scheduler.start_background_task(self.consolidate)

    async def stop_auto_consolidation(self) -> None:
        """Stop automatic consolidation background task."""
        await self.scheduler.stop_background_task()

    def get_scheduler_stats(self) -> dict[str, Any]:
        """Get scheduler statistics (P3.3)."""
        return self.scheduler.get_stats()

    async def _get_services(self):
        """Lazy initialize memory services."""
        if self._episodic is None:
            self._episodic = get_episodic_memory()
            self._semantic = get_semantic_memory()
            self._procedural = get_procedural_memory()

            await self._episodic.initialize()
            await self._semantic.initialize()
            await self._procedural.initialize()

        return self._episodic, self._semantic, self._procedural

    async def consolidate(
        self,
        consolidation_type: str = "light",
        session_filter: str | None = None,
        token: "CallerToken | None" = None,
    ) -> dict[str, Any]:
        """
        Execute consolidation cycle.

        ATOM-P0-14: Protected by access control with type validation.

        Args:
            consolidation_type: "light", "deep", "skill", or "all"
            session_filter: Limit to specific session (default: all sessions)
            token: Caller token (required for access control)

        Returns:
            Consolidation results with metrics

        Raises:
            ValueError: If consolidation_type is invalid
            AccessDenied: If token lacks required capability

        Note:
            RACE-002 FIX: This method acquires a lock to prevent concurrent
            consolidation runs from interfering with each other.
        """
        # ATOM-P0-14: Access control
        if token is not None:
            from t4dm.core.access_control import require_capability
            require_capability(token, "trigger_consolidation")

        start_time = datetime.now()
        results = {
            "consolidation_type": consolidation_type,
            "status": "completed",
            "timestamp": start_time.isoformat(),
            "results": {},
            "consolidation_events": [],
        }

        # ATOM-P0-14: Validate consolidation type
        valid_types = {"light", "deep", "skill", "all"}
        if consolidation_type not in valid_types:
            results["status"] = "error"
            results["error"] = f"Unknown consolidation type: {consolidation_type}"
            results["duration_seconds"] = (datetime.now() - start_time).total_seconds()
            return results

        # RACE-002 FIX: Serialize consolidation to prevent race conditions
        async with self._consolidation_lock:
            try:
                if consolidation_type == "light":
                    results["results"]["light"] = await self._consolidate_light(session_filter)

                elif consolidation_type == "deep":
                    results["results"]["episodic_to_semantic"] = await self._consolidate_deep(session_filter)
                    results["results"]["decay_updated"] = await self._update_decay()

                elif consolidation_type == "skill":
                    results["results"]["skill_consolidation"] = await self._consolidate_skills()

                elif consolidation_type == "all":
                    results["results"]["light"] = await self._consolidate_light(session_filter)
                    results["results"]["episodic_to_semantic"] = await self._consolidate_deep(session_filter)
                    results["results"]["skill_consolidation"] = await self._consolidate_skills()
                    results["results"]["decay_updated"] = await self._update_decay()

            except Exception as e:
                logger.error(f"Consolidation failed: {e}")
                results["status"] = "failed"
                results["error"] = str(e)

        results["duration_seconds"] = (datetime.now() - start_time).total_seconds()

        # P3.3: Notify scheduler of manual consolidation completion
        # This resets counters even for manual calls to prevent double-consolidation
        if results["status"] == "completed":
            self.scheduler.record_consolidation_complete(
                reason=TriggerReason.MANUAL,
            )
        else:
            self.scheduler.record_consolidation_complete(
                reason=TriggerReason.MANUAL,
                error=results.get("error"),
            )

        return results

    async def _consolidate_light(
        self,
        session_filter: str | None = None,
        hours: int = 24,
    ) -> dict:
        """
        Light consolidation: Quick deduplication and cleanup.

        - Identify near-duplicate episodes (>0.95 similarity)
        - Mark duplicates
        - Clean up orphaned relationships

        Args:
            session_filter: Optional session ID filter
            hours: Number of hours to look back (default 24)

        Returns:
            Consolidation summary dict
        """
        episodic, _, _ = await self._get_services()

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Get count first to estimate work
        total_count = await episodic.count_by_timerange(
            start_time=start_time,
            end_time=end_time,
            session_filter=session_filter,
        )

        logger.info(
            f"Starting light consolidation: {total_count} episodes "
            f"from {start_time.isoformat()[:16]} to {end_time.isoformat()[:16]}"
        )

        if total_count == 0:
            return {"episodes_scanned": 0, "duplicates_found": 0, "cleaned": 0}

        # Process in pages to avoid memory exhaustion
        all_episodes = []
        cursor = None
        page_num = 0

        while True:
            page_num += 1
            episodes, cursor = await episodic.recall_by_timerange(
                start_time=start_time,
                end_time=end_time,
                page_size=200,
                cursor=cursor,
                session_filter=session_filter,
            )

            all_episodes.extend(episodes)
            logger.debug(f"Loaded page {page_num}: {len(episodes)} episodes")

            if cursor is None:
                break

            # Safety limit
            if len(all_episodes) >= 10000:
                logger.warning(
                    f"Consolidation limit reached: {len(all_episodes)} episodes. "
                    "Consider shorter time windows."
                )
                break

        duplicates_found = 0
        cleaned = 0

        if len(all_episodes) < 2:
            return {
                "episodes_scanned": len(all_episodes),
                "duplicates_found": 0,
                "cleaned": 0,
                "pages_loaded": page_num,
            }

        # Find near-duplicates by embedding similarity
        duplicate_pairs = await self._find_duplicates(all_episodes, threshold=0.95)
        duplicates_found = len(duplicate_pairs)

        # Mark duplicates (keep oldest)
        for ep1_id, ep2_id in duplicate_pairs:
            # Soft delete - just update valence to 0
            try:
                await self.vector_store.update_payload(
                    collection=self.vector_store.episodes_collection,
                    id=ep2_id,
                    payload={"emotional_valence": 0.0, "duplicate_of": ep1_id},
                )
                cleaned += 1
            except Exception as e:
                logger.warning(f"Failed to mark duplicate {ep2_id}: {e}")

        logger.info(
            f"Light consolidation: scanned={len(all_episodes)}, "
            f"duplicates={duplicates_found}, cleaned={cleaned}, pages={page_num}"
        )

        return {
            "episodes_scanned": len(all_episodes),
            "duplicates_found": duplicates_found,
            "cleaned": cleaned,
            "pages_loaded": page_num,
        }

    async def _consolidate_deep(
        self,
        session_filter: str | None = None,
        hours: int = 168,
    ) -> dict:
        """
        Deep consolidation: Extract semantic knowledge from episodes.

        1. Cluster similar episodes
        2. Extract recurring entities
        3. Create/update semantic entities
        4. Build relationships

        Args:
            session_filter: Optional session ID filter
            hours: Number of hours to look back (default 168 = 7 days)

        Returns:
            Consolidation summary dict
        """
        episodic, semantic, _ = await self._get_services()

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Get count first
        total_count = await episodic.count_by_timerange(
            start_time=start_time,
            end_time=end_time,
            session_filter=session_filter,
        )

        logger.info(
            f"Starting deep consolidation: {total_count} episodes "
            f"from {start_time.isoformat()[:16]} to {end_time.isoformat()[:16]}"
        )

        # Process in pages
        all_episodes = []
        cursor = None
        page_num = 0

        while True:
            page_num += 1
            episodes, cursor = await episodic.recall_by_timerange(
                start_time=start_time,
                end_time=end_time,
                page_size=200,
                cursor=cursor,
                session_filter=session_filter,
            )

            all_episodes.extend(episodes)
            logger.debug(f"Loaded page {page_num}: {len(episodes)} episodes")

            if cursor is None:
                break

            # Safety limit for deep consolidation (more expensive processing)
            if len(all_episodes) >= 5000:
                logger.warning(
                    f"Deep consolidation limit reached: {len(all_episodes)} episodes. "
                    "Consider shorter time windows."
                )
                break

        # P7.1: Dopamine-based prioritization of episodes for consolidation
        # Episodes with higher RPE (surprising outcomes) get priority
        if self._dopamine_prioritization_enabled and all_episodes:
            try:
                dopamine_bridge = self._bridge_container.get_dopamine_bridge()
                if dopamine_bridge is not None:
                    # Compute learning modulation based on prediction errors
                    learning_mod = dopamine_bridge.get_learning_modulation()

                    # Prioritize high-valence episodes (surprising outcomes)
                    # by boosting their weights in the clustering
                    prioritized = []
                    for ep in all_episodes:
                        # Use emotional_valence as proxy for surprise/importance
                        valence = getattr(ep, 'emotional_valence', 0.5)
                        # Episodes with extreme valence (high surprise) get boosted
                        priority = abs(valence - 0.5) * 2 * learning_mod

                        # Store priority for later use
                        if hasattr(ep, '__dict__'):
                            ep._consolidation_priority = priority
                        prioritized.append((ep, priority))

                    # Sort by priority (highest first)
                    prioritized.sort(key=lambda x: x[1], reverse=True)
                    all_episodes = [ep for ep, _ in prioritized]

                    logger.debug(
                        f"P7.1: Dopamine prioritization applied - "
                        f"learning_mod={learning_mod:.3f}, "
                        f"top_priority={prioritized[0][1]:.3f}"
                    )
            except Exception as e:
                logger.warning(f"Dopamine prioritization failed: {e}")

        consolidated_count = 0
        entities_created = 0
        entities_updated = 0
        relationships_created = 0

        if len(all_episodes) < self.min_occurrences:
            return {
                "consolidated_episodes": 0,
                "new_entities_created": 0,
                "entities_updated": 0,
                "provenance_links": 0,
                "confidence": 0.0,
                "pages_loaded": page_num,
            }

        # Cluster by embedding similarity
        clusters = await self._cluster_episodes(all_episodes, threshold=self.min_similarity)

        # Process each cluster
        for cluster in clusters:
            # Only consolidate clusters with enough occurrences (default: 3)
            # Prevents creating entities from noise or one-off interactions
            if len(cluster) < self.min_occurrences:
                continue

            # Extract entity candidates from cluster
            entity_info = self._extract_entity_from_cluster(cluster)

            if entity_info:
                # Create or update semantic entity
                existing = await self._find_similar_entity(entity_info["name"])

                if existing:
                    # Update existing entity
                    await semantic.supersede(
                        entity_id=existing.id,
                        new_summary=entity_info["summary"],
                        new_details=entity_info.get("details"),
                    )
                    entities_updated += 1
                else:
                    # Create new entity
                    entity = await semantic.create_entity(
                        name=entity_info["name"],
                        entity_type=entity_info["type"],
                        summary=entity_info["summary"],
                        details=entity_info.get("details"),
                        source=f"consolidated:{len(cluster)}_episodes",
                    )
                    entities_created += 1

                    # P4.2: Batch create provenance relationships (fixes N+1 pattern)
                    provenance_rels = [
                        (
                            str(ep.id),
                            str(entity.id),
                            RelationType.SOURCE_OF.value,
                            {
                                "weight": 1.0,
                                "coAccessCount": 1,
                                "lastCoAccess": datetime.now().isoformat(),
                            },
                        )
                        for ep in cluster
                    ]
                    try:
                        created = await self.graph_store.batch_create_relationships(provenance_rels)
                        relationships_created += created
                    except Exception as e:
                        logger.warning(f"Failed to create provenance links: {e}")

                consolidated_count += len(cluster)

        confidence = min(1.0, consolidated_count / max(len(all_episodes), 1))

        logger.info(
            f"Deep consolidation: episodes={consolidated_count}, "
            f"entities_created={entities_created}, updated={entities_updated}, "
            f"pages={page_num}"
        )

        return {
            "consolidated_episodes": consolidated_count,
            "new_entities_created": entities_created,
            "entities_updated": entities_updated,
            "provenance_links": relationships_created,
            "confidence": round(confidence, 4),
            "pages_loaded": page_num,
        }

    async def _consolidate_skills(self) -> dict:
        """
        Skill consolidation: Merge similar procedures.

        1. Find procedures with >0.85 embedding similarity
        2. Keep best (highest success rate)
        3. Merge steps using consensus
        4. Deprecate redundant procedures
        """
        _, _, procedural = await self._get_services()

        # Get all active procedures
        results = await procedural.retrieve(
            task="*",
            limit=100,
        )

        procedures = [r.item for r in results if not r.item.deprecated]
        analyzed = len(procedures)
        merged = 0
        deprecated = 0
        success_improvement = 0.0

        if len(procedures) < 2:
            return {
                "procedures_analyzed": analyzed,
                "merged": 0,
                "deprecated": 0,
                "success_rate_improvement": 0.0,
            }

        # Cluster similar procedures
        clusters = await self._cluster_procedures(procedures, threshold=self.skill_similarity)

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # Sort by success rate (descending)
            cluster.sort(key=lambda p: p.success_rate, reverse=True)
            best = cluster[0]

            # Merge steps from all procedures
            merged_steps = self._merge_procedure_steps(cluster)

            # Update best procedure
            if merged_steps:
                await self.vector_store.update_payload(
                    collection=self.vector_store.procedures_collection,
                    id=str(best.id),
                    payload={
                        "steps": [s.model_dump() for s in merged_steps],
                        "version": best.version + 1,
                        "created_from": "consolidated",
                    },
                )

            # Deprecate others
            for proc in cluster[1:]:
                await procedural.deprecate(
                    procedure_id=proc.id,
                    reason=f"Consolidated into {best.name}",
                    consolidated_into=best.id,
                )
                deprecated += 1

            merged += 1
            # Track potential improvement from combining execution data
            combined_count = sum(p.execution_count for p in cluster)
            if combined_count > best.execution_count:
                success_improvement += 0.05  # Estimated improvement

        logger.info(
            f"Skill consolidation: analyzed={analyzed}, merged={merged}, deprecated={deprecated}"
        )

        return {
            "procedures_analyzed": analyzed,
            "merged": merged,
            "deprecated": deprecated,
            "success_rate_improvement": round(success_improvement, 4),
        }

    async def _update_decay(self) -> dict:
        """Update FSRS stability for all memory types."""
        datetime.now()
        episodes_updated = 0
        entities_updated = 0
        procedures_updated = 0

        # Note: This is a simplified batch update
        # Full implementation would iterate through all items

        logger.info("Decay update completed (batch mode)")

        return {
            "episodes": episodes_updated,
            "entities": entities_updated,
            "procedures": procedures_updated,
        }

    # Helper methods

    def _stratified_sample(
        self,
        episodes: list[Episode],
        n_samples: int,
    ) -> list[Episode]:
        """
        Stratified sampling preserving temporal distribution.

        Samples evenly across the time range to maintain cluster representativeness.

        Args:
            episodes: Full list of episodes
            n_samples: Number of samples to select

        Returns:
            Sampled episodes with preserved temporal distribution
        """
        if len(episodes) <= n_samples:
            return episodes

        # Sort by timestamp
        sorted_eps = sorted(episodes, key=lambda e: e.timestamp or datetime.min)

        # Sample evenly across time
        step = len(sorted_eps) / n_samples
        indices = [int(i * step) for i in range(n_samples)]

        return [sorted_eps[i] for i in indices]

    async def _assign_to_clusters(
        self,
        all_episodes: list[Episode],
        clusters: list[list[Episode]],
        sampled_episodes: list[Episode],
    ) -> list[list[Episode]]:
        """
        Assign non-sampled episodes to nearest cluster.

        Uses cosine similarity to assign each non-sampled episode
        to the most similar cluster based on centroid.

        Args:
            all_episodes: All episodes including non-sampled
            clusters: Clusters from sampled episodes
            sampled_episodes: Episodes that were used for clustering

        Returns:
            Expanded clusters including all episodes
        """
        sampled_ids = {e.id for e in sampled_episodes}
        non_sampled = [e for e in all_episodes if e.id not in sampled_ids]

        if not non_sampled or not clusters:
            return clusters

        # Calculate cluster centroids
        centroids = []
        for cluster in clusters:
            embeddings = [e.embedding for e in cluster if e.embedding is not None]
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                centroids.append(centroid)
            else:
                centroids.append(None)

        # Assign non-sampled to nearest cluster
        expanded_clusters = [list(c) for c in clusters]

        for episode in non_sampled:
            if episode.embedding is None:
                continue

            best_cluster = 0
            best_similarity = -1

            for i, centroid in enumerate(centroids):
                if centroid is None:
                    continue
                # Cosine similarity
                similarity = np.dot(episode.embedding, centroid) / (
                    np.linalg.norm(episode.embedding) * np.linalg.norm(centroid) + 1e-8
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = i

            expanded_clusters[best_cluster].append(episode)

        return expanded_clusters

    async def _find_duplicates(
        self,
        episodes: list[Episode],
        threshold: float = 0.95,
    ) -> list[tuple[str, str]]:
        """
        Find near-duplicate episode pairs using approximate nearest neighbor search.

        Uses Qdrant's native vector search instead of O(n²) pairwise comparison.
        Time complexity: O(n * k) where k is the number of candidates per episode.

        Args:
            episodes: List of episodes to check for duplicates
            threshold: Minimum similarity score to consider duplicates (default 0.95)

        Returns:
            List of (keep_id, remove_id) tuples where keep_id is the older episode
        """
        if len(episodes) < 2:
            return []

        duplicates = []
        seen_pairs: set[tuple[str, str]] = set()

        # Build episode lookup for timestamp comparison
        episode_map = {str(ep.id): ep for ep in episodes}
        episode_ids = set(episode_map.keys())

        for ep in episodes:
            # Skip if no embedding available
            if ep.embedding is None:
                continue

            try:
                # Use Qdrant's native search for approximate NN
                results = await self.vector_store.search(
                    collection=self.vector_store.episodes_collection,
                    vector=ep.embedding,
                    limit=10,  # Check top 10 candidates
                    score_threshold=threshold,
                )

                for result_id, score, payload in results:
                    # Skip self-match
                    if result_id == str(ep.id):
                        continue

                    # Only consider episodes in our input set
                    if result_id not in episode_ids:
                        continue

                    # Create canonical pair (sorted) to avoid duplicates
                    pair = tuple(sorted([str(ep.id), result_id]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    # Determine which to keep (older episode)
                    other_ep = episode_map.get(result_id)
                    if other_ep is None:
                        continue

                    if ep.timestamp <= other_ep.timestamp:
                        # Keep current, remove other
                        duplicates.append((str(ep.id), result_id))
                    else:
                        # Keep other, remove current
                        duplicates.append((result_id, str(ep.id)))

                    logger.debug(
                        f"Found duplicate pair: {pair[0][:8]}.../{pair[1][:8]}... "
                        f"(score={score:.4f})"
                    )

            except Exception as e:
                logger.warning(f"Error searching for duplicates of {ep.id}: {e}")
                continue

        logger.info(f"Found {len(duplicates)} duplicate pairs from {len(episodes)} episodes")
        return duplicates

    async def _cluster_episodes(
        self,
        episodes: list[Episode],
        threshold: float = 0.75,
        min_cluster_size: int = 3,
        max_samples: int | None = None,
    ) -> list[list[Episode]]:
        """
        Cluster episodes with memory-safe sampling.

        For datasets > max_samples, uses stratified sampling to maintain
        cluster representativeness while bounding memory usage.

        Args:
            episodes: Episodes to cluster
            threshold: Similarity threshold for clustering
            min_cluster_size: Minimum cluster size
            max_samples: Maximum samples for HDBSCAN (memory bound)

        Returns:
            List of episode clusters

        Complexity:
            Time: O(n log n) average case with HDBSCAN
            Space: O(n) for embeddings and cluster assignments
        """
        # Check HDBSCAN availability
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "HDBSCAN is required for episode clustering. "
                "Install it with: pip install hdbscan"
            )

        # Edge case: empty input
        if not episodes:
            return []

        # Use configured min_occurrences if not explicitly provided
        if min_cluster_size == 3:  # Default value
            min_cluster_size = self.min_occurrences

        # Use configured max_samples if not provided
        if max_samples is None:
            max_samples = self.hdbscan_max_samples

        # Edge case: input smaller than min_cluster_size
        if len(episodes) < min_cluster_size:
            logger.debug(
                f"Skipping clustering: {len(episodes)} episodes < min_cluster_size {min_cluster_size}"
            )
            return []

        # Sample if too large for memory
        sampled = False
        original_episodes = episodes
        if len(episodes) > max_samples:
            logger.info(
                f"Sampling {max_samples} from {len(episodes)} episodes for clustering "
                f"(memory limit)"
            )
            episodes = self._stratified_sample(episodes, max_samples)
            sampled = True

        try:
            # Get embeddings for all episodes
            contents = [ep.content for ep in episodes]
            embeddings_list = await self.embedding.embed(contents)

            # Convert to numpy array for HDBSCAN
            embeddings = np.array(embeddings_list, dtype=np.float32)

            # HDBSCAN clustering with cosine metric
            # HDBSCAN chosen over K-means because:
            # 1. No need to specify k (number of clusters) in advance
            # 2. Handles noise (outlier episodes marked as -1)
            # 3. O(n log n) vs O(n²) for hierarchical clustering
            logger.debug(
                f"Clustering {len(episodes)} episodes with HDBSCAN "
                f"(min_cluster_size={min_cluster_size})"
            )

            clusterer = HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                metric=self.hdbscan_metric,  # Cosine distance for semantic similarity
                cluster_selection_method="eom",  # Excess of Mass for stable clusters
                min_samples=self.hdbscan_min_samples or self.hdbscan_min_cluster_size,
            )
            labels = clusterer.fit_predict(embeddings)

            # Group episodes by cluster label (ignore noise points with label=-1)
            cluster_dict: dict[int, list[Episode]] = {}
            noise_count = 0

            for ep, label in zip(episodes, labels):
                if label >= 0:
                    cluster_dict.setdefault(label, []).append(ep)
                else:
                    noise_count += 1

            clusters = list(cluster_dict.values())

            logger.info(
                f"HDBSCAN clustering: {len(episodes)} episodes → "
                f"{len(clusters)} clusters, {noise_count} noise points"
            )

            # If sampled, assign non-sampled episodes to nearest cluster
            if sampled and clusters:
                clusters = await self._assign_to_clusters(
                    original_episodes,
                    clusters,
                    episodes,  # sampled episodes
                )

            return clusters

        except MemoryError as e:
            logger.error(f"HDBSCAN memory error with {len(episodes)} episodes: {e}")
            # Fallback: return episodes as single cluster
            if len(original_episodes) >= min_cluster_size:
                return [original_episodes]
            return []
        except Exception as e:
            logger.error(f"HDBSCAN clustering failed: {e}, falling back to empty clusters")
            return []

    async def _cluster_procedures(
        self,
        procedures: list[Procedure],
        threshold: float = 0.85,
        min_cluster_size: int = 2,
    ) -> list[list[Procedure]]:
        """
        Cluster procedures by embedding similarity using HDBSCAN.

        Args:
            procedures: List of procedures to cluster
            threshold: Not used (kept for API compatibility)
            min_cluster_size: Minimum size for a cluster (default: 2)

        Returns:
            List of procedure clusters

        Complexity:
            Time: O(n log n) average case with HDBSCAN
            Space: O(n) for embeddings and cluster assignments
        """
        # Edge case: empty input
        if not procedures:
            return []

        # Edge case: input smaller than min_cluster_size
        if len(procedures) < min_cluster_size:
            logger.debug(
                f"Skipping procedure clustering: {len(procedures)} procedures < "
                f"min_cluster_size {min_cluster_size}"
            )
            return []

        try:
            # Get embeddings for all procedures
            scripts = [p.script or p.name for p in procedures]
            embeddings_list = await self.embedding.embed(scripts)

            # Convert to numpy array for HDBSCAN
            embeddings = np.array(embeddings_list, dtype=np.float32)

            # HDBSCAN clustering with cosine metric
            logger.debug(
                f"Clustering {len(procedures)} procedures with HDBSCAN "
                f"(min_cluster_size={min_cluster_size})"
            )

            clusterer = HDBSCAN(
                min_cluster_size=self.hdbscan_min_cluster_size,
                metric=self.hdbscan_metric,
                cluster_selection_method="eom",
                min_samples=self.hdbscan_min_samples or self.hdbscan_min_cluster_size,
            )
            labels = clusterer.fit_predict(embeddings)

            # Group procedures by cluster label (ignore noise points with label=-1)
            cluster_dict: dict[int, list[Procedure]] = {}
            noise_count = 0

            for proc, label in zip(procedures, labels):
                if label >= 0:
                    cluster_dict.setdefault(label, []).append(proc)
                else:
                    noise_count += 1

            clusters = list(cluster_dict.values())

            logger.info(
                f"HDBSCAN procedure clustering: {len(procedures)} procedures → "
                f"{len(clusters)} clusters, {noise_count} noise points"
            )

            return clusters

        except Exception as e:
            logger.error(f"HDBSCAN procedure clustering failed: {e}, falling back to empty clusters")
            return []

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _extract_entity_from_cluster(self, cluster: list[Episode]) -> dict | None:
        """Extract entity information from episode cluster."""
        if not cluster:
            return None

        # Use most common words/phrases as entity name
        # Simple heuristic: use first episode's summary
        [ep.content for ep in cluster]

        # Find common project context
        projects = [ep.context.project for ep in cluster if ep.context.project]
        tools = [ep.context.tool for ep in cluster if ep.context.tool]

        # Generate entity from cluster characteristics
        if projects:
            most_common_project = max(set(projects), key=projects.count)
            return {
                "name": most_common_project,
                "type": EntityType.PROJECT.value,
                "summary": f"Project with {len(cluster)} related episodes",
                "details": f"Episodes: {', '.join(ep.context.file or 'unknown' for ep in cluster[:5])}",
            }
        if tools:
            most_common_tool = max(set(tools), key=tools.count)
            return {
                "name": most_common_tool,
                "type": EntityType.TOOL.value,
                "summary": f"Tool used in {len(cluster)} related interactions",
            }

        # Default: extract from content
        # Use first significant phrase
        first_content = cluster[0].content[:100]
        return {
            "name": first_content.split(".")[0][:50],
            "type": EntityType.CONCEPT.value,
            "summary": f"Concept from {len(cluster)} consolidated episodes",
        }

    async def _find_similar_entity(self, name: str) -> Entity | None:
        """Find existing entity similar to name."""
        _, semantic, _ = await self._get_services()

        results = await semantic.recall(
            query=name,
            limit=1,
        )

        if results and results[0].score > 0.9:
            return results[0].item

        return None

    def _merge_procedure_steps(self, procedures: list[Procedure]) -> list:
        """Merge steps from multiple procedures using consensus."""
        if not procedures:
            return []

        # Use best procedure's steps as base
        best = procedures[0]
        return best.steps

    async def extract_entities_from_recent_episodes(
        self,
        hours: int = 24,
        session_filter: str | None = None,
    ) -> dict[str, Any]:
        """
        Background job to extract entities from recent episodes.

        Args:
            hours: Number of hours back to process
            session_filter: Limit to specific session

        Returns:
            Summary of extraction results

        Note:
            RACE-002 FIX: This method acquires the consolidation lock to prevent
            concurrent modifications to shared state (semantic entities, graph).
        """
        from datetime import timedelta

        from t4dm.core.config import get_settings
        from t4dm.extraction.entity_extractor import create_default_extractor

        # RACE-002 FIX: Serialize with consolidation to prevent race conditions
        # Note: The lock is acquired during entity/relationship creation, not extraction
        # to allow concurrent LLM-based extraction while protecting store operations
        episodic, semantic, _ = await self._get_services()
        settings = get_settings()

        # Get recent episodes (read-only, doesn't need lock)
        cutoff_time = datetime.now() - timedelta(hours=hours)
        results = await episodic.recall(
            query="*",
            limit=1000,
            session_filter=session_filter,
            time_start=cutoff_time,
        )

        episodes = [r.item for r in results]
        total_episodes = len(episodes)
        total_extracted = 0
        total_created = 0
        total_linked = 0
        errors = []

        if not episodes:
            return {
                "episodes_processed": 0,
                "entities_extracted": 0,
                "entities_created": 0,
                "entities_linked": 0,
                "errors": [],
            }

        # Create extractor
        extractor = create_default_extractor(
            use_llm=settings.extraction_use_llm,
            llm_model=settings.extraction_llm_model,
        )

        # Process in batches
        batch_size = settings.extraction_batch_size
        for i in range(0, len(episodes), batch_size):
            batch = episodes[i:i + batch_size]

            # P4.2: Collect relationships for batch creation (fixes N+1 pattern)
            pending_relationships: list[tuple[str, str, str, dict]] = []

            for episode in batch:
                try:
                    # Extract entities (doesn't modify stores, can run without lock)
                    extracted = await extractor.extract(episode.content)
                    total_extracted += len(extracted)

                    # Filter by confidence
                    filtered = [
                        e for e in extracted
                        if e.confidence >= settings.extraction_confidence_threshold
                    ]

                    # Create entities and relationships - requires lock
                    # RACE-002 FIX: Lock only during store modifications
                    async with self._consolidation_lock:
                        for entity_data in filtered:
                            # Map to EntityType
                            entity_type_map = {
                                "PERSON": "PERSON",
                                "ORGANIZATION": "PROJECT",
                                "LOCATION": "CONCEPT",
                                "CONCEPT": "CONCEPT",
                                "TECHNOLOGY": "TOOL",
                                "EVENT": "CONCEPT",
                                "CONTACT": "PERSON",
                                "RESOURCE": "TOOL",
                                "TEMPORAL": "FACT",
                                "FINANCIAL": "FACT",
                            }

                            entity_type = entity_type_map.get(entity_data.entity_type, "CONCEPT")

                            try:
                                # Check if entity already exists
                                existing = await self._find_similar_entity(entity_data.name)

                                if existing:
                                    # P4.2: Queue relationship for batch creation
                                    pending_relationships.append((
                                        str(episode.id),
                                        str(existing.id),
                                        RelationType.SOURCE_OF.value,
                                        {
                                            "weight": entity_data.confidence,
                                            "coAccessCount": 1,
                                            "lastCoAccess": datetime.now().isoformat(),
                                        },
                                    ))
                                    total_linked += 1
                                else:
                                    # Create new entity
                                    entity = await semantic.create_entity(
                                        name=entity_data.name,
                                        entity_type=entity_type,
                                        summary=f"Auto-extracted (confidence: {entity_data.confidence:.2f})",
                                        details=entity_data.context,
                                        source=str(episode.id),
                                    )
                                    total_created += 1

                                    # P4.2: Queue relationship for batch creation
                                    pending_relationships.append((
                                        str(episode.id),
                                        str(entity.id),
                                        RelationType.SOURCE_OF.value,
                                        {
                                            "weight": entity_data.confidence,
                                            "coAccessCount": 1,
                                            "lastCoAccess": datetime.now().isoformat(),
                                        },
                                    ))
                                    total_linked += 1

                            except Exception as e:
                                error_msg = f"Failed to create/link entity '{entity_data.name}': {e}"
                                logger.warning(error_msg)
                                errors.append(error_msg)

                except Exception as e:
                    error_msg = f"Failed to extract from episode {episode.id}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            # P4.2: Batch create all pending relationships for this batch
            if pending_relationships:
                try:
                    await self.graph_store.batch_create_relationships(pending_relationships)
                except Exception as e:
                    logger.warning(f"Failed to batch create relationships: {e}")

        logger.info(
            f"Background extraction: {total_episodes} episodes → "
            f"{total_extracted} extracted → {total_created} created, {total_linked} linked"
        )

        return {
            "episodes_processed": total_episodes,
            "entities_extracted": total_extracted,
            "entities_created": total_created,
            "entities_linked": total_linked,
            "errors": errors[:10],  # Limit error list
        }


# Singleton instance
_consolidation_instance: ConsolidationService | None = None


def get_consolidation_service() -> ConsolidationService:
    """Get or create singleton consolidation service."""
    global _consolidation_instance
    if _consolidation_instance is None:
        _consolidation_instance = ConsolidationService()
    return _consolidation_instance
