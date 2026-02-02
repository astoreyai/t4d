"""
Glymphatic System for T4DM (B8).

Implements brain waste clearance during sleep following biological mechanisms:
- Xie et al. 2013: "Sleep Drives Metabolite Clearance from the Adult Brain"
- Fultz et al. 2019: CSF oscillations during NREM
- Nedergaard 2013: Glymphatic pathway discovery

Key biological principles:
1. Low NE (sleep) → astrocyte volume decrease → interstitial space expands
2. Delta oscillations (0.5-4 Hz) drive CSF flow
3. Waste clearance 2x higher during NREM vs wake
4. REM sleep shows minimal clearance (high ACh blocks AQP4)

Integration points:
- Couples to DeltaOscillator for up-state timing
- Uses WakeSleepMode for state-dependent clearance
- Modulated by NE level from locus coeruleus

References:
- Xie et al. (2013) Science 342(6156): 373-377
- Iliff et al. (2012) Science Translational Medicine 4(147)
- Fultz et al. (2019) Science 366(6465): 628-631
- Mestre et al. (2018) eLife 7:e40070
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


class WakeSleepMode(Enum):
    """Wake/sleep state enum (mirrors swr_coupling.WakeSleepMode)."""
    ACTIVE_WAKE = "active_wake"
    QUIET_WAKE = "quiet_wake"
    NREM_LIGHT = "nrem_light"
    NREM_DEEP = "nrem_deep"
    REM = "rem"


class WasteCategory(Enum):
    """Categories of neural waste to clear."""
    UNUSED_EMBEDDING = "unused_embedding"      # Embeddings not retrieved
    WEAK_CONNECTION = "weak_connection"        # Low-weight Hebbian links
    STALE_MEMORY = "stale_memory"              # Low FSRS stability
    ORPHAN_ENTITY = "orphan_entity"            # Entity with no relations
    EXPIRED_EPISODE = "expired_episode"        # Old episodic memory


@dataclass
class GlymphaticConfig:
    """Configuration for glymphatic waste clearance system.

    Biological parameters from Xie et al. 2013 and Fultz et al. 2019.
    Clearance is ~2x higher during NREM vs wake.
    """
    # Sleep-state clearance rates (biological values)
    # Xie et al. 2013: 60% higher clearance during sleep (not 90%)
    # Baseline wake clearance ~30%, NREM ~2x = 60-70%
    clearance_nrem_deep: float = 0.7      # 70% during slow-wave sleep (Xie 2013)
    clearance_nrem_light: float = 0.5     # 50% during light sleep
    clearance_quiet_wake: float = 0.3     # 30% during quiet wake
    clearance_active_wake: float = 0.1    # 10% during active wake
    clearance_rem: float = 0.05           # ~5% during REM (ACh blocks AQP4)

    # Waste identification thresholds
    unused_embedding_days: int = 30       # Prune if not retrieved in 30 days
    weak_connection_threshold: float = 0.1  # Connection weight below this
    stale_memory_stability: float = 0.2   # FSRS stability threshold
    orphan_check_enabled: bool = True     # Check for orphaned entities

    # Delta oscillator coupling
    clear_on_delta_upstate: bool = True   # Only clear during delta up-states
    delta_phase_window: float = 0.3       # Proportion of cycle for clearance

    # Neuromodulator influence
    ne_modulation: float = 0.6            # Low NE → high clearance
    ach_modulation: float = 0.4           # High ACh → low clearance (REM)

    # Clearance dynamics
    clearance_batch_size: int = 100       # Items to process per step
    min_clearance_interval: float = 60.0  # Minimum seconds between clearance

    # Safety limits
    max_clearance_fraction: float = 0.1   # Never clear more than 10% per cycle
    preserve_recent_hours: int = 24       # Never clear items < 24h old


@dataclass
class WasteState:
    """Current state of waste in the system."""
    # Waste counts by category
    unused_embeddings: int = 0
    weak_connections: int = 0
    stale_memories: int = 0
    orphan_entities: int = 0
    expired_episodes: int = 0

    # Clearance statistics
    total_cleared: int = 0
    clearance_rate: float = 0.0
    last_clearance_time: datetime | None = None

    # Per-category cleared counts
    cleared_embeddings: int = 0
    cleared_connections: int = 0
    cleared_memories: int = 0
    cleared_entities: int = 0
    cleared_episodes: int = 0

    @property
    def total_waste(self) -> int:
        """Total waste items identified."""
        return (self.unused_embeddings + self.weak_connections +
                self.stale_memories + self.orphan_entities +
                self.expired_episodes)

    @property
    def total_cleared_by_category(self) -> dict[str, int]:
        """Breakdown of cleared items by category."""
        return {
            "embeddings": self.cleared_embeddings,
            "connections": self.cleared_connections,
            "memories": self.cleared_memories,
            "entities": self.cleared_entities,
            "episodes": self.cleared_episodes,
        }


@dataclass
class ClearanceEvent:
    """Record of a single clearance operation."""
    timestamp: datetime
    category: WasteCategory
    item_id: str
    reason: str
    wake_sleep_mode: WakeSleepMode
    clearance_rate: float


class MemorySystemProtocol(Protocol):
    """Protocol for memory system integration."""

    def get_unused_embeddings(self, days: int) -> list[str]:
        """Get embedding IDs not retrieved in specified days."""
        ...

    def get_weak_connections(self, threshold: float) -> list[tuple[str, str, float]]:
        """Get connection pairs below weight threshold."""
        ...

    def get_stale_memories(self, stability_threshold: float) -> list[str]:
        """Get memory IDs with low FSRS stability."""
        ...

    def get_orphan_entities(self) -> list[str]:
        """Get entity IDs with no relations."""
        ...

    def delete_embedding(self, embedding_id: str) -> bool:
        """Delete an embedding."""
        ...

    def delete_connection(self, source_id: str, target_id: str) -> bool:
        """Delete a connection."""
        ...

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        ...

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity."""
        ...


class WasteTracker:
    """Tracks waste items for clearance.

    Maintains lists of items identified for potential clearance,
    with timestamps and access patterns.
    """

    def __init__(self, config: GlymphaticConfig):
        self.config = config
        self._waste_items: dict[WasteCategory, list[dict]] = {
            category: [] for category in WasteCategory
        }
        self._clearance_history: list[ClearanceEvent] = []
        self._last_scan_time: datetime | None = None

    def scan_for_waste(
        self,
        memory_system: MemorySystemProtocol | None = None,
        current_time: datetime | None = None
    ) -> WasteState:
        """Scan memory system for waste items.

        Args:
            memory_system: Memory system to scan (optional for testing)
            current_time: Current time for age calculations

        Returns:
            WasteState with counts of identified waste
        """
        current_time = current_time or datetime.now()
        self._last_scan_time = current_time

        state = WasteState()

        if memory_system is None:
            return state

        # Scan for unused embeddings
        try:
            unused = memory_system.get_unused_embeddings(
                self.config.unused_embedding_days
            )
            for item_id in unused:
                self._waste_items[WasteCategory.UNUSED_EMBEDDING].append({
                    "id": item_id,
                    "identified_at": current_time,
                    "reason": f"Not retrieved in {self.config.unused_embedding_days} days"
                })
            state.unused_embeddings = len(unused)
        except (AttributeError, NotImplementedError):
            pass

        # Scan for weak connections
        try:
            weak = memory_system.get_weak_connections(
                self.config.weak_connection_threshold
            )
            for source, target, weight in weak:
                self._waste_items[WasteCategory.WEAK_CONNECTION].append({
                    "id": f"{source}->{target}",
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "identified_at": current_time,
                    "reason": f"Weight {weight:.3f} below threshold {self.config.weak_connection_threshold}"
                })
            state.weak_connections = len(weak)
        except (AttributeError, NotImplementedError):
            pass

        # Scan for stale memories
        try:
            stale = memory_system.get_stale_memories(
                self.config.stale_memory_stability
            )
            for item_id in stale:
                self._waste_items[WasteCategory.STALE_MEMORY].append({
                    "id": item_id,
                    "identified_at": current_time,
                    "reason": f"FSRS stability below {self.config.stale_memory_stability}"
                })
            state.stale_memories = len(stale)
        except (AttributeError, NotImplementedError):
            pass

        # Scan for orphan entities
        if self.config.orphan_check_enabled:
            try:
                orphans = memory_system.get_orphan_entities()
                for item_id in orphans:
                    self._waste_items[WasteCategory.ORPHAN_ENTITY].append({
                        "id": item_id,
                        "identified_at": current_time,
                        "reason": "No relations found"
                    })
                state.orphan_entities = len(orphans)
            except (AttributeError, NotImplementedError):
                pass

        return state

    def get_clearance_candidates(
        self,
        category: WasteCategory,
        max_items: int | None = None
    ) -> list[dict]:
        """Get candidates for clearance in a category.

        Args:
            category: Waste category to get candidates for
            max_items: Maximum number of items to return

        Returns:
            List of waste item dictionaries
        """
        items = self._waste_items.get(category, [])
        if max_items is not None:
            return items[:max_items]
        return items

    def mark_cleared(
        self,
        category: WasteCategory,
        item_id: str,
        wake_sleep_mode: WakeSleepMode,
        clearance_rate: float
    ) -> None:
        """Mark an item as cleared.

        Args:
            category: Waste category
            item_id: ID of cleared item
            wake_sleep_mode: Current sleep state
            clearance_rate: Effective clearance rate
        """
        # Remove from waste list
        self._waste_items[category] = [
            item for item in self._waste_items[category]
            if item.get("id") != item_id
        ]

        # Record clearance event
        event = ClearanceEvent(
            timestamp=datetime.now(),
            category=category,
            item_id=item_id,
            reason="Cleared during glymphatic cycle",
            wake_sleep_mode=wake_sleep_mode,
            clearance_rate=clearance_rate
        )
        self._clearance_history.append(event)

    def get_clearance_history(
        self,
        since: datetime | None = None,
        category: WasteCategory | None = None
    ) -> list[ClearanceEvent]:
        """Get clearance history with optional filters.

        Args:
            since: Only return events after this time
            category: Only return events of this category

        Returns:
            List of ClearanceEvent objects
        """
        events = self._clearance_history

        if since is not None:
            events = [e for e in events if e.timestamp >= since]

        if category is not None:
            events = [e for e in events if e.category == category]

        return events

    def reset(self) -> None:
        """Reset all tracked waste."""
        self._waste_items = {category: [] for category in WasteCategory}
        self._clearance_history.clear()
        self._last_scan_time = None


class GlymphaticSystem:
    """Brain waste clearance during sleep (Nedergaard 2013).

    Implements the glymphatic system that clears metabolic waste
    from the brain during sleep. Key features:

    1. Sleep-state dependent clearance rates
    2. Delta oscillation coupling (clear during up-states)
    3. NE modulation (low NE = high clearance)
    4. ACh modulation (high ACh = low clearance, blocks AQP4)

    Usage:
        config = GlymphaticConfig()
        glymphatic = GlymphaticSystem(config)

        # During consolidation loop
        waste_state = glymphatic.step(
            wake_sleep_mode=WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=0.2,
            dt=0.1
        )
    """

    def __init__(
        self,
        config: GlymphaticConfig | None = None,
        memory_system: MemorySystemProtocol | None = None
    ):
        """Initialize glymphatic system.

        Args:
            config: Glymphatic configuration
            memory_system: Memory system for waste scanning/deletion
        """
        self.config = config or GlymphaticConfig()
        self.memory_system = memory_system
        self.waste_tracker = WasteTracker(self.config)

        # State tracking
        self._state = WasteState()
        self._total_steps = 0
        self._accumulated_clearance = 0.0
        self._last_step_time: datetime | None = None

        # C7: Adenosine coupling
        self._adenosine: Any | None = None
        self._adenosine_pressure_boost: float = 0.0

    @property
    def state(self) -> WasteState:
        """Current waste state."""
        return self._state

    def get_state_clearance_rate(self, wake_sleep_mode: WakeSleepMode) -> float:
        """Get base clearance rate for a wake/sleep state.

        Args:
            wake_sleep_mode: Current wake/sleep state

        Returns:
            Base clearance rate [0, 1]
        """
        rates = {
            WakeSleepMode.NREM_DEEP: self.config.clearance_nrem_deep,
            WakeSleepMode.NREM_LIGHT: self.config.clearance_nrem_light,
            WakeSleepMode.QUIET_WAKE: self.config.clearance_quiet_wake,
            WakeSleepMode.ACTIVE_WAKE: self.config.clearance_active_wake,
            WakeSleepMode.REM: self.config.clearance_rem,
        }
        return rates.get(wake_sleep_mode, 0.1)

    def compute_effective_rate(
        self,
        wake_sleep_mode: WakeSleepMode,
        delta_up_state: bool,
        ne_level: float,
        ach_level: float = 0.0
    ) -> float:
        """Compute effective clearance rate with all modulations.

        Args:
            wake_sleep_mode: Current wake/sleep state
            delta_up_state: Whether delta oscillator is in up-state
            ne_level: Norepinephrine level [0, 1]
            ach_level: Acetylcholine level [0, 1]

        Returns:
            Effective clearance rate [0, 1]
        """
        # Base rate from sleep state
        base_rate = self.get_state_clearance_rate(wake_sleep_mode)

        # Delta up-state gating
        if self.config.clear_on_delta_upstate and not delta_up_state:
            # Minimal clearance outside delta up-states
            return base_rate * 0.1

        # NE modulation: low NE = high clearance
        # Biological: NE contracts astrocytes, blocking interstitial flow
        ne_factor = 1.0 - ne_level * self.config.ne_modulation
        ne_factor = max(0.0, min(1.0, ne_factor))

        # ACh modulation: high ACh = low clearance
        # Biological: ACh blocks AQP4 water channels (Iliff 2012)
        ach_factor = 1.0 - ach_level * self.config.ach_modulation
        ach_factor = max(0.0, min(1.0, ach_factor))

        effective_rate = base_rate * ne_factor * ach_factor
        return min(effective_rate, 1.0)

    def step(
        self,
        wake_sleep_mode: WakeSleepMode,
        delta_up_state: bool,
        ne_level: float,
        ach_level: float = 0.0,
        dt: float = 1.0,
        current_time: datetime | None = None
    ) -> WasteState:
        """Execute one clearance step.

        Biological mechanism:
        1. Low NE (sleep) → astrocytes shrink → interstitial space expands
        2. Delta up-states drive CSF flow (Fultz et al. 2019)
        3. Waste swept out with CSF

        Args:
            wake_sleep_mode: Current wake/sleep state
            delta_up_state: Whether delta oscillator is in up-state
            ne_level: Norepinephrine level [0, 1]
            ach_level: Acetylcholine level [0, 1]
            dt: Time step in seconds
            current_time: Current time for tracking

        Returns:
            Updated WasteState
        """
        current_time = current_time or datetime.now()
        self._total_steps += 1
        self._last_step_time = current_time

        # Compute effective clearance rate
        effective_rate = self.compute_effective_rate(
            wake_sleep_mode=wake_sleep_mode,
            delta_up_state=delta_up_state,
            ne_level=ne_level,
            ach_level=ach_level
        )

        self._state.clearance_rate = effective_rate

        # Accumulate clearance over time
        self._accumulated_clearance += effective_rate * dt

        # Execute clearance when accumulated enough
        if self._accumulated_clearance >= 1.0:
            self._execute_clearance(
                wake_sleep_mode=wake_sleep_mode,
                effective_rate=effective_rate,
                current_time=current_time
            )
            self._accumulated_clearance = 0.0

        return self._state

    def _execute_clearance(
        self,
        wake_sleep_mode: WakeSleepMode,
        effective_rate: float,
        current_time: datetime
    ) -> None:
        """Execute actual waste clearance.

        Args:
            wake_sleep_mode: Current wake/sleep state
            effective_rate: Computed clearance rate
            current_time: Current time
        """
        if self.memory_system is None:
            return

        self._state.last_clearance_time = current_time

        # Calculate items to clear this cycle
        batch_size = min(
            self.config.clearance_batch_size,
            int(self._state.total_waste * self.config.max_clearance_fraction)
        )

        if batch_size == 0:
            return

        cleared_count = 0

        # Priority order: weak connections → stale memories → unused embeddings
        # This order minimizes data loss risk

        # Clear weak connections first (lowest impact)
        for item in self.waste_tracker.get_clearance_candidates(
            WasteCategory.WEAK_CONNECTION,
            batch_size - cleared_count
        ):
            if cleared_count >= batch_size:
                break
            try:
                source = item.get("source", "")
                target = item.get("target", "")
                if self.memory_system.delete_connection(source, target):
                    self.waste_tracker.mark_cleared(
                        WasteCategory.WEAK_CONNECTION,
                        item["id"],
                        wake_sleep_mode,
                        effective_rate
                    )
                    self._state.cleared_connections += 1
                    self._state.total_cleared += 1
                    cleared_count += 1
            except Exception:
                pass

        # Clear stale memories
        for item in self.waste_tracker.get_clearance_candidates(
            WasteCategory.STALE_MEMORY,
            batch_size - cleared_count
        ):
            if cleared_count >= batch_size:
                break
            try:
                if self.memory_system.delete_memory(item["id"]):
                    self.waste_tracker.mark_cleared(
                        WasteCategory.STALE_MEMORY,
                        item["id"],
                        wake_sleep_mode,
                        effective_rate
                    )
                    self._state.cleared_memories += 1
                    self._state.total_cleared += 1
                    cleared_count += 1
            except Exception:
                pass

        # Clear unused embeddings (most impactful, do last)
        for item in self.waste_tracker.get_clearance_candidates(
            WasteCategory.UNUSED_EMBEDDING,
            batch_size - cleared_count
        ):
            if cleared_count >= batch_size:
                break
            try:
                if self.memory_system.delete_embedding(item["id"]):
                    self.waste_tracker.mark_cleared(
                        WasteCategory.UNUSED_EMBEDDING,
                        item["id"],
                        wake_sleep_mode,
                        effective_rate
                    )
                    self._state.cleared_embeddings += 1
                    self._state.total_cleared += 1
                    cleared_count += 1
            except Exception:
                pass

    def scan(self, current_time: datetime | None = None) -> WasteState:
        """Scan for waste items.

        Args:
            current_time: Current time for age calculations

        Returns:
            WasteState with identified waste counts
        """
        self._state = self.waste_tracker.scan_for_waste(
            memory_system=self.memory_system,
            current_time=current_time
        )
        return self._state

    def get_clearance_history(
        self,
        since: datetime | None = None,
        category: WasteCategory | None = None
    ) -> list[ClearanceEvent]:
        """Get clearance history.

        Args:
            since: Only return events after this time
            category: Only return events of this category

        Returns:
            List of ClearanceEvent objects
        """
        return self.waste_tracker.get_clearance_history(since, category)

    def get_statistics(self) -> dict[str, Any]:
        """Get glymphatic system statistics.

        Returns:
            Dictionary with operational statistics
        """
        return {
            "total_steps": self._total_steps,
            "current_clearance_rate": self._state.clearance_rate,
            "total_cleared": self._state.total_cleared,
            "cleared_by_category": self._state.total_cleared_by_category,
            "total_waste_identified": self._state.total_waste,
            "waste_breakdown": {
                "unused_embeddings": self._state.unused_embeddings,
                "weak_connections": self._state.weak_connections,
                "stale_memories": self._state.stale_memories,
                "orphan_entities": self._state.orphan_entities,
                "expired_episodes": self._state.expired_episodes,
            },
            "last_clearance_time": self._state.last_clearance_time,
            "accumulated_clearance": self._accumulated_clearance,
        }

    def reset(self) -> None:
        """Reset glymphatic system state."""
        self._state = WasteState()
        self._total_steps = 0
        self._accumulated_clearance = 0.0
        self._last_step_time = None
        self.waste_tracker.reset()

    # -------------------------------------------------------------------------
    # C7: Adenosine Coupling
    # -------------------------------------------------------------------------

    def connect_adenosine(self, adenosine: Any) -> None:
        """
        C7: Connect to adenosine system for bidirectional coupling.

        High adenosine pressure increases glymphatic clearance pressure
        (drives sleep-dependent clearance). High glymphatic clearance
        reduces adenosine accumulation rate.

        Biological basis:
        - Sleep pressure (adenosine) drives sleep onset
        - Sleep activates glymphatic system (Xie et al. 2013)
        - Glymphatic clearance removes adenosine
        - Bidirectional feedback loop

        Args:
            adenosine: AdenosineDynamics instance
        """
        self._adenosine = adenosine


def create_glymphatic_system(
    clearance_nrem_deep: float = 0.7,
    clearance_wake: float = 0.3,
    unused_embedding_days: int = 30,
    weak_connection_threshold: float = 0.1,
    ne_modulation: float = 0.6,
    memory_system: MemorySystemProtocol | None = None
) -> GlymphaticSystem:
    """Factory function for creating a glymphatic system.

    Args:
        clearance_nrem_deep: Clearance rate during deep NREM
        clearance_wake: Clearance rate during quiet wake
        unused_embedding_days: Days before embedding is considered unused
        weak_connection_threshold: Weight threshold for weak connections
        ne_modulation: NE modulation strength
        memory_system: Memory system for integration

    Returns:
        Configured GlymphaticSystem instance
    """
    config = GlymphaticConfig(
        clearance_nrem_deep=clearance_nrem_deep,
        clearance_quiet_wake=clearance_wake,
        unused_embedding_days=unused_embedding_days,
        weak_connection_threshold=weak_connection_threshold,
        ne_modulation=ne_modulation,
    )
    return GlymphaticSystem(config=config, memory_system=memory_system)
