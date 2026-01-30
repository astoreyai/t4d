"""
Glymphatic-Consolidation Bridge for World Weaver.

Implements H10 (cross-region consistency) and B8 (waste clearance) by
coupling the glymphatic system with memory consolidation:

1. Sleep Spindle → Glymphatic Gating:
   - Spindles (11-16 Hz) trigger micro-clearance windows
   - Delta up-states (0.5-4 Hz) enable bulk clearance
   - Coordinated timing for optimal consolidation

2. Consolidation ↔ Waste Protection:
   - Active replay memories protected during consolidation
   - Stale/weak memories tagged for clearance
   - Post-consolidation pruning of unsuccessful patterns

3. Clearance → Learning Signal:
   - Waste removal generates negative learning signal
   - Surviving memories receive positive reinforcement
   - Dopamine signal on successful cleanup

Biological Basis:
- Xie et al. (2013): Sleep increases interstitial space for clearance
- Fultz et al. (2019): CSF flow coupled to delta oscillations
- Stickgold (2005): Sleep and memory consolidation
- Nedergaard (2013): Glymphatic system discovery

References:
- Xie et al. (2013). Sleep drives metabolite clearance from adult brain
- Fultz et al. (2019). Coupled oscillations in human sleep
- Nedergaard & Goldman (2020). Glymphatic failure and neurodegeneration
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class SleepStage(Enum):
    """Sleep stage for consolidation-clearance coordination."""

    WAKE = auto()        # No clearance
    NREM_LIGHT = auto()  # Moderate clearance, spindles active
    NREM_DEEP = auto()   # Maximum clearance, delta dominant
    REM = auto()         # Minimal clearance, consolidation active


class MemoryProtectionStatus(Enum):
    """Protection status during consolidation."""

    PROTECTED = auto()   # Being replayed, do not clear
    VULNERABLE = auto()  # Not being replayed, can be cleared
    TAGGED = auto()      # Marked for clearance
    CLEARED = auto()     # Already cleared


class ClearanceMode(Enum):
    """Clearance operation mode."""

    MICRO = auto()   # Spindle-triggered, small batches
    BULK = auto()    # Delta-triggered, large batches
    NONE = auto()    # No clearance active


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GlymphaticConsolidationConfig:
    """
    Configuration for glymphatic-consolidation coupling.

    Parameters based on:
    - Xie et al. (2013): 60% increase in interstitial space during sleep
    - Fultz et al. (2019): Delta-CSF coupling at 0.05 Hz
    - Stickgold (2005): Memory consolidation windows
    """

    # Spindle gating
    spindle_clearance_window_ms: float = 500.0  # Window after spindle
    spindle_batch_fraction: float = 0.1         # Fraction cleared per spindle

    # Delta gating
    delta_clearance_phase_range: tuple = (0.0, 0.5)  # Up-state phase
    delta_batch_fraction: float = 0.3                 # Fraction per delta cycle

    # Protection parameters
    replay_protection_duration_s: float = 30.0   # Protection after replay
    consolidation_protection_buffer: float = 1.2  # Extra protection margin

    # Clearance thresholds
    weakness_threshold: float = 0.3    # Below this = vulnerable
    staleness_threshold_days: float = 7.0  # Older than this = stale

    # Learning signals
    clearance_negative_signal: float = -0.3  # Signal for cleared items
    survival_positive_signal: float = 0.2    # Signal for surviving items
    da_clearance_boost: float = 0.1          # DA boost on cleanup

    # Timing
    min_clearance_interval_s: float = 5.0    # Minimum between clearances
    max_protected_memories: int = 100        # Cap on protected set

    def __post_init__(self):
        """Validate configuration."""
        assert 0.0 <= self.spindle_batch_fraction <= 1.0
        assert 0.0 <= self.delta_batch_fraction <= 1.0
        assert self.replay_protection_duration_s > 0
        assert 0.0 <= self.weakness_threshold <= 1.0


@dataclass
class ConsolidationBridgeState:
    """Runtime state of the glymphatic-consolidation bridge."""

    # Current stage and mode
    sleep_stage: SleepStage = SleepStage.WAKE
    clearance_mode: ClearanceMode = ClearanceMode.NONE

    # Protected memory tracking
    protected_memories: set[str] = field(default_factory=set)
    protection_timestamps: dict[str, datetime] = field(default_factory=dict)

    # Clearance tracking
    # ATOM-P2-18: Use OrderedDict for O(1) removal instead of list
    pending_clearance: OrderedDict = field(default_factory=OrderedDict)
    cleared_this_cycle: list[str] = field(default_factory=list)
    total_cleared: int = 0

    # Timing
    last_clearance_time: datetime | None = None
    last_spindle_time: datetime | None = None
    last_delta_upstate: datetime | None = None

    # Statistics
    clearance_history: list[dict[str, Any]] = field(default_factory=list)
    spindle_triggered_count: int = 0
    delta_triggered_count: int = 0


# =============================================================================
# Core Bridge Class
# =============================================================================


class GlymphaticConsolidationBridge:
    """
    Bridge between glymphatic system and memory consolidation.

    Coordinates waste clearance with memory replay to protect
    consolidating memories while clearing stale/weak ones.

    Example:
        >>> bridge = GlymphaticConsolidationBridge()
        >>> bridge.set_sleep_stage(SleepStage.NREM_DEEP)
        >>> bridge.protect_memory("episode_123")  # Being replayed
        >>> clearance = bridge.on_delta_upstate(phase=0.25)
    """

    def __init__(self, config: GlymphaticConsolidationConfig | None = None):
        """Initialize bridge."""
        self.config = config or GlymphaticConsolidationConfig()
        self.state = ConsolidationBridgeState()

        logger.info("GlymphaticConsolidationBridge initialized")

    # -------------------------------------------------------------------------
    # Sleep Stage Management
    # -------------------------------------------------------------------------

    def set_sleep_stage(self, stage: SleepStage):
        """
        Set current sleep stage.

        Updates clearance mode based on stage:
        - WAKE: No clearance
        - NREM_LIGHT: Micro clearance (spindle-gated)
        - NREM_DEEP: Bulk clearance (delta-gated)
        - REM: Minimal clearance
        """
        self.state.sleep_stage = stage

        if stage == SleepStage.WAKE:
            self.state.clearance_mode = ClearanceMode.NONE
        elif stage == SleepStage.NREM_LIGHT:
            self.state.clearance_mode = ClearanceMode.MICRO
        elif stage == SleepStage.NREM_DEEP:
            self.state.clearance_mode = ClearanceMode.BULK
        elif stage == SleepStage.REM:
            self.state.clearance_mode = ClearanceMode.NONE

        logger.debug(f"Sleep stage set to {stage.name}, mode={self.state.clearance_mode.name}")

    def infer_stage_from_neuromod(
        self,
        ach_level: float,
        ne_level: float
    ) -> SleepStage:
        """
        Infer sleep stage from ACh and NE levels.

        Based on Hobson & Pace-Schott (2002) reciprocal interaction model:
        - WAKE: High ACh, High NE
        - NREM: Low ACh, Low NE (deeper = lower)
        - REM: High ACh, Low NE
        """
        if ach_level > 0.6 and ne_level > 0.5:
            stage = SleepStage.WAKE
        elif ach_level > 0.5 and ne_level < 0.3:
            stage = SleepStage.REM
        elif ach_level < 0.3 and ne_level < 0.2:
            stage = SleepStage.NREM_DEEP
        elif ach_level < 0.4 and ne_level < 0.4:
            stage = SleepStage.NREM_LIGHT
        else:
            stage = SleepStage.WAKE

        self.set_sleep_stage(stage)
        return stage

    # -------------------------------------------------------------------------
    # Memory Protection
    # -------------------------------------------------------------------------

    def protect_memory(self, memory_id: str):
        """
        Protect a memory from clearance (being replayed).

        Protection expires after replay_protection_duration_s.
        """
        # ATOM-P2-14: Validate memory_id format
        self._validate_memory_id(memory_id)

        now = datetime.now()
        self.state.protected_memories.add(memory_id)
        self.state.protection_timestamps[memory_id] = now

        # Cap protected set size
        if len(self.state.protected_memories) > self.config.max_protected_memories:
            self._expire_oldest_protection()

        logger.debug(f"Protected memory {memory_id}")

    def unprotect_memory(self, memory_id: str):
        """Explicitly remove protection from a memory."""
        self.state.protected_memories.discard(memory_id)
        self.state.protection_timestamps.pop(memory_id, None)

    def is_protected(self, memory_id: str) -> bool:
        """Check if memory is currently protected."""
        if memory_id not in self.state.protected_memories:
            return False

        # Check expiration
        timestamp = self.state.protection_timestamps.get(memory_id)
        if timestamp is None:
            return False

        elapsed = (datetime.now() - timestamp).total_seconds()
        if elapsed > self.config.replay_protection_duration_s:
            self.unprotect_memory(memory_id)
            return False

        return True

    def _expire_oldest_protection(self):
        """Remove oldest protection to stay under cap."""
        if not self.state.protection_timestamps:
            return

        oldest_id = min(
            self.state.protection_timestamps,
            key=self.state.protection_timestamps.get
        )
        self.unprotect_memory(oldest_id)

    def get_protection_status(self, memory_id: str) -> MemoryProtectionStatus:
        """Get detailed protection status for a memory."""
        if memory_id in self.state.cleared_this_cycle:
            return MemoryProtectionStatus.CLEARED
        if memory_id in self.state.pending_clearance:
            return MemoryProtectionStatus.TAGGED
        if self.is_protected(memory_id):
            return MemoryProtectionStatus.PROTECTED
        return MemoryProtectionStatus.VULNERABLE

    # -------------------------------------------------------------------------
    # Clearance Triggering
    # -------------------------------------------------------------------------

    def on_spindle(self, spindle_power: float = 1.0) -> dict[str, Any]:
        """
        Handle spindle event - trigger micro clearance.

        Spindles during NREM_LIGHT trigger small clearance batches.

        Args:
            spindle_power: Spindle amplitude (0-1)

        Returns:
            Dict with clearance info
        """
        now = datetime.now()
        self.state.last_spindle_time = now

        # Only trigger in appropriate stage
        if self.state.sleep_stage not in [SleepStage.NREM_LIGHT, SleepStage.NREM_DEEP]:
            return {"triggered": False, "reason": "wrong_stage"}

        # Check minimum interval
        if self.state.last_clearance_time:
            elapsed = (now - self.state.last_clearance_time).total_seconds()
            if elapsed < self.config.min_clearance_interval_s:
                return {"triggered": False, "reason": "too_soon"}

        # Compute batch size based on spindle power
        batch_fraction = self.config.spindle_batch_fraction * spindle_power

        result = self._execute_clearance(
            batch_fraction=batch_fraction,
            trigger="spindle"
        )

        self.state.spindle_triggered_count += 1
        return result

    def on_delta_upstate(self, phase: float) -> dict[str, Any]:
        """
        Handle delta up-state - trigger bulk clearance.

        Delta up-states (phase 0.0-0.5) during NREM_DEEP trigger
        large clearance batches.

        Args:
            phase: Delta phase (0.0-1.0, up-state is 0.0-0.5)

        Returns:
            Dict with clearance info
        """
        now = datetime.now()
        self.state.last_delta_upstate = now

        # Only trigger in NREM_DEEP
        if self.state.sleep_stage != SleepStage.NREM_DEEP:
            return {"triggered": False, "reason": "wrong_stage"}

        # Check if in up-state phase
        phase_min, phase_max = self.config.delta_clearance_phase_range
        if not (phase_min <= phase <= phase_max):
            return {"triggered": False, "reason": "wrong_phase"}

        # Check minimum interval
        if self.state.last_clearance_time:
            elapsed = (now - self.state.last_clearance_time).total_seconds()
            if elapsed < self.config.min_clearance_interval_s:
                return {"triggered": False, "reason": "too_soon"}

        result = self._execute_clearance(
            batch_fraction=self.config.delta_batch_fraction,
            trigger="delta_upstate"
        )

        self.state.delta_triggered_count += 1
        return result

    def _execute_clearance(
        self,
        batch_fraction: float,
        trigger: str
    ) -> dict[str, Any]:
        """
        Execute clearance of vulnerable memories.

        Clears a fraction of pending (tagged) memories,
        skipping protected ones.
        """
        now = datetime.now()
        self.state.last_clearance_time = now

        # Get vulnerable memories to clear
        to_clear = []
        for memory_id in list(self.state.pending_clearance.keys()):
            if not self.is_protected(memory_id):
                to_clear.append(memory_id)

        # Compute batch size
        batch_size = max(1, int(len(to_clear) * batch_fraction))
        batch = to_clear[:batch_size]

        # Execute clearance
        # ATOM-P2-18: OrderedDict.pop() is O(1) vs list.remove() which is O(n)
        cleared = []
        for memory_id in batch:
            self.state.pending_clearance.pop(memory_id, None)
            self.state.cleared_this_cycle.append(memory_id)
            cleared.append(memory_id)

        self.state.total_cleared += len(cleared)

        # Record history
        record = {
            "timestamp": now.isoformat(),
            "trigger": trigger,
            "batch_fraction": batch_fraction,
            "cleared_count": len(cleared),
            "protected_count": len(self.state.protected_memories),
            "pending_remaining": len(self.state.pending_clearance),
        }
        self.state.clearance_history.append(record)

        # Trim history
        if len(self.state.clearance_history) > 1000:
            self.state.clearance_history = self.state.clearance_history[-1000:]

        logger.debug(
            f"Clearance executed: trigger={trigger}, cleared={len(cleared)}, "
            f"remaining={len(self.state.pending_clearance)}"
        )

        return {
            "triggered": True,
            "trigger": trigger,
            "cleared": cleared,
            "cleared_count": len(cleared),
            "protected_count": len(self.state.protected_memories),
        }

    # -------------------------------------------------------------------------
    # Memory Tagging
    # -------------------------------------------------------------------------

    def tag_for_clearance(self, memory_id: str):
        """Tag a memory for potential clearance."""
        # ATOM-P2-14: Validate memory_id format
        self._validate_memory_id(memory_id)

        # ATOM-P2-18: OrderedDict provides O(1) insertion
        if memory_id not in self.state.pending_clearance:
            self.state.pending_clearance[memory_id] = True

    def tag_weak_memories(
        self,
        memories: list[dict[str, Any]],
        strength_key: str = "strength"
    ) -> list[str]:
        """
        Tag weak memories for clearance.

        Args:
            memories: List of memory dicts with strength values
            strength_key: Key for strength value in dict

        Returns:
            List of tagged memory IDs
        """
        tagged = []
        for mem in memories:
            memory_id = mem.get("id", mem.get("memory_id"))
            strength = mem.get(strength_key, 0.5)

            if strength < self.config.weakness_threshold:
                if not self.is_protected(memory_id):
                    self.tag_for_clearance(memory_id)
                    tagged.append(memory_id)

        return tagged

    def tag_stale_memories(
        self,
        memories: list[dict[str, Any]],
        timestamp_key: str = "last_access"
    ) -> list[str]:
        """
        Tag stale memories for clearance.

        Args:
            memories: List of memory dicts with timestamp values
            timestamp_key: Key for timestamp value

        Returns:
            List of tagged memory IDs
        """
        now = datetime.now()
        threshold = timedelta(days=self.config.staleness_threshold_days)
        tagged = []

        for mem in memories:
            memory_id = mem.get("id", mem.get("memory_id"))
            timestamp = mem.get(timestamp_key)

            if timestamp is None:
                continue

            # ATOM-P3-24: Wrap datetime parsing in error handling
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except (ValueError, TypeError) as e:
                    from ww.core.validation import ValidationError
                    raise ValidationError("timestamp", f"Invalid timestamp format: {e}")

            age = now - timestamp
            if age > threshold:
                if not self.is_protected(memory_id):
                    self.tag_for_clearance(memory_id)
                    tagged.append(memory_id)

        return tagged

    # -------------------------------------------------------------------------
    # Learning Signals
    # -------------------------------------------------------------------------

    def get_clearance_learning_signals(self) -> dict[str, float]:
        """
        Get learning signals from clearance events.

        Returns signals for:
        - cleared_signal: Negative signal for cleared items
        - survival_signal: Positive signal for survivors
        - da_boost: Dopamine boost for successful cleanup
        """
        cleared_count = len(self.state.cleared_this_cycle)
        protected_count = len(self.state.protected_memories)

        # Only generate signals if clearance occurred
        if cleared_count == 0:
            return {
                "cleared_signal": 0.0,
                "survival_signal": 0.0,
                "da_boost": 0.0,
            }

        # Clearance generates negative signal (for training)
        cleared_signal = self.config.clearance_negative_signal

        # Survivors get positive reinforcement
        survival_signal = self.config.survival_positive_signal * (protected_count / (protected_count + cleared_count + 1))

        # DA boost proportional to cleanup efficiency
        efficiency = cleared_count / (cleared_count + len(self.state.pending_clearance) + 1)
        da_boost = self.config.da_clearance_boost * efficiency

        return {
            "cleared_signal": cleared_signal,
            "survival_signal": survival_signal,
            "da_boost": da_boost,
            "cleared_count": cleared_count,
            "protected_count": protected_count,
        }

    # -------------------------------------------------------------------------
    # Consolidation Integration
    # -------------------------------------------------------------------------

    def on_consolidation_start(self, replaying_memories: list[str]):
        """
        Called when consolidation begins.

        Protects all memories being replayed.
        """
        for memory_id in replaying_memories:
            self.protect_memory(memory_id)

        logger.info(f"Consolidation started, protecting {len(replaying_memories)} memories")

    def on_consolidation_end(self):
        """
        Called when consolidation ends.

        Clears protection and prepares for cleanup.
        """
        # Move cleared items to final list
        final_cleared = list(self.state.cleared_this_cycle)

        # Reset cycle state
        self.state.cleared_this_cycle = []

        # Gradually expire protections (don't clear immediately)
        # They will expire naturally based on timestamp

        logger.info(f"Consolidation ended, {len(final_cleared)} items cleared this cycle")

        return final_cleared

    def on_replay_complete(self, memory_id: str, success: bool):
        """
        Called when a single memory replay completes.

        Args:
            memory_id: ID of replayed memory
            success: Whether replay was successful
        """
        if success:
            # Extend protection slightly
            self.protect_memory(memory_id)
        else:
            # Failed replay = vulnerable
            self.unprotect_memory(memory_id)
            self.tag_for_clearance(memory_id)

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def get_state(self) -> ConsolidationBridgeState:
        """Get current bridge state."""
        return self.state

    def get_statistics(self) -> dict[str, Any]:
        """Get clearance statistics."""
        return {
            "sleep_stage": self.state.sleep_stage.name,
            "clearance_mode": self.state.clearance_mode.name,
            "protected_count": len(self.state.protected_memories),
            "pending_count": len(self.state.pending_clearance),
            "cleared_this_cycle": len(self.state.cleared_this_cycle),
            "total_cleared": self.state.total_cleared,
            "spindle_triggered": self.state.spindle_triggered_count,
            "delta_triggered": self.state.delta_triggered_count,
        }

    def reset_cycle(self):
        """Reset for new sleep cycle."""
        self.state.cleared_this_cycle = []
        self.state.last_clearance_time = None
        logger.debug("Clearance cycle reset")

    def _validate_memory_id(self, memory_id: str) -> None:
        """
        ATOM-P2-14: Validate memory ID format.

        Ensures memory_id is a non-empty string with reasonable length.

        Args:
            memory_id: Memory ID to validate

        Raises:
            ValueError: If memory_id is invalid
        """
        if not memory_id or not isinstance(memory_id, str):
            raise ValueError(f"Invalid memory_id: must be non-empty string, got {type(memory_id)}")
        if len(memory_id) > 200:
            raise ValueError(f"memory_id too long: {len(memory_id)} chars (max 200)")

    def reset(self):
        """Full reset of bridge state."""
        self.state = ConsolidationBridgeState()
        logger.debug("GlymphaticConsolidationBridge fully reset")


# =============================================================================
# Factory Functions
# =============================================================================


def create_glymphatic_consolidation_bridge(
    spindle_batch: float = 0.1,
    delta_batch: float = 0.3,
    protection_duration: float = 30.0,
    **kwargs
) -> GlymphaticConsolidationBridge:
    """
    Factory function for creating glymphatic-consolidation bridge.

    Args:
        spindle_batch: Fraction cleared per spindle
        delta_batch: Fraction cleared per delta cycle
        protection_duration: Seconds to protect replayed memories
        **kwargs: Additional config parameters

    Returns:
        Configured GlymphaticConsolidationBridge instance
    """
    config = GlymphaticConsolidationConfig(
        spindle_batch_fraction=spindle_batch,
        delta_batch_fraction=delta_batch,
        replay_protection_duration_s=protection_duration,
        **kwargs
    )

    return GlymphaticConsolidationBridge(config)
