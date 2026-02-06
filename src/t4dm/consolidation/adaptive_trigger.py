"""
Adaptive Consolidation Trigger (W1-04).

CLS-based consolidation triggering: consolidate when fast learner (MemTable)
saturates, regardless of sleep pressure.

Evidence Base:
- O'Reilly et al. (2014) "Complementary Learning Systems"
- McClelland et al. (1995) "Why There Are Complementary Learning Systems"

Key Insight (CLS Principle):
    The hippocampus (fast learner) should transfer to neocortex (slow learner)
    before capacity is exhausted. Don't wait for biological sleep pressure
    if the fast learner is filling faster than consolidation can empty it.

Mapping to T4DM:
    - MemTable = hippocampus (fast, limited capacity)
    - LSM segments = neocortex (slow, high capacity)
    - Consolidation = memory transfer during "sleep"

Trigger Priority:
    1. MemTable saturation > 70% → NREM consolidation (urgent)
    2. Encoding rate > 10/min → NREM consolidation (capacity pressure)
    3. Adenosine pressure > 0.6 → Full sleep cycle (biological signal)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConsolidationConfig:
    """CLS-based consolidation triggering configuration.

    Attributes:
        memtable_saturation_threshold: Trigger at this % full (default 70%).
        encoding_rate_threshold: Memories per minute to trigger (default 10).
        adenosine_threshold: Default sleep pressure threshold (default 0.6).
        min_interval_seconds: Minimum seconds between consolidations (default 300 = 5 min).
        encoding_window_size: Number of encoding events to track (default 100).
    """

    memtable_saturation_threshold: float = 0.7
    encoding_rate_threshold: float = 10.0
    adenosine_threshold: float = 0.6
    min_interval_seconds: float = 300.0
    encoding_window_size: int = 100


@dataclass
class ConsolidationTriggerResult:
    """Result of adaptive consolidation trigger check.

    Attributes:
        reason: Why consolidation was triggered (memtable_saturation,
                high_encoding_rate, adenosine_pressure).
        urgency: How urgent (0-1+ scale, higher = more urgent).
        phase: Recommended consolidation phase ("nrem" for urgent, "full" for normal).
    """

    reason: str
    urgency: float
    phase: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "reason": self.reason,
            "urgency": self.urgency,
            "phase": self.phase,
        }


class AdaptiveConsolidationTrigger:
    """CLS-based adaptive consolidation trigger.

    Implements the Complementary Learning Systems principle: consolidate
    when the fast learner (MemTable) approaches capacity, regardless of
    sleep pressure.

    This prevents the fast learner from becoming saturated and losing
    new memories due to lack of storage space.

    Example:
        >>> trigger = AdaptiveConsolidationTrigger(config, engine)
        >>> trigger.record_encoding()  # Called when memory stored
        >>> result = trigger.should_trigger(adenosine_pressure=0.5)
        >>> if result:
        ...     consolidation_service.consolidate(phase=result.phase)
        ...     trigger.mark_consolidation_complete()
    """

    def __init__(
        self,
        config: AdaptiveConsolidationConfig,
        engine: Any,  # T4DXEngine or mock with memtable_size() and max_memtable_size
    ):
        """Initialize adaptive trigger.

        Args:
            config: Trigger configuration.
            engine: T4DX engine for MemTable metrics.
        """
        self.config = config
        self.engine = engine
        self.last_consolidation: float = 0.0
        self.encoding_times: deque = deque(maxlen=config.encoding_window_size)

        logger.info(
            f"AdaptiveConsolidationTrigger initialized: "
            f"saturation={config.memtable_saturation_threshold}, "
            f"rate={config.encoding_rate_threshold}/min"
        )

    def should_trigger(
        self, adenosine_pressure: float
    ) -> Optional[ConsolidationTriggerResult]:
        """Decide whether to trigger consolidation.

        Checks triggers in priority order:
        1. MemTable saturation (urgent - fast learner full)
        2. Encoding rate (capacity pressure)
        3. Adenosine pressure (biological sleep signal)

        Args:
            adenosine_pressure: Current adenosine/sleep pressure (0-1).

        Returns:
            ConsolidationTriggerResult if should trigger, None otherwise.
        """
        now = time.time()

        # Respect minimum interval
        if now - self.last_consolidation < self.config.min_interval_seconds:
            return None

        # Check MemTable saturation
        saturation = self._get_memtable_saturation()
        if saturation is not None and saturation > self.config.memtable_saturation_threshold:
            logger.info(
                f"CLS trigger: MemTable saturation {saturation:.1%} "
                f"> {self.config.memtable_saturation_threshold:.1%}"
            )
            return ConsolidationTriggerResult(
                reason="memtable_saturation",
                urgency=saturation,
                phase="nrem",  # Start with NREM for fast consolidation
            )

        # Check encoding rate
        encoding_rate = self._compute_encoding_rate()
        if encoding_rate > self.config.encoding_rate_threshold:
            logger.info(
                f"CLS trigger: Encoding rate {encoding_rate:.1f}/min "
                f"> {self.config.encoding_rate_threshold:.1f}/min"
            )
            return ConsolidationTriggerResult(
                reason="high_encoding_rate",
                urgency=encoding_rate / self.config.encoding_rate_threshold,
                phase="nrem",
            )

        # Default: use adenosine pressure
        if adenosine_pressure > self.config.adenosine_threshold:
            logger.info(
                f"Adenosine trigger: pressure {adenosine_pressure:.2f} "
                f"> {self.config.adenosine_threshold:.2f}"
            )
            return ConsolidationTriggerResult(
                reason="adenosine_pressure",
                urgency=adenosine_pressure,
                phase="full",  # Full sleep cycle when not urgent
            )

        return None

    def record_encoding(self) -> None:
        """Record an encoding event for rate tracking.

        Should be called each time a memory is stored.
        """
        self.encoding_times.append(time.time())

    def mark_consolidation_complete(self) -> None:
        """Mark that consolidation has completed.

        Updates last_consolidation timestamp to enforce minimum interval.
        """
        self.last_consolidation = time.time()
        logger.debug("Consolidation complete, interval timer reset")

    def _get_memtable_saturation(self) -> Optional[float]:
        """Get current MemTable saturation ratio.

        Returns:
            Saturation ratio (0-1) or None if unavailable.
        """
        try:
            memtable_size = self.engine.memtable_size()
            max_size = self.engine.max_memtable_size

            if max_size <= 0:
                return None

            return memtable_size / max_size
        except Exception as e:
            logger.warning(f"Failed to get MemTable saturation: {e}")
            return None

    def _compute_encoding_rate(self) -> float:
        """Compute encodings per minute.

        Returns:
            Encoding rate (memories/minute).
        """
        if len(self.encoding_times) < 2:
            return 0.0

        # Duration from oldest to newest encoding
        duration = self.encoding_times[-1] - self.encoding_times[0]
        if duration < 1:  # Less than 1 second
            return 0.0

        # Convert to per minute
        return len(self.encoding_times) / (duration / 60)

    def get_stats(self) -> dict:
        """Get current trigger statistics.

        Returns:
            Dictionary with current metrics.
        """
        saturation = self._get_memtable_saturation()
        encoding_rate = self._compute_encoding_rate()
        seconds_since = time.time() - self.last_consolidation

        return {
            "memtable_saturation": saturation,
            "encoding_rate": encoding_rate,
            "seconds_since_consolidation": seconds_since,
            "encoding_count": len(self.encoding_times),
            "thresholds": {
                "saturation": self.config.memtable_saturation_threshold,
                "rate": self.config.encoding_rate_threshold,
                "adenosine": self.config.adenosine_threshold,
                "min_interval": self.config.min_interval_seconds,
            },
        }
