"""
BCM Metaplasticity with Synaptic Tagging and Capture.

Biological Basis:
- BCM (Bienenstock-Cooper-Munro) sliding threshold theory
- Synaptic tagging and capture for memory consolidation
- Three-factor learning integration

The BCM rule creates a sliding threshold that adapts to activity:
- Above threshold: LTP (strengthening)
- Below threshold: LTD (weakening)
- Threshold slides based on recent activity (metaplasticity)

Synaptic Tagging and Capture:
- Weak stimulation creates "tags" marking synapses for potential plasticity
- Strong stimulation synthesizes "plasticity-related proteins" (PRPs)
- Capture occurs when PRPs reach tagged synapses

References:
- Bienenstock, Cooper & Munro (1982): BCM theory
- Frey & Morris (1997): Synaptic tagging and LTP
- Redondo & Morris (2011): Synaptic tagging and capture hypothesis
- Clopath et al. (2008): Tag-trigger-consolidation model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class PlasticityType(Enum):
    """Type of synaptic plasticity induced."""
    NONE = "none"
    EARLY_LTP = "early_ltp"   # E-LTP: protein-synthesis independent
    LATE_LTP = "late_ltp"     # L-LTP: requires protein synthesis
    EARLY_LTD = "early_ltd"   # E-LTD: transient weakening
    LATE_LTD = "late_ltd"     # L-LTD: persistent weakening


@dataclass
class BCMConfig:
    """Configuration for BCM learning rule.

    Attributes:
        theta_m_init: Initial modification threshold
        theta_m_tau: Time constant for threshold adaptation (steps)
        theta_m_min: Minimum threshold (prevents runaway LTP)
        theta_m_max: Maximum threshold (prevents runaway LTD)
        ltp_rate: Learning rate for potentiation
        ltd_rate: Learning rate for depression
        tag_decay_tau: Tag decay time constant (seconds)
        early_tag_threshold: Activity threshold for early tag
        late_tag_threshold: Activity threshold for late tag (also triggers PRP)
        prp_base_rate: Baseline PRP synthesis rate
        prp_synthesis_threshold: Activity threshold for PRP synthesis
        prp_decay_rate: PRP decay rate per second
        capture_capacity: Maximum PRP that can be captured per tag
    """
    # BCM sliding threshold
    theta_m_init: float = 0.5
    theta_m_tau: float = 100.0
    theta_m_min: float = 0.1
    theta_m_max: float = 0.9

    # Learning rates
    ltp_rate: float = 0.01
    ltd_rate: float = 0.005

    # Synaptic tagging parameters
    tag_decay_tau: float = 7200.0       # 2 hours in seconds
    early_tag_threshold: float = 0.3
    late_tag_threshold: float = 0.7

    # Plasticity-related proteins
    prp_base_rate: float = 0.1
    prp_synthesis_threshold: float = 0.8
    prp_decay_rate: float = 0.001
    capture_capacity: float = 1.0


@dataclass
class SynapticTag:
    """A synaptic tag marking a synapse for potential consolidation.

    Biological basis:
    - Tags are created by any supra-threshold activity
    - They decay over ~2 hours if not captured
    - Capture by PRPs converts E-LTP to L-LTP
    """
    synapse_id: str
    strength: float        # Tag strength [0, 1]
    tag_type: PlasticityType
    induction_time: datetime
    is_captured: bool = False


@dataclass
class BCMState:
    """State for a single synapse under BCM rule."""
    synapse_id: str
    theta_m: float                # Current modification threshold
    weight: float                 # Current synaptic weight
    activity_history: list[float] = field(default_factory=list)
    last_plasticity: PlasticityType = PlasticityType.NONE
    tag: SynapticTag | None = None


class BCMLearningRule:
    """
    BCM sliding threshold learning rule.

    The BCM rule modulates plasticity based on postsynaptic activity:
    - phi(y) = y * (y - theta_m)
    - theta_m slides to match average activity squared

    When y > theta_m: LTP (positive phi)
    When y < theta_m: LTD (negative phi)
    When y = theta_m: no change

    This creates a sliding threshold that prevents runaway potentiation.
    """

    def __init__(self, config: BCMConfig | None = None):
        """
        Initialize BCM learning rule.

        Args:
            config: BCM configuration
        """
        self.config = config or BCMConfig()

        # Per-synapse thresholds
        self._thresholds: dict[str, float] = {}

        # Running activity averages (for theta_m adaptation)
        self._activity_ema: dict[str, float] = {}
        self._activity_ema_decay = 1.0 / self.config.theta_m_tau

        logger.info(
            f"BCMLearningRule initialized: theta_init={self.config.theta_m_init}"
        )

    def compute_update(
        self,
        pre: float,
        post: float,
        synapse_id: str,
    ) -> tuple[float, PlasticityType]:
        """
        Compute BCM weight update for a synapse.

        BCM rule: dw = eta * y * (y - theta_m) * x

        Args:
            pre: Presynaptic activity [0, 1]
            post: Postsynaptic activity [0, 1]
            synapse_id: Unique synapse identifier

        Returns:
            Tuple of (weight_delta, plasticity_type)
        """
        theta_m = self.get_threshold(synapse_id)

        # BCM modulation function: phi(y) = y * (y - theta_m)
        phi = post * (post - theta_m)

        # Compute weight update
        if phi > 0:
            # LTP
            delta = self.config.ltp_rate * phi * pre
            plasticity = (
                PlasticityType.LATE_LTP if post > self.config.late_tag_threshold
                else PlasticityType.EARLY_LTP
            )
        elif phi < 0:
            # LTD
            delta = self.config.ltd_rate * phi * pre  # phi is negative
            plasticity = (
                PlasticityType.LATE_LTD if abs(phi) > 0.5
                else PlasticityType.EARLY_LTD
            )
        else:
            delta = 0.0
            plasticity = PlasticityType.NONE

        # Update threshold
        self.update_threshold(synapse_id, post)

        return delta, plasticity

    def update_threshold(self, synapse_id: str, post_activity: float) -> float:
        """
        Update the modification threshold based on activity.

        BCM theta_m dynamics: d(theta_m)/dt = (y^2 - theta_m) / tau

        Args:
            synapse_id: Synapse identifier
            post_activity: Current postsynaptic activity

        Returns:
            New threshold value
        """
        # Get current threshold
        theta_m = self._thresholds.get(synapse_id, self.config.theta_m_init)

        # Update activity EMA
        current_ema = self._activity_ema.get(synapse_id, post_activity)
        new_ema = (
            (1 - self._activity_ema_decay) * current_ema +
            self._activity_ema_decay * (post_activity ** 2)
        )
        self._activity_ema[synapse_id] = new_ema

        # Update threshold toward y^2
        theta_update = (new_ema - theta_m) / self.config.theta_m_tau
        theta_m = theta_m + theta_update

        # Clamp to biological bounds
        theta_m = np.clip(theta_m, self.config.theta_m_min, self.config.theta_m_max)

        self._thresholds[synapse_id] = theta_m

        return theta_m

    def get_threshold(self, synapse_id: str) -> float:
        """Get current threshold for synapse."""
        return self._thresholds.get(synapse_id, self.config.theta_m_init)

    def get_stats(self) -> dict:
        """Get BCM rule statistics."""
        if not self._thresholds:
            return {"n_synapses": 0}

        thresholds = list(self._thresholds.values())
        return {
            "n_synapses": len(thresholds),
            "mean_threshold": float(np.mean(thresholds)),
            "std_threshold": float(np.std(thresholds)),
            "min_threshold": float(np.min(thresholds)),
            "max_threshold": float(np.max(thresholds)),
        }


class SynapticTaggingAndCapture:
    """
    Synaptic tagging and capture mechanism for memory consolidation.

    Biological basis (Frey & Morris 1997):
    1. Any supra-threshold stimulation creates a "tag" at the synapse
    2. Strong stimulation triggers synthesis of PRPs in the cell body
    3. PRPs can be captured by any tagged synapse
    4. Capture converts E-LTP/E-LTD to L-LTP/L-LTD

    This enables "synaptic cooperation": weak inputs can be consolidated
    if they occur near in time to a strong input.
    """

    def __init__(self, config: BCMConfig | None = None):
        """
        Initialize synaptic tagging system.

        Args:
            config: BCM configuration
        """
        self.config = config or BCMConfig()

        # Active tags
        self._tags: dict[str, SynapticTag] = {}

        # PRP pool (cell-wide resource)
        self._prp_level: float = self.config.prp_base_rate
        self._prp_last_update: datetime = datetime.now()

        # Capture history
        self._capture_history: list[dict] = []
        self._max_history = 1000

        logger.info("SynapticTaggingAndCapture initialized")

    def process_input(
        self,
        synapse_id: str,
        strength: float,
        plasticity_type: PlasticityType,
        timestamp: datetime | None = None,
    ) -> dict:
        """
        Process input and potentially create tag or trigger PRP synthesis.

        Args:
            synapse_id: Synapse identifier
            strength: Input strength [0, 1]
            plasticity_type: Type of plasticity induced
            timestamp: Event timestamp

        Returns:
            Dict with tag and PRP status
        """
        timestamp = timestamp or datetime.now()

        result = {
            "synapse_id": synapse_id,
            "tag_created": False,
            "prp_triggered": False,
            "capture_occurred": False,
        }

        # Check if input is strong enough to create a tag
        if strength >= self.config.early_tag_threshold:
            # Create or update tag
            tag = SynapticTag(
                synapse_id=synapse_id,
                strength=min(strength, 1.0),
                tag_type=plasticity_type,
                induction_time=timestamp,
            )
            self._tags[synapse_id] = tag
            result["tag_created"] = True

        # Check if input triggers PRP synthesis
        if strength >= self.config.prp_synthesis_threshold:
            # Strong input triggers protein synthesis
            prp_boost = strength * 0.5
            self._prp_level = min(self._prp_level + prp_boost, 2.0)
            result["prp_triggered"] = True

        return result

    def attempt_capture(
        self,
        timestamp: datetime | None = None,
    ) -> list[str]:
        """
        Attempt to capture PRPs at tagged synapses.

        Capture converts early plasticity to late plasticity.

        Args:
            timestamp: Current time

        Returns:
            List of synapse IDs that were captured
        """
        timestamp = timestamp or datetime.now()

        # Update PRP level (decay)
        self._update_prp_level(timestamp)

        if self._prp_level < 0.01:
            return []  # No PRPs available

        captured = []

        # Sort tags by strength (prioritize stronger tags)
        sorted_tags = sorted(
            self._tags.values(),
            key=lambda t: t.strength,
            reverse=True
        )

        for tag in sorted_tags:
            if tag.is_captured:
                continue

            # Check if tag is still valid
            tag_age = (timestamp - tag.induction_time).total_seconds()
            tag_remaining = tag.strength * np.exp(-tag_age / self.config.tag_decay_tau)

            if tag_remaining < 0.1:
                continue  # Tag too weak

            # Attempt capture
            capture_amount = min(
                tag_remaining * self.config.capture_capacity,
                self._prp_level
            )

            if capture_amount > 0.05:
                # Successful capture
                tag.is_captured = True
                self._prp_level -= capture_amount

                captured.append(tag.synapse_id)

                self._capture_history.append({
                    "synapse_id": tag.synapse_id,
                    "timestamp": timestamp.isoformat(),
                    "capture_amount": capture_amount,
                    "tag_type": tag.tag_type.value,
                })

                if self._prp_level < 0.01:
                    break  # PRPs exhausted

        # Trim history
        if len(self._capture_history) > self._max_history:
            self._capture_history = self._capture_history[-self._max_history:]

        return captured

    def decay_tags(
        self,
        timestamp: datetime | None = None,
    ) -> list[str]:
        """
        Decay and remove expired tags.

        Args:
            timestamp: Current time

        Returns:
            List of synapse IDs with expired tags
        """
        timestamp = timestamp or datetime.now()
        expired = []

        for synapse_id, tag in list(self._tags.items()):
            tag_age = (timestamp - tag.induction_time).total_seconds()
            tag_remaining = tag.strength * np.exp(-tag_age / self.config.tag_decay_tau)

            if tag_remaining < 0.01 or tag.is_captured:
                del self._tags[synapse_id]
                expired.append(synapse_id)

        return expired

    def _update_prp_level(self, timestamp: datetime) -> None:
        """Update PRP level with decay."""
        elapsed = (timestamp - self._prp_last_update).total_seconds()
        self._prp_level *= np.exp(-self.config.prp_decay_rate * elapsed)
        self._prp_level = max(self._prp_level, self.config.prp_base_rate)
        self._prp_last_update = timestamp

    def get_active_tags(self) -> list[dict]:
        """Get info on active tags."""
        now = datetime.now()
        return [
            {
                "synapse_id": t.synapse_id,
                "strength": t.strength,
                "age_seconds": (now - t.induction_time).total_seconds(),
                "type": t.tag_type.value,
                "is_captured": t.is_captured,
            }
            for t in self._tags.values()
        ]

    def get_stats(self) -> dict:
        """Get tagging system statistics."""
        return {
            "n_active_tags": len(self._tags),
            "prp_level": self._prp_level,
            "n_captured": len(self._capture_history),
        }


class BCMMetaplasticityManager:
    """
    Combined BCM metaplasticity with synaptic tagging.

    Orchestrates BCM learning rule with synaptic tagging and capture
    for biologically plausible memory consolidation.

    Usage:
        manager = BCMMetaplasticityManager()

        # On memory retrieval
        result = manager.on_retrieval(
            synapse_ids=["syn1", "syn2"],
            pre_activities=[0.8, 0.3],
            post_activities=[0.9, 0.4],
        )

        # Periodically trigger consolidation
        consolidated = manager.on_consolidation()
    """

    def __init__(self, config: BCMConfig | None = None):
        """
        Initialize BCM metaplasticity manager.

        Args:
            config: BCM configuration
        """
        self.config = config or BCMConfig()
        self.bcm = BCMLearningRule(self.config)
        self.tagging = SynapticTaggingAndCapture(self.config)

        # Synapse states
        self._states: dict[str, BCMState] = {}

        logger.info("BCMMetaplasticityManager initialized")

    def on_retrieval(
        self,
        synapse_ids: list[str],
        pre_activities: list[float],
        post_activities: list[float],
        timestamp: datetime | None = None,
    ) -> dict:
        """
        Process retrieval event through BCM and tagging.

        Args:
            synapse_ids: List of synapse IDs
            pre_activities: Presynaptic activities
            post_activities: Postsynaptic activities
            timestamp: Event timestamp

        Returns:
            Dict with weight updates and tagging events
        """
        timestamp = timestamp or datetime.now()

        updates = []
        tags_created = 0
        prp_triggered = False

        for syn_id, pre, post in zip(synapse_ids, pre_activities, post_activities):
            # Apply BCM rule
            delta, plasticity_type = self.bcm.compute_update(pre, post, syn_id)

            # Get or create state
            if syn_id not in self._states:
                self._states[syn_id] = BCMState(
                    synapse_id=syn_id,
                    theta_m=self.bcm.get_threshold(syn_id),
                    weight=0.5,
                )

            state = self._states[syn_id]
            state.weight = np.clip(state.weight + delta, 0.0, 1.0)
            state.theta_m = self.bcm.get_threshold(syn_id)
            state.last_plasticity = plasticity_type

            # Process through tagging
            tag_result = self.tagging.process_input(
                synapse_id=syn_id,
                strength=abs(delta) / self.config.ltp_rate,  # Normalize
                plasticity_type=plasticity_type,
                timestamp=timestamp,
            )

            if tag_result["tag_created"]:
                tags_created += 1
            if tag_result["prp_triggered"]:
                prp_triggered = True

            updates.append({
                "synapse_id": syn_id,
                "delta": delta,
                "new_weight": state.weight,
                "plasticity": plasticity_type.value,
            })

        return {
            "updates": updates,
            "tags_created": tags_created,
            "prp_triggered": prp_triggered,
            "timestamp": timestamp.isoformat(),
        }

    def on_consolidation(
        self,
        timestamp: datetime | None = None,
    ) -> dict:
        """
        Run consolidation phase.

        Attempts to capture PRPs at tagged synapses and
        decays expired tags.

        Args:
            timestamp: Current time

        Returns:
            Dict with consolidation results
        """
        timestamp = timestamp or datetime.now()

        # Attempt capture
        captured = self.tagging.attempt_capture(timestamp)

        # Decay expired tags
        expired = self.tagging.decay_tags(timestamp)

        return {
            "captured": captured,
            "n_captured": len(captured),
            "expired": expired,
            "n_expired": len(expired),
            "prp_level": self.tagging._prp_level,
            "timestamp": timestamp.isoformat(),
        }

    def get_plasticity_state(self, synapse_id: str) -> BCMState | None:
        """Get current state for a synapse."""
        return self._states.get(synapse_id)

    def get_stats(self) -> dict:
        """Get combined statistics."""
        return {
            "bcm": self.bcm.get_stats(),
            "tagging": self.tagging.get_stats(),
            "n_synapses_tracked": len(self._states),
        }


__all__ = [
    "BCMConfig",
    "BCMState",
    "BCMLearningRule",
    "SynapticTag",
    "SynapticTaggingAndCapture",
    "BCMMetaplasticityManager",
    "PlasticityType",
]
