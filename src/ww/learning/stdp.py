"""
P5.5: Spike-Timing-Dependent Plasticity (STDP) for World Weaver.

Implements the classic STDP learning rule where synaptic weight changes
depend on the relative timing of pre- and post-synaptic activity.

Biological Basis:
- STDP is a fundamental mechanism of synaptic plasticity in the brain
- Pre-before-post (causal): LTP (strengthen) - "fire together, wire together"
- Post-before-pre (anti-causal): LTD (weaken) - prevents spurious correlations
- The magnitude of change decreases exponentially with timing difference
- Multiplicative STDP: Weight-dependent plasticity for stability (van Rossum et al. 2000)

STDP Learning Rule:
    Δw = A+ * exp(-Δt/τ+)  if Δt > 0 (pre before post, LTP)
    Δw = -A- * exp(Δt/τ-)  if Δt < 0 (post before pre, LTD)

Multiplicative STDP (van Rossum et al. 2000):
    Δw = A+ * (w_max - w)^μ * exp(-Δt/τ+)  for LTP
    Δw = -A- * w^μ * exp(Δt/τ-)  for LTD

Where:
- Δt = t_post - t_pre
- A+, A- = amplitude parameters
- τ+, τ- = time constants
- μ = weight dependence exponent (typically 0.5-1.0)

Phase 1B Enhancement:
- Dopamine modulation of learning rates (A+, A-)
- High DA increases LTP, decreases LTD (reward learning)
- Low DA decreases LTP, increases LTD (punishment learning)

References:
- Bi & Poo (1998) "Synaptic modifications in cultured hippocampal neurons"
- Song et al. (2000) "Competitive Hebbian learning through STDP"
- Morrison et al. (2008) "Phenomenological models of synaptic plasticity"
- van Rossum et al. (2000) "Stable Hebbian learning from spike timing-dependent plasticity"
- Izhikevich (2007) "Solving the distal reward problem through linkage of STDP and DA"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class STDPConfig:
    """Configuration for STDP learning rule."""
    # Amplitude parameters
    a_plus: float = 0.01      # LTP amplitude
    a_minus: float = 0.0105   # LTD amplitude (slightly higher for stability)

    # Time constants (seconds) - biological range: 15-40ms
    # Bi & Poo (1998): tau+ ≈ 17ms, tau- ≈ 34ms (asymmetric)
    # Morrison (2008): tau+ = 16.8ms, tau- = 33.7ms
    tau_plus: float = 0.017   # LTP time window (~17ms)
    tau_minus: float = 0.034  # LTD time window (~34ms, asymmetric per literature)

    # Bounds
    min_weight: float = 0.0   # Minimum synaptic weight
    max_weight: float = 1.0   # Maximum synaptic weight
    weight_decay: float = 0.0001  # Slow decay toward baseline

    # Multiplicative STDP parameters (van Rossum et al. 2000)
    multiplicative: bool = True  # Use multiplicative STDP
    mu: float = 0.5              # Weight dependence exponent (0.5-1.0)

    # Spike history
    max_spike_history: int = 100  # Max spikes to track per entity
    spike_window_ms: float = 100.0  # Time window for STDP (ms)


@dataclass
class SpikeEvent:
    """Record of a spike (activation) event."""
    entity_id: str
    timestamp: datetime
    strength: float = 1.0  # Spike strength/magnitude


@dataclass
class STDPUpdate:
    """Result of an STDP weight update."""
    pre_id: str
    post_id: str
    delta_t_ms: float      # Time difference in milliseconds
    old_weight: float
    new_weight: float
    delta_weight: float
    update_type: str       # "ltp" or "ltd"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class STDPStats:
    """Statistics for STDP learning."""
    total_updates: int = 0
    ltp_updates: int = 0
    ltd_updates: int = 0
    total_ltp_magnitude: float = 0.0
    total_ltd_magnitude: float = 0.0
    avg_delta_t_ltp_ms: float = 0.0
    avg_delta_t_ltd_ms: float = 0.0
    last_update: datetime | None = None


class STDPLearner:
    """
    P5.5: Spike-Timing-Dependent Plasticity implementation.

    Tracks spike times for entities and computes weight updates
    based on the relative timing of pre- and post-synaptic activity.

    Supports both additive and multiplicative STDP formulations.

    Phase 1B Enhancement:
    - Optional dopamine modulation of learning rates
    - compute_stdp_delta() accepts optional da_level parameter
    - High DA increases LTP, low DA increases LTD

    Usage:
        stdp = STDPLearner()

        # Record spikes as they occur
        stdp.record_spike("neuron_a")
        stdp.record_spike("neuron_b")

        # Compute STDP update for a synapse
        update = stdp.compute_update("neuron_a", "neuron_b", current_weight=0.5)
        if update:
            new_weight = update.new_weight

        # Phase 1B: With dopamine modulation
        update = stdp.compute_update("neuron_a", "neuron_b",
                                     current_weight=0.5, da_level=0.8)
    """

    def __init__(self, config: STDPConfig | None = None):
        """
        Initialize STDP learner.

        Args:
            config: STDP configuration (uses defaults if None)
        """
        self.config = config or STDPConfig()
        self.stats = STDPStats()

        # Spike history: entity_id -> list of (timestamp, strength)
        self._spike_history: dict[str, list[tuple[datetime, float]]] = {}

        # Weight cache for synapses: (pre_id, post_id) -> weight
        self._weights: dict[tuple[str, str], float] = {}

        # Running averages for delta_t
        self._ltp_delta_sum: float = 0.0
        self._ltd_delta_sum: float = 0.0

        # Track last spike time per entity for validation
        self._last_spike_times: dict[str, float] = {}

        logger.info(
            f"STDPLearner initialized: "
            f"A+={self.config.a_plus}, A-={self.config.a_minus}, "
            f"τ+={self.config.tau_plus}s, τ-={self.config.tau_minus}s, "
            f"multiplicative={self.config.multiplicative}, μ={self.config.mu}"
        )

    def record_spike(
        self,
        entity_id: str,
        timestamp: datetime | None = None,
        strength: float = 1.0
    ) -> None:
        """
        Record a spike (activation) for an entity.

        Args:
            entity_id: ID of the entity that spiked
            timestamp: When the spike occurred (now if None)
            strength: Spike strength (default 1.0)

        Raises:
            ValidationError: If timestamp is invalid or inter-spike interval is too short
        """
        from ww.core.validation import validate_timestamp, validate_spike_interval, ValidationError

        if timestamp is None:
            timestamp = datetime.now()

        # Security: Validate timestamp is not in the future and not too old
        validate_timestamp(timestamp)

        # Convert to seconds for interval validation
        timestamp_seconds = timestamp.timestamp()

        # Security: Validate inter-spike interval if we have a previous spike
        if entity_id in self._last_spike_times:
            prev_ts = self._last_spike_times[entity_id]
            validate_spike_interval(prev_ts, timestamp_seconds)

        # Update last spike time
        self._last_spike_times[entity_id] = timestamp_seconds

        if entity_id not in self._spike_history:
            self._spike_history[entity_id] = []

        self._spike_history[entity_id].append((timestamp, strength))

        # Trim old spikes
        self._trim_spike_history(entity_id)

    def _trim_spike_history(self, entity_id: str) -> None:
        """Trim old spikes from history."""
        if entity_id not in self._spike_history:
            return

        history = self._spike_history[entity_id]

        # Keep only recent spikes
        if len(history) > self.config.max_spike_history:
            self._spike_history[entity_id] = history[-self.config.max_spike_history:]

        # Also remove very old spikes (beyond tau window)
        now = datetime.now()
        max_age = timedelta(seconds=max(self.config.tau_plus, self.config.tau_minus) * 3)
        self._spike_history[entity_id] = [
            (t, s) for t, s in self._spike_history[entity_id]
            if now - t < max_age
        ]

    def get_latest_spike(self, entity_id: str) -> tuple[datetime, float] | None:
        """Get most recent spike for an entity."""
        if entity_id not in self._spike_history:
            return None
        history = self._spike_history[entity_id]
        if not history:
            return None
        return history[-1]

    def compute_stdp_delta(
        self,
        delta_t_ms: float,
        current_weight: float | None = None,
        da_level: float | None = None
    ) -> float:
        """
        Compute weight change from timing difference.

        Supports both additive and multiplicative STDP.

        Phase 1B: Optional dopamine modulation of learning rates.
        - da_level=None: Use base A+/A- (no modulation)
        - da_level>0.5: Increase LTP, decrease LTD (reward)
        - da_level<0.5: Decrease LTP, increase LTD (punishment)

        Args:
            delta_t_ms: t_post - t_pre in milliseconds
            current_weight: Current weight (required for multiplicative STDP)
            da_level: Dopamine level [0, 1] for modulation (optional)

        Returns:
            Weight change (positive for LTP, negative for LTD)
        """
        # Convert ms to seconds for tau
        delta_t_s = delta_t_ms / 1000.0

        if abs(delta_t_ms) < 0.1:
            # Simultaneous - no update
            return 0.0

        # Phase 1B: Compute DA-modulated learning rates
        if da_level is not None:
            a_plus, a_minus = self._compute_da_modulated_rates(da_level)
        else:
            a_plus = self.config.a_plus
            a_minus = self.config.a_minus

        if self.config.multiplicative:
            # Multiplicative STDP (van Rossum et al. 2000)
            if current_weight is None:
                current_weight = 0.5  # Default if not provided

            w = np.clip(current_weight, 0.0, self.config.max_weight)
            mu = self.config.mu

            if delta_t_s > 0:
                # Pre before post: LTP
                # Δw = A+ * (w_max - w)^μ * exp(-Δt/τ+)
                weight_factor = (self.config.max_weight - w) ** mu
                return a_plus * weight_factor * np.exp(-delta_t_s / self.config.tau_plus)
            else:
                # Post before pre: LTD
                # Δw = -A- * w^μ * exp(Δt/τ-)
                weight_factor = w ** mu
                return -a_minus * weight_factor * np.exp(delta_t_s / self.config.tau_minus)
        else:
            # Additive STDP (original formulation)
            if delta_t_s > 0:
                # Pre before post: LTP
                return a_plus * np.exp(-delta_t_s / self.config.tau_plus)
            else:
                # Post before pre: LTD
                return -a_minus * np.exp(delta_t_s / self.config.tau_minus)

    def _compute_da_modulated_rates(
        self,
        da_level: float,
        ltp_gain: float = 0.5,
        ltd_gain: float = 0.3,
        baseline_da: float = 0.5
    ) -> tuple[float, float]:
        """
        Phase 1B: Compute DA-modulated learning rates.

        High DA (>baseline):
        - Increases LTP (A+ * (1 + gain * da_mod))
        - Decreases LTD (A- * (1 - gain * da_mod))

        Low DA (<baseline):
        - Decreases LTP (A+ * (1 + gain * da_mod)) [negative da_mod]
        - Increases LTD (A- * (1 - gain * da_mod)) [negative da_mod]

        Args:
            da_level: Dopamine level [0, 1]
            ltp_gain: LTP modulation strength
            ltd_gain: LTD modulation strength
            baseline_da: DA level for no modulation

        Returns:
            (a_plus_modulated, a_minus_modulated)
        """
        # Normalize around baseline: range [-1, 1]
        da_mod = (da_level - baseline_da) / baseline_da
        da_mod = np.clip(da_mod, -1.0, 1.0)

        # LTP modulation: High DA increases LTP
        ltp_mod = 1.0 + ltp_gain * da_mod
        ltp_mod = max(0.1, ltp_mod)  # Don't completely suppress

        # LTD modulation: High DA decreases LTD (inverse)
        ltd_mod = 1.0 - ltd_gain * da_mod
        ltd_mod = max(0.1, ltd_mod)  # Don't completely suppress

        a_plus_mod = self.config.a_plus * ltp_mod
        a_minus_mod = self.config.a_minus * ltd_mod

        return a_plus_mod, a_minus_mod

    def compute_weight_update(
        self,
        pre_time: float,
        post_time: float,
        current_weight: float,
        w_max: float | None = None,
        mu: float | None = None
    ) -> float:
        """
        Compute weight change for multiplicative STDP.

        Implements van Rossum et al. (2000) multiplicative rule:
        - LTP: Δw = A+ * (w_max - w)^μ * exp(-Δt/τ+)
        - LTD: Δw = -A- * w^μ * exp(Δt/τ-)

        Args:
            pre_time: Presynaptic spike time (seconds)
            post_time: Postsynaptic spike time (seconds)
            current_weight: Current synaptic weight
            w_max: Maximum weight (defaults to config.max_weight)
            mu: Weight dependence exponent (defaults to config.mu)

        Returns:
            Weight change Δw
        """
        if w_max is None:
            w_max = self.config.max_weight
        if mu is None:
            mu = self.config.mu

        dt = post_time - pre_time
        w = np.clip(current_weight, 0.0, w_max)

        if abs(dt) < 1e-6:
            return 0.0

        if dt > 0:
            # LTP - multiplicative
            weight_factor = (w_max - w) ** mu
            return self.config.a_plus * weight_factor * np.exp(-dt / self.config.tau_plus)
        else:
            # LTD - multiplicative
            weight_factor = w ** mu
            return -self.config.a_minus * weight_factor * np.exp(dt / self.config.tau_minus)

    def compute_update(
        self,
        pre_id: str,
        post_id: str,
        current_weight: float | None = None,
        da_level: float | None = None
    ) -> STDPUpdate | None:
        """
        Compute STDP weight update for a synapse.

        Phase 1B: Optional dopamine modulation via da_level parameter.

        Args:
            pre_id: Presynaptic entity ID
            post_id: Postsynaptic entity ID
            current_weight: Current synaptic weight (uses cached if None)
            da_level: Dopamine level for modulation (optional)

        Returns:
            STDPUpdate with weight change, or None if no valid update
        """
        # Get spike times
        pre_spike = self.get_latest_spike(pre_id)
        post_spike = self.get_latest_spike(post_id)

        if pre_spike is None or post_spike is None:
            return None

        pre_time, pre_strength = pre_spike
        post_time, post_strength = post_spike

        # Compute time difference in milliseconds
        delta_t = (post_time - pre_time).total_seconds() * 1000.0

        # Check if within STDP window
        if abs(delta_t) > self.config.spike_window_ms:
            return None

        # Get current weight
        synapse_key = (pre_id, post_id)
        if current_weight is None:
            current_weight = self._weights.get(synapse_key, 0.5)

        # Compute STDP delta (multiplicative if enabled, DA-modulated if da_level provided)
        base_delta = self.compute_stdp_delta(delta_t, current_weight, da_level=da_level)

        # Scale by spike strengths
        delta_w = base_delta * pre_strength * post_strength

        # Apply weight change with bounds
        new_weight = np.clip(
            current_weight + delta_w,
            self.config.min_weight,
            self.config.max_weight
        )
        actual_delta = new_weight - current_weight

        # Update cached weight
        self._weights[synapse_key] = new_weight

        # Determine update type
        if delta_t > 0:
            update_type = "ltp"
            self.stats.ltp_updates += 1
            self.stats.total_ltp_magnitude += abs(actual_delta)
            self._ltp_delta_sum += abs(delta_t)
            if self.stats.ltp_updates > 0:
                self.stats.avg_delta_t_ltp_ms = self._ltp_delta_sum / self.stats.ltp_updates
        else:
            update_type = "ltd"
            self.stats.ltd_updates += 1
            self.stats.total_ltd_magnitude += abs(actual_delta)
            self._ltd_delta_sum += abs(delta_t)
            if self.stats.ltd_updates > 0:
                self.stats.avg_delta_t_ltd_ms = self._ltd_delta_sum / self.stats.ltd_updates

        self.stats.total_updates += 1
        self.stats.last_update = datetime.now()

        update = STDPUpdate(
            pre_id=pre_id,
            post_id=post_id,
            delta_t_ms=delta_t,
            old_weight=current_weight,
            new_weight=new_weight,
            delta_weight=actual_delta,
            update_type=update_type
        )

        logger.debug(
            f"STDP update: {pre_id}->{post_id} "
            f"Δt={delta_t:.1f}ms, Δw={actual_delta:.4f}, "
            f"type={update_type}, mult={self.config.multiplicative}, "
            f"DA={da_level if da_level is not None else 'none'}"
        )

        return update

    def compute_all_updates(
        self,
        pre_ids: list[str],
        post_id: str,
        weights: dict[str, float] | None = None,
        da_level: float | None = None
    ) -> list[STDPUpdate]:
        """
        Compute STDP updates for all presynaptic inputs to a post neuron.

        Phase 1B: Optional dopamine modulation via da_level parameter.

        Args:
            pre_ids: List of presynaptic entity IDs
            post_id: Postsynaptic entity ID
            weights: Current weights {pre_id: weight}
            da_level: Dopamine level for modulation (optional)

        Returns:
            List of valid STDP updates
        """
        updates = []
        weights = weights or {}

        for pre_id in pre_ids:
            update = self.compute_update(
                pre_id,
                post_id,
                current_weight=weights.get(pre_id),
                da_level=da_level
            )
            if update:
                updates.append(update)

        return updates

    def apply_weight_decay(self, baseline: float = 0.5) -> dict[tuple[str, str], float]:
        """
        Apply slow weight decay toward baseline.

        Implements homeostatic regulation to prevent runaway weights.

        Args:
            baseline: Target weight to decay toward

        Returns:
            Dict of {synapse_key: delta_weight}
        """
        decay_rate = self.config.weight_decay
        deltas = {}

        for synapse_key, weight in self._weights.items():
            delta = decay_rate * (baseline - weight)
            self._weights[synapse_key] = weight + delta
            deltas[synapse_key] = delta

        return deltas

    def get_weight(self, pre_id: str, post_id: str) -> float:
        """Get current weight for a synapse."""
        return self._weights.get((pre_id, post_id), 0.5)

    def set_weight(self, pre_id: str, post_id: str, weight: float) -> None:
        """Set weight for a synapse."""
        self._weights[(pre_id, post_id)] = np.clip(
            weight,
            self.config.min_weight,
            self.config.max_weight
        )

    # ATOM-P2-23: Weight encapsulation accessor methods
    def get_connections(self) -> dict[tuple[str, str], float]:
        """
        Get all synaptic connections and their weights.

        Returns:
            Dictionary mapping (pre_id, post_id) tuples to weights
        """
        return dict(self._weights)

    def remove_connection(self, key: tuple[str, str]) -> None:
        """
        Remove a synaptic connection.

        Args:
            key: (pre_id, post_id) tuple identifying the synapse
        """
        self._weights.pop(key, None)

    def get_weight_from_key(self, key: tuple[str, str], default: float = 0.0) -> float:
        """
        Get weight by synapse key.

        Args:
            key: (pre_id, post_id) tuple identifying the synapse
            default: Default value if synapse not found

        Returns:
            Weight value or default
        """
        return self._weights.get(key, default)

    def clear_spikes(self, entity_id: str | None = None) -> None:
        """Clear spike history for entity or all entities."""
        if entity_id:
            self._spike_history.pop(entity_id, None)
        else:
            self._spike_history.clear()

    def get_stats(self) -> dict:
        """Get STDP statistics."""
        return {
            "total_updates": self.stats.total_updates,
            "ltp_updates": self.stats.ltp_updates,
            "ltd_updates": self.stats.ltd_updates,
            "ltp_ltd_ratio": (
                self.stats.ltp_updates / max(1, self.stats.ltd_updates)
            ),
            "total_ltp_magnitude": round(self.stats.total_ltp_magnitude, 4),
            "total_ltd_magnitude": round(self.stats.total_ltd_magnitude, 4),
            "avg_delta_t_ltp_ms": round(self.stats.avg_delta_t_ltp_ms, 2),
            "avg_delta_t_ltd_ms": round(self.stats.avg_delta_t_ltd_ms, 2),
            "active_synapses": len(self._weights),
            "tracked_entities": len(self._spike_history),
            "multiplicative": self.config.multiplicative,
            "mu": self.config.mu,
            "last_update": (
                self.stats.last_update.isoformat()
                if self.stats.last_update else None
            ),
        }

    def save_state(self) -> dict:
        """Save STDP state for persistence."""
        return {
            "config": {
                "a_plus": self.config.a_plus,
                "a_minus": self.config.a_minus,
                "tau_plus": self.config.tau_plus,
                "tau_minus": self.config.tau_minus,
                "min_weight": self.config.min_weight,
                "max_weight": self.config.max_weight,
                "multiplicative": self.config.multiplicative,
                "mu": self.config.mu,
            },
            "weights": {
                f"{pre}|{post}": w
                for (pre, post), w in self._weights.items()
            },
            "stats": self.get_stats(),
        }

    def load_state(self, state: dict) -> None:
        """Load STDP state from dictionary."""
        # Load weights
        weights_data = state.get("weights", {})
        for key, weight in weights_data.items():
            pre, post = key.split("|")
            self._weights[(pre, post)] = weight

        logger.info(f"STDPLearner state loaded: {len(self._weights)} synapses")


class PairBasedSTDP(STDPLearner):
    """
    Pair-based STDP implementation with nearest-spike pairing.

    Uses only the nearest pre-post spike pair for weight updates,
    which is more biologically plausible than all-pairs.
    """

    def __init__(self, config: STDPConfig | None = None):
        super().__init__(config)
        self._last_update_time: dict[tuple[str, str], datetime] = {}

    def compute_update(
        self,
        pre_id: str,
        post_id: str,
        current_weight: float | None = None,
        da_level: float | None = None
    ) -> STDPUpdate | None:
        """
        Compute STDP update using nearest-spike pairing.

        Only considers the most recent spike pair to avoid
        multiple updates from the same spike.

        Phase 1B: Optional dopamine modulation via da_level.
        """
        synapse_key = (pre_id, post_id)

        # Get spike times
        pre_spike = self.get_latest_spike(pre_id)
        post_spike = self.get_latest_spike(post_id)

        if pre_spike is None or post_spike is None:
            return None

        pre_time, _ = pre_spike
        post_time, _ = post_spike

        # Check if we've already processed this spike pair
        last_update = self._last_update_time.get(synapse_key)
        if last_update:
            # Skip if both spikes are older than last update
            if pre_time < last_update and post_time < last_update:
                return None

        # Get update from parent class (with DA modulation if provided)
        update = super().compute_update(pre_id, post_id, current_weight, da_level=da_level)

        if update:
            self._last_update_time[synapse_key] = datetime.now()

        return update


class TripletSTDP(STDPLearner):
    """
    Triplet STDP implementation for more accurate LTP modeling.

    Extends pair-based STDP by considering triplets of spikes,
    which better captures the dependence of LTP on post-synaptic
    firing rate (Pfister & Gerstner, 2006).

    Triplet contribution: Pre-Post-Pre and Post-Pre-Post sequences
    modify the standard pair-based rule.
    """

    def __init__(
        self,
        config: STDPConfig | None = None,
        triplet_a_plus: float = 0.005,  # Triplet LTP amplitude
        triplet_a_minus: float = 0.002,  # Triplet LTD amplitude
        tau_triplet: float = 40.0        # Triplet time constant
    ):
        super().__init__(config)
        self.triplet_a_plus = triplet_a_plus
        self.triplet_a_minus = triplet_a_minus
        self.tau_triplet = tau_triplet

    def compute_update(
        self,
        pre_id: str,
        post_id: str,
        current_weight: float | None = None,
        da_level: float | None = None
    ) -> STDPUpdate | None:
        """
        Compute triplet STDP update.

        Adds triplet term to standard pair-based STDP.

        Phase 1B: Optional dopamine modulation via da_level.
        """
        # Get base pair update (with DA modulation if provided)
        base_update = super().compute_update(pre_id, post_id, current_weight, da_level=da_level)

        if base_update is None:
            return None

        # Check for triplet contribution
        pre_history = self._spike_history.get(pre_id, [])
        post_history = self._spike_history.get(post_id, [])

        if len(pre_history) < 2 and len(post_history) < 2:
            return base_update

        # Compute triplet contribution
        triplet_delta = 0.0
        now = datetime.now()

        # Post-Pre-Post triplet (enhances LTP)
        if len(post_history) >= 2 and len(pre_history) >= 1:
            post1_time = post_history[-2][0]
            pre_time = pre_history[-1][0]
            post2_time = post_history[-1][0]

            if post1_time < pre_time < post2_time:
                dt1 = (pre_time - post1_time).total_seconds()
                dt2 = (post2_time - pre_time).total_seconds()
                triplet_delta += self.triplet_a_plus * np.exp(-dt2 / self.tau_triplet)

        # Pre-Post-Pre triplet (enhances LTD)
        if len(pre_history) >= 2 and len(post_history) >= 1:
            pre1_time = pre_history[-2][0]
            post_time = post_history[-1][0]
            pre2_time = pre_history[-1][0]

            if pre1_time < post_time < pre2_time:
                dt1 = (post_time - pre1_time).total_seconds()
                dt2 = (pre2_time - post_time).total_seconds()
                triplet_delta -= self.triplet_a_minus * np.exp(-dt1 / self.tau_triplet)

        # Apply triplet correction
        if triplet_delta != 0:
            synapse_key = (pre_id, post_id)
            new_weight = np.clip(
                base_update.new_weight + triplet_delta,
                self.config.min_weight,
                self.config.max_weight
            )
            self._weights[synapse_key] = new_weight

            base_update.new_weight = new_weight
            base_update.delta_weight = new_weight - base_update.old_weight

        return base_update


# Singleton instance for global use
_stdp_learner: STDPLearner | None = None


def get_stdp_learner() -> STDPLearner:
    """Get or create global STDPLearner instance."""
    global _stdp_learner
    if _stdp_learner is None:
        _stdp_learner = STDPLearner()
    return _stdp_learner


def reset_stdp_learner() -> None:
    """Reset global STDP learner (for testing)."""
    global _stdp_learner
    _stdp_learner = None
