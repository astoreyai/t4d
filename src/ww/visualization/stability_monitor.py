"""
Stability Monitor Visualization for World Weaver NCA.

Comprehensive stability analysis and monitoring:
- Jacobian eigenvalue tracking (stability indicator)
- Lyapunov exponent estimation (chaos detection)
- Bifurcation detection (regime changes)
- Convergence monitoring (attractor approach)
- Oscillation detection (limit cycles)

This is CRITICAL for detecting pathological dynamics
and ensuring the NCA operates in stable regimes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ww.nca.coupling import LearnableCoupling
    from ww.nca.energy import EnergyLandscape

logger = logging.getLogger(__name__)

# Stability type from eigenvalue analysis
class StabilityType(Enum):
    """Classification of dynamical stability."""
    STABLE_NODE = auto()      # All eigenvalues negative real
    STABLE_FOCUS = auto()     # Complex eigenvalues, negative real parts
    UNSTABLE_NODE = auto()    # All eigenvalues positive real
    UNSTABLE_FOCUS = auto()   # Complex eigenvalues, positive real parts
    SADDLE = auto()           # Mixed sign real parts
    CENTER = auto()           # Pure imaginary eigenvalues
    BIFURCATION = auto()      # Near-zero eigenvalue


@dataclass
class StabilitySnapshot:
    """Snapshot of system stability state."""

    timestamp: datetime
    nt_state: np.ndarray  # Current NT state

    # Jacobian eigenvalues
    eigenvalues: np.ndarray  # Complex eigenvalues
    max_real_eigenvalue: float
    min_real_eigenvalue: float
    spectral_abscissa: float  # max(Re(eigenvalues))

    # Stability classification
    stability_type: StabilityType
    is_stable: bool
    stability_margin: float  # Distance from instability

    # Lyapunov analysis
    lyapunov_exponent: float
    is_chaotic: bool

    # Oscillation metrics
    has_oscillations: bool
    oscillation_frequency: float  # Hz (from imaginary parts)
    damping_ratio: float

    # Convergence
    energy: float
    energy_gradient_norm: float
    convergence_rate: float  # Estimated from eigenvalues


@dataclass
class BifurcationEvent:
    """Record of a stability regime change."""

    timestamp: datetime
    nt_state: np.ndarray
    old_type: StabilityType
    new_type: StabilityType
    eigenvalue_crossing: float  # Which eigenvalue crossed zero
    parameter_sensitivity: float  # How close to bifurcation


class StabilityMonitor:
    """
    Real-time stability monitoring dashboard.

    Tracks dynamical stability of the NCA system using
    Jacobian eigenvalue analysis, Lyapunov exponents,
    and bifurcation detection.
    """

    def __init__(
        self,
        coupling: LearnableCoupling | None = None,
        energy_landscape: EnergyLandscape | None = None,
        window_size: int = 1000,
        tau: float = 0.1,
        alert_stability_margin: float = 0.1,
        alert_lyapunov: float = 0.01,
    ):
        """
        Initialize stability monitor.

        Args:
            coupling: LearnableCoupling for Jacobian computation
            energy_landscape: EnergyLandscape for energy metrics
            window_size: Number of snapshots to retain
            tau: Time constant for dynamics
            alert_stability_margin: Threshold for stability alerts
            alert_lyapunov: Threshold for chaos alerts
        """
        self.coupling = coupling
        self.energy_landscape = energy_landscape
        self.window_size = window_size
        self.tau = tau
        self.alert_stability_margin = alert_stability_margin
        self.alert_lyapunov = alert_lyapunov

        # History tracking
        self._snapshots: list[StabilitySnapshot] = []
        self._bifurcations: list[BifurcationEvent] = []
        self._eigenvalue_history: list[np.ndarray] = []

        # Alerts
        self._active_alerts: list[str] = []

        # Previous state for bifurcation detection
        self._prev_stability_type: StabilityType | None = None

        logger.info("StabilityMonitor initialized")

    def record_state(
        self,
        nt_state: np.ndarray,
        jacobian: np.ndarray | None = None,
        energy: float | None = None,
        gradient: np.ndarray | None = None,
    ) -> StabilitySnapshot:
        """
        Record stability state at current NT configuration.

        Args:
            nt_state: Current 6D NT state
            jacobian: Pre-computed Jacobian (computed if None)
            energy: Pre-computed energy (computed if None)
            gradient: Pre-computed gradient (computed if None)

        Returns:
            StabilitySnapshot with all metrics
        """
        now = datetime.now()
        nt = np.asarray(nt_state, dtype=np.float32)

        # Compute Jacobian if not provided
        if jacobian is None:
            jacobian = self._compute_jacobian(nt)

        # Eigenvalue analysis
        eigenvalues = np.linalg.eigvals(jacobian)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        max_real = float(np.max(real_parts))
        min_real = float(np.min(real_parts))
        spectral_abscissa = max_real

        # Stability classification
        stability_type = self._classify_stability(eigenvalues)
        is_stable = max_real < 0
        stability_margin = -max_real if is_stable else max_real

        # Lyapunov exponent (approximated by spectral abscissa for linear)
        lyapunov_exponent = spectral_abscissa
        is_chaotic = lyapunov_exponent > self.alert_lyapunov

        # Oscillation analysis
        has_oscillations = np.any(np.abs(imag_parts) > 1e-6)
        oscillation_frequency = 0.0
        damping_ratio = 0.0

        if has_oscillations:
            # Find dominant oscillation
            osc_idx = np.argmax(np.abs(imag_parts))
            omega = np.abs(imag_parts[osc_idx])
            sigma = real_parts[osc_idx]
            oscillation_frequency = float(omega / (2 * np.pi))  # Hz

            # Damping ratio: zeta = -sigma / sqrt(sigma^2 + omega^2)
            if omega > 0:
                damping_ratio = float(-sigma / np.sqrt(sigma**2 + omega**2))

        # Energy metrics
        if energy is None and self.energy_landscape is not None:
            energy = self.energy_landscape.compute_total_energy(nt)
        energy = energy if energy is not None else 0.0

        if gradient is None and self.energy_landscape is not None:
            gradient = self.energy_landscape.compute_energy_gradient(nt)
        gradient_norm = float(np.linalg.norm(gradient)) if gradient is not None else 0.0

        # Convergence rate (dominant eigenvalue magnitude)
        convergence_rate = float(-max_real) if is_stable else 0.0

        snapshot = StabilitySnapshot(
            timestamp=now,
            nt_state=nt.copy(),
            eigenvalues=eigenvalues,
            max_real_eigenvalue=max_real,
            min_real_eigenvalue=min_real,
            spectral_abscissa=spectral_abscissa,
            stability_type=stability_type,
            is_stable=is_stable,
            stability_margin=stability_margin,
            lyapunov_exponent=lyapunov_exponent,
            is_chaotic=is_chaotic,
            has_oscillations=has_oscillations,
            oscillation_frequency=oscillation_frequency,
            damping_ratio=damping_ratio,
            energy=float(energy),
            energy_gradient_norm=gradient_norm,
            convergence_rate=convergence_rate,
        )

        # Store
        self._snapshots.append(snapshot)
        self._eigenvalue_history.append(eigenvalues.copy())

        # Maintain window
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)
            self._eigenvalue_history.pop(0)

        # Bifurcation detection
        self._detect_bifurcation(snapshot)

        # Check alerts
        self._check_alerts(snapshot)

        return snapshot

    def _compute_jacobian(self, nt_state: np.ndarray) -> np.ndarray:
        """Compute dynamics Jacobian at given state."""
        if self.coupling is not None:
            return self.coupling.compute_jacobian(nt_state, self.tau)

        # Default: simple diagonal decay
        return -np.eye(6) / self.tau

    def _classify_stability(self, eigenvalues: np.ndarray) -> StabilityType:
        """Classify stability type from eigenvalues."""
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        tol = 1e-8

        has_positive = np.any(real_parts > tol)
        has_negative = np.any(real_parts < -tol)
        has_zero = np.any(np.abs(real_parts) < tol)
        has_complex = np.any(np.abs(imag_parts) > tol)

        if has_zero:
            return StabilityType.BIFURCATION
        elif has_positive and has_negative:
            return StabilityType.SADDLE
        elif has_positive:
            if has_complex:
                return StabilityType.UNSTABLE_FOCUS
            return StabilityType.UNSTABLE_NODE
        elif has_negative:
            if has_complex:
                return StabilityType.STABLE_FOCUS
            return StabilityType.STABLE_NODE
        else:
            return StabilityType.CENTER

    def _detect_bifurcation(self, snapshot: StabilitySnapshot) -> None:
        """Detect and record bifurcation events."""
        if self._prev_stability_type is None:
            self._prev_stability_type = snapshot.stability_type
            return

        if snapshot.stability_type != self._prev_stability_type:
            # Stability type changed - bifurcation!
            event = BifurcationEvent(
                timestamp=snapshot.timestamp,
                nt_state=snapshot.nt_state.copy(),
                old_type=self._prev_stability_type,
                new_type=snapshot.stability_type,
                eigenvalue_crossing=snapshot.max_real_eigenvalue,
                parameter_sensitivity=abs(snapshot.stability_margin),
            )
            self._bifurcations.append(event)

            if len(self._bifurcations) > 100:
                self._bifurcations.pop(0)

            logger.warning(
                f"Bifurcation detected: {self._prev_stability_type.name} -> "
                f"{snapshot.stability_type.name}"
            )

        self._prev_stability_type = snapshot.stability_type

    def _check_alerts(self, snapshot: StabilitySnapshot) -> None:
        """Check for alert conditions."""
        self._active_alerts.clear()

        if not snapshot.is_stable:
            self._active_alerts.append(
                f"UNSTABLE: {snapshot.stability_type.name} "
                f"(max_real={snapshot.max_real_eigenvalue:.4f})"
            )

        if snapshot.stability_margin < self.alert_stability_margin and snapshot.is_stable:
            self._active_alerts.append(
                f"MARGINAL STABILITY: margin={snapshot.stability_margin:.4f}"
            )

        if snapshot.is_chaotic:
            self._active_alerts.append(
                f"CHAOTIC: Lyapunov={snapshot.lyapunov_exponent:.4f}"
            )

        if snapshot.stability_type == StabilityType.BIFURCATION:
            self._active_alerts.append("BIFURCATION POINT: Near-zero eigenvalue")

    def get_alerts(self) -> list[str]:
        """Get current active alerts."""
        return self._active_alerts.copy()

    # -------------------------------------------------------------------------
    # Enhanced Anomaly Detection (P0-3)
    # -------------------------------------------------------------------------

    def detect_excitotoxicity(self, nt_state: np.ndarray) -> list[str]:
        """
        Detect excitotoxic conditions from NT state.

        Excitotoxicity occurs when excessive glutamate causes neuronal damage.
        We detect:
        - Glutamate surge (Glu > 0.85)
        - E/I imbalance (Glu/GABA ratio > 3.0, seizure-like)
        - GABA depletion (GABA < 0.2)

        NT indices: [DA, 5-HT, ACh, NE, GABA, Glu]

        Args:
            nt_state: 6D neurotransmitter state array

        Returns:
            List of alert strings (empty if none)
        """
        alerts = []
        nt = np.asarray(nt_state, dtype=np.float32)

        if len(nt) < 6:
            return alerts

        glutamate = float(nt[5])
        gaba = float(nt[4])

        # Glutamate surge - excitotoxic
        if glutamate > 0.85:
            alerts.append(f"EXCITOTOXICITY: Glu={glutamate:.3f} > 0.85")

        # E/I imbalance - seizure-like activity
        ei_ratio = glutamate / max(gaba, 0.01)
        if ei_ratio > 3.0:
            alerts.append(f"E/I IMBALANCE: ratio={ei_ratio:.2f} > 3.0 (seizure-like)")

        # GABA depletion - loss of inhibition
        if gaba < 0.2:
            alerts.append(f"GABA DEPLETION: GABA={gaba:.3f} < 0.2")

        # Log if critical
        for alert in alerts:
            logger.warning(f"Excitotoxicity alert: {alert}")

        return alerts

    def detect_forgetting_risk(self, hippocampus_state: dict) -> list[str]:
        """
        Detect catastrophic forgetting risk from hippocampal state.

        Monitors:
        - CA3 pattern capacity (near full = interference)
        - Pattern overlap/similarity (high overlap = forgetting)
        - Memory collision rate

        Args:
            hippocampus_state: Dict with CA3 pattern info:
                - ca3_patterns: list of stored patterns
                - ca3_max_patterns: maximum capacity
                - pattern_similarity: average pairwise similarity

        Returns:
            List of alert strings
        """
        alerts = []

        # CA3 capacity check
        if "ca3_patterns" in hippocampus_state:
            n_patterns = len(hippocampus_state["ca3_patterns"])
            max_patterns = hippocampus_state.get("ca3_max_patterns", 1000)

            if max_patterns > 0:
                capacity_fraction = n_patterns / max_patterns

                if capacity_fraction > 0.95:
                    alerts.append(
                        f"CA3 CRITICAL CAPACITY: {capacity_fraction*100:.1f}% "
                        f"({n_patterns}/{max_patterns} patterns)"
                    )
                elif capacity_fraction > 0.9:
                    alerts.append(
                        f"CA3 NEAR CAPACITY: {capacity_fraction*100:.1f}% "
                        f"({n_patterns}/{max_patterns} patterns)"
                    )

        # Pattern overlap check (high similarity = interference)
        if "pattern_similarity" in hippocampus_state:
            avg_similarity = float(hippocampus_state["pattern_similarity"])

            if avg_similarity > 0.7:
                alerts.append(
                    f"SEVERE PATTERN OVERLAP: similarity={avg_similarity:.3f} > 0.7 "
                    "(high interference risk)"
                )
            elif avg_similarity > 0.6:
                alerts.append(
                    f"PATTERN OVERLAP: similarity={avg_similarity:.3f} > 0.6 "
                    "(interference risk)"
                )

        # Memory collision rate
        if "collision_rate" in hippocampus_state:
            collision_rate = float(hippocampus_state["collision_rate"])
            if collision_rate > 0.1:
                alerts.append(
                    f"MEMORY COLLISION: rate={collision_rate:.3f} > 0.1"
                )

        for alert in alerts:
            logger.warning(f"Forgetting risk: {alert}")

        return alerts

    def detect_nt_depletion(self, nt_state: np.ndarray) -> list[str]:
        """
        Detect neuromodulator depletion states.

        Low neuromodulator levels impair learning and memory:
        - Low DA: reduced reward learning, motivation
        - Low NE: reduced attention, consolidation
        - Low ACh: reduced encoding, pattern separation
        - Low 5-HT: mood/behavioral instability

        NT indices: [DA, 5-HT, ACh, NE, GABA, Glu]

        Args:
            nt_state: 6D neurotransmitter state array

        Returns:
            List of alert strings
        """
        alerts = []
        nt = np.asarray(nt_state, dtype=np.float32)

        if len(nt) < 6:
            return alerts

        da = float(nt[0])
        serotonin = float(nt[1])
        ach = float(nt[2])
        ne = float(nt[3])

        # Depletion thresholds (biological homeostatic setpoints ~0.3-0.5)
        if da < 0.15:
            alerts.append(f"DA DEPLETION: DA={da:.3f} < 0.15 (impaired reward learning)")

        if ne < 0.15:
            alerts.append(f"NE DEPLETION: NE={ne:.3f} < 0.15 (impaired attention)")

        if ach < 0.2:
            alerts.append(f"ACh DEPLETION: ACh={ach:.3f} < 0.2 (impaired encoding)")

        if serotonin < 0.15:
            alerts.append(f"5-HT DEPLETION: 5-HT={serotonin:.3f} < 0.15 (mood instability)")

        # Also check for pathological elevation
        if da > 0.9:
            alerts.append(f"DA SURGE: DA={da:.3f} > 0.9 (hyperdopaminergic)")

        if ne > 0.85:
            alerts.append(f"NE SURGE: NE={ne:.3f} > 0.85 (hyperarousal)")

        for alert in alerts:
            logger.warning(f"NT depletion: {alert}")

        return alerts

    def detect_oscillation_pathology(self, nt_state: np.ndarray) -> list[str]:
        """
        Detect pathological oscillation patterns.

        Monitors for:
        - Seizure-like high-frequency oscillations
        - Pathological synchronization
        - Loss of normal rhythms

        Args:
            nt_state: Current NT state

        Returns:
            List of alert strings
        """
        alerts = []

        if not self._snapshots:
            return alerts

        current = self._snapshots[-1]

        # Very high oscillation frequency (seizure-like)
        if current.oscillation_frequency > 100:  # > 100 Hz
            alerts.append(
                f"HIGH FREQUENCY OSCILLATION: {current.oscillation_frequency:.1f} Hz "
                "(seizure-like)"
            )

        # Very low damping with oscillations (sustained pathological rhythm)
        if current.has_oscillations and current.damping_ratio < 0.05:
            alerts.append(
                f"UNDAMPED OSCILLATION: ζ={current.damping_ratio:.3f} "
                "(pathological rhythm)"
            )

        # Check for oscillation instability (growing amplitude)
        if len(self._snapshots) >= 10:
            recent_damping = [s.damping_ratio for s in self._snapshots[-10:]]
            if all(d < 0 for d in recent_damping):
                alerts.append("GROWING OSCILLATION: negative damping trend")

        return alerts

    def detect_anomalies(
        self,
        nt_state: np.ndarray | None = None,
        hippocampus_state: dict | None = None,
    ) -> dict[str, list[str]]:
        """
        Comprehensive anomaly detection combining all checks.

        Args:
            nt_state: Current 6D NT state (uses last recorded if None)
            hippocampus_state: Optional hippocampus state dict

        Returns:
            Dict mapping category to list of alerts:
            - "stability": Jacobian/eigenvalue alerts
            - "excitotoxicity": Glutamate/GABA alerts
            - "forgetting": CA3 capacity/overlap alerts
            - "depletion": Neuromodulator depletion alerts
            - "oscillation": Pathological rhythm alerts
        """
        anomalies: dict[str, list[str]] = {
            "stability": [],
            "excitotoxicity": [],
            "forgetting": [],
            "depletion": [],
            "oscillation": [],
        }

        # Use last recorded state if not provided
        if nt_state is None and self._snapshots:
            nt_state = self._snapshots[-1].nt_state

        # Stability alerts (from existing mechanism)
        anomalies["stability"] = self.get_alerts()

        # Enhanced anomaly detection
        if nt_state is not None:
            anomalies["excitotoxicity"] = self.detect_excitotoxicity(nt_state)
            anomalies["depletion"] = self.detect_nt_depletion(nt_state)
            anomalies["oscillation"] = self.detect_oscillation_pathology(nt_state)

        if hippocampus_state is not None:
            anomalies["forgetting"] = self.detect_forgetting_risk(hippocampus_state)

        return anomalies

    def get_anomaly_summary(
        self,
        nt_state: np.ndarray | None = None,
        hippocampus_state: dict | None = None,
    ) -> dict:
        """
        Get summary of current anomaly state.

        Returns:
            Dict with:
            - total_alerts: Total number of alerts
            - by_category: Count per category
            - critical: Whether any critical alerts
            - all_alerts: Flattened list of all alerts
        """
        anomalies = self.detect_anomalies(nt_state, hippocampus_state)

        all_alerts = []
        by_category = {}
        critical = False

        for category, alerts in anomalies.items():
            by_category[category] = len(alerts)
            all_alerts.extend(alerts)

            # Check for critical alerts
            for alert in alerts:
                if any(kw in alert for kw in ["CRITICAL", "EXCITOTOXICITY", "SEIZURE", "UNSTABLE"]):
                    critical = True

        return {
            "total_alerts": len(all_alerts),
            "by_category": by_category,
            "critical": critical,
            "all_alerts": all_alerts,
        }

    # -------------------------------------------------------------------------
    # Current State Access
    # -------------------------------------------------------------------------

    def get_current_snapshot(self) -> StabilitySnapshot | None:
        """Get most recent stability snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def is_stable(self) -> bool:
        """Check if system is currently stable."""
        if not self._snapshots:
            return True
        return self._snapshots[-1].is_stable

    def get_stability_type(self) -> StabilityType | None:
        """Get current stability classification."""
        if not self._snapshots:
            return None
        return self._snapshots[-1].stability_type

    def get_current_eigenvalues(self) -> np.ndarray | None:
        """Get current Jacobian eigenvalues."""
        if not self._snapshots:
            return None
        return self._snapshots[-1].eigenvalues.copy()

    # -------------------------------------------------------------------------
    # Time Series Access
    # -------------------------------------------------------------------------

    def get_eigenvalue_traces(self) -> dict[str, tuple[list[datetime], list[float]]]:
        """
        Get time series of eigenvalue statistics.

        Returns:
            Dict with max_real, min_real, spectral_abscissa traces
        """
        if not self._snapshots:
            return {}

        timestamps = [s.timestamp for s in self._snapshots]

        return {
            "max_real": (timestamps, [s.max_real_eigenvalue for s in self._snapshots]),
            "min_real": (timestamps, [s.min_real_eigenvalue for s in self._snapshots]),
            "spectral_abscissa": (timestamps, [s.spectral_abscissa for s in self._snapshots]),
        }

    def get_stability_margin_trace(self) -> tuple[list[datetime], list[float]]:
        """Get stability margin time series."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        margins = [s.stability_margin for s in self._snapshots]
        return timestamps, margins

    def get_lyapunov_trace(self) -> tuple[list[datetime], list[float]]:
        """Get Lyapunov exponent time series."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        lyapunov = [s.lyapunov_exponent for s in self._snapshots]
        return timestamps, lyapunov

    def get_oscillation_traces(self) -> dict[str, tuple[list[datetime], list[float]]]:
        """Get oscillation metrics time series."""
        if not self._snapshots:
            return {}

        timestamps = [s.timestamp for s in self._snapshots]

        return {
            "frequency": (timestamps, [s.oscillation_frequency for s in self._snapshots]),
            "damping_ratio": (timestamps, [s.damping_ratio for s in self._snapshots]),
        }

    def get_convergence_trace(self) -> tuple[list[datetime], list[float]]:
        """Get convergence rate time series."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        rates = [s.convergence_rate for s in self._snapshots]
        return timestamps, rates

    def get_energy_trace(self) -> tuple[list[datetime], list[float]]:
        """Get energy time series."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        energies = [s.energy for s in self._snapshots]
        return timestamps, energies

    # -------------------------------------------------------------------------
    # Eigenvalue Evolution
    # -------------------------------------------------------------------------

    def get_eigenvalue_evolution(self, component: str = "real") -> np.ndarray:
        """
        Get evolution of all 6 eigenvalues over time.

        Args:
            component: "real", "imag", or "abs"

        Returns:
            Array of shape [n_snapshots, 6]
        """
        if not self._eigenvalue_history:
            return np.array([]).reshape(0, 6)

        eigenvalues = np.array(self._eigenvalue_history)

        if component == "real":
            return np.real(eigenvalues)
        elif component == "imag":
            return np.imag(eigenvalues)
        else:  # abs
            return np.abs(eigenvalues)

    def get_eigenvalue_trajectory(self) -> list[tuple[float, float]]:
        """
        Get trajectory of dominant eigenvalue in complex plane.

        Returns:
            List of (real, imag) tuples
        """
        if not self._eigenvalue_history:
            return []

        trajectory = []
        for eigenvalues in self._eigenvalue_history:
            # Find dominant (largest magnitude)
            idx = np.argmax(np.abs(eigenvalues))
            trajectory.append((
                float(np.real(eigenvalues[idx])),
                float(np.imag(eigenvalues[idx])),
            ))

        return trajectory

    # -------------------------------------------------------------------------
    # Bifurcation Analysis
    # -------------------------------------------------------------------------

    def get_bifurcation_events(self) -> list[BifurcationEvent]:
        """Get list of detected bifurcation events."""
        return self._bifurcations.copy()

    def get_bifurcation_count(self) -> int:
        """Get number of bifurcations detected."""
        return len(self._bifurcations)

    def get_stability_type_distribution(self) -> dict[str, int]:
        """Get distribution of stability types over history."""
        if not self._snapshots:
            return {}

        counts = {}
        for s in self._snapshots:
            name = s.stability_type.name
            counts[name] = counts.get(name, 0) + 1

        return counts

    # -------------------------------------------------------------------------
    # Stability Metrics
    # -------------------------------------------------------------------------

    def compute_stability_statistics(self) -> dict:
        """Compute comprehensive stability statistics."""
        if not self._snapshots:
            return {}

        recent = self._snapshots[-min(100, len(self._snapshots)):]

        margins = [s.stability_margin for s in recent]
        lyapunov = [s.lyapunov_exponent for s in recent]
        stable_frac = sum(1 for s in recent if s.is_stable) / len(recent)

        return {
            "current_stability_margin": self._snapshots[-1].stability_margin,
            "mean_stability_margin": float(np.mean(margins)),
            "min_stability_margin": float(np.min(margins)),
            "margin_trend": self._compute_trend(margins),
            "current_lyapunov": self._snapshots[-1].lyapunov_exponent,
            "mean_lyapunov": float(np.mean(lyapunov)),
            "max_lyapunov": float(np.max(lyapunov)),
            "stable_fraction": stable_frac,
            "bifurcation_count": len(self._bifurcations),
            "current_type": self._snapshots[-1].stability_type.name,
            "is_stable": self._snapshots[-1].is_stable,
            "is_chaotic": self._snapshots[-1].is_chaotic,
        }

    def _compute_trend(self, values: list[float]) -> float:
        """Compute linear trend."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return float(coeffs[0])

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_data(self) -> dict:
        """Export all visualization data."""
        current = self.get_current_snapshot()
        stats = self.compute_stability_statistics()

        eigenvalue_traces = self.get_eigenvalue_traces()
        timestamps_margin, margins = self.get_stability_margin_trace()

        return {
            "current_state": {
                "stability_type": current.stability_type.name if current else None,
                "is_stable": current.is_stable if current else True,
                "stability_margin": current.stability_margin if current else 0.0,
                "lyapunov_exponent": current.lyapunov_exponent if current else 0.0,
                "eigenvalues_real": np.real(current.eigenvalues).tolist() if current else [],
                "eigenvalues_imag": np.imag(current.eigenvalues).tolist() if current else [],
            },
            "traces": {
                "stability_margin": {
                    "timestamps": [t.isoformat() for t in timestamps_margin],
                    "values": margins,
                },
                "max_real_eigenvalue": {
                    "timestamps": [t.isoformat() for t in eigenvalue_traces.get("max_real", ([], []))[0]],
                    "values": eigenvalue_traces.get("max_real", ([], []))[1],
                },
            },
            "statistics": stats,
            "bifurcations": [
                {
                    "timestamp": b.timestamp.isoformat(),
                    "old_type": b.old_type.name,
                    "new_type": b.new_type.name,
                }
                for b in self._bifurcations
            ],
            "type_distribution": self.get_stability_type_distribution(),
            "alerts": self.get_alerts(),
            "n_samples": len(self._snapshots),
        }

    def clear_history(self) -> None:
        """Clear all history."""
        self._snapshots.clear()
        self._bifurcations.clear()
        self._eigenvalue_history.clear()
        self._active_alerts.clear()
        self._prev_stability_type = None


# =============================================================================
# Standalone Plot Functions
# =============================================================================


def plot_eigenvalue_spectrum(
    monitor: StabilityMonitor,
    ax=None,
    show_history: bool = True,
    history_alpha: float = 0.3,
):
    """
    Plot eigenvalues in complex plane with stability regions.

    Args:
        monitor: StabilityMonitor instance
        ax: Matplotlib axes
        show_history: Show trajectory of eigenvalues
        history_alpha: Alpha for historical points

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    eigenvalues = monitor.get_current_eigenvalues()
    if eigenvalues is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    # Shade stability regions
    ax.axvspan(-10, 0, alpha=0.1, color="green", label="Stable region")
    ax.axvspan(0, 10, alpha=0.1, color="red", label="Unstable region")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

    # Plot historical eigenvalues
    if show_history and len(monitor._eigenvalue_history) > 1:
        for i, hist_eig in enumerate(monitor._eigenvalue_history[:-1]):
            alpha = history_alpha * (i / len(monitor._eigenvalue_history))
            ax.scatter(
                np.real(hist_eig), np.imag(hist_eig),
                c="gray", s=20, alpha=alpha, marker="."
            )

    # Plot current eigenvalues
    ax.scatter(
        np.real(eigenvalues), np.imag(eigenvalues),
        c="blue", s=100, marker="x", linewidths=2, label="Current eigenvalues"
    )

    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title("Jacobian Eigenvalue Spectrum")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Set reasonable limits
    max_abs = max(np.abs(eigenvalues).max() * 1.5, 0.5)
    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)

    return ax


def plot_stability_timeline(
    monitor: StabilityMonitor,
    ax=None,
    show_threshold: bool = True,
):
    """
    Plot stability margin over time.

    Args:
        monitor: StabilityMonitor instance
        ax: Matplotlib axes
        show_threshold: Show alert threshold

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    timestamps, margins = monitor.get_stability_margin_trace()
    if not timestamps:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    t0 = timestamps[0]
    t_seconds = [(t - t0).total_seconds() for t in timestamps]

    # Color by stability
    colors = ["green" if m > 0 else "red" for m in margins]

    ax.scatter(t_seconds, margins, c=colors, s=10, alpha=0.7)
    ax.plot(t_seconds, margins, "b-", linewidth=0.5, alpha=0.5)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=2)

    if show_threshold:
        ax.axhline(
            y=monitor.alert_stability_margin,
            color="orange", linestyle="--", linewidth=1.5,
            label=f"Alert threshold ({monitor.alert_stability_margin})"
        )

    ax.fill_between(t_seconds, 0, margins, where=[m > 0 for m in margins],
                    alpha=0.2, color="green", label="Stable")
    ax.fill_between(t_seconds, 0, margins, where=[m <= 0 for m in margins],
                    alpha=0.2, color="red", label="Unstable")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stability Margin")
    ax.set_title("Stability Margin Over Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return ax


def plot_lyapunov_timeline(
    monitor: StabilityMonitor,
    ax=None,
):
    """
    Plot Lyapunov exponent over time.

    Args:
        monitor: StabilityMonitor instance
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    timestamps, lyapunov = monitor.get_lyapunov_trace()
    if not timestamps:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    t0 = timestamps[0]
    t_seconds = [(t - t0).total_seconds() for t in timestamps]

    ax.plot(t_seconds, lyapunov, "b-", linewidth=1.5)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=2)
    ax.axhline(y=monitor.alert_lyapunov, color="red", linestyle="--",
               label=f"Chaos threshold ({monitor.alert_lyapunov})")

    ax.fill_between(t_seconds, 0, lyapunov, where=[l > 0 for l in lyapunov],
                    alpha=0.2, color="red")
    ax.fill_between(t_seconds, 0, lyapunov, where=[l <= 0 for l in lyapunov],
                    alpha=0.2, color="green")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lyapunov Exponent")
    ax.set_title("Lyapunov Exponent (λ > 0 = Chaotic)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    return ax


def plot_eigenvalue_evolution(
    monitor: StabilityMonitor,
    ax=None,
):
    """
    Plot evolution of all 6 eigenvalue real parts.

    Args:
        monitor: StabilityMonitor instance
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    evolution = monitor.get_eigenvalue_evolution(component="real")
    if evolution.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    colors = plt.cm.viridis(np.linspace(0, 1, 6))

    for i in range(6):
        ax.plot(evolution[:, i], color=colors[i], linewidth=1.5,
               label=f"λ{i+1}", alpha=0.8)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=2, label="Stability boundary")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Real(eigenvalue)")
    ax.set_title("Eigenvalue Evolution")
    ax.legend(loc="upper right", ncol=4)
    ax.grid(True, alpha=0.3)

    return ax


def plot_bifurcation_diagram(
    monitor: StabilityMonitor,
    ax=None,
):
    """
    Plot bifurcation events on stability timeline.

    Args:
        monitor: StabilityMonitor instance
        ax: Matplotlib axes

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    timestamps, margins = monitor.get_stability_margin_trace()
    if not timestamps:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    t0 = timestamps[0]
    t_seconds = [(t - t0).total_seconds() for t in timestamps]

    ax.plot(t_seconds, margins, "b-", linewidth=1, alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Mark bifurcation events
    bifurcations = monitor.get_bifurcation_events()
    for bif in bifurcations:
        bif_t = (bif.timestamp - t0).total_seconds()
        ax.axvline(x=bif_t, color="red", linestyle=":", linewidth=2, alpha=0.8)
        ax.annotate(
            f"{bif.old_type.name[:3]}→{bif.new_type.name[:3]}",
            xy=(bif_t, 0),
            xytext=(bif_t, ax.get_ylim()[1] * 0.8),
            fontsize=8,
            rotation=90,
            ha="center",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stability Margin")
    ax.set_title(f"Bifurcation Events ({len(bifurcations)} detected)")
    ax.grid(True, alpha=0.3)

    return ax


def plot_oscillation_metrics(
    monitor: StabilityMonitor,
    ax=None,
):
    """
    Plot oscillation frequency and damping ratio.

    Args:
        monitor: StabilityMonitor instance
        ax: Matplotlib axes (creates 2 subplots if None)

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    traces = monitor.get_oscillation_traces()
    if not traces:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Frequency
    timestamps, frequencies = traces["frequency"]
    if timestamps:
        t0 = timestamps[0]
        t_seconds = [(t - t0).total_seconds() for t in timestamps]

        axes[0].plot(t_seconds, frequencies, "b-", linewidth=1.5)
        axes[0].set_ylabel("Frequency (Hz)")
        axes[0].set_title("Oscillation Frequency")
        axes[0].grid(True, alpha=0.3)

    # Damping
    timestamps, damping = traces["damping_ratio"]
    if timestamps:
        t0 = timestamps[0]
        t_seconds = [(t - t0).total_seconds() for t in timestamps]

        axes[1].plot(t_seconds, damping, "g-", linewidth=1.5)
        axes[1].axhline(y=1.0, color="gray", linestyle="--", label="Critical damping")
        axes[1].axhline(y=0, color="red", linestyle="--", label="No damping")
        axes[1].set_ylabel("Damping Ratio")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title("Damping Ratio (ζ)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_stability_dashboard(
    monitor: StabilityMonitor,
    figsize: tuple[int, int] = (16, 12),
):
    """
    Create comprehensive stability monitoring dashboard.

    Args:
        monitor: StabilityMonitor instance
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig)

    # Eigenvalue spectrum (complex plane)
    ax_spectrum = fig.add_subplot(gs[0, 0])
    plot_eigenvalue_spectrum(monitor, ax=ax_spectrum, show_history=True)

    # Eigenvalue evolution
    ax_evolution = fig.add_subplot(gs[0, 1:])
    plot_eigenvalue_evolution(monitor, ax=ax_evolution)

    # Stability timeline
    ax_stability = fig.add_subplot(gs[1, :2])
    plot_stability_timeline(monitor, ax=ax_stability)

    # Lyapunov timeline
    ax_lyapunov = fig.add_subplot(gs[2, :2])
    plot_lyapunov_timeline(monitor, ax=ax_lyapunov)

    # Metrics panel
    ax_metrics = fig.add_subplot(gs[1:, 2])
    ax_metrics.axis("off")

    stats = monitor.compute_stability_statistics()
    alerts = monitor.get_alerts()
    type_dist = monitor.get_stability_type_distribution()

    if stats:
        text_lines = [
            "Stability Metrics",
            "=" * 35,
            f"Current Type: {stats.get('current_type', 'N/A')}",
            f"Is Stable: {stats.get('is_stable', True)}",
            f"Stability Margin: {stats.get('current_stability_margin', 0):.4f}",
            f"Mean Margin: {stats.get('mean_stability_margin', 0):.4f}",
            f"Min Margin: {stats.get('min_stability_margin', 0):.4f}",
            "",
            "Lyapunov Analysis",
            "-" * 35,
            f"Current λ: {stats.get('current_lyapunov', 0):.4f}",
            f"Max λ: {stats.get('max_lyapunov', 0):.4f}",
            f"Is Chaotic: {stats.get('is_chaotic', False)}",
            "",
            f"Bifurcations: {stats.get('bifurcation_count', 0)}",
            f"Stable %: {stats.get('stable_fraction', 1.0)*100:.1f}%",
            "",
            "Type Distribution",
            "-" * 35,
        ]

        for stype, count in type_dist.items():
            text_lines.append(f"  {stype}: {count}")

        if alerts:
            text_lines.extend(["", "ALERTS", "-" * 35])
            text_lines.extend(alerts[:4])

        text = "\n".join(text_lines)
        ax_metrics.text(0.05, 0.95, text, fontsize=9, family="monospace",
                       verticalalignment="top", transform=ax_metrics.transAxes)

    plt.tight_layout()
    return fig


__all__ = [
    "StabilityMonitor",
    "StabilitySnapshot",
    "StabilityType",
    "BifurcationEvent",
    "plot_eigenvalue_spectrum",
    "plot_stability_timeline",
    "plot_lyapunov_timeline",
    "plot_eigenvalue_evolution",
    "plot_bifurcation_diagram",
    "plot_oscillation_metrics",
    "create_stability_dashboard",
]
