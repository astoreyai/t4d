"""
Coupling Matrix Dynamics Visualization for World Weaver NCA.

Hinton-inspired visualization of learnable coupling evolution:
- 6x6 heatmap of current K values with biological bounds overlay
- Time series of coupling matrix Frobenius norm
- Spectral radius tracking (stability indicator)
- Eligibility trace heatmap (credit assignment flow)
- E/I balance monitoring (GABA-Glu homeostasis)

This is ESSENTIAL for understanding:
- What NT interactions the system is learning
- Whether coupling remains within biological bounds
- Stability of the learned dynamics
- How credit flows through eligibility traces
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from t4dm.nca.coupling import LearnableCoupling

logger = logging.getLogger(__name__)

# NT labels for visualization
NT_LABELS = ["DA", "5-HT", "ACh", "NE", "GABA", "Glu"]


@dataclass
class CouplingSnapshot:
    """Snapshot of coupling matrix state."""

    timestamp: datetime
    K: np.ndarray  # Current coupling matrix [6, 6]
    frobenius_norm: float
    spectral_radius: float
    max_eigenvalue_real: float
    min_eigenvalue_real: float
    ei_balance: float  # GABA-Glu coupling sum
    bounds_violations: int  # Number of entries at bounds
    gradient_norm: float  # Learning activity
    eligibility_entropy: float  # Credit assignment spread


@dataclass
class LearningEvent:
    """Record of a learning update."""

    timestamp: datetime
    update_type: str  # "rpe", "energy", "scaling"
    gradient_norm: float
    coupling_change: float  # Frobenius norm of delta K
    rpe: float | None = None


class CouplingDynamicsVisualizer:
    """
    Visualizes coupling matrix learning dynamics.

    Tracks evolution of the learnable coupling matrix K,
    monitors stability via spectral analysis, and visualizes
    credit assignment through eligibility traces.
    """

    def __init__(
        self,
        coupling: LearnableCoupling | None = None,
        window_size: int = 1000,
        alert_spectral_radius: float = 0.95,
        alert_ei_imbalance: float = 0.5,
    ):
        """
        Initialize coupling dynamics visualizer.

        Args:
            coupling: LearnableCoupling instance to monitor
            window_size: Number of snapshots to retain
            alert_spectral_radius: Threshold for stability alert
            alert_ei_imbalance: Threshold for E/I balance alert
        """
        self.coupling = coupling
        self.window_size = window_size
        self.alert_spectral_radius = alert_spectral_radius
        self.alert_ei_imbalance = alert_ei_imbalance

        # History tracking
        self._snapshots: list[CouplingSnapshot] = []
        self._learning_events: list[LearningEvent] = []
        self._K_history: list[np.ndarray] = []

        # Alerts
        self._active_alerts: list[str] = []

        logger.info("CouplingDynamicsVisualizer initialized")

    def record_state(
        self,
        K: np.ndarray | None = None,
        eligibility_trace: np.ndarray | None = None,
        gradient: np.ndarray | None = None,
    ) -> CouplingSnapshot:
        """
        Record current coupling matrix state.

        Args:
            K: Coupling matrix (uses self.coupling.K if None)
            eligibility_trace: Current eligibility trace
            gradient: Recent gradient (for learning activity)

        Returns:
            CouplingSnapshot with all metrics
        """
        now = datetime.now()

        # Get coupling matrix
        if K is None and self.coupling is not None:
            K = self.coupling.K.copy()
        elif K is None:
            K = np.zeros((6, 6), dtype=np.float32)
        else:
            K = np.asarray(K).copy()

        # Get eligibility trace
        if eligibility_trace is None and self.coupling is not None:
            eligibility_trace = self.coupling.get_eligibility_trace()
        elif eligibility_trace is None:
            eligibility_trace = np.zeros((6, 6), dtype=np.float32)

        # Compute metrics
        frobenius_norm = float(np.linalg.norm(K))

        eigenvalues = np.linalg.eigvals(K)
        spectral_radius = float(np.abs(eigenvalues).max())
        max_eigenvalue_real = float(np.max(np.real(eigenvalues)))
        min_eigenvalue_real = float(np.min(np.real(eigenvalues)))

        # E/I balance: sum of GABA-Glu interactions (should be negative)
        ei_balance = float(K[4, 5] + K[5, 4])  # GABA->Glu + Glu->GABA

        # Count bounds violations
        bounds_violations = 0
        if self.coupling is not None:
            bounds = self.coupling.bounds
            at_min = np.sum(np.abs(K - bounds.K_MIN) < 1e-6)
            at_max = np.sum(np.abs(K - bounds.K_MAX) < 1e-6)
            bounds_violations = int(at_min + at_max)

        # Gradient norm
        gradient_norm = 0.0
        if gradient is not None:
            gradient_norm = float(np.linalg.norm(gradient))
        elif self.coupling is not None and len(self.coupling._gradient_history) > 0:
            gradient_norm = float(np.linalg.norm(self.coupling._gradient_history[-1]))

        # Eligibility entropy (spread of credit assignment)
        eligibility_entropy = self._compute_entropy(eligibility_trace)

        # Create snapshot
        snapshot = CouplingSnapshot(
            timestamp=now,
            K=K,
            frobenius_norm=frobenius_norm,
            spectral_radius=spectral_radius,
            max_eigenvalue_real=max_eigenvalue_real,
            min_eigenvalue_real=min_eigenvalue_real,
            ei_balance=ei_balance,
            bounds_violations=bounds_violations,
            gradient_norm=gradient_norm,
            eligibility_entropy=eligibility_entropy,
        )

        # Store snapshot
        self._snapshots.append(snapshot)
        self._K_history.append(K.copy())

        # Maintain window
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)
        if len(self._K_history) > self.window_size:
            self._K_history.pop(0)

        # Check alerts
        self._check_alerts(snapshot)

        return snapshot

    def record_learning_event(
        self,
        update_type: str,
        gradient: np.ndarray,
        K_before: np.ndarray,
        K_after: np.ndarray,
        rpe: float | None = None,
    ) -> LearningEvent:
        """
        Record a learning update event.

        Args:
            update_type: Type of update ("rpe", "energy", "scaling")
            gradient: Gradient used for update
            K_before: Coupling matrix before update
            K_after: Coupling matrix after update
            rpe: Reward prediction error (if applicable)

        Returns:
            LearningEvent record
        """
        event = LearningEvent(
            timestamp=datetime.now(),
            update_type=update_type,
            gradient_norm=float(np.linalg.norm(gradient)),
            coupling_change=float(np.linalg.norm(K_after - K_before)),
            rpe=rpe,
        )

        self._learning_events.append(event)

        if len(self._learning_events) > self.window_size:
            self._learning_events.pop(0)

        return event

    def _compute_entropy(self, matrix: np.ndarray) -> float:
        """Compute entropy of matrix values (normalized)."""
        flat = np.abs(matrix.flatten())
        total = flat.sum()
        if total < 1e-10:
            return 0.0
        probs = flat / total
        probs = probs[probs > 1e-10]  # Avoid log(0)
        entropy = -np.sum(probs * np.log(probs))
        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log(len(flat))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _check_alerts(self, snapshot: CouplingSnapshot) -> None:
        """Check for alert conditions."""
        self._active_alerts.clear()

        if snapshot.spectral_radius > self.alert_spectral_radius:
            self._active_alerts.append(
                f"STABILITY: Spectral radius {snapshot.spectral_radius:.3f} > {self.alert_spectral_radius}"
            )

        if snapshot.ei_balance > -self.alert_ei_imbalance:
            self._active_alerts.append(
                f"E/I BALANCE: GABA-Glu sum {snapshot.ei_balance:.3f} > -{self.alert_ei_imbalance}"
            )

        if snapshot.bounds_violations > 10:
            self._active_alerts.append(
                f"BOUNDS: {snapshot.bounds_violations} entries at biological limits"
            )

    def get_alerts(self) -> list[str]:
        """Get current active alerts."""
        return self._active_alerts.copy()

    # -------------------------------------------------------------------------
    # Coupling Matrix Analysis
    # -------------------------------------------------------------------------

    def get_current_K(self) -> np.ndarray | None:
        """Get current coupling matrix."""
        if self._snapshots:
            return self._snapshots[-1].K.copy()
        elif self.coupling is not None:
            return self.coupling.K.copy()
        return None

    def get_K_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get biological bounds (K_MIN, K_MAX)."""
        if self.coupling is not None:
            return self.coupling.bounds.K_MIN, self.coupling.bounds.K_MAX
        else:
            from t4dm.nca.coupling import BiologicalBounds
            bounds = BiologicalBounds()
            return bounds.K_MIN, bounds.K_MAX

    def get_K_normalized(self) -> np.ndarray | None:
        """
        Get K normalized to [-1, 1] within biological bounds.

        Useful for visualizing how close values are to their limits.
        """
        K = self.get_current_K()
        if K is None:
            return None

        K_min, K_max = self.get_K_bounds()
        K_range = K_max - K_min
        K_range[K_range == 0] = 1  # Avoid division by zero

        # Normalize to [0, 1] then shift to [-1, 1]
        K_norm = (K - K_min) / K_range
        return 2 * K_norm - 1

    def get_coupling_change_rate(self, window: int = 10) -> np.ndarray:
        """
        Compute rate of change for each coupling entry.

        Args:
            window: Number of recent snapshots to consider

        Returns:
            6x6 matrix of change rates
        """
        if len(self._K_history) < 2:
            return np.zeros((6, 6), dtype=np.float32)

        recent = self._K_history[-window:]
        if len(recent) < 2:
            return np.zeros((6, 6), dtype=np.float32)

        # Compute mean absolute change per step
        changes = []
        for i in range(1, len(recent)):
            changes.append(np.abs(recent[i] - recent[i-1]))

        return np.mean(changes, axis=0).astype(np.float32)

    # -------------------------------------------------------------------------
    # Spectral Analysis
    # -------------------------------------------------------------------------

    def get_spectral_radius_trace(self) -> tuple[list[datetime], list[float]]:
        """Get time series of spectral radius."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        radii = [s.spectral_radius for s in self._snapshots]
        return timestamps, radii

    def get_eigenvalue_evolution(self) -> dict[str, list[float]]:
        """Get evolution of eigenvalue statistics."""
        return {
            "max_real": [s.max_eigenvalue_real for s in self._snapshots],
            "min_real": [s.min_eigenvalue_real for s in self._snapshots],
            "spectral_radius": [s.spectral_radius for s in self._snapshots],
        }

    def get_current_eigenvalues(self) -> np.ndarray:
        """Get eigenvalues of current coupling matrix."""
        K = self.get_current_K()
        if K is None:
            return np.array([])
        return np.linalg.eigvals(K)

    def is_stable(self) -> bool:
        """Check if current coupling is stable (spectral radius < 1)."""
        if not self._snapshots:
            return True
        return self._snapshots[-1].spectral_radius < 1.0

    # -------------------------------------------------------------------------
    # E/I Balance Analysis
    # -------------------------------------------------------------------------

    def get_ei_balance_trace(self) -> tuple[list[datetime], list[float]]:
        """Get time series of E/I balance."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        balances = [s.ei_balance for s in self._snapshots]
        return timestamps, balances

    def get_ei_components(self) -> dict[str, float]:
        """Get detailed E/I balance components."""
        K = self.get_current_K()
        if K is None:
            return {}

        return {
            "gaba_to_glu": float(K[4, 5]),  # GABA inhibits Glu
            "glu_to_gaba": float(K[5, 4]),  # Glu inhibits GABA
            "gaba_total_out": float(K[4, :].sum()),  # Total GABA output
            "glu_total_out": float(K[5, :].sum()),   # Total Glu output
            "inhibitory_sum": float(K[K < 0].sum()),  # All inhibitory
            "excitatory_sum": float(K[K > 0].sum()),  # All excitatory
            "ei_ratio": float(K[K > 0].sum() / max(abs(K[K < 0].sum()), 1e-10)),
        }

    # -------------------------------------------------------------------------
    # Eligibility Trace Analysis
    # -------------------------------------------------------------------------

    def get_eligibility_trace(self) -> np.ndarray | None:
        """Get current eligibility trace matrix."""
        if self.coupling is not None:
            return self.coupling.get_eligibility_trace()
        return None

    def get_eligibility_entropy_trace(self) -> tuple[list[datetime], list[float]]:
        """Get time series of eligibility entropy."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        entropies = [s.eligibility_entropy for s in self._snapshots]
        return timestamps, entropies

    def get_credit_flow(self) -> dict[str, float]:
        """
        Analyze where credit is flowing in the eligibility trace.

        Returns dict mapping NT pair -> eligibility magnitude.
        """
        trace = self.get_eligibility_trace()
        if trace is None:
            return {}

        flow = {}
        for i in range(6):
            for j in range(6):
                key = f"{NT_LABELS[i]}->{NT_LABELS[j]}"
                flow[key] = float(trace[i, j])

        # Sort by magnitude
        return dict(sorted(flow.items(), key=lambda x: -abs(x[1])))

    # -------------------------------------------------------------------------
    # Learning Activity Analysis
    # -------------------------------------------------------------------------

    def get_learning_activity_trace(self) -> tuple[list[datetime], list[float]]:
        """Get time series of learning activity (gradient norm)."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        norms = [s.gradient_norm for s in self._snapshots]
        return timestamps, norms

    def get_frobenius_norm_trace(self) -> tuple[list[datetime], list[float]]:
        """Get time series of coupling matrix norm."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        norms = [s.frobenius_norm for s in self._snapshots]
        return timestamps, norms

    def get_learning_summary(self) -> dict:
        """Get summary of learning activity."""
        if not self._learning_events:
            return {"total_updates": 0}

        rpe_events = [e for e in self._learning_events if e.update_type == "rpe"]
        energy_events = [e for e in self._learning_events if e.update_type == "energy"]

        return {
            "total_updates": len(self._learning_events),
            "rpe_updates": len(rpe_events),
            "energy_updates": len(energy_events),
            "mean_gradient_norm": float(np.mean([e.gradient_norm for e in self._learning_events])),
            "mean_coupling_change": float(np.mean([e.coupling_change for e in self._learning_events])),
            "total_coupling_change": float(sum(e.coupling_change for e in self._learning_events)),
        }

    # -------------------------------------------------------------------------
    # Stability Metrics
    # -------------------------------------------------------------------------

    def get_stability_metrics(self) -> dict:
        """Get comprehensive stability metrics."""
        if not self._snapshots:
            return {}

        recent = self._snapshots[-min(100, len(self._snapshots)):]

        return {
            "current_spectral_radius": self._snapshots[-1].spectral_radius,
            "mean_spectral_radius": float(np.mean([s.spectral_radius for s in recent])),
            "max_spectral_radius": float(np.max([s.spectral_radius for s in recent])),
            "spectral_trend": self._compute_trend([s.spectral_radius for s in recent]),
            "current_ei_balance": self._snapshots[-1].ei_balance,
            "mean_ei_balance": float(np.mean([s.ei_balance for s in recent])),
            "bounds_violations": self._snapshots[-1].bounds_violations,
            "is_stable": self._snapshots[-1].spectral_radius < 1.0,
            "stability_margin": 1.0 - self._snapshots[-1].spectral_radius,
        }

    def _compute_trend(self, values: list[float]) -> float:
        """Compute linear trend of values."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return float(coeffs[0])

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_data(self) -> dict:
        """Export visualization data for external rendering."""
        K = self.get_current_K()
        K_norm = self.get_K_normalized()
        K_min, K_max = self.get_K_bounds()
        eligibility = self.get_eligibility_trace()

        timestamps_sr, spectral_radii = self.get_spectral_radius_trace()
        timestamps_ei, ei_balances = self.get_ei_balance_trace()

        return {
            "coupling_matrix": {
                "K": K.tolist() if K is not None else None,
                "K_normalized": K_norm.tolist() if K_norm is not None else None,
                "K_min": K_min.tolist(),
                "K_max": K_max.tolist(),
                "labels": NT_LABELS,
            },
            "eligibility_trace": eligibility.tolist() if eligibility is not None else None,
            "spectral_radius": {
                "timestamps": [t.isoformat() for t in timestamps_sr],
                "values": spectral_radii,
            },
            "ei_balance": {
                "timestamps": [t.isoformat() for t in timestamps_ei],
                "values": ei_balances,
                "components": self.get_ei_components(),
            },
            "stability_metrics": self.get_stability_metrics(),
            "learning_summary": self.get_learning_summary(),
            "alerts": self.get_alerts(),
            "change_rate": self.get_coupling_change_rate().tolist(),
        }

    def clear_history(self) -> None:
        """Clear all history."""
        self._snapshots.clear()
        self._learning_events.clear()
        self._K_history.clear()
        self._active_alerts.clear()


# =============================================================================
# Standalone Plot Functions
# =============================================================================


def plot_coupling_heatmap(
    visualizer: CouplingDynamicsVisualizer,
    ax=None,
    show_bounds: bool = True,
    show_values: bool = True,
    cmap: str = "RdBu_r",
):
    """
    Plot coupling matrix as heatmap with biological bounds overlay.

    Args:
        visualizer: CouplingDynamicsVisualizer instance
        ax: Matplotlib axes
        show_bounds: Show bound indicators
        show_values: Annotate with values
        cmap: Colormap (diverging recommended)

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    K = visualizer.get_current_K()
    if K is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    K_min, K_max = visualizer.get_K_bounds()

    # Determine symmetric color limits
    vmax = max(abs(K.min()), abs(K.max()), 0.3)
    vmin = -vmax

    # Plot heatmap
    im = ax.imshow(K, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Coupling Strength")

    # Add grid
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(NT_LABELS)
    ax.set_yticklabels(NT_LABELS)
    ax.set_xlabel("Target NT")
    ax.set_ylabel("Source NT")

    # Add value annotations
    if show_values:
        for i in range(6):
            for j in range(6):
                val = K[i, j]
                color = "white" if abs(val) > vmax * 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    # Add bounds indicators
    if show_bounds:
        for i in range(6):
            for j in range(6):
                # At minimum bound
                if abs(K[i, j] - K_min[i, j]) < 1e-6:
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2,
                                      edgecolor="blue", facecolor="none", linestyle="--")
                    ax.add_patch(rect)
                # At maximum bound
                elif abs(K[i, j] - K_max[i, j]) < 1e-6:
                    rect = Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2,
                                      edgecolor="red", facecolor="none", linestyle="--")
                    ax.add_patch(rect)

    ax.set_title("Coupling Matrix K\n(blue outline=at min, red=at max)")

    return ax


def plot_eligibility_heatmap(
    visualizer: CouplingDynamicsVisualizer,
    ax=None,
    cmap: str = "YlOrRd",
):
    """
    Plot eligibility trace as heatmap.

    Args:
        visualizer: CouplingDynamicsVisualizer instance
        ax: Matplotlib axes
        cmap: Colormap

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    trace = visualizer.get_eligibility_trace()
    if trace is None:
        ax.text(0.5, 0.5, "No eligibility data", ha="center", va="center")
        return ax

    im = ax.imshow(trace, cmap=cmap, aspect="equal")
    plt.colorbar(im, ax=ax, label="Eligibility")

    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(NT_LABELS)
    ax.set_yticklabels(NT_LABELS)
    ax.set_xlabel("Target NT")
    ax.set_ylabel("Source NT")
    ax.set_title("Eligibility Trace\n(Credit Assignment Flow)")

    return ax


def plot_spectral_radius_timeline(
    visualizer: CouplingDynamicsVisualizer,
    ax=None,
    show_threshold: bool = True,
):
    """
    Plot spectral radius over time with stability threshold.

    Args:
        visualizer: CouplingDynamicsVisualizer instance
        ax: Matplotlib axes
        show_threshold: Show stability threshold line

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    timestamps, radii = visualizer.get_spectral_radius_trace()
    if not timestamps:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    # Convert to relative seconds
    t0 = timestamps[0]
    t_seconds = [(t - t0).total_seconds() for t in timestamps]

    ax.plot(t_seconds, radii, "b-", linewidth=1.5, label="Spectral Radius")

    if show_threshold:
        ax.axhline(y=1.0, color="r", linestyle="--", linewidth=2, label="Stability Limit")
        ax.axhline(y=visualizer.alert_spectral_radius, color="orange",
                   linestyle=":", linewidth=1.5, label=f"Alert ({visualizer.alert_spectral_radius})")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spectral Radius")
    ax.set_title("Coupling Stability (Spectral Radius)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Color background based on stability
    ax.axhspan(1.0, ax.get_ylim()[1], alpha=0.1, color="red")
    ax.axhspan(0, 1.0, alpha=0.1, color="green")

    return ax


def plot_ei_balance_timeline(
    visualizer: CouplingDynamicsVisualizer,
    ax=None,
):
    """
    Plot E/I balance over time.

    Args:
        visualizer: CouplingDynamicsVisualizer instance
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
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    timestamps, balances = visualizer.get_ei_balance_trace()
    if not timestamps:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    t0 = timestamps[0]
    t_seconds = [(t - t0).total_seconds() for t in timestamps]

    ax.plot(t_seconds, balances, "g-", linewidth=1.5)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GABA-Glu Balance")
    ax.set_title("E/I Balance (should be negative)")
    ax.grid(True, alpha=0.3)

    # Color: green for inhibitory-dominant, red for excitatory-dominant
    ax.axhspan(0, ax.get_ylim()[1], alpha=0.1, color="red")
    ax.axhspan(ax.get_ylim()[0], 0, alpha=0.1, color="green")

    return ax


def plot_eigenvalue_spectrum(
    visualizer: CouplingDynamicsVisualizer,
    ax=None,
):
    """
    Plot eigenvalues in complex plane.

    Args:
        visualizer: CouplingDynamicsVisualizer instance
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
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    eigenvalues = visualizer.get_current_eigenvalues()
    if len(eigenvalues) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    # Plot unit circle (stability boundary)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "r--", alpha=0.5, label="Unit Circle")

    # Plot eigenvalues
    ax.scatter(np.real(eigenvalues), np.imag(eigenvalues),
               s=100, c="blue", marker="x", linewidths=2, label="Eigenvalues")

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)

    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title("Eigenvalue Spectrum")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return ax


def create_coupling_dashboard(
    visualizer: CouplingDynamicsVisualizer,
    figsize: tuple[int, int] = (16, 12),
):
    """
    Create comprehensive coupling dynamics dashboard.

    Args:
        visualizer: CouplingDynamicsVisualizer instance
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

    # Coupling matrix heatmap (large)
    ax_coupling = fig.add_subplot(gs[0, :2])
    plot_coupling_heatmap(visualizer, ax=ax_coupling)

    # Eligibility trace
    ax_elig = fig.add_subplot(gs[0, 2])
    plot_eligibility_heatmap(visualizer, ax=ax_elig)

    # Spectral radius timeline
    ax_spectral = fig.add_subplot(gs[1, :2])
    plot_spectral_radius_timeline(visualizer, ax=ax_spectral)

    # Eigenvalue spectrum
    ax_eigen = fig.add_subplot(gs[1, 2])
    plot_eigenvalue_spectrum(visualizer, ax=ax_eigen)

    # E/I balance timeline
    ax_ei = fig.add_subplot(gs[2, :2])
    plot_ei_balance_timeline(visualizer, ax=ax_ei)

    # Metrics text box
    ax_metrics = fig.add_subplot(gs[2, 2])
    ax_metrics.axis("off")

    metrics = visualizer.get_stability_metrics()
    ei_comp = visualizer.get_ei_components()
    alerts = visualizer.get_alerts()

    if metrics:
        text_lines = [
            "Stability Metrics",
            "-" * 25,
            f"Spectral Radius: {metrics.get('current_spectral_radius', 0):.4f}",
            f"Stability Margin: {metrics.get('stability_margin', 0):.4f}",
            f"E/I Balance: {metrics.get('current_ei_balance', 0):.4f}",
            f"Bounds Violations: {metrics.get('bounds_violations', 0)}",
            "",
            "E/I Components",
            "-" * 25,
            f"GABA→Glu: {ei_comp.get('gaba_to_glu', 0):.3f}",
            f"Glu→GABA: {ei_comp.get('glu_to_gaba', 0):.3f}",
            f"E/I Ratio: {ei_comp.get('ei_ratio', 0):.2f}",
        ]

        if alerts:
            text_lines.extend(["", "⚠️ ALERTS", "-" * 25])
            text_lines.extend(alerts[:3])  # Limit to 3

        text = "\n".join(text_lines)
        ax_metrics.text(0.05, 0.95, text, fontsize=9, family="monospace",
                        verticalalignment="top", transform=ax_metrics.transAxes)

    plt.tight_layout()
    return fig


__all__ = [
    "CouplingDynamicsVisualizer",
    "CouplingSnapshot",
    "LearningEvent",
    "plot_coupling_heatmap",
    "plot_eligibility_heatmap",
    "plot_spectral_radius_timeline",
    "plot_ei_balance_timeline",
    "plot_eigenvalue_spectrum",
    "create_coupling_dashboard",
]
