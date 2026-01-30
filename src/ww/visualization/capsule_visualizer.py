"""
Capsule Network Visualization for World Weaver NCA.

Hinton-inspired visualization of capsule dynamics:
- Routing coefficient matrix (c_ij)
- Pose vector directions (3D glyph representation)
- Entity probability bar chart (capsule lengths)
- Part-whole hierarchy tree
- NT modulation effects on routing

This is CRITICAL for understanding:
- How lower capsules route to higher capsules
- Pose parameter consistency across hierarchy
- Entity detection confidence
- Neuromodulator effects on routing dynamics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ww.nca.capsules import CapsuleNetwork

logger = logging.getLogger(__name__)


@dataclass
class CapsuleSnapshot:
    """Snapshot of capsule network state."""

    timestamp: datetime
    layer_idx: int
    n_capsules: int
    capsule_lengths: np.ndarray  # Entity probabilities
    pose_vectors: np.ndarray  # Capsule directions
    routing_coefficients: np.ndarray | None = None  # c_ij matrix
    routing_iterations: int = 3
    nt_modulation: dict = field(default_factory=dict)


@dataclass
class RoutingEvent:
    """Record of a routing iteration."""

    timestamp: datetime
    iteration: int
    source_layer: int
    target_layer: int
    agreement_scores: np.ndarray  # How well predictions match outputs
    coefficient_entropy: float  # Spread of routing coefficients
    dominant_routes: list[tuple[int, int, float]]  # Top (src, dst, coeff) tuples


class CapsuleVisualizer:
    """
    Visualizes capsule network dynamics.

    Tracks routing coefficients, pose vectors, entity probabilities,
    and neuromodulator effects on capsule processing.
    """

    def __init__(
        self,
        capsule_network: CapsuleNetwork | None = None,
        window_size: int = 500,
        alert_low_agreement: float = 0.3,
        alert_high_entropy: float = 0.9,
    ):
        """
        Initialize capsule visualizer.

        Args:
            capsule_network: CapsuleNetwork instance to monitor
            window_size: Number of snapshots to retain
            alert_low_agreement: Alert if routing agreement too low
            alert_high_entropy: Alert if routing too diffuse
        """
        self.capsule_network = capsule_network
        self.window_size = window_size
        self.alert_low_agreement = alert_low_agreement
        self.alert_high_entropy = alert_high_entropy

        # History tracking
        self._snapshots: list[CapsuleSnapshot] = []
        self._routing_events: list[RoutingEvent] = []
        self._length_history: dict[int, list[np.ndarray]] = {}  # layer -> history
        self._pose_history: dict[int, list[np.ndarray]] = {}

        # Alerts
        self._active_alerts: list[str] = []

        logger.info("CapsuleVisualizer initialized")

    def record_layer_state(
        self,
        layer_idx: int,
        capsule_lengths: np.ndarray,
        pose_vectors: np.ndarray,
        routing_coefficients: np.ndarray | None = None,
        routing_iterations: int = 3,
        da_level: float = 0.5,
        ne_level: float = 0.5,
        ach_level: float = 0.5,
        sht_level: float = 0.5,
    ) -> CapsuleSnapshot:
        """
        Record state of a capsule layer.

        Args:
            layer_idx: Index of the layer
            capsule_lengths: Entity probabilities (||v||)
            pose_vectors: Capsule directions (normalized v)
            routing_coefficients: c_ij routing matrix
            routing_iterations: Number of routing iterations used
            da_level: Dopamine (routing temperature)
            ne_level: Norepinephrine (squash threshold)
            ach_level: Acetylcholine (encode/retrieve mode)
            sht_level: Serotonin (routing patience)

        Returns:
            CapsuleSnapshot with all metrics
        """
        now = datetime.now()

        nt_modulation = {
            "da_routing_temp": 1.0 / (1.0 + da_level),  # High DA = sharp routing
            "ne_squash_thresh": 0.5 + 0.5 * ne_level,  # High NE = higher threshold
            "ach_mode": "encoding" if ach_level > 0.5 else "retrieval",
            "sht_patience": int(routing_iterations * (1 + sht_level)),
        }

        snapshot = CapsuleSnapshot(
            timestamp=now,
            layer_idx=layer_idx,
            n_capsules=len(capsule_lengths),
            capsule_lengths=np.asarray(capsule_lengths).copy(),
            pose_vectors=np.asarray(pose_vectors).copy(),
            routing_coefficients=routing_coefficients.copy() if routing_coefficients is not None else None,
            routing_iterations=routing_iterations,
            nt_modulation=nt_modulation,
        )

        # Store snapshot
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

        # Store layer-specific history
        if layer_idx not in self._length_history:
            self._length_history[layer_idx] = []
            self._pose_history[layer_idx] = []

        self._length_history[layer_idx].append(capsule_lengths.copy())
        self._pose_history[layer_idx].append(pose_vectors.copy())

        if len(self._length_history[layer_idx]) > self.window_size:
            self._length_history[layer_idx].pop(0)
            self._pose_history[layer_idx].pop(0)

        self._check_alerts(snapshot)

        return snapshot

    def record_routing_iteration(
        self,
        iteration: int,
        source_layer: int,
        target_layer: int,
        predictions: np.ndarray,
        outputs: np.ndarray,
        coefficients: np.ndarray,
    ) -> RoutingEvent:
        """
        Record a single routing iteration.

        Args:
            iteration: Routing iteration number
            source_layer: Source layer index
            target_layer: Target layer index
            predictions: u_hat predictions from source capsules
            outputs: v outputs of target capsules
            coefficients: c_ij coupling coefficients

        Returns:
            RoutingEvent record
        """
        # Compute agreement scores (dot product of prediction and output)
        if len(predictions.shape) == 3:  # (n_source, n_target, cap_dim)
            # Compute agreement for each source-target pair
            agreement_scores = np.sum(predictions * outputs, axis=-1)
        else:
            agreement_scores = np.zeros((coefficients.shape[0], coefficients.shape[1]))

        # Compute entropy of routing coefficients
        # Higher entropy = more diffuse routing
        coeff_flat = coefficients.flatten()
        coeff_flat = coeff_flat[coeff_flat > 1e-10]
        if len(coeff_flat) > 0:
            entropy = -np.sum(coeff_flat * np.log(coeff_flat + 1e-10))
            max_entropy = np.log(len(coeff_flat))
            coefficient_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        else:
            coefficient_entropy = 0.0

        # Find dominant routes
        n_top = min(5, coefficients.size)
        flat_indices = np.argsort(coefficients.flatten())[-n_top:][::-1]
        dominant_routes = []
        for idx in flat_indices:
            src = idx // coefficients.shape[1]
            dst = idx % coefficients.shape[1]
            coeff = coefficients[src, dst]
            dominant_routes.append((int(src), int(dst), float(coeff)))

        event = RoutingEvent(
            timestamp=datetime.now(),
            iteration=iteration,
            source_layer=source_layer,
            target_layer=target_layer,
            agreement_scores=agreement_scores,
            coefficient_entropy=coefficient_entropy,
            dominant_routes=dominant_routes,
        )

        self._routing_events.append(event)
        if len(self._routing_events) > self.window_size:
            self._routing_events.pop(0)

        return event

    def _check_alerts(self, snapshot: CapsuleSnapshot) -> None:
        """Check for alert conditions."""
        self._active_alerts.clear()

        # Check for low entity probabilities (no confident detections)
        if snapshot.capsule_lengths.max() < 0.3:
            self._active_alerts.append(
                f"LOW CONFIDENCE: Layer {snapshot.layer_idx} max prob "
                f"{snapshot.capsule_lengths.max():.3f} < 0.3"
            )

        # Check routing entropy from recent events
        recent_events = [e for e in self._routing_events
                         if e.target_layer == snapshot.layer_idx][-10:]
        if recent_events:
            mean_entropy = np.mean([e.coefficient_entropy for e in recent_events])
            if mean_entropy > self.alert_high_entropy:
                self._active_alerts.append(
                    f"DIFFUSE ROUTING: Layer {snapshot.layer_idx} entropy "
                    f"{mean_entropy:.3f} > {self.alert_high_entropy}"
                )

    def get_alerts(self) -> list[str]:
        """Get current active alerts."""
        return self._active_alerts.copy()

    # -------------------------------------------------------------------------
    # Entity Probability Analysis
    # -------------------------------------------------------------------------

    def get_current_probabilities(self, layer_idx: int = -1) -> np.ndarray:
        """Get current entity probabilities (capsule lengths)."""
        layer_snapshots = [s for s in self._snapshots if s.layer_idx == layer_idx]
        if layer_snapshots:
            return layer_snapshots[-1].capsule_lengths.copy()
        return np.array([])

    def get_top_entities(self, layer_idx: int = -1, n: int = 5) -> list[tuple[int, float]]:
        """Get top N entities by probability."""
        probs = self.get_current_probabilities(layer_idx)
        if len(probs) == 0:
            return []
        top_indices = np.argsort(probs)[-n:][::-1]
        return [(int(i), float(probs[i])) for i in top_indices]

    def get_probability_history(self, layer_idx: int) -> list[np.ndarray]:
        """Get history of entity probabilities for a layer."""
        return [arr.copy() for arr in self._length_history.get(layer_idx, [])]

    # -------------------------------------------------------------------------
    # Pose Vector Analysis
    # -------------------------------------------------------------------------

    def get_current_poses(self, layer_idx: int = -1) -> np.ndarray:
        """Get current pose vectors."""
        layer_snapshots = [s for s in self._snapshots if s.layer_idx == layer_idx]
        if layer_snapshots:
            return layer_snapshots[-1].pose_vectors.copy()
        return np.array([])

    def get_pose_variance(self, layer_idx: int) -> float:
        """Get variance of pose vectors over time (stability measure)."""
        history = self._pose_history.get(layer_idx, [])
        if len(history) < 2:
            return 0.0
        # Compute mean variance across all pose dimensions
        stacked = np.stack(history, axis=0)
        return float(np.mean(np.var(stacked, axis=0)))

    def get_pose_alignment(self, layer_idx: int) -> float:
        """Get alignment between consecutive poses (consistency)."""
        history = self._pose_history.get(layer_idx, [])
        if len(history) < 2:
            return 1.0
        # Compute mean cosine similarity between consecutive poses
        alignments = []
        for i in range(1, len(history)):
            prev = history[i-1].flatten()
            curr = history[i].flatten()
            norm_prev = np.linalg.norm(prev)
            norm_curr = np.linalg.norm(curr)
            if norm_prev > 1e-10 and norm_curr > 1e-10:
                cos_sim = np.dot(prev, curr) / (norm_prev * norm_curr)
                alignments.append(cos_sim)
        return float(np.mean(alignments)) if alignments else 1.0

    # -------------------------------------------------------------------------
    # Routing Analysis
    # -------------------------------------------------------------------------

    def get_routing_matrix(self, layer_idx: int) -> np.ndarray | None:
        """Get current routing coefficient matrix for a layer."""
        layer_snapshots = [s for s in self._snapshots if s.layer_idx == layer_idx]
        if layer_snapshots and layer_snapshots[-1].routing_coefficients is not None:
            return layer_snapshots[-1].routing_coefficients.copy()
        return None

    def get_routing_entropy_history(self) -> list[float]:
        """Get history of routing entropy values."""
        return [e.coefficient_entropy for e in self._routing_events]

    def get_dominant_routes_summary(self) -> dict[tuple[int, int], float]:
        """Get summary of dominant routing patterns."""
        route_counts = {}
        for event in self._routing_events:
            for src, dst, coeff in event.dominant_routes:
                key = (src, dst)
                route_counts[key] = route_counts.get(key, 0) + coeff
        return route_counts

    # -------------------------------------------------------------------------
    # Neuromodulator Effects
    # -------------------------------------------------------------------------

    def get_nt_modulation_summary(self) -> dict[str, dict]:
        """Get summary of neuromodulator effects on capsules."""
        if not self._snapshots:
            return {}

        recent = self._snapshots[-min(100, len(self._snapshots)):]

        return {
            "da_routing_temp": {
                "mean": float(np.mean([s.nt_modulation.get("da_routing_temp", 1.0)
                                       for s in recent])),
                "effect": "High DA = sharper routing (lower temperature)",
            },
            "ne_squash_thresh": {
                "mean": float(np.mean([s.nt_modulation.get("ne_squash_thresh", 0.5)
                                       for s in recent])),
                "effect": "High NE = higher squash threshold (arousal)",
            },
            "ach_mode": {
                "encoding_ratio": sum(1 for s in recent
                                      if s.nt_modulation.get("ach_mode") == "encoding") / len(recent),
                "effect": "ACh > 0.5 = encoding mode, else retrieval",
            },
            "sht_patience": {
                "mean": float(np.mean([s.nt_modulation.get("sht_patience", 3)
                                       for s in recent])),
                "effect": "High 5-HT = more routing iterations (patience)",
            },
        }

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_data(self) -> dict:
        """Export visualization data for external rendering."""
        # Collect layer indices
        layer_indices = set(s.layer_idx for s in self._snapshots)

        layer_data = {}
        for idx in layer_indices:
            layer_data[idx] = {
                "probabilities": self.get_current_probabilities(idx).tolist(),
                "poses": self.get_current_poses(idx).tolist(),
                "routing": self.get_routing_matrix(idx).tolist() if self.get_routing_matrix(idx) is not None else None,
                "top_entities": self.get_top_entities(idx),
                "pose_variance": self.get_pose_variance(idx),
                "pose_alignment": self.get_pose_alignment(idx),
            }

        return {
            "layers": layer_data,
            "routing_entropy_history": self.get_routing_entropy_history(),
            "dominant_routes": {f"{k[0]}->{k[1]}": v
                                for k, v in self.get_dominant_routes_summary().items()},
            "nt_modulation": self.get_nt_modulation_summary(),
            "alerts": self.get_alerts(),
        }

    def clear_history(self) -> None:
        """Clear all history."""
        self._snapshots.clear()
        self._routing_events.clear()
        self._length_history.clear()
        self._pose_history.clear()
        self._active_alerts.clear()


# =============================================================================
# Standalone Plot Functions
# =============================================================================


def plot_entity_probabilities(
    visualizer: CapsuleVisualizer,
    layer_idx: int = -1,
    ax=None,
    n_show: int = 16,
):
    """
    Plot entity probabilities as bar chart.

    Args:
        visualizer: CapsuleVisualizer instance
        layer_idx: Layer to plot (-1 for last output layer)
        ax: Matplotlib axes
        n_show: Maximum number of entities to show

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

    probs = visualizer.get_current_probabilities(layer_idx)

    if len(probs) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    # Sort by probability and take top n_show
    indices = np.argsort(probs)[-n_show:][::-1]
    sorted_probs = probs[indices]

    x = np.arange(len(sorted_probs))
    colors = plt.cm.viridis(sorted_probs)

    ax.bar(x, sorted_probs, color=colors)
    ax.set_xlabel("Entity (sorted by probability)")
    ax.set_ylabel("Probability (||v||)")
    ax.set_title(f"Capsule Entity Probabilities (Layer {layer_idx})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i}" for i in indices], rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_routing_heatmap(
    visualizer: CapsuleVisualizer,
    layer_idx: int,
    ax=None,
    cmap: str = "Blues",
):
    """
    Plot routing coefficient matrix as heatmap.

    Args:
        visualizer: CapsuleVisualizer instance
        layer_idx: Target layer index
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
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    routing = visualizer.get_routing_matrix(layer_idx)

    if routing is None:
        ax.text(0.5, 0.5, "No routing data", ha="center", va="center")
        return ax

    im = ax.imshow(routing, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Routing Coefficient")

    ax.set_xlabel("Target Capsule")
    ax.set_ylabel("Source Capsule")
    ax.set_title(f"Routing Coefficients (to Layer {layer_idx})")

    return ax


def plot_pose_vectors(
    visualizer: CapsuleVisualizer,
    layer_idx: int = -1,
    ax=None,
    n_show: int = 8,
):
    """
    Plot pose vectors as arrow glyphs (2D projection).

    Args:
        visualizer: CapsuleVisualizer instance
        layer_idx: Layer to plot
        ax: Matplotlib axes
        n_show: Number of capsules to show

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

    poses = visualizer.get_current_poses(layer_idx)
    probs = visualizer.get_current_probabilities(layer_idx)

    if len(poses) == 0:
        ax.text(0.5, 0.5, "No pose data", ha="center", va="center")
        return ax

    # Take top entities by probability
    top_indices = np.argsort(probs)[-n_show:][::-1]

    # Use first 2 dimensions of pose for 2D plot
    for i, idx in enumerate(top_indices):
        pose = poses[idx]
        prob = probs[idx]

        # Position capsules in a grid
        row = i // 4
        col = i % 4
        x_base = col * 2.5
        y_base = -row * 2.5

        # Use first 2 pose dimensions as arrow direction
        if len(pose) >= 2:
            dx = pose[0] * prob * 2  # Scale by probability
            dy = pose[1] * prob * 2
        else:
            dx, dy = prob, 0

        # Draw arrow
        ax.arrow(x_base, y_base, dx, dy, head_width=0.15, head_length=0.1,
                 fc=plt.cm.viridis(prob), ec="black", linewidth=1.5)

        # Label
        ax.text(x_base, y_base - 0.3, f"E{idx}\n({prob:.2f})",
                ha="center", va="top", fontsize=8)

    ax.set_xlim(-1, 10)
    ax.set_ylim(-6, 2)
    ax.set_aspect("equal")
    ax.set_title(f"Pose Vectors (Layer {layer_idx})\nArrow = pose direction, Length = probability")
    ax.axis("off")

    return ax


def plot_routing_entropy(
    visualizer: CapsuleVisualizer,
    ax=None,
):
    """
    Plot routing entropy over time.

    Args:
        visualizer: CapsuleVisualizer instance
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

    entropy_history = visualizer.get_routing_entropy_history()

    if not entropy_history:
        ax.text(0.5, 0.5, "No routing data", ha="center", va="center")
        return ax

    ax.plot(entropy_history, "b-", linewidth=1.5)
    ax.axhline(y=visualizer.alert_high_entropy, color="r",
               linestyle="--", label=f"High entropy threshold ({visualizer.alert_high_entropy})")

    ax.set_xlabel("Routing Iteration")
    ax.set_ylabel("Coefficient Entropy")
    ax.set_title("Routing Entropy (lower = more focused routing)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def create_capsule_dashboard(
    visualizer: CapsuleVisualizer,
    layer_idx: int = -1,
    figsize: tuple[int, int] = (16, 10),
):
    """
    Create comprehensive capsule network dashboard.

    Args:
        visualizer: CapsuleVisualizer instance
        layer_idx: Layer to focus on
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
    gs = GridSpec(2, 3, figure=fig)

    # Entity probabilities
    ax_probs = fig.add_subplot(gs[0, 0])
    plot_entity_probabilities(visualizer, layer_idx=layer_idx, ax=ax_probs)

    # Routing heatmap
    ax_routing = fig.add_subplot(gs[0, 1])
    plot_routing_heatmap(visualizer, layer_idx=layer_idx, ax=ax_routing)

    # Pose vectors
    ax_poses = fig.add_subplot(gs[0, 2])
    plot_pose_vectors(visualizer, layer_idx=layer_idx, ax=ax_poses)

    # Routing entropy
    ax_entropy = fig.add_subplot(gs[1, :2])
    plot_routing_entropy(visualizer, ax=ax_entropy)

    # Statistics text box
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.axis("off")

    top_entities = visualizer.get_top_entities(layer_idx)
    pose_var = visualizer.get_pose_variance(layer_idx)
    pose_align = visualizer.get_pose_alignment(layer_idx)
    nt_mod = visualizer.get_nt_modulation_summary()
    alerts = visualizer.get_alerts()

    text_lines = [
        "Capsule Statistics",
        "=" * 30,
        "",
        "Top Entities:",
    ]
    for idx, prob in top_entities[:5]:
        text_lines.append(f"  E{idx}: {prob:.3f}")

    text_lines.extend([
        "",
        f"Pose Variance: {pose_var:.4f}",
        f"Pose Alignment: {pose_align:.3f}",
        "",
        "NT Modulation:",
    ])

    if nt_mod:
        text_lines.append(f"  DA temp: {nt_mod.get('da_routing_temp', {}).get('mean', 1.0):.3f}")
        text_lines.append(f"  NE thresh: {nt_mod.get('ne_squash_thresh', {}).get('mean', 0.5):.3f}")

    if alerts:
        text_lines.extend(["", "ALERTS:", "-" * 20])
        text_lines.extend(alerts[:3])

    text = "\n".join(text_lines)
    ax_stats.text(0.05, 0.95, text, fontsize=9, family="monospace",
                  verticalalignment="top", transform=ax_stats.transAxes)

    plt.tight_layout()
    return fig


__all__ = [
    "CapsuleVisualizer",
    "CapsuleSnapshot",
    "RoutingEvent",
    "plot_entity_probabilities",
    "plot_routing_heatmap",
    "plot_pose_vectors",
    "plot_routing_entropy",
    "create_capsule_dashboard",
]
