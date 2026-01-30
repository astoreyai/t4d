"""
Forward-Forward Algorithm Visualization for World Weaver NCA.

Hinton-inspired visualization of FF learning dynamics:
- Layer-wise goodness bars (G(h) = sum(h_i^2))
- Positive vs negative phase comparison
- Threshold adaptation over time
- Sleep/wake phase transitions
- Neuromodulator coupling effects

This is CRITICAL for understanding:
- How layers differentiate positive/negative data
- Whether thresholds are calibrated correctly
- Learning progression across phases
- Neuromodulator effects on FF learning
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ww.nca.forward_forward import ForwardForwardNetwork

logger = logging.getLogger(__name__)


@dataclass
class FFSnapshot:
    """Snapshot of Forward-Forward layer state."""

    timestamp: datetime
    layer_idx: int
    goodness: float
    threshold: float
    is_positive: bool
    confidence: float  # |goodness - threshold|
    activation_norm: float  # ||h||
    phase: str  # "positive", "negative", "inference"
    learning_rate: float
    da_modulation: float  # Dopamine effect on LR
    ne_modulation: float  # NE effect on threshold


@dataclass
class FFTrainingEvent:
    """Record of an FF training step."""

    timestamp: datetime
    phase: str
    layer_goodnesses: list[float]
    layer_thresholds: list[float]
    mean_goodness: float
    mean_threshold: float
    accuracy: float  # Classification accuracy
    positive_margin: float  # Mean(goodness - theta) for positive
    negative_margin: float  # Mean(theta - goodness) for negative


class ForwardForwardVisualizer:
    """
    Visualizes Forward-Forward algorithm dynamics.

    Tracks layer goodness, thresholds, and learning events
    across positive and negative phases.
    """

    def __init__(
        self,
        ff_network: ForwardForwardNetwork | None = None,
        window_size: int = 1000,
        alert_low_margin: float = 0.1,
        alert_threshold_drift: float = 0.5,
    ):
        """
        Initialize FF visualizer.

        Args:
            ff_network: ForwardForwardNetwork instance to monitor
            window_size: Number of snapshots to retain
            alert_low_margin: Alert if positive/negative margin too low
            alert_threshold_drift: Alert if threshold changes too much
        """
        self.ff_network = ff_network
        self.window_size = window_size
        self.alert_low_margin = alert_low_margin
        self.alert_threshold_drift = alert_threshold_drift

        # History tracking
        self._snapshots: list[FFSnapshot] = []
        self._training_events: list[FFTrainingEvent] = []
        self._goodness_history: dict[int, list[float]] = {}  # layer_idx -> history
        self._threshold_history: dict[int, list[float]] = {}

        # Alerts
        self._active_alerts: list[str] = []

        logger.info("ForwardForwardVisualizer initialized")

    def record_layer_state(
        self,
        layer_idx: int,
        goodness: float,
        threshold: float,
        activation: np.ndarray,
        phase: str = "inference",
        learning_rate: float = 0.0,
        da_level: float = 0.5,
        ne_level: float = 0.5,
    ) -> FFSnapshot:
        """
        Record state of a single FF layer.

        Args:
            layer_idx: Index of the layer
            goodness: Current goodness G(h) = sum(h_i^2)
            threshold: Current threshold theta
            activation: Layer activations h
            phase: "positive", "negative", or "inference"
            learning_rate: Current learning rate
            da_level: Dopamine level (modulates LR)
            ne_level: Norepinephrine level (modulates threshold)

        Returns:
            FFSnapshot with all metrics
        """
        now = datetime.now()

        is_positive = goodness > threshold
        confidence = abs(goodness - threshold)
        activation_norm = float(np.linalg.norm(activation))

        # DA modulates learning rate (higher DA = faster learning)
        da_modulation = 1.0 + 0.5 * (da_level - 0.5)  # 0.75 to 1.25

        # NE modulates threshold (higher NE = higher threshold)
        ne_modulation = 1.0 + 0.2 * (ne_level - 0.5)  # 0.9 to 1.1

        snapshot = FFSnapshot(
            timestamp=now,
            layer_idx=layer_idx,
            goodness=goodness,
            threshold=threshold,
            is_positive=is_positive,
            confidence=confidence,
            activation_norm=activation_norm,
            phase=phase,
            learning_rate=learning_rate,
            da_modulation=da_modulation,
            ne_modulation=ne_modulation,
        )

        # Store snapshot
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

        # Store layer-specific history
        if layer_idx not in self._goodness_history:
            self._goodness_history[layer_idx] = []
            self._threshold_history[layer_idx] = []

        self._goodness_history[layer_idx].append(goodness)
        self._threshold_history[layer_idx].append(threshold)

        if len(self._goodness_history[layer_idx]) > self.window_size:
            self._goodness_history[layer_idx].pop(0)
            self._threshold_history[layer_idx].pop(0)

        self._check_alerts(snapshot)

        return snapshot

    def record_training_step(
        self,
        phase: str,
        layer_goodnesses: list[float],
        layer_thresholds: list[float],
        accuracy: float = 0.0,
    ) -> FFTrainingEvent:
        """
        Record a complete training step across all layers.

        Args:
            phase: "positive" or "negative"
            layer_goodnesses: Goodness values for each layer
            layer_thresholds: Threshold values for each layer
            accuracy: Classification accuracy

        Returns:
            FFTrainingEvent record
        """
        mean_goodness = float(np.mean(layer_goodnesses))
        mean_threshold = float(np.mean(layer_thresholds))

        # Compute margins
        if phase == "positive":
            positive_margin = mean_goodness - mean_threshold
            negative_margin = 0.0
        else:
            positive_margin = 0.0
            negative_margin = mean_threshold - mean_goodness

        event = FFTrainingEvent(
            timestamp=datetime.now(),
            phase=phase,
            layer_goodnesses=layer_goodnesses,
            layer_thresholds=layer_thresholds,
            mean_goodness=mean_goodness,
            mean_threshold=mean_threshold,
            accuracy=accuracy,
            positive_margin=positive_margin,
            negative_margin=negative_margin,
        )

        self._training_events.append(event)
        if len(self._training_events) > self.window_size:
            self._training_events.pop(0)

        return event

    def _check_alerts(self, snapshot: FFSnapshot) -> None:
        """Check for alert conditions."""
        self._active_alerts.clear()

        if snapshot.confidence < self.alert_low_margin:
            self._active_alerts.append(
                f"LOW MARGIN: Layer {snapshot.layer_idx} confidence "
                f"{snapshot.confidence:.3f} < {self.alert_low_margin}"
            )

        # Check threshold drift
        if snapshot.layer_idx in self._threshold_history:
            history = self._threshold_history[snapshot.layer_idx]
            if len(history) > 10:
                drift = abs(history[-1] - history[-10]) / max(abs(history[-10]), 1e-6)
                if drift > self.alert_threshold_drift:
                    self._active_alerts.append(
                        f"THRESHOLD DRIFT: Layer {snapshot.layer_idx} "
                        f"drifted {drift:.1%} in 10 steps"
                    )

    def get_alerts(self) -> list[str]:
        """Get current active alerts."""
        return self._active_alerts.copy()

    # -------------------------------------------------------------------------
    # Goodness Analysis
    # -------------------------------------------------------------------------

    def get_layer_goodness_trace(
        self, layer_idx: int
    ) -> tuple[list[datetime], list[float]]:
        """Get time series of goodness for a specific layer."""
        layer_snapshots = [s for s in self._snapshots if s.layer_idx == layer_idx]
        timestamps = [s.timestamp for s in layer_snapshots]
        goodnesses = [s.goodness for s in layer_snapshots]
        return timestamps, goodnesses

    def get_all_layer_goodnesses(self) -> dict[int, list[float]]:
        """Get goodness history for all layers."""
        return {k: v.copy() for k, v in self._goodness_history.items()}

    def get_current_goodnesses(self) -> dict[int, float]:
        """Get most recent goodness for each layer."""
        result = {}
        for layer_idx, history in self._goodness_history.items():
            if history:
                result[layer_idx] = history[-1]
        return result

    def get_goodness_statistics(self) -> dict[int, dict[str, float]]:
        """Get statistics of goodness for each layer."""
        stats = {}
        for layer_idx, history in self._goodness_history.items():
            if history:
                stats[layer_idx] = {
                    "mean": float(np.mean(history)),
                    "std": float(np.std(history)),
                    "min": float(np.min(history)),
                    "max": float(np.max(history)),
                    "current": history[-1],
                }
        return stats

    # -------------------------------------------------------------------------
    # Threshold Analysis
    # -------------------------------------------------------------------------

    def get_layer_threshold_trace(
        self, layer_idx: int
    ) -> tuple[list[datetime], list[float]]:
        """Get time series of threshold for a specific layer."""
        layer_snapshots = [s for s in self._snapshots if s.layer_idx == layer_idx]
        timestamps = [s.timestamp for s in layer_snapshots]
        thresholds = [s.threshold for s in layer_snapshots]
        return timestamps, thresholds

    def get_current_thresholds(self) -> dict[int, float]:
        """Get most recent threshold for each layer."""
        result = {}
        for layer_idx, history in self._threshold_history.items():
            if history:
                result[layer_idx] = history[-1]
        return result

    # -------------------------------------------------------------------------
    # Phase Analysis
    # -------------------------------------------------------------------------

    def get_phase_breakdown(self) -> dict[str, int]:
        """Get count of snapshots by phase."""
        breakdown = {"positive": 0, "negative": 0, "inference": 0}
        for s in self._snapshots:
            breakdown[s.phase] = breakdown.get(s.phase, 0) + 1
        return breakdown

    def get_accuracy_by_phase(self) -> dict[str, float]:
        """Get classification accuracy for each phase."""
        result = {}
        for phase in ["positive", "negative"]:
            phase_events = [e for e in self._training_events if e.phase == phase]
            if phase_events:
                result[phase] = float(np.mean([e.accuracy for e in phase_events]))
        return result

    def get_margin_history(self) -> dict[str, list[float]]:
        """Get positive and negative margin history."""
        pos_margins = [e.positive_margin for e in self._training_events
                       if e.phase == "positive"]
        neg_margins = [e.negative_margin for e in self._training_events
                       if e.phase == "negative"]
        return {"positive": pos_margins, "negative": neg_margins}

    # -------------------------------------------------------------------------
    # Neuromodulator Effects
    # -------------------------------------------------------------------------

    def get_neuromod_effects(self) -> dict[str, dict[str, float]]:
        """Analyze neuromodulator effects on FF learning."""
        if not self._snapshots:
            return {}

        recent = self._snapshots[-min(100, len(self._snapshots)):]

        return {
            "dopamine": {
                "mean_modulation": float(np.mean([s.da_modulation for s in recent])),
                "effect_on_lr": "Increases learning rate with surprise",
            },
            "norepinephrine": {
                "mean_modulation": float(np.mean([s.ne_modulation for s in recent])),
                "effect_on_threshold": "Raises threshold with arousal",
            },
        }

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------

    def export_data(self) -> dict:
        """Export visualization data for external rendering."""
        return {
            "layer_goodnesses": self.get_all_layer_goodnesses(),
            "layer_thresholds": {
                k: v.copy() for k, v in self._threshold_history.items()
            },
            "current_goodnesses": self.get_current_goodnesses(),
            "current_thresholds": self.get_current_thresholds(),
            "goodness_statistics": self.get_goodness_statistics(),
            "phase_breakdown": self.get_phase_breakdown(),
            "accuracy_by_phase": self.get_accuracy_by_phase(),
            "margin_history": self.get_margin_history(),
            "neuromod_effects": self.get_neuromod_effects(),
            "alerts": self.get_alerts(),
        }

    def clear_history(self) -> None:
        """Clear all history."""
        self._snapshots.clear()
        self._training_events.clear()
        self._goodness_history.clear()
        self._threshold_history.clear()
        self._active_alerts.clear()


# =============================================================================
# Standalone Plot Functions
# =============================================================================


def plot_goodness_bars(
    visualizer: ForwardForwardVisualizer,
    ax=None,
    show_threshold: bool = True,
):
    """
    Plot current goodness for each layer as bar chart.

    Args:
        visualizer: ForwardForwardVisualizer instance
        ax: Matplotlib axes
        show_threshold: Overlay threshold line

    Returns:
        Matplotlib axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return None

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    goodnesses = visualizer.get_current_goodnesses()
    thresholds = visualizer.get_current_thresholds()

    if not goodnesses:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    layers = sorted(goodnesses.keys())
    x = np.arange(len(layers))
    g_values = [goodnesses[l] for l in layers]
    t_values = [thresholds.get(l, 0) for l in layers]

    # Color by positive/negative classification
    colors = ["green" if g > t else "red" for g, t in zip(g_values, t_values)]

    ax.bar(x, g_values, color=colors, alpha=0.7, label="Goodness G(h)")

    if show_threshold:
        ax.plot(x, t_values, "k--", linewidth=2, marker="s",
                markersize=8, label="Threshold θ")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Goodness")
    ax.set_title("Layer Goodness (green=positive, red=negative)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_goodness_timeline(
    visualizer: ForwardForwardVisualizer,
    layer_idx: int = 0,
    ax=None,
):
    """
    Plot goodness over time for a specific layer.

    Args:
        visualizer: ForwardForwardVisualizer instance
        layer_idx: Layer to plot
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

    timestamps_g, goodnesses = visualizer.get_layer_goodness_trace(layer_idx)
    timestamps_t, thresholds = visualizer.get_layer_threshold_trace(layer_idx)

    if not timestamps_g:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    t0 = timestamps_g[0]
    t_seconds_g = [(t - t0).total_seconds() for t in timestamps_g]
    t_seconds_t = [(t - t0).total_seconds() for t in timestamps_t]

    ax.plot(t_seconds_g, goodnesses, "b-", linewidth=1.5, label="Goodness G(h)")
    ax.plot(t_seconds_t, thresholds, "r--", linewidth=1.5, label="Threshold θ")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.set_title(f"Layer {layer_idx} Goodness vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Shade positive/negative regions
    for i in range(len(goodnesses)):
        if i < len(thresholds):
            if goodnesses[i] > thresholds[i]:
                ax.axvspan(t_seconds_g[i], t_seconds_g[min(i+1, len(t_seconds_g)-1)],
                           alpha=0.1, color="green")
            else:
                ax.axvspan(t_seconds_g[i], t_seconds_g[min(i+1, len(t_seconds_g)-1)],
                           alpha=0.1, color="red")

    return ax


def plot_margin_evolution(
    visualizer: ForwardForwardVisualizer,
    ax=None,
):
    """
    Plot positive and negative margin evolution.

    Args:
        visualizer: ForwardForwardVisualizer instance
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

    margins = visualizer.get_margin_history()

    if not margins["positive"] and not margins["negative"]:
        ax.text(0.5, 0.5, "No training data", ha="center", va="center")
        return ax

    if margins["positive"]:
        ax.plot(margins["positive"], "g-", linewidth=1.5,
                label="Positive margin (G > θ)")
    if margins["negative"]:
        ax.plot(margins["negative"], "r-", linewidth=1.5,
                label="Negative margin (θ > G)")

    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Margin")
    ax.set_title("Learning Margin Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_phase_comparison(
    visualizer: ForwardForwardVisualizer,
    ax=None,
):
    """
    Plot goodness distribution comparison between phases.

    Args:
        visualizer: ForwardForwardVisualizer instance
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
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Get goodnesses by phase
    pos_goodnesses = [s.goodness for s in visualizer._snapshots
                      if s.phase == "positive"]
    neg_goodnesses = [s.goodness for s in visualizer._snapshots
                      if s.phase == "negative"]

    if not pos_goodnesses and not neg_goodnesses:
        ax.text(0.5, 0.5, "No phase data", ha="center", va="center")
        return ax

    positions = []
    data = []
    labels = []

    if pos_goodnesses:
        positions.append(1)
        data.append(pos_goodnesses)
        labels.append("Positive\n(real)")
    if neg_goodnesses:
        positions.append(2)
        data.append(neg_goodnesses)
        labels.append("Negative\n(fake)")

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)

    colors = ["green", "red"][:len(data)]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Goodness")
    ax.set_title("Goodness Distribution by Phase")
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def create_ff_dashboard(
    visualizer: ForwardForwardVisualizer,
    figsize: tuple[int, int] = (16, 10),
):
    """
    Create comprehensive FF algorithm dashboard.

    Args:
        visualizer: ForwardForwardVisualizer instance
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

    # Goodness bars (current state)
    ax_bars = fig.add_subplot(gs[0, 0])
    plot_goodness_bars(visualizer, ax=ax_bars)

    # Phase comparison
    ax_phase = fig.add_subplot(gs[0, 1])
    plot_phase_comparison(visualizer, ax=ax_phase)

    # Margin evolution
    ax_margin = fig.add_subplot(gs[0, 2])
    plot_margin_evolution(visualizer, ax=ax_margin)

    # Goodness timeline (layer 0)
    ax_timeline = fig.add_subplot(gs[1, :2])
    plot_goodness_timeline(visualizer, layer_idx=0, ax=ax_timeline)

    # Statistics text box
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.axis("off")

    stats = visualizer.get_goodness_statistics()
    phase_breakdown = visualizer.get_phase_breakdown()
    accuracy = visualizer.get_accuracy_by_phase()
    neuromod = visualizer.get_neuromod_effects()
    alerts = visualizer.get_alerts()

    text_lines = [
        "Forward-Forward Statistics",
        "=" * 30,
    ]

    for layer_idx, layer_stats in stats.items():
        text_lines.append(f"Layer {layer_idx}: G={layer_stats['mean']:.3f} "
                          f"(±{layer_stats['std']:.3f})")

    text_lines.extend([
        "",
        "Phase Breakdown:",
        f"  Positive: {phase_breakdown.get('positive', 0)}",
        f"  Negative: {phase_breakdown.get('negative', 0)}",
        f"  Inference: {phase_breakdown.get('inference', 0)}",
    ])

    if accuracy:
        text_lines.extend([
            "",
            "Classification Accuracy:",
        ])
        for phase, acc in accuracy.items():
            text_lines.append(f"  {phase}: {acc:.1%}")

    if alerts:
        text_lines.extend(["", "ALERTS:", "-" * 20])
        text_lines.extend(alerts[:3])

    text = "\n".join(text_lines)
    ax_stats.text(0.05, 0.95, text, fontsize=9, family="monospace",
                  verticalalignment="top", transform=ax_stats.transAxes)

    plt.tight_layout()
    return fig


__all__ = [
    "ForwardForwardVisualizer",
    "FFSnapshot",
    "FFTrainingEvent",
    "plot_goodness_bars",
    "plot_goodness_timeline",
    "plot_margin_evolution",
    "plot_phase_comparison",
    "create_ff_dashboard",
]
