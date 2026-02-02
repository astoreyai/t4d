"""
Oscillator phase bias injection visualization for spiking cortical blocks.

Visualizes theta/gamma/delta phase states and their projected bias currents,
based on the OscillatorBias module from spiking/oscillator_bias.py.

Neural oscillations:
- theta (4-8 Hz): Gates episodic encoding
- gamma (30-100 Hz): Binding and attention
- delta (0.5-4 Hz): Deep sleep consolidation

Author: Claude Opus 4.5
Date: 2026-02-02
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

OSC_NAMES = ["theta", "gamma", "delta"]
OSC_COLORS = {"theta": "#e67e22", "gamma": "#8e44ad", "delta": "#2c3e50"}


@dataclass
class OscillatorSnapshot:
    """Snapshot of oscillator phase and bias injection state."""

    theta_phase: float  # [0, 2*pi]
    gamma_phase: float
    delta_phase: float
    bias_values: np.ndarray  # projected bias current vector
    timestamp: float


class OscillatorInjectionVisualizer:
    """
    Visualizes theta/gamma/delta phase bias injection into spiking blocks.

    Provides polar phase plots, bias timelines, and phase coupling analysis.
    """

    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self._snapshots: list[OscillatorSnapshot] = []

    def record_snapshot(self, snapshot: OscillatorSnapshot) -> None:
        """Record an oscillator snapshot."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.max_history:
            self._snapshots.pop(0)

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def plot_phase_polar(self, ax=None) -> Any:
        """
        Polar plot showing current theta/gamma/delta phases.

        Each oscillator is a vector from origin at its current phase angle.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

        if not self._snapshots:
            ax.set_title("Oscillator Phases (no data)")
            return ax

        s = self._snapshots[-1]
        phases = [s.theta_phase, s.gamma_phase, s.delta_phase]
        radii = [1.0, 0.8, 0.6]  # visual separation

        for name, phase, r in zip(OSC_NAMES, phases, radii):
            color = OSC_COLORS[name]
            ax.annotate(
                "",
                xy=(phase, r),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.5),
            )
            ax.plot(phase, r, "o", color=color, markersize=8, label=name)

        ax.set_title("Oscillator Phases")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

        return ax

    def plot_bias_timeline(self, ax=None) -> Any:
        """Line chart of mean bias magnitude injected over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return ax

        timestamps = [s.timestamp for s in self._snapshots]
        mean_bias = [float(np.mean(np.abs(s.bias_values))) for s in self._snapshots]
        max_bias = [float(np.max(np.abs(s.bias_values))) for s in self._snapshots]

        ax.plot(timestamps, mean_bias, label="Mean |bias|", color="steelblue", linewidth=1.5)
        ax.fill_between(timestamps, 0, mean_bias, alpha=0.2, color="steelblue")
        ax.plot(timestamps, max_bias, label="Max |bias|", color="coral", linewidth=1, linestyle="--")

        ax.set_xlabel("Time")
        ax.set_ylabel("Bias Magnitude")
        ax.set_title("Oscillator Bias Injection Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_phase_coupling(self, ax=None) -> Any:
        """
        Theta-gamma phase relationship scatter plot.

        Shows theta-gamma coupling (cross-frequency coupling), a key
        biological phenomenon for memory encoding.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return ax

        theta_phases = [s.theta_phase for s in self._snapshots]
        gamma_phases = [s.gamma_phase for s in self._snapshots]
        timestamps = [s.timestamp for s in self._snapshots]

        # Color by time (older = lighter)
        t_arr = np.array(timestamps)
        if t_arr.max() > t_arr.min():
            t_norm = (t_arr - t_arr.min()) / (t_arr.max() - t_arr.min())
        else:
            t_norm = np.ones_like(t_arr)

        sc = ax.scatter(theta_phases, gamma_phases, c=t_norm, cmap="viridis",
                        s=15, alpha=0.7, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Time (normalized)")

        ax.set_xlabel("Theta Phase (rad)")
        ax.set_ylabel("Gamma Phase (rad)")
        ax.set_title("Theta-Gamma Phase Coupling")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 2 * np.pi)
        ax.grid(True, alpha=0.3)

        return ax

    def export_data(self) -> dict:
        """Export all recorded data as JSON-serializable dict."""
        return {
            "snapshot_count": len(self._snapshots),
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "theta_phase": s.theta_phase,
                    "gamma_phase": s.gamma_phase,
                    "delta_phase": s.delta_phase,
                    "bias_mean": float(np.mean(np.abs(s.bias_values))),
                    "bias_max": float(np.max(np.abs(s.bias_values))),
                    "bias_dim": len(s.bias_values),
                }
                for s in self._snapshots
            ],
            "oscillator_names": OSC_NAMES,
        }


__all__ = [
    "OscillatorSnapshot",
    "OscillatorInjectionVisualizer",
    "OSC_NAMES",
    "OSC_COLORS",
]
