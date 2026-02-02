"""
Kappa Gradient Visualization for T4DM.

Visualizes kappa (consolidation level) distribution and flow:
- Distribution histogram with color-coded bands
- Stacked area chart of item counts per band over time
- LSM level distribution bar chart

Kappa bands (from CLAUDE.md):
- 0.0-0.15: Episodic (just encoded)
- 0.15-0.4: Replayed (NREM strengthened)
- 0.4-0.85: Transitional (being abstracted)
- 0.85-1.0: Semantic (fully consolidated)

Author: Claude Opus 4.5
Date: 2026-02-02
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Kappa band definitions
KAPPA_BANDS = {
    "episodic": (0.0, 0.15),
    "replayed": (0.15, 0.4),
    "transitional": (0.4, 0.85),
    "semantic": (0.85, 1.0),
}

KAPPA_COLORS = {
    "episodic": "royalblue",
    "replayed": "seagreen",
    "transitional": "orange",
    "semantic": "firebrick",
}


@dataclass
class KappaSnapshot:
    """Snapshot of kappa distribution state."""

    kappa_values: list[float]
    item_count: int
    timestamp: float
    level_counts: dict[int, int] = field(default_factory=dict)


def _classify_kappa(value: float) -> str:
    """Classify a kappa value into its band."""
    for band_name, (lo, hi) in KAPPA_BANDS.items():
        if lo <= value < hi or (band_name == "semantic" and value == 1.0):
            return band_name
    return "episodic"


class KappaGradientVisualizer:
    """
    Visualizes kappa consolidation gradient distribution and flow.

    Tracks how items move through kappa bands over time,
    providing insight into the consolidation pipeline health.
    """

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self._snapshots: list[KappaSnapshot] = []
        logger.info("KappaGradientVisualizer initialized")

    def record_snapshot(self, snapshot: KappaSnapshot) -> None:
        """Append a snapshot to history, maintaining window size."""
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

    def _band_counts(self, kappa_values: list[float]) -> dict[str, int]:
        """Count items per kappa band."""
        counts = {name: 0 for name in KAPPA_BANDS}
        for v in kappa_values:
            counts[_classify_kappa(v)] += 1
        return counts

    def plot_kappa_distribution(self, ax: Any = None) -> Any:
        """
        Histogram of kappa values with color bands.

        Blue = episodic, green = replayed, orange = transitional, red = semantic.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        # Use latest snapshot
        values = self._snapshots[-1].kappa_values
        if not values:
            ax.text(0.5, 0.5, "No kappa values", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        bins = np.linspace(0, 1, 41)
        n, bins_out, patches = ax.hist(values, bins=bins, edgecolor="black", alpha=0.8)

        # Color patches by band
        for patch, b in zip(patches, bins_out[:-1]):
            band = _classify_kappa(float(b))
            patch.set_facecolor(KAPPA_COLORS[band])

        # Add band boundaries
        for boundary in [0.15, 0.4, 0.85]:
            ax.axvline(x=boundary, color="black", linestyle="--", alpha=0.5)

        ax.set_xlabel("Kappa (consolidation level)")
        ax.set_ylabel("Count")
        ax.set_title("Kappa Distribution")
        ax.grid(True, alpha=0.3)

        return ax

    def plot_kappa_flow(self, ax: Any = None) -> Any:
        """Stacked area chart of item counts per kappa band over time."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        times = [s.timestamp for s in self._snapshots]
        band_series = {name: [] for name in KAPPA_BANDS}

        for s in self._snapshots:
            counts = self._band_counts(s.kappa_values)
            for name in KAPPA_BANDS:
                band_series[name].append(counts[name])

        arrays = [np.array(band_series[name]) for name in KAPPA_BANDS]
        colors = [KAPPA_COLORS[name] for name in KAPPA_BANDS]
        labels = list(KAPPA_BANDS.keys())

        ax.stackplot(times, *arrays, labels=labels, colors=colors, alpha=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel("Item Count")
        ax.set_title("Kappa Flow (items per consolidation band)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        return ax

    def plot_level_distribution(self, ax: Any = None) -> Any:
        """Bar chart of LSM level counts from latest snapshot."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        if not self._snapshots:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        level_counts = self._snapshots[-1].level_counts
        if not level_counts:
            ax.text(0.5, 0.5, "No level data", ha="center", va="center",
                    transform=ax.transAxes)
            return ax

        levels = sorted(level_counts.keys())
        counts = [level_counts[l] for l in levels]

        ax.bar([str(l) for l in levels], counts, color="steelblue", edgecolor="black")
        ax.set_xlabel("LSM Level")
        ax.set_ylabel("Segment Count")
        ax.set_title("LSM Level Distribution")
        ax.grid(True, alpha=0.3, axis="y")

        return ax

    def export_data(self) -> dict:
        """Export visualization data for external analysis."""
        snapshots_data = []
        for s in self._snapshots:
            band_counts = self._band_counts(s.kappa_values)
            snapshots_data.append({
                "timestamp": s.timestamp,
                "item_count": s.item_count,
                "band_counts": band_counts,
                "level_counts": s.level_counts,
                "kappa_mean": float(np.mean(s.kappa_values)) if s.kappa_values else 0.0,
                "kappa_std": float(np.std(s.kappa_values)) if s.kappa_values else 0.0,
            })
        return {
            "snapshots": snapshots_data,
            "bands": {name: list(bounds) for name, bounds in KAPPA_BANDS.items()},
            "meta": {
                "window_size": self.window_size,
                "total_snapshots": len(self._snapshots),
            },
        }


__all__ = [
    "KappaSnapshot",
    "KappaGradientVisualizer",
    "KAPPA_BANDS",
    "KAPPA_COLORS",
]
