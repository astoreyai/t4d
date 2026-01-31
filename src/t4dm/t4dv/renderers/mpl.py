"""Matplotlib renderer adapters wrapping existing viz modules."""

from __future__ import annotations

from typing import Any

import numpy as np


class _MplRenderer:
    def __init__(self, view_id: str):
        self.view_id = view_id

    def _get_fig_ax(self, kwargs: dict) -> tuple:
        import matplotlib.pyplot as plt
        fig = kwargs.pop("fig", None)
        ax = kwargs.pop("ax", None)
        if fig is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 6)))
        elif ax is None:
            ax = fig.add_subplot(111)
        return fig, ax


class SpikeRasterRenderer(_MplRenderer):
    """Raster plot: time x block, color = firing_rate."""

    def __init__(self):
        super().__init__("spiking.raster")

    def render(self, data: dict[str, Any], **kwargs: Any) -> Any:
        fig, ax = self._get_fig_ax(kwargs)
        events = data.get("events", [])
        if not events:
            ax.text(0.5, 0.5, "No spike events", ha="center", va="center")
            ax.set_title("Spike Raster")
            return fig

        times = [e.get("timestamp_offset", i) for i, e in enumerate(events)]
        blocks = [e.get("block_index", 0) for e in events]
        rates = [e.get("firing_rate", 0.0) for e in events]

        sc = ax.scatter(times, blocks, c=rates, cmap="hot", s=4, vmin=0, vmax=1)
        fig.colorbar(sc, ax=ax, label="Firing Rate")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Block Index")
        ax.set_title("Spike Raster")
        return fig


class NeuromodDashboardRenderer(_MplRenderer):
    """NT levels over time."""

    def __init__(self):
        super().__init__("neuromod.dashboard")

    def render(self, data: dict[str, Any], **kwargs: Any) -> Any:
        fig, ax = self._get_fig_ax(kwargs)
        series = data.get("series", {})
        for nt_name, values in series.items():
            ax.plot(values, label=nt_name, alpha=0.8)
        ax.set_xlabel("Sample")
        ax.set_ylabel("Level")
        ax.set_title("Neuromodulator Dashboard")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig


class StorageOpsRenderer(_MplRenderer):
    """Ops/sec bar chart."""

    def __init__(self):
        super().__init__("storage.ops")

    def render(self, data: dict[str, Any], **kwargs: Any) -> Any:
        fig, ax = self._get_fig_ax(kwargs)
        ops = data.get("op_counts", {})
        if ops:
            ax.barh(list(ops.keys()), list(ops.values()), color="steelblue")
        ax.set_xlabel("Count")
        ax.set_title("Storage Operations")
        return fig


class KappaDistributionRenderer(_MplRenderer):
    """Kappa histogram."""

    def __init__(self):
        super().__init__("kappa.distribution")

    def render(self, data: dict[str, Any], **kwargs: Any) -> Any:
        fig, ax = self._get_fig_ax(kwargs)
        kappas = data.get("kappas", [])
        if kappas:
            ax.hist(kappas, bins=50, range=(0, 1), color="teal", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Kappa")
        ax.set_ylabel("Count")
        ax.set_title("Kappa Distribution")
        return fig


class EnergyLandscapeRenderer(_MplRenderer):
    """Wrapper around existing energy_landscape module."""

    def __init__(self):
        super().__init__("energy.landscape")

    def render(self, data: dict[str, Any], **kwargs: Any) -> Any:
        fig, ax = self._get_fig_ax(kwargs)
        energies = data.get("energies", [])
        if energies:
            ax.plot(energies, color="purple", alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Energy")
        ax.set_title("Energy Landscape")
        ax.grid(True, alpha=0.3)
        return fig


def register_mpl_renderers(registry: Any) -> None:
    """Register all matplotlib renderers in *registry*."""
    for cls in (
        SpikeRasterRenderer,
        NeuromodDashboardRenderer,
        StorageOpsRenderer,
        KappaDistributionRenderer,
        EnergyLandscapeRenderer,
    ):
        registry.register(cls())
