"""
Visualization Streamer for T4DM.

Collects snapshots from registered visualizers and produces
JSON-serializable data for WebSocket streaming.
"""

import logging
import time
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ExportableVisualizer(Protocol):
    """Protocol for visualizers that can export data."""

    def export_data(self) -> dict[str, Any]: ...


class VisualizationStreamer:
    """
    Collects from registered visualizers and produces JSON-serializable snapshots.

    Decouples WebSocket routes from specific visualizer implementations.
    """

    def __init__(self) -> None:
        self._visualizers: dict[str, ExportableVisualizer] = {}

    def register_visualizer(self, name: str, visualizer: ExportableVisualizer) -> None:
        """
        Register a visualizer with an export_data() method.

        Args:
            name: Module name (e.g. "spiking", "kappa", "neuromod")
            visualizer: Object implementing export_data() -> dict
        """
        self._visualizers[name] = visualizer
        logger.info(f"Visualizer registered: {name}")

    def unregister_visualizer(self, name: str) -> None:
        """Remove a registered visualizer."""
        self._visualizers.pop(name, None)

    @property
    def registered_modules(self) -> list[str]:
        """Return names of all registered visualizer modules."""
        return list(self._visualizers.keys())

    def get_snapshot(self, modules: list[str] | None = None) -> dict[str, Any]:
        """
        Collect snapshot data from registered visualizers.

        Args:
            modules: If provided, only collect from these modules.
                     If None, collect from all registered visualizers.

        Returns:
            Dict with module names as keys and their exported data as values,
            plus a timestamp.
        """
        targets = modules if modules is not None else list(self._visualizers.keys())
        snapshot: dict[str, Any] = {
            "timestamp": time.time(),
            "modules": {},
        }

        for name in targets:
            visualizer = self._visualizers.get(name)
            if visualizer is None:
                continue
            try:
                data = visualizer.export_data()
                snapshot["modules"][name] = data
            except Exception as e:
                logger.warning(f"Failed to collect snapshot from {name}: {e}")
                snapshot["modules"][name] = {"error": str(e)}

        return snapshot


__all__ = [
    "ExportableVisualizer",
    "VisualizationStreamer",
]
