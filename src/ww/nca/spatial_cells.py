"""
Place Cells and Grid Cells for Spatial Representation.

P4-3: Spatial prediction and navigation in embedding space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpatialConfig:
    embedding_dim: int = 1024
    spatial_dim: int = 2
    n_place_cells: int = 100
    place_field_sigma: float = 0.15
    place_sparsity: float = 0.01
    n_grid_modules: int = 3
    grid_scales: tuple[float, ...] = (0.3, 0.5, 0.8)
    cells_per_module: int = 32
    velocity_gain: float = 0.1
    position_decay: float = 0.01
    place_learning_rate: float = 0.01
    projection_learning_rate: float = 0.001
    boundary_mode: str = "wrap"  # ATOM-P3-29: "wrap" (toroidal) or "clip"
    arena_size: float = 2.0  # ATOM-P3-29: Arena size for wrap-around


@dataclass
class Position2D:
    x: float = 0.0
    y: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Position2D:
        return cls(x=float(arr[0]), y=float(arr[1]))

    def distance_to(self, other: Position2D) -> float:
        return float(np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2))


@dataclass
class PlaceCell:
    cell_id: int
    center: Position2D
    sigma: float
    activation: float = 0.0

    def compute_activation(self, position: Position2D) -> float:
        dist = position.distance_to(self.center)
        self.activation = float(np.exp(-(dist ** 2) / (2 * self.sigma ** 2)))
        return self.activation


@dataclass
class GridModule:
    module_id: int
    scale: float
    orientation: float
    phase_x: float = 0.0
    phase_y: float = 0.0

    def compute_response(self, position: Position2D, boundary_mode: str = "wrap", arena_size: float = 2.0) -> float:
        """
        Compute grid cell response at position.

        ATOM-P3-29: Handles boundary effects with configurable mode.

        Args:
            position: 2D position
            boundary_mode: "wrap" for toroidal, "clip" for hard boundaries
            arena_size: Size of arena for wrap-around

        Returns:
            Grid response [0, 1]
        """
        x, y = position.x, position.y

        # ATOM-P3-29: Apply boundary mode
        if boundary_mode == "wrap":
            x = x % arena_size
            y = y % arena_size
        # else: "clip" mode uses position as-is

        cos_o = np.cos(self.orientation)
        sin_o = np.sin(self.orientation)
        rx = x * cos_o + y * sin_o
        ry = -x * sin_o + y * cos_o
        rx += self.phase_x
        ry += self.phase_y
        k = 2 * np.pi / self.scale
        response = (
            np.cos(k * rx)
            + np.cos(k * (rx * 0.5 + ry * np.sqrt(3) / 2))
            + np.cos(k * (rx * 0.5 - ry * np.sqrt(3) / 2))
        ) / 3.0
        return float((response + 1) / 2)


@dataclass
class SpatialState:
    position: Position2D = field(default_factory=Position2D)
    velocity: Position2D = field(default_factory=Position2D)
    place_activations: np.ndarray = field(default_factory=lambda: np.zeros(100, dtype=np.float32))
    grid_responses: np.ndarray = field(default_factory=lambda: np.zeros(96, dtype=np.float32))

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": {"x": self.position.x, "y": self.position.y},
            "velocity": {"x": self.velocity.x, "y": self.velocity.y},
            "place_sparsity": float(np.mean(self.place_activations > 0.5)),
            "grid_mean": float(np.mean(self.grid_responses)),
        }


class SpatialCellSystem:
    """Place and Grid Cell System for Spatial Cognition."""

    def __init__(self, config: SpatialConfig | None = None):
        self.config = config or SpatialConfig()
        self.state = SpatialState()

        np.random.seed(44)
        self._projection = np.random.randn(self.config.embedding_dim, self.config.spatial_dim).astype(np.float32)
        self._projection /= np.linalg.norm(self._projection, axis=0, keepdims=True)

        self._place_cells: list[PlaceCell] = []
        # ATOM-P4-1: Arena-size adaptive sigma (scales with arena size)
        arena_size = 4.0  # Current arena spans [-1,1] in each dimension
        reference_size = 1.0  # Reference for base sigma
        sigma_scaled = self.config.place_field_sigma * np.sqrt(arena_size / reference_size)
        for i in range(self.config.n_place_cells):
            center = Position2D(x=np.random.uniform(-1, 1), y=np.random.uniform(-1, 1))
            self._place_cells.append(PlaceCell(cell_id=i, center=center, sigma=sigma_scaled))

        self._grid_modules: list[GridModule] = []
        for i, scale in enumerate(self.config.grid_scales):
            module = GridModule(
                module_id=i,
                scale=scale,
                orientation=np.random.uniform(0, np.pi / 3),
                phase_x=np.random.uniform(0, scale),
                phase_y=np.random.uniform(0, scale),
            )
            self._grid_modules.append(module)

        self._position_history: list[tuple[UUID | None, Position2D]] = []
        self._max_history = 1000
        self._last_embedding: np.ndarray | None = None
        self._total_updates = 0

        logger.info(f"SpatialCellSystem initialized: {self.config.n_place_cells} place cells, {len(self._grid_modules)} grid modules")

    def encode_position(self, embedding: np.ndarray, episode_id: UUID | None = None) -> Position2D:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        pos_2d = embedding @ self._projection
        position = Position2D(x=float(pos_2d[0]), y=float(pos_2d[1]))

        if self._last_embedding is not None:
            delta = embedding - self._last_embedding
            vel_2d = delta @ self._projection * self.config.velocity_gain
            self.state.velocity = Position2D(x=float(vel_2d[0]), y=float(vel_2d[1]))
        else:
            self.state.velocity = Position2D()

        position.x = position.x * (1 - self.config.position_decay)
        position.y = position.y * (1 - self.config.position_decay)

        self.state.position = position
        self._last_embedding = embedding.copy()

        self._update_place_cells(position)
        self._update_grid_cells(position)

        self._position_history.append((episode_id, position))
        if len(self._position_history) > self._max_history:
            self._position_history = self._position_history[-self._max_history:]

        self._total_updates += 1
        return position

    def _update_place_cells(self, position: Position2D) -> None:
        activations = np.zeros(self.config.n_place_cells, dtype=np.float32)
        for i, cell in enumerate(self._place_cells):
            activations[i] = cell.compute_activation(position)
        threshold = np.percentile(activations, (1 - self.config.place_sparsity) * 100)
        activations[activations < threshold] *= 0.1
        self.state.place_activations = activations

    def _update_grid_cells(self, position: Position2D) -> None:
        """
        Update grid cell responses.

        ATOM-P3-29: Passes boundary mode and arena size to grid cells.
        """
        responses = []
        for module in self._grid_modules:
            base_response = module.compute_response(
                position,
                boundary_mode=self.config.boundary_mode,
                arena_size=self.config.arena_size
            )
            for j in range(self.config.cells_per_module):
                phase_shift = (j / self.config.cells_per_module) * module.scale
                shifted_pos = Position2D(x=position.x + phase_shift, y=position.y)
                responses.append(module.compute_response(
                    shifted_pos,
                    boundary_mode=self.config.boundary_mode,
                    arena_size=self.config.arena_size
                ))
        self.state.grid_responses = np.array(responses, dtype=np.float32)

    def get_place_activations(self) -> np.ndarray:
        return self.state.place_activations.copy()

    def get_grid_responses(self) -> np.ndarray:
        return self.state.grid_responses.copy()

    def get_combined_spatial_code(self) -> np.ndarray:
        return np.concatenate([self.state.place_activations, self.state.grid_responses])

    def find_neighbors(self, k: int = 5, radius: float | None = None) -> list[tuple[UUID, float]]:
        current = self.state.position
        neighbors = []
        for episode_id, pos in self._position_history:
            if episode_id is None:
                continue
            dist = current.distance_to(pos)
            if radius is None or dist <= radius:
                neighbors.append((episode_id, dist))
        neighbors.sort(key=lambda x: x[1])
        return neighbors[:k]

    def predict_next_position(self, velocity_hint: Position2D | None = None) -> Position2D:
        velocity = velocity_hint or self.state.velocity
        current = self.state.position
        return Position2D(x=current.x + velocity.x, y=current.y + velocity.y)

    def get_position_sequence(self, n: int = 10) -> list[Position2D]:
        return [pos for _, pos in self._position_history[-n:]]

    def get_statistics(self) -> dict[str, Any]:
        active_place = np.sum(self.state.place_activations > 0.5)
        return {
            "total_updates": self._total_updates,
            "position": self.state.to_dict()["position"],
            "velocity": self.state.to_dict()["velocity"],
            "active_place_cells": int(active_place),
            "place_sparsity": float(active_place / self.config.n_place_cells),
            "mean_grid_response": float(np.mean(self.state.grid_responses)),
            "history_size": len(self._position_history),
        }

    def save_state(self) -> dict[str, Any]:
        return {
            "projection": self._projection.tolist(),
            "place_centers": [{"x": c.center.x, "y": c.center.y, "sigma": c.sigma} for c in self._place_cells],
            "grid_modules": [{"scale": m.scale, "orientation": m.orientation, "phase_x": m.phase_x, "phase_y": m.phase_y} for m in self._grid_modules],
            "statistics": self.get_statistics(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if "projection" in state:
            self._projection = np.array(state["projection"], dtype=np.float32)
        if "place_centers" in state:
            for i, center_data in enumerate(state["place_centers"]):
                if i < len(self._place_cells):
                    self._place_cells[i].center = Position2D(x=center_data["x"], y=center_data["y"])
                    if "sigma" in center_data:
                        self._place_cells[i].sigma = center_data["sigma"]

    # =========================================================================
    # Grid Cell Hexagonal Pattern Validation (Phase 3 - B7)
    # =========================================================================

    def validate_hexagonal_pattern(
        self,
        resolution: int = 50,
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        """
        Validate that grid cells produce hexagonal firing patterns.

        Biological basis (Moser et al., 2008; Nobel Prize 2014):
        - Grid cells fire at vertices of regular hexagonal lattice
        - Hexagonal pattern has 6-fold rotational symmetry
        - Spatial autocorrelation shows 6 peaks at 60-degree intervals

        Reference: Sargolini et al. (2006) define gridness score.

        Args:
            resolution: Grid resolution for sampling (default 50)
            threshold: Minimum gridness score (0.3 = typical for grid cells)

        Returns:
            Validation results with gridness score and diagnostics
        """
        results: dict[str, Any] = {
            "modules": [],
            "overall_gridness": 0.0,
            "passes_threshold": False,
            "six_fold_symmetry": False,
        }

        for module in self._grid_modules:
            # Sample grid responses across 2D space
            x = np.linspace(-1, 1, resolution)
            y = np.linspace(-1, 1, resolution)
            responses = np.zeros((resolution, resolution), dtype=np.float32)

            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    pos = Position2D(x=float(xi), y=float(yj))
                    responses[i, j] = module.compute_response(pos)

            # Compute spatial autocorrelation
            autocorr = self._compute_autocorrelation(responses)

            # Compute gridness score
            gridness = self.compute_gridness_score(autocorr)

            # Check 6-fold symmetry
            has_symmetry = self._check_sixfold_symmetry(autocorr)

            results["modules"].append({
                "module_id": module.module_id,
                "scale": module.scale,
                "gridness": gridness,
                "has_symmetry": has_symmetry,
            })

        # Overall gridness is mean across modules
        module_gridness = [m["gridness"] for m in results["modules"]]
        results["overall_gridness"] = float(np.mean(module_gridness))
        results["passes_threshold"] = results["overall_gridness"] > threshold
        results["six_fold_symmetry"] = all(
            m["has_symmetry"] for m in results["modules"]
        )

        return results

    def compute_gridness_score(self, autocorr: np.ndarray) -> float:
        """
        Compute gridness score following Sargolini et al. (2006).

        Gridness = min(correlation at 60,120 degrees) - max(correlation at 30,90,150 degrees)

        A positive score indicates hexagonal pattern.
        Score > 0.3 is typical for grid cells.

        Args:
            autocorr: 2D spatial autocorrelation

        Returns:
            Gridness score (higher = more hexagonal)
        """
        center = np.array(autocorr.shape) // 2

        def get_rotated_correlation(angle_deg: float) -> float:
            """Find peak correlation at this rotation angle."""
            angle_rad = np.deg2rad(angle_deg)
            r = min(center) // 2  # Radius to sample
            x_idx = int(center[0] + r * np.cos(angle_rad))
            y_idx = int(center[1] + r * np.sin(angle_rad))
            if 0 <= x_idx < autocorr.shape[0] and 0 <= y_idx < autocorr.shape[1]:
                return float(autocorr[x_idx, y_idx])
            return 0.0

        # Correlations at hexagonal angles (60, 120)
        hex_angles = [60.0, 120.0]
        hex_corrs = [get_rotated_correlation(a) for a in hex_angles]

        # Correlations at non-hexagonal angles (30, 90, 150)
        non_hex_angles = [30.0, 90.0, 150.0]
        non_hex_corrs = [get_rotated_correlation(a) for a in non_hex_angles]

        # Gridness = min(hex) - max(non-hex)
        gridness = min(hex_corrs) - max(non_hex_corrs)

        return float(gridness)

    def _compute_autocorrelation(self, responses: np.ndarray) -> np.ndarray:
        """
        Compute 2D spatial autocorrelation using FFT.

        Args:
            responses: 2D array of grid cell responses

        Returns:
            2D autocorrelation centered
        """
        # Normalize
        responses = responses - np.mean(responses)

        # FFT-based autocorrelation
        f = np.fft.fft2(responses)
        autocorr = np.fft.ifft2(f * np.conj(f)).real
        autocorr = np.fft.fftshift(autocorr)

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(autocorr))
        if max_val > 0:
            autocorr = autocorr / max_val

        return autocorr.astype(np.float32)

    def _check_sixfold_symmetry(
        self,
        autocorr: np.ndarray,
        tolerance: float = 0.15,
    ) -> bool:
        """
        Check if autocorrelation has 6-fold rotational symmetry.

        Uses simplified approach: compare correlations at 0, 60, 120 degrees.

        Args:
            autocorr: 2D autocorrelation array
            tolerance: Correlation tolerance (default 0.15)

        Returns:
            True if 6-fold symmetry detected
        """
        center = np.array(autocorr.shape) // 2
        r = min(center) // 2

        # Sample at 0, 60, 120 degrees
        samples = []
        for angle in [0.0, 60.0, 120.0]:
            angle_rad = np.deg2rad(angle)
            x_idx = int(center[0] + r * np.cos(angle_rad))
            y_idx = int(center[1] + r * np.sin(angle_rad))
            if 0 <= x_idx < autocorr.shape[0] and 0 <= y_idx < autocorr.shape[1]:
                samples.append(autocorr[x_idx, y_idx])

        if len(samples) < 3:
            return False

        # Check if all samples are similar (within tolerance)
        mean_val = np.mean(samples)
        if mean_val <= 0:
            return False

        max_deviation = max(abs(s - mean_val) for s in samples)
        relative_deviation = max_deviation / abs(mean_val)

        return relative_deviation < tolerance
