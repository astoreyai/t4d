"""
Energy Landscape Visualization for World Weaver NCA.

Hinton-inspired visualization of energy-based dynamics:
- 2D projection of 6D NT space (PCA/t-SNE)
- Contour plots of energy surface
- Gradient vector fields
- Attractor basin boundaries
- State trajectory overlays

This is CRITICAL for understanding:
- What attractors the system has learned
- How states flow through the energy landscape
- Whether learning is creating appropriate basins
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ww.nca.attractors import StateTransitionManager
    from ww.nca.coupling import LearnableCoupling
    from ww.nca.energy import EnergyLandscape

logger = logging.getLogger(__name__)

# NT labels for visualization
NT_LABELS = ["DA", "5-HT", "ACh", "NE", "GABA", "Glu"]


@dataclass
class EnergySnapshot:
    """Snapshot of energy landscape state."""

    timestamp: datetime
    nt_state: np.ndarray  # Current NT state [6]
    total_energy: float
    hopfield_energy: float
    boundary_energy: float
    attractor_energy: float
    gradient_norm: float
    nearest_attractor: str | None = None
    attractor_distance: float = 0.0


@dataclass
class TrajectoryPoint:
    """Single point in state trajectory."""

    timestamp: datetime
    nt_state: np.ndarray
    energy: float
    cognitive_state: str | None = None


class EnergyLandscapeVisualizer:
    """
    Visualizes energy landscape dynamics of the NCA system.

    Maintains trajectory history and provides methods for
    generating energy surface plots, gradient fields, and
    attractor basin visualizations.
    """

    def __init__(
        self,
        energy_landscape: EnergyLandscape | None = None,
        coupling: LearnableCoupling | None = None,
        state_manager: StateTransitionManager | None = None,
        window_size: int = 1000,
        grid_resolution: int = 50,
    ):
        """
        Initialize energy landscape visualizer.

        Args:
            energy_landscape: Energy computation module
            coupling: Learnable coupling matrix
            state_manager: Attractor state manager
            window_size: Number of trajectory points to track
            grid_resolution: Resolution for energy surface grid
        """
        self.energy_landscape = energy_landscape
        self.coupling = coupling
        self.state_manager = state_manager
        self.window_size = window_size
        self.grid_resolution = grid_resolution

        # Trajectory history
        self._trajectory: list[TrajectoryPoint] = []
        self._snapshots: list[EnergySnapshot] = []

        # Cached projection matrix (from PCA)
        self._projection_matrix: np.ndarray | None = None
        self._projection_mean: np.ndarray | None = None

        # Cached energy surface
        self._energy_surface: np.ndarray | None = None
        self._surface_x: np.ndarray | None = None
        self._surface_y: np.ndarray | None = None
        self._surface_timestamp: datetime | None = None

        logger.info("EnergyLandscapeVisualizer initialized")

    def record_state(
        self,
        nt_state: np.ndarray,
        cognitive_state: str | None = None,
    ) -> EnergySnapshot:
        """
        Record current NT state and compute energy metrics.

        Args:
            nt_state: Current NT concentration vector [6]
            cognitive_state: Optional cognitive state label

        Returns:
            Energy snapshot with all metrics
        """
        nt_state = np.asarray(nt_state).flatten()
        now = datetime.now()

        # Compute energy components
        if self.energy_landscape is not None:
            total_energy = self.energy_landscape.compute_total_energy(nt_state)
            hopfield_energy = self.energy_landscape.compute_hopfield_energy(nt_state)
            boundary_energy = self.energy_landscape.compute_boundary_energy(nt_state)
            attractor_energy = self.energy_landscape.compute_attractor_energy(nt_state)
            gradient = self.energy_landscape.compute_energy_gradient(nt_state)
            gradient_norm = float(np.linalg.norm(gradient))
        else:
            total_energy = self._compute_energy_from_coupling(nt_state)
            hopfield_energy = total_energy
            boundary_energy = 0.0
            attractor_energy = 0.0
            gradient_norm = 0.0

        # Find nearest attractor
        nearest_attractor = None
        attractor_distance = float("inf")
        if self.state_manager is not None:
            for state, basin in self.state_manager.attractors.items():
                dist = float(np.linalg.norm(nt_state - basin.center))
                if dist < attractor_distance:
                    attractor_distance = dist
                    nearest_attractor = state.name

        # Create snapshot
        snapshot = EnergySnapshot(
            timestamp=now,
            nt_state=nt_state.copy(),
            total_energy=total_energy,
            hopfield_energy=hopfield_energy,
            boundary_energy=boundary_energy,
            attractor_energy=attractor_energy,
            gradient_norm=gradient_norm,
            nearest_attractor=nearest_attractor,
            attractor_distance=attractor_distance,
        )

        # Add to trajectory
        trajectory_point = TrajectoryPoint(
            timestamp=now,
            nt_state=nt_state.copy(),
            energy=total_energy,
            cognitive_state=cognitive_state or nearest_attractor,
        )

        self._trajectory.append(trajectory_point)
        self._snapshots.append(snapshot)

        # Maintain window
        if len(self._trajectory) > self.window_size:
            self._trajectory.pop(0)
        if len(self._snapshots) > self.window_size:
            self._snapshots.pop(0)

        return snapshot

    def _compute_energy_from_coupling(self, nt_state: np.ndarray) -> float:
        """Compute Hopfield energy directly from coupling matrix."""
        if self.coupling is None:
            return 0.0
        K = self.coupling.K
        return -0.5 * float(nt_state @ K @ nt_state)

    # -------------------------------------------------------------------------
    # PCA Projection (6D → 2D)
    # -------------------------------------------------------------------------

    def compute_pca_projection(
        self,
        include_attractors: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute PCA projection matrix from trajectory and attractors.

        Returns:
            Tuple of (projection_matrix [2, 6], mean [6])
        """
        points = []

        # Add trajectory points
        for tp in self._trajectory:
            points.append(tp.nt_state)

        # Add attractor centers
        if include_attractors and self.state_manager is not None:
            for basin in self.state_manager.attractors.values():
                points.append(basin.center)

        if len(points) < 3:
            # Not enough points, use default projection (first 2 PCs)
            self._projection_matrix = np.array([
                [1, 0, 0, 0, 0, 0],  # DA axis
                [0, 0, 0, 1, 0, 0],  # NE axis
            ], dtype=np.float32)
            self._projection_mean = np.array([0.5] * 6, dtype=np.float32)
            return self._projection_matrix, self._projection_mean

        X = np.array(points)
        mean = X.mean(axis=0)
        X_centered = X - mean

        # SVD for PCA
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Top 2 principal components
        self._projection_matrix = Vt[:2].astype(np.float32)
        self._projection_mean = mean.astype(np.float32)

        return self._projection_matrix, self._projection_mean

    def project_to_2d(self, nt_state: np.ndarray) -> np.ndarray:
        """
        Project 6D NT state to 2D visualization space.

        Args:
            nt_state: NT state vector [6]

        Returns:
            2D projected point [2]
        """
        if self._projection_matrix is None:
            self.compute_pca_projection()

        centered = nt_state - self._projection_mean
        return self._projection_matrix @ centered

    def project_trajectory(self) -> np.ndarray:
        """
        Project entire trajectory to 2D.

        Returns:
            Array of 2D points [N, 2]
        """
        if not self._trajectory:
            return np.zeros((0, 2))

        if self._projection_matrix is None:
            self.compute_pca_projection()

        points_6d = np.array([tp.nt_state for tp in self._trajectory])
        centered = points_6d - self._projection_mean
        return centered @ self._projection_matrix.T

    # -------------------------------------------------------------------------
    # Energy Surface Computation
    # -------------------------------------------------------------------------

    def compute_energy_surface(
        self,
        grid_range: tuple[float, float] = (-1.5, 1.5),
        force_recompute: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D energy surface for contour plotting.

        Samples energy at grid points in the 2D PCA space,
        projecting back to 6D for energy computation.

        Args:
            grid_range: Range for x and y axes in PCA space
            force_recompute: Force recomputation even if cached

        Returns:
            Tuple of (X grid, Y grid, Energy values)
        """
        # Check cache validity (recompute every 10 seconds)
        now = datetime.now()
        if (
            not force_recompute
            and self._energy_surface is not None
            and self._surface_timestamp is not None
            and (now - self._surface_timestamp).total_seconds() < 10
        ):
            return self._surface_x, self._surface_y, self._energy_surface

        if self._projection_matrix is None:
            self.compute_pca_projection()

        # Create grid in 2D PCA space
        x = np.linspace(grid_range[0], grid_range[1], self.grid_resolution)
        y = np.linspace(grid_range[0], grid_range[1], self.grid_resolution)
        X, Y = np.meshgrid(x, y)

        # Compute energy at each grid point
        energy = np.zeros_like(X)

        # Pseudo-inverse for back-projection
        proj_pinv = np.linalg.pinv(self._projection_matrix)

        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                # Project 2D point back to 6D
                point_2d = np.array([X[i, j], Y[i, j]])
                point_6d = proj_pinv @ point_2d + self._projection_mean

                # Clamp to valid range
                point_6d = np.clip(point_6d, 0.0, 1.0)

                # Compute energy
                if self.energy_landscape is not None:
                    energy[i, j] = self.energy_landscape.compute_total_energy(point_6d)
                else:
                    energy[i, j] = self._compute_energy_from_coupling(point_6d)

        self._surface_x = X
        self._surface_y = Y
        self._energy_surface = energy
        self._surface_timestamp = now

        return X, Y, energy

    def compute_gradient_field(
        self,
        grid_range: tuple[float, float] = (-1.5, 1.5),
        subsample: int = 5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradient vector field for visualization.

        Args:
            grid_range: Range for grid
            subsample: Subsample factor for arrows (every Nth point)

        Returns:
            Tuple of (X, Y, U, V) for quiver plot
        """
        if self._projection_matrix is None:
            self.compute_pca_projection()

        n_points = self.grid_resolution // subsample
        x = np.linspace(grid_range[0], grid_range[1], n_points)
        y = np.linspace(grid_range[0], grid_range[1], n_points)
        X, Y = np.meshgrid(x, y)

        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        proj_pinv = np.linalg.pinv(self._projection_matrix)

        for i in range(n_points):
            for j in range(n_points):
                point_2d = np.array([X[i, j], Y[i, j]])
                point_6d = proj_pinv @ point_2d + self._projection_mean
                point_6d = np.clip(point_6d, 0.0, 1.0)

                # Compute 6D gradient
                if self.energy_landscape is not None:
                    grad_6d = self.energy_landscape.compute_energy_gradient(point_6d)
                elif self.coupling is not None:
                    # Approximate gradient from coupling
                    grad_6d = -self.coupling.K @ point_6d
                else:
                    grad_6d = np.zeros(6)

                # Project gradient to 2D
                grad_2d = self._projection_matrix @ grad_6d

                # Negative gradient (points toward lower energy)
                U[i, j] = -grad_2d[0]
                V[i, j] = -grad_2d[1]

        # Normalize for visualization
        magnitude = np.sqrt(U**2 + V**2)
        max_mag = magnitude.max()
        if max_mag > 0:
            U = U / max_mag * 0.1  # Scale arrows
            V = V / max_mag * 0.1

        return X, Y, U, V

    # -------------------------------------------------------------------------
    # Attractor Basin Visualization
    # -------------------------------------------------------------------------

    def get_attractor_positions(self) -> dict[str, tuple[float, float, float]]:
        """
        Get 2D positions and sizes of attractor basins.

        Returns:
            Dict mapping state name -> (x, y, radius)
        """
        if self.state_manager is None:
            return {}

        if self._projection_matrix is None:
            self.compute_pca_projection()

        positions = {}
        for state, basin in self.state_manager.attractors.items():
            pos_2d = self.project_to_2d(basin.center)
            # Project width as approximate radius
            radius = basin.width * np.linalg.norm(self._projection_matrix[0])
            positions[state.name] = (float(pos_2d[0]), float(pos_2d[1]), float(radius))

        return positions

    def classify_basin(self, nt_state: np.ndarray) -> str | None:
        """
        Classify which attractor basin a state belongs to.

        Args:
            nt_state: NT state vector [6]

        Returns:
            Name of nearest attractor basin, or None
        """
        if self.state_manager is None:
            return None

        min_dist = float("inf")
        nearest = None

        for state, basin in self.state_manager.attractors.items():
            dist = float(np.linalg.norm(nt_state - basin.center))
            if dist < basin.width and dist < min_dist:
                min_dist = dist
                nearest = state.name

        return nearest

    # -------------------------------------------------------------------------
    # Metrics and Analysis
    # -------------------------------------------------------------------------

    def get_energy_trace(self) -> tuple[list[datetime], list[float]]:
        """Get time series of total energy."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        energies = [s.total_energy for s in self._snapshots]
        return timestamps, energies

    def get_gradient_norm_trace(self) -> tuple[list[datetime], list[float]]:
        """Get time series of gradient norm (learning activity)."""
        if not self._snapshots:
            return [], []
        timestamps = [s.timestamp for s in self._snapshots]
        norms = [s.gradient_norm for s in self._snapshots]
        return timestamps, norms

    def get_basin_occupancy(self) -> dict[str, float]:
        """
        Compute fraction of time spent in each attractor basin.

        Returns:
            Dict mapping basin name -> occupancy fraction
        """
        if not self._trajectory:
            return {}

        counts: dict[str, int] = {}
        total = len(self._trajectory)

        for tp in self._trajectory:
            basin = self.classify_basin(tp.nt_state)
            if basin:
                counts[basin] = counts.get(basin, 0) + 1

        return {k: v / total for k, v in counts.items()}

    def get_energy_components(self) -> dict[str, list[float]]:
        """Get time series of energy components."""
        return {
            "total": [s.total_energy for s in self._snapshots],
            "hopfield": [s.hopfield_energy for s in self._snapshots],
            "boundary": [s.boundary_energy for s in self._snapshots],
            "attractor": [s.attractor_energy for s in self._snapshots],
        }

    def get_stability_metrics(self) -> dict[str, float]:
        """
        Compute stability metrics from recent trajectory.

        Returns:
            Dict with mean energy, variance, basin transitions, etc.
        """
        if len(self._snapshots) < 2:
            return {}

        energies = [s.total_energy for s in self._snapshots]
        gradient_norms = [s.gradient_norm for s in self._snapshots]

        # Count basin transitions
        transitions = 0
        prev_basin = None
        for s in self._snapshots:
            if s.nearest_attractor != prev_basin and prev_basin is not None:
                transitions += 1
            prev_basin = s.nearest_attractor

        return {
            "mean_energy": float(np.mean(energies)),
            "energy_variance": float(np.var(energies)),
            "mean_gradient_norm": float(np.mean(gradient_norms)),
            "basin_transitions": transitions,
            "transition_rate": transitions / len(self._snapshots),
            "energy_trend": float(np.polyfit(range(len(energies)), energies, 1)[0]),
        }

    # -------------------------------------------------------------------------
    # Export and Serialization
    # -------------------------------------------------------------------------

    def export_data(self) -> dict:
        """Export visualization data for external rendering."""
        X, Y, E = self.compute_energy_surface()
        Xg, Yg, U, V = self.compute_gradient_field()
        trajectory_2d = self.project_trajectory()

        return {
            "surface": {
                "x": X.tolist(),
                "y": Y.tolist(),
                "energy": E.tolist(),
            },
            "gradient_field": {
                "x": Xg.tolist(),
                "y": Yg.tolist(),
                "u": U.tolist(),
                "v": V.tolist(),
            },
            "trajectory": trajectory_2d.tolist(),
            "attractors": self.get_attractor_positions(),
            "metrics": self.get_stability_metrics(),
            "basin_occupancy": self.get_basin_occupancy(),
        }

    def clear_history(self) -> None:
        """Clear trajectory and snapshot history."""
        self._trajectory.clear()
        self._snapshots.clear()
        self._energy_surface = None
        self._surface_timestamp = None


# =============================================================================
# Standalone Plot Functions
# =============================================================================


def plot_energy_contour(
    visualizer: EnergyLandscapeVisualizer,
    ax=None,
    show_gradient: bool = True,
    show_trajectory: bool = True,
    show_attractors: bool = True,
    cmap: str = "viridis",
    levels: int = 20,
):
    """
    Plot energy contour with gradient field and trajectory overlay.

    Args:
        visualizer: EnergyLandscapeVisualizer instance
        ax: Matplotlib axes (creates new if None)
        show_gradient: Show gradient vector field
        show_trajectory: Show state trajectory
        show_attractors: Show attractor basin markers
        cmap: Colormap for contours
        levels: Number of contour levels

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

    # Energy surface contour
    X, Y, E = visualizer.compute_energy_surface()
    contour = ax.contourf(X, Y, E, levels=levels, cmap=cmap, alpha=0.8)
    ax.contour(X, Y, E, levels=levels, colors="white", linewidths=0.5, alpha=0.3)

    # Gradient field
    if show_gradient:
        Xg, Yg, U, V = visualizer.compute_gradient_field()
        ax.quiver(Xg, Yg, U, V, color="white", alpha=0.6, scale=2)

    # Trajectory
    if show_trajectory and len(visualizer._trajectory) > 1:
        traj = visualizer.project_trajectory()
        ax.plot(traj[:, 0], traj[:, 1], "r-", linewidth=1.5, alpha=0.7, label="Trajectory")
        ax.scatter(traj[-1, 0], traj[-1, 1], c="red", s=100, marker="*", zorder=5, label="Current")

    # Attractor basins
    if show_attractors:
        positions = visualizer.get_attractor_positions()
        colors = plt.cm.Set1(np.linspace(0, 1, len(positions)))
        for (name, (x, y, r)), color in zip(positions.items(), colors):
            circle = plt.Circle((x, y), r, fill=False, color=color, linewidth=2, linestyle="--")
            ax.add_patch(circle)
            ax.annotate(name, (x, y), fontsize=9, ha="center", va="center", color=color, weight="bold")

    ax.set_xlabel("PC1 (DA-dominated)")
    ax.set_ylabel("PC2 (NE-dominated)")
    ax.set_title("Energy Landscape")
    plt.colorbar(contour, ax=ax, label="Energy")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")

    return ax


def plot_energy_timeline(
    visualizer: EnergyLandscapeVisualizer,
    ax=None,
    show_components: bool = True,
):
    """
    Plot energy over time with component breakdown.

    Args:
        visualizer: EnergyLandscapeVisualizer instance
        ax: Matplotlib axes (creates new if None)
        show_components: Show individual energy components

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

    timestamps, energies = visualizer.get_energy_trace()
    if not timestamps:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    # Convert timestamps to relative seconds
    t0 = timestamps[0]
    t_seconds = [(t - t0).total_seconds() for t in timestamps]

    ax.plot(t_seconds, energies, "b-", linewidth=2, label="Total Energy")

    if show_components:
        components = visualizer.get_energy_components()
        ax.plot(t_seconds, components["hopfield"], "--", alpha=0.7, label="Hopfield")
        ax.plot(t_seconds, components["boundary"], "--", alpha=0.7, label="Boundary")
        ax.plot(t_seconds, components["attractor"], "--", alpha=0.7, label="Attractor")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy")
    ax.set_title("Energy Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_basin_occupancy(
    visualizer: EnergyLandscapeVisualizer,
    ax=None,
):
    """
    Plot pie chart of basin occupancy.

    Args:
        visualizer: EnergyLandscapeVisualizer instance
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

    occupancy = visualizer.get_basin_occupancy()
    if not occupancy:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax

    labels = list(occupancy.keys())
    sizes = list(occupancy.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Basin Occupancy")

    return ax


def create_energy_dashboard(
    visualizer: EnergyLandscapeVisualizer,
    figsize: tuple[int, int] = (16, 10),
):
    """
    Create comprehensive energy landscape dashboard.

    Args:
        visualizer: EnergyLandscapeVisualizer instance
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
    gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1])

    # Main energy contour (large)
    ax_contour = fig.add_subplot(gs[0, :2])
    plot_energy_contour(visualizer, ax=ax_contour)

    # Basin occupancy pie chart
    ax_pie = fig.add_subplot(gs[0, 2])
    plot_basin_occupancy(visualizer, ax=ax_pie)

    # Energy timeline
    ax_timeline = fig.add_subplot(gs[1, :2])
    plot_energy_timeline(visualizer, ax=ax_timeline)

    # Metrics text box
    ax_metrics = fig.add_subplot(gs[1, 2])
    ax_metrics.axis("off")

    metrics = visualizer.get_stability_metrics()
    if metrics:
        text = "\n".join([
            "Stability Metrics",
            "-" * 20,
            f"Mean Energy: {metrics.get('mean_energy', 0):.3f}",
            f"Energy Var: {metrics.get('energy_variance', 0):.4f}",
            f"Gradient Norm: {metrics.get('mean_gradient_norm', 0):.3f}",
            f"Transitions: {metrics.get('basin_transitions', 0)}",
            f"Trans. Rate: {metrics.get('transition_rate', 0):.3f}",
            f"Energy Trend: {metrics.get('energy_trend', 0):.4f}",
        ])
        ax_metrics.text(0.1, 0.5, text, fontsize=10, family="monospace",
                        verticalalignment="center", transform=ax_metrics.transAxes)

    plt.tight_layout()
    return fig


__all__ = [
    "EnergyLandscapeVisualizer",
    "EnergySnapshot",
    "TrajectoryPoint",
    "plot_energy_contour",
    "plot_energy_timeline",
    "plot_basin_occupancy",
    "create_energy_dashboard",
]
