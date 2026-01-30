"""
Transmission Delay System for World Weaver NCA.

Implements biologically-plausible signal transmission delays including:
1. Axonal conduction delays (distance and myelination dependent)
2. Synaptic delays (NT-specific release/binding kinetics)
3. Region-to-region communication delays
4. Delay differential equation support for neural field

Biological Basis:
- Axonal conduction: 1-100 m/s depending on fiber diameter and myelination
  - Unmyelinated C-fibers: 0.5-2 m/s
  - Myelinated A-fibers: 5-120 m/s
  - Typical cortical: 1-10 m/s

- Synaptic delay components:
  - Presynaptic Ca2+ influx: 0.2-0.5 ms
  - Vesicle fusion: 0.1-0.3 ms
  - Diffusion across cleft: 0.1-0.2 ms
  - Receptor binding: 0.1-0.5 ms
  - Total: ~0.5-2 ms for fast synapses, up to 100+ ms for metabotropic

- Long-range delays:
  - Intra-cortical: 1-10 ms
  - Cortico-cortical: 5-20 ms
  - Cortico-subcortical: 10-50 ms
  - Transcallosal: 10-30 ms

Implementation:
- Circular buffer for delay lines (memory efficient)
- Distance matrix for region-to-region delays
- NT-specific synaptic delay profiles
- Integration with neural field PDE solver

References:
- Swadlow (2000) - Axonal conduction delays
- Kandel et al. - Principles of Neural Science
- Deco et al. (2008) - Delays in large-scale brain models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class FiberType(Enum):
    """Axon fiber types with characteristic conduction velocities."""
    C_UNMYELINATED = "c_unmyelinated"      # 0.5-2 m/s (pain, temperature)
    A_DELTA = "a_delta"                     # 5-30 m/s (fast pain, touch)
    A_BETA = "a_beta"                       # 30-70 m/s (touch, pressure)
    A_ALPHA = "a_alpha"                     # 70-120 m/s (motor, proprioception)
    CORTICAL_LOCAL = "cortical_local"       # 0.5-2 m/s (local circuits)
    CORTICAL_LONG = "cortical_long"         # 3-10 m/s (long-range)


@dataclass
class DelayConfig:
    """Configuration for transmission delay system.

    All times in milliseconds, distances in millimeters.
    """
    # Time resolution
    dt_ms: float = 1.0                      # Timestep in ms
    max_delay_ms: float = 100.0             # Maximum delay to buffer

    # Axonal conduction
    default_velocity_m_s: float = 5.0       # Default conduction velocity
    velocity_variability: float = 0.1       # CV of velocity (noise)

    # Synaptic delays (ms)
    synaptic_delay_glu: float = 1.0         # Fast glutamatergic
    synaptic_delay_gaba: float = 1.5        # GABAergic (slightly slower)
    synaptic_delay_da: float = 20.0         # Dopaminergic (volume transmission)
    synaptic_delay_5ht: float = 25.0        # Serotonergic (volume transmission)
    synaptic_delay_ach: float = 5.0         # Cholinergic (mixed)
    synaptic_delay_ne: float = 15.0         # Noradrenergic (volume transmission)

    # Region distances (mm) - simplified cortical model
    intra_region_distance: float = 2.0      # Within a region
    inter_region_distance: float = 20.0     # Between adjacent regions
    long_range_distance: float = 100.0      # Long-range connections

    # Plasticity of delays (activity-dependent myelination)
    delay_plasticity: bool = False          # Enable delay learning
    plasticity_rate: float = 0.001          # Learning rate for delays

    @property
    def buffer_size(self) -> int:
        """Compute buffer size needed for max delay."""
        return int(self.max_delay_ms / self.dt_ms) + 1

    def get_synaptic_delay(self, nt_index: int) -> float:
        """Get synaptic delay for NT by index [DA, 5HT, ACh, NE, GABA, Glu]."""
        delays = [
            self.synaptic_delay_da,      # 0: DA
            self.synaptic_delay_5ht,     # 1: 5-HT
            self.synaptic_delay_ach,     # 2: ACh
            self.synaptic_delay_ne,      # 3: NE
            self.synaptic_delay_gaba,    # 4: GABA
            self.synaptic_delay_glu,     # 5: Glu
        ]
        return delays[nt_index] if 0 <= nt_index < 6 else 1.0


@dataclass
class DelayState:
    """Current state of delay system."""
    total_buffered_ms: float = 0.0          # Total time in buffers
    active_delays: int = 0                   # Number of active delay lines
    mean_delay_ms: float = 0.0              # Average delay
    max_active_delay_ms: float = 0.0        # Maximum active delay


class CircularDelayBuffer:
    """
    Efficient circular buffer for signal delay.

    Stores historical values and retrieves delayed versions
    without data copying.
    """

    def __init__(self, size: int, shape: tuple[int, ...] = ()):
        """
        Initialize delay buffer.

        Args:
            size: Number of timesteps to buffer
            shape: Shape of each timestep's data
        """
        self.size = max(2, size)
        self.shape = shape
        self._buffer = np.zeros((self.size,) + shape, dtype=np.float32)
        self._head = 0  # Points to oldest entry (next to overwrite)

    def push(self, value: np.ndarray) -> None:
        """
        Push new value, overwriting oldest.

        Args:
            value: New value to add (must match shape)
        """
        self._buffer[self._head] = value
        self._head = (self._head + 1) % self.size

    def get_delayed(self, delay_steps: int) -> np.ndarray:
        """
        Get value from delay_steps ago.

        Args:
            delay_steps: How many steps back (0 = current, size-1 = oldest)

        Returns:
            Delayed value
        """
        delay_steps = np.clip(delay_steps, 0, self.size - 1)
        # Current position is _head - 1 (just written)
        # delay_steps ago is _head - 1 - delay_steps
        idx = (self._head - 1 - delay_steps) % self.size
        return self._buffer[idx]

    def get_current(self) -> np.ndarray:
        """Get most recently pushed value."""
        return self.get_delayed(0)

    def get_oldest(self) -> np.ndarray:
        """Get oldest value in buffer."""
        return self._buffer[self._head]

    def interpolate_delay(self, delay_steps: float) -> np.ndarray:
        """
        Get interpolated value for fractional delay.

        Args:
            delay_steps: Fractional delay steps

        Returns:
            Linearly interpolated value
        """
        lower = int(np.floor(delay_steps))
        upper = int(np.ceil(delay_steps))
        frac = delay_steps - lower

        if lower == upper:
            return self.get_delayed(lower)

        val_lower = self.get_delayed(lower)
        val_upper = self.get_delayed(upper)

        return val_lower * (1 - frac) + val_upper * frac

    def clear(self) -> None:
        """Reset buffer to zeros."""
        self._buffer.fill(0)
        self._head = 0


class DistanceMatrix:
    """
    Manages distances between brain regions for delay computation.

    Distances are used with conduction velocity to compute
    transmission delays.
    """

    def __init__(
        self,
        n_regions: int,
        default_distance: float = 20.0  # mm
    ):
        """
        Initialize distance matrix.

        Args:
            n_regions: Number of brain regions
            default_distance: Default inter-region distance (mm)
        """
        self.n_regions = n_regions
        self.default_distance = default_distance

        # Initialize with default distance, 0 on diagonal
        self._distances = np.full(
            (n_regions, n_regions),
            default_distance,
            dtype=np.float32
        )
        np.fill_diagonal(self._distances, 0.0)

    def set_distance(self, i: int, j: int, distance: float) -> None:
        """Set distance between regions (symmetric)."""
        self._distances[i, j] = distance
        self._distances[j, i] = distance

    def get_distance(self, i: int, j: int) -> float:
        """Get distance between regions."""
        return float(self._distances[i, j])

    def get_delay_ms(
        self,
        i: int,
        j: int,
        velocity_m_s: float = 5.0
    ) -> float:
        """
        Compute transmission delay in ms.

        delay = distance / velocity

        Args:
            i: Source region
            j: Target region
            velocity_m_s: Conduction velocity in m/s

        Returns:
            Delay in milliseconds
        """
        distance_mm = self.get_distance(i, j)
        # Convert: mm / (m/s) = mm / (mm/ms) = ms
        distance_m = distance_mm / 1000.0
        delay_s = distance_m / velocity_m_s
        delay_ms = delay_s * 1000.0
        return delay_ms

    def get_all_delays(self, velocity_m_s: float = 5.0) -> np.ndarray:
        """Get full delay matrix in ms."""
        distance_m = self._distances / 1000.0
        delay_s = distance_m / velocity_m_s
        return delay_s * 1000.0

    @classmethod
    def from_coordinates(
        cls,
        coordinates: np.ndarray
    ) -> DistanceMatrix:
        """
        Create distance matrix from 3D coordinates.

        Args:
            coordinates: Array of shape (n_regions, 3) with XYZ positions in mm

        Returns:
            DistanceMatrix with Euclidean distances
        """
        n_regions = len(coordinates)
        dm = cls(n_regions)

        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                dm.set_distance(i, j, dist)

        return dm


class TransmissionDelaySystem:
    """
    Complete transmission delay system for neural field.

    Manages:
    1. Per-NT delay buffers (synaptic delays)
    2. Region-to-region axonal delays
    3. Delay-dependent coupling
    4. Optional delay plasticity
    """

    def __init__(
        self,
        config: DelayConfig | None = None,
        n_regions: int = 1,
        grid_shape: tuple[int, ...] | None = None
    ):
        """
        Initialize transmission delay system.

        Args:
            config: Delay configuration
            n_regions: Number of brain regions (for multi-region models)
            grid_shape: Spatial grid shape for field delays
        """
        self.config = config or DelayConfig()
        self.n_regions = n_regions
        self.grid_shape = grid_shape or (16,)

        # Per-NT synaptic delay buffers (6 NTs)
        self._nt_buffers: dict[int, CircularDelayBuffer] = {}
        for nt_idx in range(6):
            self._nt_buffers[nt_idx] = CircularDelayBuffer(
                size=self.config.buffer_size,
                shape=self.grid_shape
            )

        # Region-to-region distance matrix
        self._distance_matrix = DistanceMatrix(
            n_regions=n_regions,
            default_distance=self.config.inter_region_distance
        )

        # Region delay buffers (if multi-region)
        self._region_buffers: dict[tuple[int, int], CircularDelayBuffer] = {}
        if n_regions > 1:
            for i in range(n_regions):
                for j in range(n_regions):
                    if i != j:
                        self._region_buffers[(i, j)] = CircularDelayBuffer(
                            size=self.config.buffer_size,
                            shape=(6,) + self.grid_shape  # All NTs
                        )

        # Conduction velocities per pathway (can be learned)
        self._velocities = np.full(
            (n_regions, n_regions),
            self.config.default_velocity_m_s,
            dtype=np.float32
        )

        # Statistics
        self._step_count = 0

        logger.info(
            f"TransmissionDelaySystem initialized: "
            f"n_regions={n_regions}, "
            f"grid_shape={grid_shape}, "
            f"buffer_size={self.config.buffer_size}, "
            f"max_delay={self.config.max_delay_ms}ms"
        )

    def push_nt_state(self, fields: np.ndarray) -> None:
        """
        Push current NT fields into delay buffers.

        Args:
            fields: NT concentration fields [6, *grid_shape]
        """
        for nt_idx in range(6):
            self._nt_buffers[nt_idx].push(fields[nt_idx])

    def get_delayed_nt(
        self,
        nt_index: int,
        custom_delay_ms: float | None = None
    ) -> np.ndarray:
        """
        Get delayed NT field.

        Args:
            nt_index: NT index [0=DA, 1=5HT, 2=ACh, 3=NE, 4=GABA, 5=Glu]
            custom_delay_ms: Override default synaptic delay

        Returns:
            Delayed NT field
        """
        if custom_delay_ms is not None:
            delay_ms = custom_delay_ms
        else:
            delay_ms = self.config.get_synaptic_delay(nt_index)

        delay_steps = delay_ms / self.config.dt_ms

        return self._nt_buffers[nt_index].interpolate_delay(delay_steps)

    def get_all_delayed_nts(self) -> np.ndarray:
        """
        Get all NT fields with their respective synaptic delays.

        Returns:
            Delayed NT fields [6, *grid_shape]
        """
        delayed = np.zeros((6,) + self.grid_shape, dtype=np.float32)
        for nt_idx in range(6):
            delayed[nt_idx] = self.get_delayed_nt(nt_idx)
        return delayed

    def push_region_state(
        self,
        source_region: int,
        target_region: int,
        fields: np.ndarray
    ) -> None:
        """
        Push fields for region-to-region transmission.

        Args:
            source_region: Source region index
            target_region: Target region index
            fields: NT fields to transmit [6, *grid_shape]
        """
        key = (source_region, target_region)
        if key in self._region_buffers:
            self._region_buffers[key].push(fields)

    def get_delayed_region_input(
        self,
        source_region: int,
        target_region: int
    ) -> np.ndarray:
        """
        Get delayed input from another region.

        Args:
            source_region: Source region index
            target_region: Target region index

        Returns:
            Delayed NT fields [6, *grid_shape]
        """
        key = (source_region, target_region)
        if key not in self._region_buffers:
            return np.zeros((6,) + self.grid_shape, dtype=np.float32)

        # Compute delay based on distance and velocity
        delay_ms = self._distance_matrix.get_delay_ms(
            source_region,
            target_region,
            self._velocities[source_region, target_region]
        )

        delay_steps = delay_ms / self.config.dt_ms
        return self._region_buffers[key].interpolate_delay(delay_steps)

    def compute_delayed_coupling(
        self,
        fields: np.ndarray,
        coupling_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute coupling with appropriate delays.

        Each NT's contribution to others is delayed by its
        synaptic delay.

        Args:
            fields: Current NT fields [6, *grid_shape]
            coupling_matrix: Coupling strengths [6, 6]

        Returns:
            Delayed coupling contribution [6, *grid_shape]
        """
        # Push current state
        self.push_nt_state(fields)

        # Get delayed versions of each NT
        delayed_fields = self.get_all_delayed_nts()

        # Compute coupling: for each target NT, sum contributions from delayed sources
        output = np.zeros_like(fields)

        for target_nt in range(6):
            for source_nt in range(6):
                if abs(coupling_matrix[target_nt, source_nt]) > 1e-6:
                    output[target_nt] += (
                        coupling_matrix[target_nt, source_nt] *
                        delayed_fields[source_nt]
                    )

        return output

    def step(self, fields: np.ndarray) -> None:
        """
        Advance delay system by one timestep.

        Args:
            fields: Current NT fields to buffer
        """
        self.push_nt_state(fields)
        self._step_count += 1

    def set_region_distance(
        self,
        i: int,
        j: int,
        distance_mm: float
    ) -> None:
        """Set distance between regions."""
        self._distance_matrix.set_distance(i, j, distance_mm)

    def set_velocity(
        self,
        source: int,
        target: int,
        velocity_m_s: float
    ) -> None:
        """Set conduction velocity for a pathway."""
        self._velocities[source, target] = velocity_m_s
        self._velocities[target, source] = velocity_m_s

    def get_delay_ms(self, source: int, target: int) -> float:
        """Get transmission delay between regions."""
        return self._distance_matrix.get_delay_ms(
            source, target,
            self._velocities[source, target]
        )

    def update_velocity_plasticity(
        self,
        source: int,
        target: int,
        activity: float
    ) -> None:
        """
        Update conduction velocity based on activity (myelination plasticity).

        High activity increases myelination → faster conduction.

        Args:
            source: Source region
            target: Target region
            activity: Activity level (affects velocity change)
        """
        if not self.config.delay_plasticity:
            return

        current = self._velocities[source, target]

        # Activity increases velocity (up to biological max ~100 m/s)
        max_velocity = 100.0
        change = self.config.plasticity_rate * activity * (max_velocity - current)

        self._velocities[source, target] += change
        self._velocities[target, source] = self._velocities[source, target]

    def reset(self) -> None:
        """Reset all delay buffers."""
        for buffer in self._nt_buffers.values():
            buffer.clear()
        for buffer in self._region_buffers.values():
            buffer.clear()
        self._step_count = 0

    def get_state(self) -> DelayState:
        """Get current delay system state."""
        all_delays = []
        for nt_idx in range(6):
            all_delays.append(self.config.get_synaptic_delay(nt_idx))

        if self.n_regions > 1:
            for i in range(self.n_regions):
                for j in range(self.n_regions):
                    if i != j:
                        all_delays.append(self.get_delay_ms(i, j))

        return DelayState(
            total_buffered_ms=self._step_count * self.config.dt_ms,
            active_delays=len(all_delays),
            mean_delay_ms=float(np.mean(all_delays)) if all_delays else 0.0,
            max_active_delay_ms=float(np.max(all_delays)) if all_delays else 0.0
        )

    def get_stats(self) -> dict:
        """Get delay system statistics."""
        state = self.get_state()
        return {
            "step_count": self._step_count,
            "buffer_size": self.config.buffer_size,
            "max_delay_ms": self.config.max_delay_ms,
            "n_regions": self.n_regions,
            "grid_shape": self.grid_shape,
            "total_buffered_ms": state.total_buffered_ms,
            "active_delays": state.active_delays,
            "mean_delay_ms": state.mean_delay_ms,
            "max_active_delay_ms": state.max_active_delay_ms,
            "synaptic_delays": {
                "da": self.config.synaptic_delay_da,
                "5ht": self.config.synaptic_delay_5ht,
                "ach": self.config.synaptic_delay_ach,
                "ne": self.config.synaptic_delay_ne,
                "gaba": self.config.synaptic_delay_gaba,
                "glu": self.config.synaptic_delay_glu,
            },
            "plasticity_enabled": self.config.delay_plasticity,
        }


class DelayDifferentialOperator:
    """
    Implements delay differential equation (DDE) terms for neural field.

    Converts the standard PDE:
        ∂U/∂t = f(U(t))

    To a delay differential equation:
        ∂U/∂t = f(U(t), U(t-τ₁), U(t-τ₂), ...)

    Where τᵢ are transmission delays.
    """

    def __init__(
        self,
        delay_system: TransmissionDelaySystem,
        delay_weights: np.ndarray | None = None
    ):
        """
        Initialize delay differential operator.

        Args:
            delay_system: Transmission delay system
            delay_weights: Weights for delayed terms [6, 6] (default: identity)
        """
        self.delay_system = delay_system

        if delay_weights is not None:
            self.delay_weights = delay_weights
        else:
            # Default: each NT influenced by its own delayed version
            self.delay_weights = np.eye(6, dtype=np.float32) * 0.1

    def compute_delay_term(
        self,
        current_fields: np.ndarray
    ) -> np.ndarray:
        """
        Compute the delay contribution to field dynamics.

        This term represents the effect of past activity on
        current dynamics (e.g., feedback loops with delay).

        Args:
            current_fields: Current NT fields [6, *grid_shape]

        Returns:
            Delay contribution [6, *grid_shape]
        """
        # Get delayed fields with NT-specific delays
        delayed = self.delay_system.get_all_delayed_nts()

        # Compute difference from current (delay-induced dynamics)
        delay_term = np.zeros_like(current_fields)

        for target in range(6):
            for source in range(6):
                weight = self.delay_weights[target, source]
                if abs(weight) > 1e-6:
                    # Delayed feedback: pulls toward delayed state
                    delay_term[target] += weight * (delayed[source] - current_fields[target])

        return delay_term

    def step(self, fields: np.ndarray) -> np.ndarray:
        """
        Advance delay operator and return delay term.

        Args:
            fields: Current NT fields

        Returns:
            Delay contribution to add to dynamics
        """
        self.delay_system.step(fields)
        return self.compute_delay_term(fields)


# Convenience functions
def create_delay_system(
    n_regions: int = 1,
    grid_size: int = 16,
    max_delay_ms: float = 100.0
) -> TransmissionDelaySystem:
    """
    Create a transmission delay system.

    Args:
        n_regions: Number of brain regions
        grid_size: Spatial grid size
        max_delay_ms: Maximum delay to support

    Returns:
        Configured TransmissionDelaySystem
    """
    config = DelayConfig(max_delay_ms=max_delay_ms)
    return TransmissionDelaySystem(
        config=config,
        n_regions=n_regions,
        grid_shape=(grid_size,)
    )


def compute_axonal_delay(
    distance_mm: float,
    fiber_type: FiberType = FiberType.CORTICAL_LONG
) -> float:
    """
    Compute axonal conduction delay.

    Args:
        distance_mm: Distance in millimeters
        fiber_type: Type of axon fiber

    Returns:
        Delay in milliseconds
    """
    # Velocities in m/s
    velocities = {
        FiberType.C_UNMYELINATED: 1.0,
        FiberType.A_DELTA: 15.0,
        FiberType.A_BETA: 50.0,
        FiberType.A_ALPHA: 90.0,
        FiberType.CORTICAL_LOCAL: 1.0,
        FiberType.CORTICAL_LONG: 5.0,
    }

    velocity = velocities.get(fiber_type, 5.0)
    distance_m = distance_mm / 1000.0
    delay_s = distance_m / velocity
    return delay_s * 1000.0  # Convert to ms
