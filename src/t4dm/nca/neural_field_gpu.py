"""
GPU-Accelerated Neural Field PDE Solver.

PyTorch-based implementation of the neural field solver for GPU acceleration.
This module provides significant speedups over the NumPy/SciPy implementation
when running on CUDA-capable hardware.

Core Equation:
    dU(x,t)/dt = -alpha*U + D*nabla^2(U) + S(x,t) + C(U)

where:
    U = neurotransmitter concentration field
    alpha = decay rate (NT-specific)
    D = diffusion coefficient
    S = external stimulus
    C = coupling function

This implementation uses:
- PyTorch tensor operations for GPU parallelism
- Efficient convolution-based Laplacian computation
- Batched operations for multiple NT fields simultaneously
- Memory-efficient in-place operations where possible
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from t4dm.nca.neural_field import NeuralFieldConfig

logger = logging.getLogger(__name__)


class BoundaryCondition(str, Enum):
    """Boundary condition types for the PDE solver."""

    NO_FLUX = "no_flux"  # Neumann: dU/dn = 0
    PERIODIC = "periodic"  # Wrap around
    DIRICHLET = "dirichlet"  # Fixed value at boundary


class IntegrationMethod(str, Enum):
    """Numerical integration methods."""

    EULER = "euler"  # Forward Euler (explicit)
    SEMI_IMPLICIT = "semi_implicit"  # Semi-implicit for stability
    RK4 = "rk4"  # Runge-Kutta 4th order
    ADAMS_BASHFORTH = "adams_bashforth"  # Multi-step method


@dataclass
class GPUFieldConfig:
    """Configuration for GPU neural field solver."""

    # Grid parameters
    spatial_dims: int = 1
    grid_size: int = 32
    dx: float = 1.0

    # Time parameters
    dt: float = 0.001
    dt_min: float = 0.0001
    dt_max: float = 0.01

    # Boundary conditions
    boundary: BoundaryCondition = BoundaryCondition.NO_FLUX

    # Integration
    method: IntegrationMethod = IntegrationMethod.SEMI_IMPLICIT
    adaptive_dt: bool = True

    # Stability
    max_concentration: float = 1.0
    min_concentration: float = 0.0
    stability_threshold: float = 0.5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32


class LaplacianKernel(nn.Module):
    """
    Efficient Laplacian kernel using convolution.

    Computes the discrete Laplacian (nabla^2) using:
    - 1D: [-1, 2, -1] / dx^2
    - 2D: [[0,-1,0],[-1,4,-1],[0,-1,0]] / dx^2
    - 3D: 3D stencil with 6-point connectivity
    """

    def __init__(
        self,
        spatial_dims: int,
        dx: float = 1.0,
        boundary: BoundaryCondition = BoundaryCondition.NO_FLUX,
    ):
        """
        Initialize Laplacian kernel.

        Args:
            spatial_dims: Number of spatial dimensions (1, 2, or 3)
            dx: Grid spacing
            boundary: Boundary condition type
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.dx = dx
        self.boundary = boundary

        # Create Laplacian kernel
        kernel = self._create_kernel()
        self.register_buffer("kernel", kernel)

        # Padding mode based on boundary condition
        if boundary == BoundaryCondition.PERIODIC:
            self.padding_mode = "circular"
        else:
            self.padding_mode = "replicate"  # Approximates no-flux

    def _create_kernel(self) -> Tensor:
        """Create the Laplacian stencil kernel."""
        if self.spatial_dims == 1:
            # 1D Laplacian: d2u/dx2 ≈ (u[i-1] - 2u[i] + u[i+1]) / dx^2
            kernel = torch.tensor([1.0, -2.0, 1.0]) / (self.dx ** 2)
            kernel = kernel.view(1, 1, 3)

        elif self.spatial_dims == 2:
            # 2D Laplacian (5-point stencil)
            kernel = torch.tensor([
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0, 1.0, 0.0],
            ]) / (self.dx ** 2)
            kernel = kernel.view(1, 1, 3, 3)

        elif self.spatial_dims == 3:
            # 3D Laplacian (7-point stencil)
            kernel = torch.zeros(3, 3, 3)
            kernel[1, 1, 0] = 1.0
            kernel[1, 1, 2] = 1.0
            kernel[1, 0, 1] = 1.0
            kernel[1, 2, 1] = 1.0
            kernel[0, 1, 1] = 1.0
            kernel[2, 1, 1] = 1.0
            kernel[1, 1, 1] = -6.0
            kernel = kernel / (self.dx ** 2)
            kernel = kernel.view(1, 1, 3, 3, 3)

        else:
            raise ValueError(f"Unsupported spatial_dims: {self.spatial_dims}")

        return kernel

    def forward(self, field: Tensor) -> Tensor:
        """
        Compute Laplacian of the field.

        Args:
            field: Input field (batch, channels, *spatial_dims)

        Returns:
            Laplacian (batch, channels, *spatial_dims)
        """
        # Ensure proper shape
        if field.dim() == self.spatial_dims:
            field = field.unsqueeze(0).unsqueeze(0)
        elif field.dim() == self.spatial_dims + 1:
            field = field.unsqueeze(0)

        # Apply padding
        if self.spatial_dims == 1:
            padded = F.pad(field, (1, 1), mode=self.padding_mode)
            # Apply convolution
            laplacian = F.conv1d(padded, self.kernel.expand(field.size(1), -1, -1), groups=field.size(1))

        elif self.spatial_dims == 2:
            padded = F.pad(field, (1, 1, 1, 1), mode=self.padding_mode)
            laplacian = F.conv2d(padded, self.kernel.expand(field.size(1), -1, -1, -1), groups=field.size(1))

        elif self.spatial_dims == 3:
            padded = F.pad(field, (1, 1, 1, 1, 1, 1), mode=self.padding_mode)
            laplacian = F.conv3d(padded, self.kernel.expand(field.size(1), -1, -1, -1, -1), groups=field.size(1))

        return laplacian.squeeze(0)


class NeuralFieldGPU(nn.Module):
    """
    GPU-accelerated neural field PDE solver.

    This class provides the same functionality as NeuralFieldSolver but
    implemented in PyTorch for GPU acceleration.
    """

    # NT indices (same as CPU implementation)
    DA = 0  # Dopamine
    HT5 = 1  # Serotonin
    ACH = 2  # Acetylcholine
    NE = 3  # Norepinephrine
    GABA = 4  # GABA
    GLU = 5  # Glutamate
    NUM_NTS = 6

    def __init__(
        self,
        config: GPUFieldConfig | None = None,
        coupling_matrix: Tensor | None = None,
    ):
        """
        Initialize GPU neural field solver.

        Args:
            config: Solver configuration
            coupling_matrix: 6x6 NT coupling matrix (optional)
        """
        super().__init__()
        self.config = config or GPUFieldConfig()

        # Create Laplacian kernel
        self.laplacian = LaplacianKernel(
            self.config.spatial_dims,
            self.config.dx,
            self.config.boundary,
        )

        # NT-specific parameters
        # Decay rates (1/s)
        alphas = torch.tensor([
            10.0,   # DA: ~100ms timescale
            2.0,    # 5-HT: ~500ms timescale
            20.0,   # ACh: ~50ms timescale
            5.0,    # NE: ~200ms timescale
            100.0,  # GABA: ~10ms timescale
            200.0,  # Glu: ~5ms timescale
        ], dtype=self.config.dtype)

        # Diffusion coefficients (mm^2/s)
        diffusions = torch.tensor([
            0.1,    # DA
            0.2,    # 5-HT
            0.05,   # ACh
            0.15,   # NE
            0.03,   # GABA
            0.02,   # Glu
        ], dtype=self.config.dtype)

        # Register as buffers
        self.register_buffer("alphas", alphas)
        self.register_buffer("diffusions", diffusions)

        # Coupling matrix (6x6)
        if coupling_matrix is not None:
            self.register_buffer("coupling", coupling_matrix)
        else:
            # Default: no coupling
            self.register_buffer("coupling", torch.zeros(6, 6, dtype=self.config.dtype))

        # Initialize state
        self._init_fields()

        # Tracking
        self._time = 0.0
        self._step_count = 0
        self._current_dt = self.config.dt

        logger.info(
            f"NeuralFieldGPU initialized: "
            f"dims={self.config.spatial_dims}, "
            f"grid={self.config.grid_size}, "
            f"device={self.config.device}"
        )

    def _init_fields(self) -> None:
        """Initialize NT concentration fields at baseline."""
        shape = (self.NUM_NTS, *([self.config.grid_size] * self.config.spatial_dims))
        fields = torch.full(
            shape, 0.5,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        self.register_buffer("fields", fields)

    @property
    def device(self) -> torch.device:
        """Get the device of the fields."""
        return self.fields.device

    def to_device(self, device: str | torch.device) -> "NeuralFieldGPU":
        """Move solver to specified device."""
        self.to(device)
        return self

    def _compute_diffusion(self) -> Tensor:
        """
        Compute diffusion term D * nabla^2(U) for all NT fields.

        Returns:
            Diffusion contribution (6, *spatial_dims)
        """
        # Compute Laplacian for each NT field
        laplacian = self.laplacian(self.fields.unsqueeze(0)).squeeze(0)

        # Scale by diffusion coefficients
        # Reshape diffusions for broadcasting: (6, 1, 1, ...)
        diff_shape = (6,) + (1,) * self.config.spatial_dims
        diffusions = self.diffusions.view(*diff_shape)

        return diffusions * laplacian

    def _compute_decay(self) -> Tensor:
        """
        Compute decay term -alpha * U.

        Returns:
            Decay contribution (6, *spatial_dims)
        """
        # Reshape alphas for broadcasting
        alpha_shape = (6,) + (1,) * self.config.spatial_dims
        alphas = self.alphas.view(*alpha_shape)

        return -alphas * self.fields

    def _compute_coupling(self) -> Tensor:
        """
        Compute coupling term C(U).

        Returns:
            Coupling contribution (6, *spatial_dims)
        """
        # Get mean concentrations for coupling
        mean_state = self.get_mean_state()  # (6,)

        # Apply coupling matrix: C * U_mean
        coupling_effect = self.coupling @ mean_state  # (6,)

        # Broadcast to spatial dimensions
        coupling_shape = (6,) + (1,) * self.config.spatial_dims
        coupling_field = coupling_effect.view(*coupling_shape)
        coupling_field = coupling_field.expand_as(self.fields)

        return coupling_field

    def _compute_lateral_inhibition(self) -> Tensor:
        """
        Compute GABA-mediated lateral inhibition.

        Returns:
            Lateral inhibition contribution (6, *spatial_dims)
        """
        # Get GABA and Glu fields
        gaba = self.fields[self.GABA]
        glu = self.fields[self.GLU]

        # Apply Gaussian smoothing to GABA for surround effect
        if self.config.spatial_dims == 1:
            # Simple averaging for 1D
            kernel = torch.tensor([0.25, 0.5, 0.25], device=self.device, dtype=self.config.dtype)
            kernel = kernel.view(1, 1, 3)
            gaba_padded = F.pad(gaba.unsqueeze(0).unsqueeze(0), (1, 1), mode="replicate")
            gaba_spread = F.conv1d(gaba_padded, kernel).squeeze()

        elif self.config.spatial_dims == 2:
            # Gaussian kernel for 2D
            kernel = torch.tensor([
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625],
            ], device=self.device, dtype=self.config.dtype)
            kernel = kernel.view(1, 1, 3, 3)
            gaba_padded = F.pad(gaba.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="replicate")
            gaba_spread = F.conv2d(gaba_padded, kernel).squeeze()

        else:
            # Skip for 3D (computational cost)
            gaba_spread = gaba

        # Inhibition: higher GABA reduces Glu
        inhibition_strength = 0.3
        inhibition = inhibition_strength * gaba_spread * glu

        # Create output (only affects Glu)
        lateral = torch.zeros_like(self.fields)
        lateral[self.GLU] = -inhibition

        return lateral

    def step(
        self,
        stimulus: Tensor | None = None,
        dt: float | None = None,
    ) -> Tensor:
        """
        Advance neural field by one time step.

        Args:
            stimulus: External input to NT fields (6, *spatial_dims)
            dt: Override time step

        Returns:
            Spatially-averaged NT state (6,)
        """
        dt = dt or self._current_dt

        # Compute all terms
        diffusion = self._compute_diffusion()
        coupling = self._compute_coupling()
        lateral = self._compute_lateral_inhibition()

        # External stimulus
        if stimulus is not None:
            stim = stimulus.to(self.device)
        else:
            stim = torch.zeros_like(self.fields)

        # Choose integration method
        if self.config.method == IntegrationMethod.EULER:
            # Forward Euler (explicit)
            decay = self._compute_decay()
            dU = decay + diffusion + coupling + lateral + stim
            self.fields = self.fields + dt * dU

        elif self.config.method == IntegrationMethod.SEMI_IMPLICIT:
            # Semi-implicit: decay treated implicitly
            explicit_terms = diffusion + coupling + lateral + stim

            # U^(n+1) = (U^n + dt*explicit) / (1 + dt*alpha)
            alpha_shape = (6,) + (1,) * self.config.spatial_dims
            alphas = self.alphas.view(*alpha_shape)
            numerator = self.fields + dt * explicit_terms
            denominator = 1.0 + dt * alphas
            self.fields = numerator / denominator

        elif self.config.method == IntegrationMethod.RK4:
            # 4th order Runge-Kutta
            self.fields = self._rk4_step(stimulus, dt)

        # Clamp to biological range
        self.fields = self.fields.clamp(
            self.config.min_concentration,
            self.config.max_concentration,
        )

        # Adaptive timestepping
        if self.config.adaptive_dt:
            self._adapt_timestep()

        # Update tracking
        self._time += dt
        self._step_count += 1

        return self.get_mean_state()

    def _rk4_step(self, stimulus: Tensor | None, dt: float) -> Tensor:
        """Perform one RK4 integration step."""

        def compute_derivative(fields: Tensor) -> Tensor:
            # Temporarily set fields
            old_fields = self.fields
            self.fields = fields

            decay = self._compute_decay()
            diffusion = self._compute_diffusion()
            coupling = self._compute_coupling()
            lateral = self._compute_lateral_inhibition()

            # Restore
            self.fields = old_fields

            dU = decay + diffusion + coupling + lateral
            if stimulus is not None:
                dU = dU + stimulus.to(self.device)
            return dU

        k1 = compute_derivative(self.fields)
        k2 = compute_derivative(self.fields + 0.5 * dt * k1)
        k3 = compute_derivative(self.fields + 0.5 * dt * k2)
        k4 = compute_derivative(self.fields + dt * k3)

        return self.fields + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _adapt_timestep(self) -> None:
        """Adapt timestep based on field changes."""
        # Compute maximum rate of change
        diffusion = self._compute_diffusion()
        max_rate = diffusion.abs().max().item()

        # CFL condition
        cfl_dt = self.config.dx ** 2 / (2 * self.diffusions.max().item() * self.config.spatial_dims + 1e-10)

        # Adjust dt
        if max_rate * self._current_dt > 0.5:
            self._current_dt = max(self._current_dt * 0.5, self.config.dt_min)
        elif max_rate * self._current_dt < 0.1:
            self._current_dt = min(self._current_dt * 1.2, self.config.dt_max, cfl_dt)

    def get_mean_state(self) -> Tensor:
        """
        Get spatially-averaged NT state.

        Returns:
            Mean concentrations (6,)
        """
        # Average over spatial dimensions
        dims = tuple(range(1, self.fields.dim()))
        return self.fields.mean(dim=dims)

    def get_field(self, nt_index: int) -> Tensor:
        """
        Get spatial field for specific neurotransmitter.

        Args:
            nt_index: NT index (0=DA, 1=5HT, 2=ACh, 3=NE, 4=GABA, 5=Glu)

        Returns:
            Spatial field (*spatial_dims)
        """
        return self.fields[nt_index].clone()

    def inject_stimulus(
        self,
        nt_index: int,
        magnitude: float,
        location: tuple[int, ...] | None = None,
    ) -> None:
        """
        Inject external stimulus into specific NT field.

        Args:
            nt_index: NT index
            magnitude: Stimulus strength [-1, 1]
            location: Spatial location (None = global)
        """
        if location is None:
            self.fields[nt_index] += magnitude
        else:
            self.fields[nt_index][location] += magnitude

        # Clamp
        self.fields = self.fields.clamp(
            self.config.min_concentration,
            self.config.max_concentration,
        )

    def inject_rpe(self, rpe: float, scale: float = 0.3) -> None:
        """
        Inject reward prediction error into DA field.

        Args:
            rpe: Reward prediction error [-1, 1]
            scale: Scale factor
        """
        self.inject_stimulus(self.DA, rpe * scale)

    def reset(self) -> None:
        """Reset fields to baseline."""
        self._init_fields()
        self._time = 0.0
        self._step_count = 0
        self._current_dt = self.config.dt

    def get_stats(self) -> dict:
        """Get solver statistics."""
        return {
            "time": self._time,
            "step_count": self._step_count,
            "current_dt": self._current_dt,
            "mean_state": self.get_mean_state().tolist(),
            "device": str(self.device),
        }

    @classmethod
    def from_cpu_config(cls, cpu_config: NeuralFieldConfig) -> "NeuralFieldGPU":
        """
        Create GPU solver from CPU solver config.

        Args:
            cpu_config: NeuralFieldConfig from neural_field.py

        Returns:
            Configured NeuralFieldGPU instance
        """
        # Map boundary condition
        boundary_map = {
            "no-flux": BoundaryCondition.NO_FLUX,
            "periodic": BoundaryCondition.PERIODIC,
            "dirichlet": BoundaryCondition.DIRICHLET,
        }
        boundary = boundary_map.get(cpu_config.boundary_type, BoundaryCondition.NO_FLUX)

        # Map integration method
        method_map = {
            "euler": IntegrationMethod.EULER,
            "strang": IntegrationMethod.SEMI_IMPLICIT,
        }
        method = method_map.get(cpu_config.splitting_method, IntegrationMethod.SEMI_IMPLICIT)

        gpu_config = GPUFieldConfig(
            spatial_dims=cpu_config.spatial_dims,
            grid_size=cpu_config.grid_size,
            dx=cpu_config.dx,
            dt=cpu_config.dt,
            dt_min=cpu_config.dt_min,
            dt_max=cpu_config.dt_max,
            boundary=boundary,
            method=method,
            adaptive_dt=cpu_config.adaptive_timestepping,
            max_concentration=cpu_config.max_concentration,
            min_concentration=cpu_config.min_concentration,
        )

        return cls(gpu_config)


# Utility function for benchmarking
def benchmark_gpu_vs_cpu(
    num_steps: int = 1000,
    grid_size: int = 64,
    spatial_dims: int = 2,
) -> dict:
    """
    Benchmark GPU vs CPU neural field solver.

    Args:
        num_steps: Number of simulation steps
        grid_size: Grid size per dimension
        spatial_dims: Number of spatial dimensions

    Returns:
        Dictionary with timing results
    """
    import time

    results = {}

    # GPU solver
    if torch.cuda.is_available():
        gpu_config = GPUFieldConfig(
            spatial_dims=spatial_dims,
            grid_size=grid_size,
            device="cuda",
        )
        gpu_solver = NeuralFieldGPU(gpu_config)

        # Warmup
        for _ in range(10):
            gpu_solver.step()

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_steps):
            gpu_solver.step()
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start

        results["gpu_time"] = gpu_time
        results["gpu_steps_per_sec"] = num_steps / gpu_time

    # CPU solver
    cpu_config = GPUFieldConfig(
        spatial_dims=spatial_dims,
        grid_size=grid_size,
        device="cpu",
    )
    cpu_solver = NeuralFieldGPU(cpu_config)

    start = time.perf_counter()
    for _ in range(num_steps):
        cpu_solver.step()
    cpu_time = time.perf_counter() - start

    results["cpu_time"] = cpu_time
    results["cpu_steps_per_sec"] = num_steps / cpu_time

    if "gpu_time" in results:
        results["speedup"] = cpu_time / gpu_time

    return results


__all__ = [
    "NeuralFieldGPU",
    "GPUFieldConfig",
    "BoundaryCondition",
    "IntegrationMethod",
    "LaplacianKernel",
    "benchmark_gpu_vs_cpu",
]
